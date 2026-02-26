# RAG 최적화 상세 구현 가이드

## 파트 1: 단기 최적화 (즉시 실행)

### 1. DB 연결 풀 확대

**파일:** `backend/app/core/database.py`

```python
# Before
engine = create_async_engine(
    settings.DATABASE_URL_ASYNC,
    echo=settings.DEBUG,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
)

# After
engine = create_async_engine(
    settings.DATABASE_URL_ASYNC,
    echo=settings.DEBUG,
    pool_size=10,           # ← 5 → 10
    max_overflow=20,        # ← 10 → 20
    pool_pre_ping=True,
    pool_recycle=3600,      # ← 추가: 1시간마다 연결 재활용
)

# Sync engine도 동일 적용
sync_engine = create_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=3600,
)
```

**테스트:**
```bash
cd backend
uv run python -c "
from app.core.database import engine, sync_engine
print(f'Async pool size: {engine.pool.pool_size}')
print(f'Sync pool size: {sync_engine.pool.pool_size}')
"
```

**예상 성능 변화:**
- 동시 연결: 5→10 (2배 증가)
- 응답 시간: -15-20%

---

### 2. LanceDB 벡터 인덱싱 활성화

**Step 1: .env 파일 수정**

```bash
# backend/.env
LANCEDB_INDEX_TYPE=IVF_FLAT
LANCEDB_NPROBES=20  # 기본값 40에서 20으로 축소 (속도 우선)
```

**Step 2: 서버 시작 시 자동 인덱싱 (이미 구현됨)**

`backend/app/main.py`의 lifespan에서 자동으로 처리:

```python
# app/main.py line 57-73 참조
if settings.LANCEDB_INDEX_TYPE:
    try:
        lance_store = LanceDBStore()
        created = lance_store.create_vector_index(settings.LANCEDB_INDEX_TYPE)
        if created:
            logger.info("LanceDB 벡터 인덱스 생성 완료: %s", settings.LANCEDB_INDEX_TYPE)
    except Exception as e:
        logger.error("LanceDB 벡터 인덱스 생성 실패 (brute-force로 동작): %s", e)
```

**Step 3: 서버 시작**

```bash
# 첫 시작 시 인덱스 생성 (1-5분 소요, 호스트 성능에 따라)
cd backend
uv run uvicorn app.main:app --reload

# 로그 확인
# [INFO] LanceDB 벡터 인덱스 생성 완료: IVF_FLAT
```

**Step 4: nprobes 조정 검증**

```python
# backend/tests/performance/test_lancedb_index.py
import time
from app.tools.vectorstore.lancedb import LanceDBStore

def test_search_with_index():
    store = LanceDBStore()
    query_vec = [0.1] * 1024  # 임베딩 벡터 (1024 차원)

    start = time.perf_counter()
    results = store.search(query_vec, n_results=10)
    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"검색 시간: {elapsed_ms:.1f}ms")
    print(f"결과 수: {len(results['ids'][0]) if results['ids'] else 0}")

    # 기대: 30-50ms (이전: 100-200ms)
    assert elapsed_ms < 60, f"검색 시간 초과: {elapsed_ms}ms"

if __name__ == "__main__":
    test_search_with_index()
```

**Recall 모니터링 (중요):**

IVF 인덱스 사용으로 recall이 약간 감소할 수 있습니다. 모니터링이 필수입니다:

```python
# backend/evaluation/metrics.py에서 기존 평가 시스템 활용
# uv run python -m evaluation.evaluate_search --metric recall
```

**Fallback 전략:**

IVF 인덱스로 recall이 목표 미달이면 nprobes 증가:
```bash
# backend/.env
LANCEDB_NPROBES=30  # 20 → 30 (정확도 향상, 속도 약간 저하)
```

---

### 3. LRU 캐싱 (결과 캐시)

**파일 생성:** `backend/app/services/rag/cache.py`

```python
"""
쿼리 결과 LRU 캐시

- 상위 1,000개 쿼리 캐싱
- TTL: 5분 (기본값)
- 캐시 키: 쿼리 + doc_type 기반 해시
"""

import hashlib
import json
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class QueryResultCache:
    """쿼리 검색 결과 LRU 캐시"""

    def __init__(self, maxsize: int = 1000, ttl_seconds: int = 300):
        """
        Args:
            maxsize: 최대 캐시 항목 수
            ttl_seconds: 캐시 만료 시간 (초)
        """
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        self.cache: dict[str, tuple[dict, float]] = {}  # {key: (result, timestamp)}
        self.hit_count = 0
        self.miss_count = 0

    def _get_key(self, query: str, doc_type: Optional[str]) -> str:
        """쿼리와 doc_type으로 캐시 키 생성"""
        # 정규화: 공백 정리, 소문자화
        normalized_query = " ".join(query.lower().split())
        data = json.dumps(
            {"query": normalized_query, "doc_type": doc_type},
            sort_keys=True,
        )
        return hashlib.md5(data.encode()).hexdigest()

    def get(self, query: str, doc_type: Optional[str]) -> Optional[dict]:
        """캐시에서 결과 조회"""
        key = self._get_key(query, doc_type)
        if key in self.cache:
            result, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl_seconds:
                self.hit_count += 1
                logger.debug(f"캐시 히트: {query[:50]}... (key={key})")
                return result
            else:
                # 만료된 항목 삭제
                del self.cache[key]
                self.miss_count += 1
                return None
        self.miss_count += 1
        return None

    def set(self, query: str, doc_type: Optional[str], result: dict) -> None:
        """캐시에 결과 저장"""
        key = self._get_key(query, doc_type)

        # 캐시 크기 초과 시 LRU 항목 제거
        if len(self.cache) >= self.maxsize:
            oldest_key = min(self.cache, key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
            logger.debug(f"LRU 항목 제거: {oldest_key}")

        self.cache[key] = (result, time.time())
        logger.debug(f"캐시 저장: {query[:50]}... (항목 수={len(self.cache)})")

    def clear(self) -> None:
        """캐시 전체 삭제"""
        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0

    def get_stats(self) -> dict:
        """캐시 통계 반환"""
        total = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total * 100) if total > 0 else 0
        return {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": f"{hit_rate:.1f}%",
            "size": len(self.cache),
            "maxsize": self.maxsize,
        }


# 글로벌 캐시 인스턴스
_cache_instance: Optional[QueryResultCache] = None


def get_cache() -> QueryResultCache:
    """글로벌 캐시 인스턴스 반환 (lazy initialization)"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = QueryResultCache(maxsize=1000, ttl_seconds=300)
    return _cache_instance
```

**파일 수정:** `backend/app/services/rag/pipeline.py`

```python
# 추가 import
from app.services.rag.cache import get_cache

# RAGPipeline 클래스에 캐시 사용 추가
class RAGPipeline:
    def __init__(self):
        self.cache = get_cache()

    async def execute_async(
        self,
        query: str,
        config: Optional[PipelineConfig] = None,
    ) -> PipelineResult:
        """파이프라인 실행 (비동기) - 캐시 지원"""
        config = config or PipelineConfig()

        # Step 0: 캐시 확인
        cached_result = self.cache.get(query, config.doc_type)
        if cached_result is not None:
            logger.info("캐시 히트: %s", query)
            return cached_result

        # Step 1-4: 기존 로직 (await asyncio.to_thread 사용)
        result = await asyncio.to_thread(self.execute, query, config)

        # Step 5: 캐시 저장
        self.cache.set(query, config.doc_type, result)

        return result
```

**API 엔드포인트 추가 (캐시 통계):**

```python
# backend/app/api/router/chat.py 또는 별도 admin 라우터
@router.get("/admin/cache-stats")
async def get_cache_stats() -> dict:
    """캐시 통계 조회"""
    from app.services.rag.cache import get_cache
    cache = get_cache()
    return cache.get_stats()

@router.post("/admin/cache-clear")
async def clear_cache() -> dict:
    """캐시 초기화"""
    from app.services.rag.cache import get_cache
    cache = get_cache()
    cache.clear()
    return {"message": "캐시가 초기화되었습니다"}
```

**테스트:**

```bash
# backend/tests/unit/test_rag_cache.py
import pytest
from app.services.rag.cache import QueryResultCache

def test_cache_hit():
    cache = QueryResultCache(maxsize=10, ttl_seconds=60)

    query = "손해배상 책임"
    doc_type = "precedent"
    result = {"documents": [...]}

    # 캐시 저장
    cache.set(query, doc_type, result)

    # 캐시 조회
    cached = cache.get(query, doc_type)
    assert cached == result
    assert cache.hit_count == 1

def test_cache_ttl():
    cache = QueryResultCache(maxsize=10, ttl_seconds=1)

    query = "손해배상"
    result = {"documents": [...]}

    cache.set(query, None, result)
    assert cache.get(query, None) is not None

    # 1초 대기 후 만료 확인
    import time
    time.sleep(1.1)
    assert cache.get(query, None) is None
```

**성능 검증:**

```bash
cd backend

# 캐시 히트율 테스트
uv run python -c "
from app.services.rag.cache import get_cache
cache = get_cache()

# 동일 쿼리 100번 호출
for i in range(100):
    cache.get('손해배상', 'precedent')

stats = cache.get_stats()
print(f\"Hit Rate: {stats['hit_rate']}\")
print(f\"Cache Size: {stats['size']}\")
"
```

---

## 파트 2: 중기 최적화 (다음주)

### 4. 벡터/FTS 병렬 검색

**파일 수정:** `backend/app/services/rag/retrieval.py`

```python
# 현재 (순차 실행)
def search_without_content(
    query: str,
    n_results: int = 5,
    doc_type: Optional[str] = None,
) -> list[dict[str, Any]]:
    vector_results = _search_vector_ids(query, vector_fetch, doc_type)

    if not settings.USE_HYBRID_SEARCH:
        return vector_results[:n_results]

    keyword_results = search_by_keyword(query, n_results=vector_fetch, doc_type=doc_type)
    # ... RRF 병합

# 개선안 (비동기 병렬)
async def search_without_content_async(
    query: str,
    n_results: int = 5,
    doc_type: Optional[str] = None,
) -> list[dict[str, Any]]:
    """하이브리드 검색 (병렬 실행)"""
    vector_fetch = n_results * 3 if settings.USE_HYBRID_SEARCH else n_results

    # 벡터 검색 태스크
    vector_task = asyncio.to_thread(
        _search_vector_ids, query, vector_fetch, doc_type
    )

    # FTS 검색 태스크 (USE_HYBRID_SEARCH=true인 경우만)
    if not settings.USE_HYBRID_SEARCH:
        vector_results = await vector_task
        return vector_results[:n_results]

    from app.services.rag.keyword_search import (
        is_fts_available_sync,
        search_by_keyword,
    )

    if not is_fts_available_sync():
        vector_results = await vector_task
        return vector_results[:n_results]

    keyword_task = asyncio.to_thread(
        search_by_keyword, query, n_results=vector_fetch, doc_type=doc_type
    )

    # 병렬 실행: max(vector_time, keyword_time)
    vector_results, keyword_results = await asyncio.gather(
        vector_task, keyword_task, return_exceptions=True
    )

    # 예외 처리
    if isinstance(vector_results, Exception):
        logger.error("벡터 검색 실패: %s", vector_results)
        vector_results = []
    if isinstance(keyword_results, Exception):
        logger.error("FTS 검색 실패: %s", keyword_results)
        keyword_results = []

    if not keyword_results:
        return vector_results[:n_results]

    # RRF 병합 (기존 로직)
    from app.services.rag.fusion import reciprocal_rank_fusion

    vector_source_ids = _unique_source_ids(vector_results)
    keyword_source_ids = _unique_source_ids(keyword_results)
    fused_source_ids = reciprocal_rank_fusion(vector_source_ids, keyword_source_ids)

    vector_best = _best_doc_per_source(vector_results)
    keyword_best = _best_doc_per_source(keyword_results)

    merged: list[dict[str, Any]] = []
    for sid in fused_source_ids:
        if sid in vector_best:
            merged.append(vector_best[sid])
        elif sid in keyword_best:
            merged.append(keyword_best[sid])

        if len(merged) >= n_results:
            break

    return merged
```

**파일 수정:** `backend/app/services/rag/pipeline.py`

```python
# 기존 동기 버전은 유지, 비동기 버전만 추가

async def execute_async(
    self,
    query: str,
    config: Optional[PipelineConfig] = None,
) -> PipelineResult:
    """파이프라인 실행 (비동기) - 병렬 검색"""
    config = config or PipelineConfig()
    pipeline_start = time.monotonic()

    result = PipelineResult(original_query=query)
    metrics = result.metrics

    # 캐시 확인 (추가됨)
    cached_result = self.cache.get(query, config.doc_type)
    if cached_result is not None:
        return cached_result

    # Step 1: 쿼리 리라이팅 (변경 없음)
    queries = [query]
    if config.enable_rewrite:
        queries = rewrite_query(query, config.num_rewrite_queries, config.use_llm_rewrite)
        result.rewritten_queries = queries

    # Step 2: 병렬 검색 (변경됨)
    search_start = time.monotonic()

    all_documents: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    search_fn = (
        search_without_content_async  # ← 비동기 버전 사용
        if config.enable_rerank
        else search_relevant_documents_async
    )

    # 여러 쿼리 병렬 처리
    search_tasks = [
        search_fn(q, config.n_results, config.doc_type)
        for q in queries
    ]

    search_results_list = await asyncio.gather(*search_tasks)

    for docs in search_results_list:
        for doc in docs:
            doc_id = doc.get("metadata", {}).get("doc_id", "")
            if doc_id and doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_documents.append(doc)

    metrics.search_time_ms = (time.monotonic() - search_start) * 1000
    metrics.total_searched = len(all_documents)
    result.total_retrieved = len(all_documents)

    # Step 3-5: 기존 로직
    # ...
```

**벤치마크:**

```python
# backend/tests/performance/test_parallel_search.py
import asyncio
import time
from app.services.rag.pipeline import search_with_pipeline_async, PipelineConfig

async def benchmark_parallel():
    queries = [
        "손해배상 책임",
        "임금 체불",
        "부정거래",
    ]

    config = PipelineConfig(enable_rerank=True)

    start = time.perf_counter()
    results = await asyncio.gather(*[
        search_with_pipeline_async(q, config) for q in queries
    ])
    elapsed_sec = time.perf_counter() - start

    print(f"3개 쿼리 병렬 처리: {elapsed_sec:.2f}초")
    print(f"평균 시간: {elapsed_sec / len(queries):.2f}초/쿼리")
    # 기대: 3개 순차 1.2초 → 병렬 0.8초

if __name__ == "__main__":
    asyncio.run(benchmark_parallel())
```

---

### 5. 임베딩 모델 싱글톤 캐싱

**파일 수정:** `backend/app/services/rag/embedding.py`

```python
"""
임베딩 모델 관리

- 프로세스당 1회 로드 (thread-safe)
- 캐싱으로 요청마다 로드 오버헤드 제거
"""

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

_embedding_model_lock = threading.Lock()
_embedding_model_cache: Optional[object] = None


def _load_embedding_model() -> object:
    """임베딩 모델 로드"""
    from app.core.config import settings

    if not settings.USE_LOCAL_EMBEDDING:
        raise RuntimeError("USE_LOCAL_EMBEDDING=false, local embedding 불가")

    logger.info("임베딩 모델 로딩: %s", settings.LOCAL_EMBEDDING_MODEL)
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(settings.LOCAL_EMBEDDING_MODEL)
        logger.info("임베딩 모델 로드 완료")
        return model
    except Exception as e:
        logger.error("임베딩 모델 로드 실패: %s", e)
        raise


def get_embedding_model() -> object:
    """임베딩 모델 싱글톤 반환 (thread-safe)"""
    global _embedding_model_cache

    with _embedding_model_lock:
        if _embedding_model_cache is None:
            _embedding_model_cache = _load_embedding_model()

    return _embedding_model_cache


def create_query_embedding(query: str) -> list[float]:
    """쿼리 임베딩 생성"""
    model = get_embedding_model()
    embedding = model.encode(query, normalize_embeddings=True)
    return embedding.tolist()


# 기존 함수 (backward compatibility)
def check_embedding_model_availability() -> bool:
    """임베딩 모델 사용 가능 여부 확인"""
    # ... 기존 코드
```

**lifespan에서 미리 로드 (app/main.py 이미 구현됨):**

```python
# app/main.py line 33-39 참조
if model_available and settings.USE_LOCAL_EMBEDDING:
    logger.info("임베딩 모델을 미리 로드합니다...")
    try:
        model = get_local_model()  # ← get_embedding_model() 사용
        model.encode("warm-up", normalize_embeddings=True)
        logger.info("임베딩 모델 로드 + warm-up 완료")
    except Exception as e:
        logger.error("임베딩 모델 로드 실패: %s", e)
```

**성능 측정:**

```bash
# backend/tests/performance/test_embedding_cache.py
import time
from app.services.rag.embedding import create_query_embedding

# 첫 호출: 모델 로드 (~1-2초)
start = time.perf_counter()
vec1 = create_query_embedding("손해배상")
elapsed1 = (time.perf_counter() - start) * 1000
print(f"첫 호출: {elapsed1:.0f}ms")

# 두 번째 호출: 캐시 사용 (~10ms)
start = time.perf_counter()
vec2 = create_query_embedding("임금 체불")
elapsed2 = (time.perf_counter() - start) * 1000
print(f"두 번째 호출: {elapsed2:.0f}ms")

# 1,000번 호출 합계
start = time.perf_counter()
for i in range(1000):
    create_query_embedding(f"쿼리 {i}")
elapsed_total = (time.perf_counter() - start)
print(f"1,000개 쿼리: {elapsed_total:.2f}초 ({elapsed_total/1000*1000:.1f}ms/개)")
```

---

### 6. 쿼리 리라이팅 캐싱 (선택)

**파일 수정:** `backend/app/services/rag/query_rewrite.py`

```python
"""
쿼리 리라이팅 캐싱

- 동일 쿼리의 LLM 호출 감소
- 정규화: 공백 정리, 소문자화
"""

from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


@lru_cache(maxsize=500)
def rewrite_query_cached(
    query: str,
    num_queries: int = 3,
    use_llm: bool = True,
) -> tuple[str, ...]:
    """
    쿼리 리라이팅 (캐시 지원)

    Returns:
        리라이팅된 쿼리 튜플 (hashable for lru_cache)
    """
    # 정규화
    normalized = " ".join(query.lower().split())

    # 기존 rewrite_query 로직
    rewritten = rewrite_query(normalized, num_queries, use_llm)

    return tuple(rewritten)  # 튜플로 반환 (hashable)


# 기존 동기 버전
def rewrite_query(query: str, num_queries: int = 3, use_llm: bool = True) -> list[str]:
    """
    쿼리 리라이팅 (기존 로직)
    """
    # ... 기존 구현
```

**사용:**

```python
# pipeline.py에서
if config.enable_rewrite:
    # 캐시 지원 버전 사용
    queries_tuple = rewrite_query_cached(
        query, config.num_rewrite_queries, config.use_llm_rewrite
    )
    result.rewritten_queries = list(queries_tuple)
```

---

## 파트 3: 구현 체크리스트

### Phase 1 체크리스트 (단기)

- [ ] DB pool 파일 수정
  - [ ] pool_size: 5→10
  - [ ] max_overflow: 10→20
  - [ ] pool_recycle 추가
  - [ ] sync_engine도 동일 적용

- [ ] LanceDB 인덱싱
  - [ ] .env 파일에 LANCEDB_INDEX_TYPE=IVF_FLAT 추가
  - [ ] LANCEDB_NPROBES=20 설정
  - [ ] 서버 시작 및 인덱스 생성 확인
  - [ ] 로그에서 "벡터 인덱스 생성 완료" 확인

- [ ] LRU 캐싱
  - [ ] cache.py 파일 생성
  - [ ] pipeline.py 수정 (캐시 추가)
  - [ ] 테스트 작성 및 실행
  - [ ] 캐시 통계 API 추가

- [ ] 성능 테스트
  - [ ] 벤치마크 스크립트 실행
  - [ ] 응답 시간 기록 (250ms → 125ms 확인)
  - [ ] Recall@10 검증 (≥0.80)

### Phase 2 체크리스트 (중기)

- [ ] 병렬 검색
  - [ ] search_without_content_async 구현
  - [ ] execute_async 수정
  - [ ] 테스트 및 벤치마크

- [ ] 임베딩 모델 캐싱
  - [ ] get_embedding_model() 싱글톤 구현
  - [ ] create_query_embedding() 수정
  - [ ] 성능 테스트

- [ ] 쿼리 리라이팅 캐싱
  - [ ] @lru_cache 데코레이터 추가
  - [ ] 정규화 로직 추가
  - [ ] 테스트

- [ ] 통합 테스트
  - [ ] 모든 변경사항 함께 테스트
  - [ ] 응답 시간 기록 (125ms → 80ms 확인)

---

## 파트 4: 모니터링 및 검증

### 응답 시간 모니터링

```python
# backend/app/core/logging.py 수정 또는 확장
import logging
import time

class RAGMetricsFilter(logging.Filter):
    def filter(self, record):
        # 요청별 응답 시간 로깅
        if hasattr(record, 'elapsed_ms'):
            return True
        return False

# 로거 설정
rag_logger = logging.getLogger("app.services.rag")
rag_logger.addFilter(RAGMetricsFilter())
```

### 성능 대시보드 설정

```bash
# Prometheus 메트릭 (선택)
# backend/app/core/metrics.py
from prometheus_client import Histogram

search_duration = Histogram(
    'rag_search_duration_seconds',
    'RAG 검색 소요 시간',
    buckets=(0.05, 0.1, 0.2, 0.5, 1.0)
)

cache_hits = Counter(
    'rag_cache_hits_total',
    'RAG 캐시 히트 수'
)
```

### 테스트 자동화

```bash
# backend/scripts/performance_test.sh
#!/bin/bash

echo "=== RAG 성능 테스트 ==="

# 단위 테스트
uv run pytest tests/unit/test_rag_cache.py -v

# 성능 벤치마크
uv run python tests/performance/benchmark_rag.py

# 결과 수집
echo "=== 테스트 완료 ==="
```

---

## 마무리

**다음 단계:**
1. 위 가이드를 따라 단기 최적화 3가지 구현
2. 성능 벤치마크 실행
3. 모니터링 설정
4. 결과 검증 후 PR 제출

**문의:**
- bottleneck 분석: `eval/CLAUDE.md` RAG 평가 시스템 참조
- 배포 전략: `docs/operations/backup-restore.md` 참조
