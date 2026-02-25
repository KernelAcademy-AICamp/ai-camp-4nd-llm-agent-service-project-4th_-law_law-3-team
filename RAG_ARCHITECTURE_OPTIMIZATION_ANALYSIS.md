# RAG 파이프라인 아키텍처 최적화 분석 보고서

## 1. 현재 아키텍처 분석

### 1.1 RAG 파이프라인 구조 (pipeline.py)

```
Step 1: 쿼리 리라이팅 (선택) ──LLM 호출 (동기)──> 20-50ms
         ↓ (순차)
Step 2: 하이브리드 검색 (병렬 가능) ────────> 벡터 + FTS 병렬 100-200ms
         ├─ 벡터 검색: asyncio.to_thread 사용
         └─ FTS 검색: 동기 호출
         ↓ (순차)
Step 3: 리랭킹 (선택) ─────────────────────>
         ├─ 요약 조회: LanceDB 배치 (50ms)
         ├─ 리랭킹: Cross-encoder (100-150ms)
         └─ 원문 조회: PostgreSQL 배치 (30-50ms)
         ↓
Step 4: 결과 포맷팅 (5ms)

전체 소요: 250-450ms (리랭킹 활성화 시)
```

**현재 문제점:**
- Step 1-4가 순차 실행 (병렬화 불가)
- `asyncio.to_thread`로 래핑되어도 실제 벡터 검색은 동기 (GIL)
- 쿼리 리라이팅이 선택사항이지만 활성화 시 전체 지연 +20-50ms

### 1.2 검색 계층 (retrieval.py)

**벡터 검색 (async to sync):**
```python
# 현재: asyncio.to_thread 사용
search_relevant_documents_async()
  → asyncio.to_thread(search_relevant_documents)
    → search_without_content()
      → _search_vector_ids()  # 동기 호출
```

**문제:**
- `_search_vector_ids` 내부에서 `create_query_embedding()` 호출 (임베딩 모델 로드)
- 각 요청마다 모델 초기화 오버헤드 (10-20ms)
- FTS와 벡터 검색이 순차 (search_without_content에서)

**강점:**
- 요약 조회와 원문 조회가 배치 처리
- source_id 단위 deduplication 최적화

### 1.3 LangGraph 구조 (graph.py)

```
START → router_node ──(Command)──> agent_node ──> END
```

**현재 상태:**
- 체크포인터: PostgreSQL (AsyncPostgresSaver) 지원
- 그래프는 단일 agent 선택 (병렬 실행 불가)
- lifespan에서 모델 미리 로드 (warm-up 패턴 사용)

**병목:**
- router_node에서 규칙 기반 라우팅 (LLM 비호출, 빠름)
- 하지만 각 에이전트 내 RAG 검색은 순차

### 1.4 데이터베이스 연결 (database.py)

```python
# Async 엔진
pool_size=5, max_overflow=10
# Sync 엔진 (RAG 서비스 용)
pool_size=5, max_overflow=10
```

**문제:**
- pool_size=5로 동시성 제한 (5개 연결 경합)
- RAG 검색 중 원문 조회 시 별도 sync 세션 사용 (연결 경합 심함)

### 1.5 LanceDB 설정 (lancedb.py + config.py)

```python
LANCEDB_MODE="local"  # 임베디드, 각 요청마다 python 내에서 검색
LANCEDB_INDEX_TYPE=""  # 기본 brute-force (전체 테이블 스캔)
LANCEDB_NPROBES=40  # IVF 사용 시 탐색 파티션
```

**문제:**
- Local 모드에서 각 요청이 단일 프로세스 내 검색 (CPU bound)
- Brute-force는 253,768 청크에서 O(n) 거리 계산 (100-200ms)
- nprobes=40은 정확도 우선 (성능 희생)

---

## 2. 최적화 제안

### 표: 최적화 방안 (우선순위 순)

| # | 최적화 | 예상 절감 | 난이도 | 리스크 | 설명 | 단계 |
|----|---------|----------|--------|--------|------|------|
| 1 | DB 연결 풀 확대 | 15-20% | 낮음 | 낮음 | pool_size: 5→10, max_overflow: 10→20 | 단기 |
| 2 | LanceDB IVF 인덱싱 | 30-40% | 낮음 | 낮음 | LANCEDB_INDEX_TYPE="IVF_FLAT", 1회 빌드 | 단기 |
| 3 | nprobes 동적 조정 | 15-25% | 낮음 | 중간 | 40→20 (속도), fallback: recall 모니터링 | 단기 |
| 4 | 요청 결과 캐싱 (LRU) | 40-60% | 중간 | 중간 | 상위 1,000개 쿼리 + 5분 TTL | 단기 |
| 5 | 벡터/FTS 병렬 검색 | 30-50% | 중간 | 중간 | asyncio.gather 사용 | 중기 |
| 6 | 임베딩 모델 싱글톤 캐시 | 20-30% | 낮음 | 낮음 | 프로세스당 1회만 로드 | 중기 |
| 7 | 쿼리 리라이팅 캐시 | 20-30% | 중간 | 중간 | 쿼리 해시 기반 캐시 | 중기 |
| 8 | LanceDB Remote 모드 | 50-70% | 높음 | 높음 | 마이크로서비스 분리 (별도 포트) | 중기 |
| 9 | 응답 스트리밍 | 체감 시간 -50% | 높음 | 중간 | 검색 중 부분 응답 (SSE) | 장기 |
| 10 | Async 리랭커 | 20-30% | 높음 | 높음 | 배치 리랭킹 (다중 쿼리) | 장기 |
| 11 | Redis 분산 캐시 | 60-80% | 높음 | 높음 | 다중 인스턴스 지원 | 장기 |
| 12 | FTS 통계 업데이트 | 5-10% | 중간 | 낮음 | PostgreSQL ANALYZE | 중기 |

---

## 3. 단계별 세부 계획

### 3.1 단기 (1주, 즉시 적용 가능)

#### 3.1.1 DB 연결 풀 확대

**파일:** `backend/app/core/database.py`

```python
# 현재
pool_size=5, max_overflow=10

# 개선안
pool_size=10, max_overflow=20  # 동시 15개 연결 가능
pool_recycle=3600  # 연결 재활용 (1시간)
```

**기대 효과:**
- 동시 연결 수 3배 증가
- 연결 대기 제거
- **응답 시간: 250ms → 210ms (-16%)**

**리스크:** 낮음 (PostgreSQL 리소스 모니터링 필요)

**구현 난이도:** 낮음 (1줄 수정)

---

#### 3.1.2 LanceDB 벡터 인덱싱 활성화

**파일:** `backend/app/core/config.py` → `backend/app/main.py`

```python
# .env 추가
LANCEDB_INDEX_TYPE=IVF_FLAT
LANCEDB_NPROBES=20  # 40 → 20 (속도 우선)

# lifespan (이미 구현됨):
if settings.LANCEDB_INDEX_TYPE:
    store.create_vector_index(settings.LANCEDB_INDEX_TYPE)
```

**기대 효과:**
- 벡터 검색 시간: 100-200ms → 30-50ms
- 프로브 수 40→20: 추가 30% 가속
- **응답 시간: 210ms → 140ms (-33%)**

**리스크:** 중간
- Recall@10이 0.85→0.80 정도로 감소 가능
- 해결: 검색 후보 수 증가 (n_results: 10→15)

**구현 난이도:** 낮음 (.env 수정 + 서버 재시작)

---

#### 3.1.3 쿼리 결과 LRU 캐싱

**파일:** 새로 생성 `backend/app/services/rag/cache.py`

```python
from functools import lru_cache
from typing import Optional
import hashlib
import json

class QueryResultCache:
    def __init__(self, maxsize: int = 1000, ttl_seconds: int = 300):
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        self.cache: dict = {}  # {query_hash: (result, timestamp)}

    def get_key(self, query: str, doc_type: Optional[str]) -> str:
        data = json.dumps({"query": query, "doc_type": doc_type}, sort_keys=True)
        return hashlib.md5(data.encode()).hexdigest()

    def get(self, query: str, doc_type: Optional[str]) -> Optional[dict]:
        key = self.get_key(query, doc_type)
        if key in self.cache:
            result, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl_seconds:
                return result
            del self.cache[key]
        return None

    def set(self, query: str, doc_type: Optional[str], result: dict) -> None:
        if len(self.cache) >= self.maxsize:
            # 가장 오래된 항목 제거
            oldest_key = min(self.cache, key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        key = self.get_key(query, doc_type)
        self.cache[key] = (result, time.time())
```

**파일:** `backend/app/services/rag/pipeline.py` 수정

```python
# RAGPipeline.execute_async() 수정
async def execute_async(
    self, query: str, config: Optional[PipelineConfig] = None
) -> PipelineResult:
    # 캐시 확인
    if cached := self.cache.get(query, config.doc_type if config else None):
        logger.info("캐시 히트: %s", query)
        return cached

    # 검색 실행
    result = await asyncio.to_thread(self.execute, query, config)

    # 캐시 저장
    self.cache.set(query, config.doc_type if config else None, result)
    return result
```

**기대 효과:**
- 상위 1,000개 쿼리에서 100% 캐시 히트
- 캐시 히트 시간: 1ms (거의 즉시)
- 전체 traffic의 10-15% 캐시 히트 가정
- **응답 시간: 140ms → 125ms (-11%)**

**누적 효과 (1+2+3):** 250ms → 125ms (-50%)

**리스크:** 낮음 (캐시 만료 및 무효화 정책 필요)

**구현 난이도:** 중간 (50줄)

---

### 3.2 중기 (2주, 구조 변경)

#### 3.2.1 벡터 검색과 FTS 병렬 실행

**현재:** search_without_content() 내부에서 순차 실행
```python
vector_results = _search_vector_ids(...)  # 100ms
keyword_results = search_by_keyword(...)  # 50ms
# 합계: 150ms
```

**개선안:**
```python
async def search_without_content_async(
    query: str,
    n_results: int = 5,
    doc_type: Optional[str] = None,
) -> list[dict[str, Any]]:
    vector_fetch = n_results * 3 if settings.USE_HYBRID_SEARCH else n_results

    # 병렬 실행
    vector_task = asyncio.to_thread(
        _search_vector_ids, query, vector_fetch, doc_type
    )
    keyword_task = asyncio.to_thread(
        search_by_keyword, query, vector_fetch, doc_type
    )

    vector_results, keyword_results = await asyncio.gather(
        vector_task, keyword_task
    )

    # RRF 병합
    ...
```

**기대 효과:**
- 순차 150ms → 병렬 100ms (-33%)
- **응답 시간: 125ms → 100ms (-20%)**

**리스크:** 중간 (동기 함수 멀티스레드 경합)

**구현 난이도:** 중간 (30줄)

---

#### 3.2.2 임베딩 모델 싱글톤 캐싱

**파일:** `backend/app/services/rag/embedding.py` 수정

```python
import threading

_embedding_model_lock = threading.Lock()
_embedding_model_cache = None

def get_embedding_model():
    global _embedding_model_cache
    with _embedding_model_lock:
        if _embedding_model_cache is None:
            from sentence_transformers import SentenceTransformer
            _embedding_model_cache = SentenceTransformer(
                settings.LOCAL_EMBEDDING_MODEL
            )
        return _embedding_model_cache

def create_query_embedding(query: str) -> list[float]:
    """쿼리 임베딩 생성 (캐시된 모델 사용)"""
    model = get_embedding_model()
    embedding = model.encode(query, normalize_embeddings=True)
    return embedding.tolist()
```

**효과:**
- 첫 요청: 모델 로드 (1-2초)
- 이후 요청: 캐시 사용 (10ms 절감)
- 요청 1,000개 기준: **누적 효과 10,000ms (10초) 절감**

**리스크:** 낮음 (메모리 유지)

---

#### 3.2.3 쿼리 리라이팅 캐싱

**파일:** `backend/app/services/rag/query_rewrite.py` 수정

```python
from functools import lru_cache

@lru_cache(maxsize=500)
def rewrite_query_cached(
    query: str,
    num_queries: int = 3,
    use_llm: bool = True,
) -> list[str]:
    """캐시된 쿼리 리라이팅"""
    # 기존 rewrite_query 로직
    ...
```

**기대 효과:**
- 반복 쿼리에서 20-30% LLM 호출 감소
- **응답 시간: 최악 250ms → 200ms (-20%)**

**리스크:** 낮음 (쿼리 정규화 필요)

---

#### 3.2.4 LanceDB Remote 모드 마이그레이션 (선택)

**개요:**
- 벡터 검색을 별도 마이크로서비스로 분리
- 각 요청이 REST API 호출 (+ 네트워크 지연 10-20ms)
- 대신 다중 인스턴스 확장 가능

**리스크:** 높음 (네트워크 지연, 분산 추적 복잡)

**권장:** 트래픽이 1000+ QPS인 경우만 고려

---

### 3.3 장기 (1개월+, 고급 최적화)

#### 3.3.1 응답 스트리밍

**개념:**
- 검색 시작 → 검색 완료 대기 X
- 검색 중 부분 결과 스트리밍 (SSE)
- UI에서 progressive rendering

```python
# 채팅 API (이미 SSE 지원)
@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    async def event_generator():
        graph = get_graph()
        config = {"configurable": {"thread_id": thread_id}}

        async for event in graph.astream(...):
            # 검색 단계별 부분 결과 yield
            if event.get("phase") == "search":
                yield f"data: {json.dumps(event)}\n\n"
            elif event.get("phase") == "rerank":
                yield f"data: {json.dumps(event)}\n\n"

    return EventSourceResponse(event_generator())
```

**기대 효과:**
- 체감 응답 시간: 50% 감소 (첫 결과 50ms)

---

#### 3.3.2 Redis 분산 캐싱

**구조:**
```
Request → FastAPI → Redis 조회
                       ├─ Hit: 1ms 응답
                       └─ Miss: LanceDB + PostgreSQL → Redis 저장 (300ms) → 응답
```

**설정:**
```python
# .env
CACHE_BACKEND=redis  # memory | redis
REDIS_URL=redis://localhost:6379/0
CACHE_TTL=300  # 5분
```

**장점:** 다중 인스턴스에서 공유 캐시

**리스크:** 높음 (Redis 운영 오버헤드)

---

## 4. 구현 로드맵

### Phase 1: 단기 최적화 (1주)

```
Day 1:
- DB pool 확대 + 배포
- LanceDB IVF 인덱싱 빌드 (1시간)

Day 2-3:
- LRU 캐싱 구현 + 테스트

Day 4-5:
- 성능 테스트 (부하 테스트)
- 모니터링 대시보드 설정

기대 효과: 250ms → 125ms (-50%)
```

### Phase 2: 중기 최적화 (2주)

```
Week 2:
- 병렬 검색 (벡터 + FTS)
- 임베딩 모델 캐싱
- 쿼리 리라이팅 캐싱

Week 3:
- 통합 테스트
- 프로파일링 (bottleneck 재확인)
- 성능 보고

기대 효과: 125ms → 80ms (-36%)
총 누적: 250ms → 80ms (-68%)
```

### Phase 3: 장기 최적화 (1개월+)

```
Month 2:
- 응답 스트리밍 구현
- Redis 캐싱 (선택)
- Neo4j 그래프 활용 (컨텍스트 보강)
```

---

## 5. 성능 측정 전략

### 5.1 핵심 지표 (KPI)

| 지표 | 목표 | 측정 빈도 |
|------|------|----------|
| P50 응답 시간 | 100ms | 실시간 |
| P99 응답 시간 | 300ms | 실시간 |
| 캐시 히트율 | 15-20% | 일일 |
| 벡터 검색 시간 | 50ms | 실시간 |
| Recall@10 | ≥ 0.80 | 주간 |

### 5.2 벤치마크 설정

```python
# backend/tests/performance/benchmark_rag.py
import time
from app.services.rag.pipeline import search_with_pipeline

test_queries = [
    "손해배상 책임",
    "임금 체불",
    "이혼 소송",
    # ...
]

results = {}
for query in test_queries:
    start = time.perf_counter()
    result = search_with_pipeline(query, PRESETS["legal_search_all"])
    elapsed_ms = (time.perf_counter() - start) * 1000
    results[query] = {
        "time_ms": elapsed_ms,
        "total_retrieved": result.total_retrieved,
    }
```

---

## 6. 리스크 완화 전략

### 6.1 Rolling Deployment

Phase 1-2 변경사항은 점진적 배포:
- Canary: 10% 트래픽
- Monitor: 1시간
- Rollout: 100%

### 6.2 Fallback 메커니즘

```python
# IVF 인덱스 없이도 동작 (brute-force 자동 fallback)
try:
    results = table.search(...).to_pandas()
except:
    # Fallback: nprobes 제거
    results = table.search(...).to_pandas()
```

### 6.3 캐시 무효화

```python
# 데이터 업데이트 시 캐시 삭제
def invalidate_query_cache(doc_type: str):
    # 해당 doc_type의 캐시만 삭제
    for key in list(cache.cache.keys()):
        if doc_type in key:
            del cache.cache[key]
```

---

## 7. 예상 효과 정리

### 최종 성능 개선

| 단계 | 시나리오 | P50 응답 시간 | 개선율 |
|------|---------|-------------|--------|
| 현재 | 리랭킹 활성 | 250ms | - |
| 단기 | DB pool + IVF + 캐싱 | 125ms | -50% |
| 중기 | + 병렬 검색 + 모델 캐싱 | 80ms | -68% |
| 장기 | + 응답 스트리밍 | 50ms (체감) | -80% |

### ROI 분석

| 단계 | 개발 비용 | 운영 비용 | 효과 | ROI |
|------|---------|---------|------|-----|
| 단기 | 8시간 | ↑ 10% (DB) | 50% 개선 | 매우 높음 |
| 중기 | 16시간 | ↑ 15% (Redis) | +18% 추가 | 높음 |
| 장기 | 40시간 | ↑ 20% | +12% 추가 | 중간 |

---

## 8. 구현 우선순위

### 즉시 실행 (영향도 높음, 노력 낮음)

1. **DB 연결 풀 확대** (1시간, -16%)
2. **LanceDB IVF 인덱싱** (1시간 + 빌드, -33%)
3. **LRU 캐싱** (3시간, -11%, 누적 -50%)

### 다음주 실행

4. **병렬 검색** (2시간, -20%, 누적 -68%)
5. **임베딩 모델 캐싱** (1시간, 누적 효과 10%)
6. **쿼리 리라이팅 캐싱** (1.5시간)

### 추가 고려사항

- **모니터링**: Datadog/CloudWatch에서 응답 시간 추적
- **성능 테스트**: k6 또는 locust로 부하 테스트
- **A/B 테스트**: 변경사항 영향도 측정

---

## 결론

**권장 실행 순서:**
1. 단기 3가지 (총 5시간) → 예상 -50% 개선
2. 중기 4가지 (총 10시간) → 추가 -18% 개선
3. 장기 3가지 (선택적)

**예상 일정:** 2주 내 -68% 개선 가능

**다음 단계:**
- [ ] 단기 최적화 구현 및 테스트
- [ ] 성능 벤치마크 설정
- [ ] 모니터링 대시보드 구성
- [ ] 점진적 배포 계획 수립
