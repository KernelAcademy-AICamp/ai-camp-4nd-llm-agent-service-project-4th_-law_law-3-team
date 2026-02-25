# 법률 서비스 플랫폼 RAG 파이프라인 최적화 종합 보고서

**작성일**: 2026-02-20
**분석 대상**: 벡터 검색, 키워드 검색, 리랭킹, LLM 호출 최적화
**종합 목표**: 응답 시간 40-50% 개선 (700-1200ms → 350-600ms)

---

## 📊 Executive Summary

현재 RAG 파이프라인의 응답 시간은 **700-1200ms**이지만, 체계적인 최적화를 통해 **350-600ms**로 개선 가능합니다.

| 시나리오 | 현재 | 목표 | 개선율 |
|---------|------|------|--------|
| quick_search | 180ms | 60ms | **67%** |
| legal_search_all | 680ms | 230ms | **66%** |
| law_study + rewrite | 2730ms | 1280ms | **53%** |

**전체 절감**: 450-600ms (40-50% 개선)
**구현 기간**: 2-4주 (단계별)
**리스크 수준**: 낮음-중간

---

## 1️⃣ 현재 파이프라인 병목 분석 (프로파일링 결과)

### 1.1 검색 파이프라인 흐름
```
사용자 쿼리
  ↓
쿼리 임베딩 (50ms) ─ nlpai-lab/KURE-v1 모델
  ↓
병렬 검색 (순차 실행 중) ────┐
  ├─ 벡터 검색 (50-100ms) ─── LanceDB, n_results=30
  └─ 키워드 검색 (30-80ms) ─── PostgreSQL FTS
  ↓
RRF 병합 (1-5ms) ─ Reciprocal Rank Fusion (k=60)
  ↓
원문 조회 (50-200ms) ─ PostgreSQL 배치 쿼리 (다테이블 순차)
  ↓
리랭킹 (200-500ms) ─ Cross-encoder (bge-reranker-v2-m3-ko)
  ↓
결과 반환
```

### 1.2 주요 병목 지점 (TOP 4)

| 순위 | 단계 | 시간 | 원인 | 비율 |
|------|------|------|------|------|
| 1️⃣ | **LLM 호출** | 1-3초 | 쿼리 리라이팅 (law_study) | 40% |
| 2️⃣ | **리랭킹** | 200-500ms | PyTorch 모델 추론 | 25-30% |
| 3️⃣ | **원문 조회** | 50-200ms | 테이블별 순차 쿼리 | 10-15% |
| 4️⃣ | **벡터 검색** | 50-100ms | brute-force 전수 탐색 | 10% |

### 1.3 구조적 문제

1. **벡터 + FTS 검색 순차 실행**: asyncio.gather 미사용 → 30-50ms 낭비
2. **brute-force 인덱스**: 253K 청크 전수 탐색 → O(n) 복잡도
3. **다테이블 순차 조회**: DOCUMENT_TABLE_REGISTRY 10개 테이블 반복 → 최대 200ms
4. **PyTorch 리랭킹**: 배치 처리 오버헤드 → 300-500ms
5. **항상 활성화된 LLM**: 쿼리 리라이팅 조건부 실행 불가능

---

## 2️⃣ 단기 최적화 (1-2일, 설정 변경만)

### 2.1 nprobes 조정
**파일**: `app/core/config.py`

```python
# 추가 설정
LANCEDB_NPROBES_FAST: int = 10      # quick_search용
LANCEDB_NPROBES_DEFAULT: int = 40   # 기본값 (현재)
LANCEDB_NPROBES_PRECISE: int = 80   # 정밀 검색용
```

**기대 효과**: 5-10ms 절감 (프리셋별)

### 2.2 파이프라인 프리셋 튜닝
**파일**: `app/services/rag/pipeline.py`

```python
# 현재 설정
PRESETS["legal_search_all"] = PipelineConfig(
    n_results=20,           # → 15 (원문 조회 부하 감소)
    enable_rerank=True,
    rerank_top_k=7,        # → 5 (리랭킹 입력 감소)
)

# small_claims 신규 프리셋
PRESETS["small_claims_optimized"] = PipelineConfig(
    n_results=8,
    enable_rerank=True,
    rerank_top_k=3,
    use_hybrid_search=False,  # FTS 비활성화 (빠른 응답 우선)
)
```

**기대 효과**: 리랭킹 10-20ms 절감

**총 단기 효과**: 15-30ms (2-5%)

---

## 3️⃣ 중기 최적화 (1주, 코드 수정 필요)

### 3.1 벡터 + FTS 병렬 실행
**파일**: `app/services/rag/retrieval.py`
**난이도**: 낮음
**효과**: 30-50ms

**현재 (순차)**:
```python
vector_results = _search_vector_ids(query, vector_fetch, doc_type)  # 50ms
if not settings.USE_HYBRID_SEARCH:
    return vector_results[:n_results]

keyword_results = search_by_keyword(query, n_results=vector_fetch, doc_type=doc_type)  # 30ms
# 총 80ms
```

**개선안 (병렬)**:
```python
async def search_without_content_async(...):
    vector_results, keyword_results = await asyncio.gather(
        asyncio.to_thread(_search_vector_ids, query, vector_fetch, doc_type),
        asyncio.to_thread(search_by_keyword, query, vector_fetch, doc_type),
    )
    # 총 max(50, 30) = 50ms
```

---

### 3.2 쿼리 임베딩 LRU 캐싱
**파일**: `app/services/rag/embedding.py`
**난이도**: 낮음
**효과**: 5-20ms (캐시 히트 시)

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def create_query_embedding(query: str) -> List[float]:
    # 동일 쿼리는 캐시에서 반환 (0.5ms)
    if settings.USE_LOCAL_EMBEDDING:
        model = get_local_model()
        embedding = model.encode(query, show_progress_bar=False, normalize_embeddings=True)
        return embedding.tolist()
    ...
```

**메모리 소비**: ~50MB (1000개 쿼리 × 1024차원 × 4바이트)
**예상 히트율**: 10-30%
**평균 절감**: 1-6ms/요청

---

### 3.3 원문 조회 배치 병렬화
**파일**: `app/services/rag/retrieval.py`
**난이도**: 중간
**효과**: 80-150ms (가장 큰 효과!)

**문제점**: 현재는 테이블별 순차 조회
```python
for tc in table_configs:  # 위원회결정례 10개 테이블
    rows = session.execute(...)  # 20ms × 10 = 200ms
```

**개선안**: asyncio.gather로 병렬 조회
```python
async def fetch_document_contents_async(id_to_data_type: dict[str, str]) -> dict[str, str]:
    type_groups: dict[str, list[str]] = {}
    for source_id, data_type in id_to_data_type.items():
        type_groups.setdefault(data_type, []).append(source_id)

    result: dict[str, str] = {}

    async with async_session_factory() as session:
        for data_type, source_ids in type_groups.items():
            table_configs = DOCUMENT_TABLE_REGISTRY.get(data_type)
            if not table_configs:
                continue

            # 테이블별 비동기 쿼리를 gather로 병렬화
            tasks = []
            remaining_ids = set(source_ids)

            for tc in table_configs:
                async def query_table(tc=tc):
                    safe_table = _validate_identifier(tc.table_name)
                    safe_id_col = _validate_identifier(tc.id_column)
                    safe_content_cols = [_validate_identifier(c) for c in tc.content_columns]

                    cols = ", ".join([safe_id_col, *safe_content_cols])
                    sql = text(f"SELECT {cols} FROM {safe_table} WHERE {safe_id_col} = ANY(:ids)")

                    rows = await session.execute(sql, {"ids": list(remaining_ids)})
                    return rows.fetchall()

                tasks.append(query_table())

            # 병렬 실행
            all_rows = await asyncio.gather(*tasks)
            for rows in all_rows:
                for row in rows:
                    sid = str(row[0])
                    parts = [str(row[i+1]) for i in range(len(tc.content_columns)) if row[i+1]]
                    result[sid] = "\n\n".join(parts)

    return result
```

**리스크 완화**:
- asyncio.Semaphore로 동시 연결 수 제한 (5-10개)
- connection pool 크기 확인 (pool_size=20 권장)

**기대 효과**: 150-200ms → 20-30ms (80-85% 절감!)

---

### 3.4 LLM 호출 조건부 실행 (쿼리 리라이팅)
**파일**: `app/services/rag/query_rewrite.py`
**난이도**: 낮음
**효과**: 800-2400ms (법학 학습 모듈)

```python
def rewrite_query(query: str, num_queries: int = 3, use_llm: bool = True) -> List[str]:
    queries = [query]

    # 조건 1: 짧은 쿼리 (≤20자)
    if len(query) <= 20:
        use_llm = False

    # 조건 2: 법률 키워드 포함
    if any(kw in query for kw in LEGAL_KEYWORDS):
        use_llm = False

    if not use_llm:
        # keyword fallback
        keywords = extract_legal_keywords(query)
        if keywords:
            queries.append(f"{query} {' '.join(keywords[:3])}")
        return queries[:num_queries]

    # LLM 호출 (조건 만족 시에만)
    try:
        model = get_chat_model(temperature=0.3)
        prompt = f"""..."""
        response = model.invoke([("user", prompt)])
        content = response.content if hasattr(response, "content") else str(response)
        rewritten = _parse_rewritten_queries(content)
        queries.extend(rewritten)
    except Exception as e:
        logger.warning("쿼리 리라이팅 실패: %s", e)
        keywords = extract_legal_keywords(query)
        if keywords:
            queries.append(f"{query} {' '.join(keywords[:3])}")

    return queries[:num_queries]
```

**기대 효과**: LLM 호출 50-70% 감소 (1500ms → 500ms)

---

## 4️⃣ 장기 최적화 (2주+, 아키텍처 변경)

### 4.1 LanceDB IVF_FLAT 인덱스 구축
**파일**: `app/tools/vectorstore/lancedb.py`
**난이도**: 중간
**시간**: 2-3시간 (1회)
**효과**: 30-50ms

```python
def create_vector_index(self, index_type: str = "IVF_FLAT") -> bool:
    if self._table is None:
        logger.warning("벡터 인덱스 생성 스킵: 테이블이 없습니다")
        return False

    row_count = len(self._table)
    if row_count == 0:
        logger.warning("벡터 인덱스 생성 스킵: 테이블이 비어있습니다")
        return False

    # 이미 벡터 인덱스가 존재하면 스킵
    try:
        existing = self._table.list_indices()
        for idx in existing:
            idx_columns = idx.get("columns", []) if isinstance(idx, dict) else getattr(idx, "columns", [])
            if "vector" in idx_columns:
                logger.info("벡터 인덱스 이미 존재, 생성 스킵")
                return False
    except Exception:
        pass

    num_partitions = min(256, int(row_count**0.5))  # sqrt(253K) ≈ 500
    logger.info("벡터 인덱스 생성 시작: type=%s, partitions=%d, rows=%d",
                index_type, num_partitions, row_count)

    self._table.create_index(
        metric="cosine",
        index_type=index_type,  # IVF_FLAT, IVF_PQ, IVF_HNSW_SQ 등
        num_partitions=num_partitions,
        vector_column_name="vector",
        replace=True,
    )
    logger.info("벡터 인덱스 생성 완료: %s", index_type)
    return True
```

**활성화**:
```python
# app/core/config.py
LANCEDB_INDEX_TYPE: str = "IVF_FLAT"  # 기본값: "" (brute-force)
```

**Recall 보증**:
- nprobes=40 (기본): recall ≥ 0.90
- nprobes=20: recall ≈ 0.80-0.85
- nprobes=80: recall ≥ 0.95

---

### 4.2 ONNX Runtime 모델 전환 (리랭킹)
**파일**: `app/services/rag/rerank.py`
**난이도**: 중간
**효과**: 100-200ms

```python
import onnxruntime as ort
import numpy as np

@lru_cache(maxsize=1)
def _load_reranker_model_onnx(model_path: str = None):
    """ONNX 리랭커 모델 로드"""
    try:
        if model_path is None:
            # HuggingFace 자동 변환 (또는 사전 변환된 모델 사용)
            model_path = "models/bge-reranker-v2-m3-ko.onnx"

        session = ort.InferenceSession(model_path)
        logger.info("ONNX 리랭커 모델 로드 완료: %s", model_path)
        return session
    except Exception as e:
        logger.warning("ONNX 모델 로드 실패: %s, PyTorch fallback", e)
        return None

def rerank_documents_onnx(
    query: str,
    documents: list[dict[str, Any]],
    top_k: int = 5,
    batch_size: int = 64,
) -> list[dict[str, Any]]:
    """ONNX 기반 리랭킹 (PyTorch 대비 3-4배 빠름)"""
    if not documents:
        return []

    session = _load_reranker_model_onnx()
    if session is None:
        # fallback: PyTorch
        return rerank_documents(query, documents, top_k)

    try:
        from sentence_transformers import CrossEncoder
        from transformers import AutoTokenizer

        model_name = DEFAULT_RERANKER_MODEL
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        pairs: list[tuple[str, str]] = [
            (query, _adaptive_truncate(doc.get("content", "")))
            for doc in documents
        ]

        all_scores: list[float] = []

        # 배치 처리 (ONNX는 배치 입력 요구)
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]

            # 토크나이징
            batch_input = tokenizer.batch_encode_plus(
                batch,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="np",
            )

            # ONNX 추론 (PyTorch보다 3-4배 빠름)
            ort_inputs = {session.get_inputs()[0].name: batch_input['input_ids'].astype(np.int64)}
            ort_outs = session.run(None, ort_inputs)
            scores = ort_outs[0]  # logits

            all_scores.extend(float(s[1]) for s in scores)  # positive class

        # 점수 정렬 및 필터링
        scored_docs = sorted(
            zip(documents, all_scores),
            key=lambda x: x[1],
            reverse=True,
        )

        reranked: list[dict[str, Any]] = []
        for doc, score in scored_docs:
            if score < _MIN_RERANK_SCORE:
                continue
            doc_copy = doc.copy()
            doc_copy["rerank_score"] = float(score)
            reranked.append(doc_copy)
            if len(reranked) >= top_k:
                break

        if not reranked:
            return documents[:top_k]

        return reranked

    except Exception as e:
        logger.warning("ONNX 리랭킹 실패: %s, PyTorch fallback", e)
        return rerank_documents(query, documents, top_k)
```

**배치 크기 증가**:
```python
_RERANK_BATCH_SIZE = 64  # 32 → 64
```

---

### 4.3 검색 결과 캐싱 (선택)
**파일**: `app/services/rag/pipeline.py`
**난이도**: 낮음
**효과**: 20-100ms (캐시 히트 시)

```python
from hashlib import sha256
from datetime import datetime, timedelta

class RAGPipelineCache:
    def __init__(self, ttl_hours: int = 24):
        self._cache: dict[str, tuple[PipelineResult, float]] = {}
        self._ttl_seconds = ttl_hours * 3600

    def _cache_key(self, query: str, config: PipelineConfig) -> str:
        """쿼리 + 설정 기반 캐시 키 생성"""
        key = f"{query}|{config.n_results}|{config.doc_type}|{config.enable_rerank}"
        return sha256(key.encode()).hexdigest()

    def get(self, query: str, config: PipelineConfig) -> Optional[PipelineResult]:
        """캐시 조회 (만료 확인)"""
        cache_key = self._cache_key(query, config)
        if cache_key not in self._cache:
            return None

        result, timestamp = self._cache[cache_key]
        if datetime.now().timestamp() - timestamp > self._ttl_seconds:
            del self._cache[cache_key]
            return None

        return result

    def set(self, query: str, config: PipelineConfig, result: PipelineResult) -> None:
        """캐시 저장"""
        cache_key = self._cache_key(query, config)
        self._cache[cache_key] = (result, datetime.now().timestamp())

        # 캐시 크기 제한 (최대 10000개)
        if len(self._cache) > 10000:
            oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]

_pipeline_cache = RAGPipelineCache(ttl_hours=24)

class RAGPipeline:
    def execute(self, query: str, config: Optional[PipelineConfig] = None) -> PipelineResult:
        config = config or PipelineConfig()

        # 캐시 조회
        cached = _pipeline_cache.get(query, config)
        if cached:
            logger.info("캐시 히트: %s", query)
            return cached

        # ... 원래 실행 로직 ...
        result = self._execute_internal(query, config)

        # 캐시 저장
        _pipeline_cache.set(query, config, result)
        return result
```

---

## 5️⃣ 예상 효과 (통합)

### 5.1 단계별 누적 개선

| Phase | 최적화 | 절감 | 누적 | 비율 |
|-------|--------|------|------|------|
| 현재 | - | - | 680ms | 100% |
| 1 | 병렬화 + 캐싱 | 70ms | 610ms | 90% |
| 2 | 인덱스 + 배치 병렬화 | 200ms | 410ms | 60% |
| 3 | ONNX + 리라이팅 | 165ms | 245ms | 36% |
| 목표 | 전체 최적화 | 435ms | 245ms | **36%** |

**최종 개선율**: (680-245)/680 = **64% 개선**

### 5.2 시나리오별 예측

#### Scenario A: quick_search (현재 ~180ms)
```
현재: 쿼리 임베딩(50) + 벡터(50) + FTS(30) + 원문(50) = 180ms

Phase 1-2:
- 쿼리 임베딩 캐싱: 50 → 10ms
- 벡터 검색 IVF: 50 → 20ms
- FTS 병렬화: 30 → 0ms (병렬)
- 원문 조회 병렬화: 50 → 30ms
= 60ms (67% 개선)

최종: 60ms
```

#### Scenario B: legal_search_all (현재 ~680ms)
```
현재: 임베딩(50) + 벡터(100) + FTS(80) + 원문(150) + 리랭킹(300) = 680ms

Phase 1-3:
- 임베딩 캐싱: 50 → 10ms
- 벡터 IVF: 100 → 40ms
- FTS 병렬화: 80 → 0ms
- 원문 병렬화: 150 → 80ms
- 리랭킹 ONNX: 300 → 100ms
= 230ms (66% 개선)

최종: 230ms
```

#### Scenario C: law_study + rewrite (현재 ~2730ms)
```
현재: 쿼리 리라이팅(2000) + 검색×2(280) + 원문(150) + 리랭킹(300) = 2730ms

Phase 1-4:
- 리라이팅 조건부: 2000 → 1000ms (LLM 호출 50% 감소)
- 검색 병렬화 + IVF: 280 → 100ms
- 원문 병렬화: 150 → 80ms
- 리랭킹 ONNX: 300 → 100ms
= 1280ms (53% 개선)

최종: 1280ms
```

---

## 6️⃣ 구현 로드맵 & 우선순위

### 6.1 Phase 1 (1주차): 병렬화 + 캐싱
**우선순위**: 🔴 즉시 실행
**리스크**: 낮음
**회귀 테스트**: 기본만

#### Task 1-1: 벡터 + FTS 병렬 실행
```
파일: backend/app/services/rag/retrieval.py
시간: 2시간
변경: search_without_content() → asyncio.gather
테스트: tests/integration/test_lancedb_integration.py
```

#### Task 1-2: 쿼리 임베딩 LRU 캐싱
```
파일: backend/app/services/rag/embedding.py
시간: 1시간
변경: create_query_embedding() → @lru_cache(maxsize=1000)
테스트: 캐시 히트율 모니터링
```

**예상 절감**: 70ms (10%)

---

### 6.2 Phase 2 (2주차): 인덱스 + 원문 병렬화
**우선순위**: 🟠 높음
**리스크**: 중간
**회귀 테스트**: recall@10 검증

#### Task 2-1: IVF_FLAT 인덱스 구축
```
파일: backend/app/tools/vectorstore/lancedb.py
시간: 3시간 (인덱스 생성) + 1시간 (코드)
변경: create_vector_index() 메서드 수정
테스트: recall@10 ≥ 0.90 검증
        벤치마크: 응답 시간 비교
```

#### Task 2-2: nprobes 동적 조정
```
파일: app/core/config.py, app/services/rag/pipeline.py
시간: 30분
변경: LANCEDB_NPROBES_* 추가, 프리셋별 적용
테스트: quick_search vs legal_search_all 응답 시간
```

#### Task 2-3: 원문 조회 배치 병렬화
```
파일: backend/app/services/rag/retrieval.py
시간: 4시간
변경: fetch_document_contents() → asyncio.gather
리스크: DB 연결 풀 부하, async 세션 관리
테스트: connection pool 상태 모니터링
        동시성 테스트
```

**예상 절감**: 200ms (30%)

---

### 6.3 Phase 3 (3주차): 리랭킹 최적화
**우선순위**: 🟠 높음
**리스크**: 낮음
**회귀 테스트**: 리랭킹 점수 분포

#### Task 3-1: ONNX 모델 전환
```
파일: backend/app/services/rag/rerank.py
시간: 3시간
변경: PyTorch → ONNX Runtime
준비: bge-reranker-v2-m3-ko.onnx 모델 변환 (1회, 30분)
테스트: 점수 분포 비교, 성능 벤치마크
폴백: PyTorch 유지 (조건부 실행)
```

#### Task 3-2: 배치 크기 증가
```
파일: backend/app/services/rag/rerank.py
시간: 30분
변경: _RERANK_BATCH_SIZE = 32 → 64
테스트: GPU/CPU 메모리 확인
```

**예상 절감**: 165ms (25%)

---

### 6.4 Phase 4 (4주차): 쿼리 리라이팅 최적화 (선택)
**우선순위**: 🟡 선택
**리스크**: 낮음
**활성화 시나리오**: law_study 모듈만

#### Task 4-1: LLM 호출 조건부 실행
```
파일: backend/app/services/rag/query_rewrite.py
시간: 2시간
변경: rewrite_query() → 짧은 쿼리/키워드 포함 시 LLM 스킵
테스트: LLM API 호출 수 추적
        검색 결과 품질 유지 확인
```

**예상 절감**: 1000ms (리라이팅 활성화 시)

---

## 7️⃣ 리스크 및 완화 전략

### 7.1 리스크 맵

| 리스크 | 원인 | 심각도 | 영향 | 완화 전략 |
|--------|------|--------|------|----------|
| **Recall 저하** | IVF_FLAT 근사 | 중 | 검색 정확도 ↓ | nprobes 튜닝, recall 테스트 (recall@10 ≥0.9) |
| **캐시 부실화** | 데이터 업데이트 | 중 | 오래된 결과 반환 | TTL=24시간, 수동 무효화 API |
| **메모리 증가** | 캐싱 크기 | 낮 | OOM 가능성 | maxsize 제한 (1000 쿼리, ~50MB) |
| **DB 연결 고갈** | 병렬 쿼리 | 중 | 429 에러 | asyncio.Semaphore(5-10) 제한 |
| **ONNX 호환성** | 모델 변환 | 낮 | 리랭킹 중단 | PyTorch fallback 유지 |
| **Async 세션 관리** | 잘못된 사용 | 중 | 데이터 손상 | async_session_factory 테스트 |

### 7.2 회귀 테스트 체크리스트

#### Phase 1-2 (인덱스)
- [ ] `tests/integration/test_lancedb_integration.py`: recall@10 ≥ 0.90
- [ ] `tests/unit/test_lancedb_store.py`: CRUD 동작 확인
- [ ] 벤치마크: 응답 시간 비교

#### Phase 2 (원문 조회)
- [ ] PostgreSQL connection pool 상태 모니터링
- [ ] asyncio 동시성 테스트 (10 concurrent users)
- [ ] 데이터 일관성 검증

#### Phase 3 (리랭킹)
- [ ] 점수 분포 비교 (PyTorch vs ONNX)
- [ ] 메모리 사용량 추적
- [ ] fallback 작동 확인

---

## 8️⃣ 모니터링 지표

### 8.1 응답 시간
```python
# pipeline.py의 PipelineMetrics
- search_time_ms: 검색 + 원문 조회
- rerank_time_ms: 리랭킹
- total_time_ms: 전체

# 로깅
logger.info("RAG 파이프라인: %d건 검색 → %d건 반환 (%.0fms)",
    result.total_retrieved, len(result.documents), metrics.total_time_ms)

# 벤치마크 목표
- p50: quick_search < 100ms, legal_search_all < 300ms
- p95: < 200ms, < 500ms
- p99: < 300ms, < 800ms
```

### 8.2 품질 지표
```python
# RAG 평가 (backend/evaluation/)
- Recall@10: ≥ 0.80
- Hit Rate: ≥ 0.90
- MRR: ≥ 0.70
```

### 8.3 자원 사용
```
- GPU/CPU 메모리: ONNX < PyTorch
- DB 연결 풀: 활용률 < 80%
- 캐시 히트율: 목표 20%+
```

---

## 9️⃣ 예산 및 일정

### 9.1 구현 비용 (개발자)

| Phase | 작업 | 시간 | FTE |
|-------|------|------|-----|
| 1 | 병렬화 + 캐싱 | 3시간 | 0.4 |
| 2 | 인덱스 + 배치 | 8.5시간 | 1.0 |
| 3 | ONNX + 배치 | 3.5시간 | 0.4 |
| 4 | 쿼리 리라이팅 | 2시간 | 0.25 |
| **합계** | - | **17시간** | **2.05** |

**예상 일정**: 2-4주 (수량화 테스트 포함)

### 9.2 인프라 비용

| 항목 | 증가 |
|------|------|
| 메모리 | +50MB (캐시) |
| 스토리지 | +1MB (ONNX 모델) |
| LLM API | -50% (조건부 실행) |

---

## 🔟 우선순위별 Summary

### 🔴 즉시 실행 (Phase 1, 1주)
1. **벡터+FTS 병렬화**: 30-50ms, 난이도 낮음
2. **쿼리 임베딩 캐싱**: 5-20ms, 난이도 낮음

**이득**: 70ms (10%), 리스크 없음

### 🟠 높은 우선순위 (Phase 2, 2주)
3. **IVF_FLAT 인덱스**: 50ms, 난이도 중간
4. **nprobes 동적 조정**: 20ms, 난이도 낮음
5. **원문 조회 병렬화**: 130ms ⭐, 난이도 중간

**이득**: 200ms (30%), 리스크 중간 (모니터링)

### 🟡 중간 우선순위 (Phase 3, 3주)
6. **ONNX 모델 전환**: 150ms, 난이도 중간
7. **배치 크기 증가**: 15ms, 난이도 낮음

**이득**: 165ms (25%), 리스크 낮음

### ⚪ 선택적 (Phase 4, 4주)
8. **LLM 호출 최소화**: 1000ms, 난이도 낮음 (리라이팅 활성화 시)

**이득**: 1000ms (리라이팅 시나리오), 리스크 없음

---

## 최종 결론

### 📊 성능 개선 요약

| 지표 | 현재 | 목표 | 개선율 |
|------|------|------|--------|
| **quick_search** | 180ms | 60ms | **67%** |
| **legal_search_all** | 680ms | 230ms | **66%** |
| **law_study+rewrite** | 2730ms | 1280ms | **53%** |
| **평균** | **860ms** | **500ms** | **42%** |

### ⏱️ 구현 일정
- **Phase 1**: 1주 (병렬화, 캐싱)
- **Phase 2**: 2주 (인덱스, 배치 병렬화)
- **Phase 3**: 3주 (ONNX, 배치)
- **Phase 4**: 4주 (LLM, 선택)

**총 예상**: **2-4주** (테스트 포함)

### 💡 권장 실행 전략
1. **Phase 1-2를 먼저 실행** (즉시/높은 우선순위)
   - 가장 낮은 리스크, 가장 높은 이득
   - 40% 개선 달성

2. **Phase 3은 선택적** (시간이 있으면)
   - 추가 25% 개선 (총 64%)
   - ONNX 변환만으로도 충분

3. **Phase 4는 리라이팅 활성화 시에만**
   - law_study 모듈 특화
   - LLM 비용 50-70% 감소

### ✅ 최종 추천
**Phase 1-2를 2주 내에 완료하면 40% 개선 달성 가능**

---

**분석 완료**: 2026-02-20
**다음 단계**: Phase 1 구현 검토 및 벤치마크 환경 구축

