# RAG 파이프라인 프로파일링 분석

## 1. 전체 파이프라인 흐름도

```
사용자 질문 (message)
    ↓
[LegalSearchAgent._prepare_rag_data] 진입
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 1-1. 대화형 쿼리 리라이팅 (동기)                              │
│ rewrite_conversational_query(message, history)              │
│ → _is_followup_query() 검사 (정규식 + 키워드)                 │
│ → follow-up 아니면 원본 그대로 반환                            │
│ → follow-up이면 LLM 호출 (temperature=0.0)                   │
│                                                              │
│ 소요 시간: 1-3ms (follow-up 아님) or 1000-1500ms (LLM)      │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 1-2. RAGPipeline 병렬 실행 (2개 동시)                         │
│ await search_with_pipeline_async(query, precedent_config)  │
│ await search_with_pipeline_async(query, law_config)        │
│                                                              │
│ [각 Pipeline 내부 순차 실행]                                  │
│ ├─ Step 1: 쿼리 리라이팅 (선택, enable_rewrite=False)       │
│ ├─ Step 2: 하이브리드 검색 (순차)                             │
│ │  ├─ for q in queries:                                     │
│ │  │  └─ search_without_content() [LanceDB 벡터 검색]      │
│ │  │     → embedding 생성 (50-100ms)                        │
│ │  │     → LanceDB 검색 (100-150ms)                         │
│ │  │     → 각 쿼리당 총 200-250ms                            │
│ │  └─ 중복 제거 (set 기반)                                    │
│ ├─ Step 3: 리랭킹 (enable_rerank=True)                      │
│ │  ├─ fetch_lancedb_summaries() (50-100ms)                 │
│ │  ├─ _populate_content() (메모리)                           │
│ │  ├─ rerank_documents() (Cross-encoder, 배치)              │
│ │  │  → 배치당 32개, 32개 이상이면 2배치 → 500-800ms      │
│ │  └─ fetch_document_contents() (PostgreSQL)                │
│ │     → 배치 조회 (~50ms)                                    │
│ └─ Step 4: 정렬 + 반환 (메모리)                              │
│                                                              │
│ [각 Pipeline]                                                │
│ 판례 검색: 15→4 결과 (with rerank) ≈ 1500-2000ms           │
│ 법령 검색: 1→1 결과 (no rerank) ≈ 250-400ms                │
│                                                              │
│ 병렬 실행 시간: max(1500-2000, 250-400) ≈ 1500-2000ms       │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 1-3. 판례 상세 정보 조회 (비동기)                             │
│ precedent_service.get_details(source_ids)                  │
│ → PostgreSQL 배치 조회                                       │
│ → 4개 판례 ≈ 50-100ms                                        │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 1-4. 그래프 컨텍스트 (비활성화)                               │
│ graph_contexts = {} (Neo4j 연결 비용 vs 효과 미미)           │
│ 소요 시간: 0ms                                                │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 1-5. 컨텍스트 구성 (메모리, 빠름)                             │
│ _build_context() + _format_sources()                       │
│ → 판례 요약, 법령 요약                                        │
│ 소요 시간: 10-20ms                                            │
└─────────────────────────────────────────────────────────────┘
    ↓
[RAGPipeline 종료] → context, sources 반환
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. LLM 응답 생성 (동기 또는 스트리밍)                         │
│ _generate_response() 또는 process_stream()                  │
│ → get_chat_model() [OpenAI/Anthropic]                      │
│ → model.invoke(messages) 또는 model.astream()              │
│                                                              │
│ 메시지 크기: context 포함 → ~5KB-10KB                       │
│ 소요 시간: 2000-5000ms (프로바이더별 상이)                   │
└─────────────────────────────────────────────────────────────┘
    ↓
AgentResult 반환
```

## 2. 병렬도

**병렬 실행 구간:**
1. `search_with_pipeline_async(precedent_config)` (Coroutine 1)
2. `search_with_pipeline_async(law_config)` (Coroutine 2)
3. `precedent_service.get_details()` (병렬 기회 있음, 현재 순차)
4. LLM 응답 생성 (RAG 완료 후)

**블로킹 구간:**
- embedding 생성 (CPU 로컬 모델): 50-100ms
- LanceDB 벡터 검색: 100-150ms
- rerank_documents (Cross-encoder): 500-800ms
- PostgreSQL 배치 조회: 50-100ms
- LLM API 호출: 2000-5000ms

---

## 3. 각 단계별 추정 소요 시간

| # | 단계 | 구현부 | 소요시간(ms) | 설명 |
|---|------|--------|------------|------|
| 1 | **대화형 쿼리 리라이팅** | `rewrite_conversational_query()` | 1-3 (non-followup)<br>1000-1500 (followup) | 키워드 검사: 1-3ms<br>LLM 호출: 1000-1500ms |
| 2 | **판례 RAG 검색** | `search_with_pipeline_async(precedent_config)` | **1500-2000** | 자체 분석 참조 |
| 3 | **법령 RAG 검색** | `search_with_pipeline_async(law_config)` | 250-400 | 리랭킹 없음, 간단함 |
| 4 | **판례 상세 조회** | `precedent_service.get_details()` | 50-100 | PostgreSQL 배치 조회 (4건) |
| 5 | **그래프 컨텍스트** | 비활성화 | 0 | Neo4j 불필요 |
| 6 | **컨텍스트/소스 포맷** | 메모리 작업 | 10-20 | 문자열 조립 |
| 7 | **LLM 응답** | `model.invoke()` | **2000-5000** | OpenAI/Anthropic |
| | **전체** | | **3800-8220** | 병렬 고려: (2) + max(1,3,4,5,6) + (7) |

### 3-1. 판례 RAG 검색 상세 분석 (1500-2000ms)

| # | 서브단계 | 소요시간(ms) | 설명 |
|---|---------|------------|------|
| 2-1 | Embedding 생성 | 50-100 | `create_query_embedding()` 로컬 모델 |
| 2-2 | LanceDB 벡터 검색 | 100-150 | n_results=15, cosine 거리 |
| 2-3 | 요약문 조회 | 50-100 | `fetch_lancedb_summaries()` 배치 |
| 2-4 | **Cross-encoder 리랭킹** | **500-800** | bge-reranker-v2-m3-ko, 32 배치 |
| 2-5 | PostgreSQL 원문 조회 | 50-100 | top-5만 배치 조회 |
| 2-6 | 메모리 작업 | 20-30 | 정렬, 중복 제거 |
| | **소계** | **770-1280** | 1회 쿼리당 |
| | **×2 쿼리** | **1540-2560** | 원본 + LLM 리라이팅 |

**병렬화:**
- 원본 쿼리 + 리라이팅 쿼리 순차 실행
- 각 LanceDB 검색은 I/O bound → 비동기 가능

---

## 4. 병목 TOP 3

### 1️⃣ **LLM 응답 생성 (2000-5000ms, 약 30-60%)**

**상황:**
- OpenAI GPT-4o-mini: 2000-3000ms
- Anthropic Claude-3.5: 2500-4000ms
- 다국어 지원 필요로 모델 크기 증가

**원인 분석:**
```python
# app/tools/llm/__init__.py
return ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,  # 일관성 vs 유연성 트레이드오프
    request_timeout=settings.LLM_TIMEOUT_SECONDS,  # default: 60s
)
```

**최적화 기회:**
- 모델 다운그레이드: gpt-4o-mini → gpt-3.5-turbo (500ms 단축)
- 캐싱: 자주 묻는 질문 (법률 용어 설명) → Redis 캐시
- 병렬 요청: 여러 agent 동시 실행 (현재는 단일 agent)

---

### 2️⃣ **Cross-encoder 리랭킹 (500-800ms, 약 10-20%)**

**상황:**
```python
# app/services/rag/rerank.py
model = CrossEncoder(
    "dragonkue/bge-reranker-v2-m3-ko",
    activation_fn=torch.nn.Sigmoid(),
)
# 최대 4000자 × 15개 문서 = 60,000자
# 배치 크기 32 → 대기 시간 발생
```

**병목 포인트:**
- **문서 개수**: 15개 후보 × 4000자 = 대량의 쌍 처리
- **배치 처리**: 32개 배치 → 15개만 처리 (배치 미활용)
- **GPU 없음**: 로컬 CPU 모델 (가정)

**최적화 기회:**
- 리랭킹 전 pre-filter: similarity > 0.5 (10개 → 5개 축소)
- GPU 사용: CUDA/MPS → 50-70% 단축
- 경량 모델 대체: bge-reranker-v2-m3-ko → bge-reranker-v2-xs-ko (3배 빠름)

---

### 3️⃣ **LanceDB 벡터 검색 (100-150ms, 약 3-5%)**

**상황:**
```python
# app/services/rag/retrieval.py
def search_without_content(...):
    # 벡터 검색 2회 (판례 + 법령 쿼리)
    # n_results=15 (리랭킹용 후보)
    # 인덱스: brute-force (현재) or IVF (설정 가능)
```

**병목 포인트:**
- **전문 데이터셋**: LanceDB legal_chunks 테이블 253,768개 청크
- **인덱스 미사용**: LANCEDB_INDEX_TYPE="" (brute-force, O(n))
- **쿼리 수**: 원본 + 리라이팅 쿼리 (2회)

**최적화 기회:**
- IVF 인덱스 활성화: brute-force → IVF (50-70% 단축)
- 쿼리 결합: 2개 쿼리 → 1개 (후보 통합 후 리랭킹)
- Embedding 캐싱: 자주 쓰는 검색어 (법률 용어)

---

## 5. 현재 코드의 타이밍/로깅 위치

### 5-1. RAGPipeline 메트릭

```python
# backend/app/services/rag/pipeline.py:158-249

class PipelineMetrics:
    search_time_ms: float = 0.0      # 검색 + 원문 조회
    rerank_time_ms: float = 0.0      # 리랭킹
    total_time_ms: float = 0.0       # 전체
    total_searched: int = 0
    total_reranked: int = 0

# 측정 포인트
pipeline_start = time.monotonic()  # L165
search_start = time.monotonic()    # L181
metrics.search_time_ms = ...       # L203
rerank_start = time.monotonic()    # L217
metrics.rerank_time_ms = ...       # L223
metrics.total_time_ms = ...        # L240

# 로깅
logger.info(
    "RAG 파이프라인 완료: %d건 검색 → %d건 반환 (%.0fms)",
    result.total_retrieved,
    len(result.documents),
    metrics.total_time_ms,
)
```

**출력 예시:**
```
RAG 파이프라인 완료: 20건 검색 → 5건 반환 (1847.0ms)
```

### 5-2. LegalSearchAgent 로깅

```python
# backend/app/multi_agent/agents/legal_search_agent.py

# 현재: 명시적 타이밍 없음
# LLM 호출은 langchain에 위임 (자동 로깅 없음)

# 추천: 다음처럼 개선 가능
import time
start = time.monotonic()
response = await self._generate_response(...)
logger.info("LLM 응답 생성: %.0fms", (time.monotonic() - start) * 1000)
```

### 5-3. 데이터 로드 단계에서의 타이밍

```python
# backend/app/services/rag/retrieval.py:154-250

# 측정 없음 (추천: 각 단계 타이밍 추가)
def search_without_content(...):
    # 1. Embedding 생성 (측정 없음)
    embedding = create_query_embedding(query)

    # 2. LanceDB 검색 (측정 없음)
    results = vector_store.search(...)
```

---

## 6. 콜드스타트 vs 웜스타트 분석

### 6-1. 콜드스타트 (첫 요청)

```
전체 시간 분포:
┌─────────────────────────────────────────────────┐
│ 1. 임베딩 모델 로드 (첫 1회만)      ~1000-2000ms │
│ 2. LanceDB 연결 초기화               ~100-200ms  │
│ 3. Cross-encoder 모델 로드 (첫 1회) ~500-1000ms │
│ 4. RAG 파이프라인 실행               ~1500-2000ms│
│ 5. LLM 응답                          ~2000-5000ms│
│                                                   │
│ **총 콜드스타트: 5000-10000ms (5-10초)**          │
└─────────────────────────────────────────────────┘

메모리: ~2.3GB (KURE-v1 임베딩 모델)
       + ~500MB (Cross-encoder 모델)
       + ~100MB (LanceDB 캐시)
```

### 6-2. 웜스타트 (2번째 이후)

```
전체 시간 분포:
┌─────────────────────────────────────────────────┐
│ 1. 임베딩 모델 (캐시됨)                ~10-20ms  │
│ 2. LanceDB 연결 (재사용)              ~5-10ms   │
│ 3. Cross-encoder (캐시됨)             ~10-20ms  │
│ 4. RAG 파이프라인 실행                ~1500-2000ms│
│ 5. LLM 응답                          ~2000-5000ms│
│                                                   │
│ **총 웜스타트: 3800-8220ms (3.8-8.2초)**         │
└─────────────────────────────────────────────────┘

메모리: ~2.9GB (모든 모델 메모리)
      → CPU 메모리: ~3.5GB 필요
      → 추천 시스템 RAM: 8GB 이상
```

### 6-3. 성능 개선 효과

| 최적화 | 콜드스타트 영향 | 웜스타트 영향 | 난이도 |
|--------|----------------|-------------|------|
| **임베딩 모델 경량화** | ↓ 1000-2000ms | ↓ 10-20ms | 높음 |
| **IVF 인덱스 활성화** | ↓ 200-300ms | ↓ 200-300ms | 중간 |
| **Cross-encoder 최적화** | ↓ 500ms | ↓ 500-800ms | 중간 |
| **LLM 모델 다운그레이드** | ↓ 500ms | ↓ 500-1000ms | 낮음 |
| **쿼리 캐싱** | ↓ 100-200ms | ↓ 1500-2000ms | 중간 |

**시뮬레이션:** 모든 최적화 적용 시
```
콜드스타트: 5-10초 → 2-3초 (50-70% 개선)
웜스타트:   3.8-8.2초 → 1.5-3초 (50-70% 개선)
```

---

## 7. 현재 코드의 이슈 및 개선안

### Issue 1: 리라이팅 쿼리 중복 처리

**현재 코드:**
```python
# pipeline.py:172-179
queries = [query]
if config.enable_rewrite:
    queries = rewrite_query(...)  # 원본 + N개 리라이팅
    result.rewritten_queries = queries

# 검색 루프: 원본 + 리라이팅 쿼리 모두 검색
for q in queries:
    docs = search_fn(...)  # LanceDB 검색 2회 (판례), 1회 (법령)
```

**문제:**
- 판례 검색: 1 + 2개 리라이팅 = 3회 (활성화 안 됨, but 설정에서 enable_rewrite=False)
- 법령 검색: 1회만 (enable_rewrite 없음)
- 판례 상세의 경우 리라이팅이 disabled이므로 실제로는 **1회만 검색**

**개선안:**
```python
# 유의: legal_search_agent.py에서는 쿼리 리라이팅이 disable 됨
# enable_rewrite=False (default)이므로 현재 영향 없음
# 하지만 향후 활성화 시 이슈 발생 가능
```

### Issue 2: 병렬 실행 미활용

**현재 코드:**
```python
# legal_search_agent.py:112-120
precedent_result = await search_with_pipeline_async(
    message, self.precedent_config
)
law_result = await search_with_pipeline_async(
    message, self.law_config
)
```

**좋은 점:** 이미 `await asyncio.gather()` 패턴 가능 (현재는 순차)

**개선안:**
```python
precedent_result, law_result = await asyncio.gather(
    search_with_pipeline_async(message, self.precedent_config),
    search_with_pipeline_async(message, self.law_config),
)
# 예상 단축: 250-400ms (두 요청 병렬화)
```

### Issue 3: 리랭킹 전 pre-filtering 없음

**현재 코드:**
```python
# pipeline.py:208-225
if config.enable_rerank and all_documents:
    reranked = rerank_documents(
        query=query,
        documents=all_documents,  # 15개 모두
        top_k=config.rerank_top_k,  # 5개만 반환
    )
```

**문제:**
- 15개 문서 모두 Cross-encoder 처리 (500-800ms)
- 최종 5개만 사용 → 비효율

**개선안:**
```python
# 상위 similarity > threshold 만 리랭킹
filtered = [
    d for d in all_documents
    if d.get('similarity', 0) > 0.5
]
reranked = rerank_documents(
    query=query,
    documents=filtered[:10],  # 최대 10개로 제한
    top_k=config.rerank_top_k,
)
# 예상 단축: 250-400ms
```

---

## 8. 최적화 우선순위

### Phase 1: 즉시 적용 (코드 변경, 5-10분)

1. **LLM 응답 병렬화**
   - 예상 단축: 100-200ms
   - 코드: `asyncio.gather()` 사용
   - 난이도: 낮음

2. **리랭킹 pre-filtering**
   - 예상 단축: 250-400ms
   - 코드: similarity threshold 추가
   - 난이도: 낮음

3. **판례 상세 조회 병렬화**
   - 예상 단축: 20-50ms
   - 코드: `asyncio.gather()` 사용
   - 난이도: 낮음

### Phase 2: 중기 최적화 (1-2주)

4. **LanceDB IVF 인덱스 활성화**
   - 예상 단축: 50-150ms
   - 코드: LANCEDB_INDEX_TYPE="ivf"
   - 난이도: 중간 (재인덱싱 필요)

5. **경량 Cross-encoder 모델**
   - 예상 단축: 200-400ms
   - 코드: bge-reranker-v2-xs-ko 변경
   - 난이도: 낮음

6. **응답 캐싱 (Redis)**
   - 예상 단축: 3500-8000ms (캐시 히트)
   - 코드: 쿼리 해시 → Redis 저장
   - 난이도: 중간

### Phase 3: 장기 아키텍처 (1개월+)

7. **LLM 모델 다운그레이드**
   - 예상 단축: 500-1000ms
   - 코드: gpt-3.5-turbo 변경
   - 난이도: 낮음 (성능 저하 위험)

8. **마이크로서비스 분리**
   - RAG 파이프라인 전용 워커 분리
   - 예상 개선: 동시 요청 처리 +50%
   - 난이도: 높음
