# Plan: RAG 파이프라인 성능 최적화

> **Feature**: rag-optimization
> **Phase**: Plan
> **Created**: 2026-02-20
> **Status**: Draft
> **관련 분석**: `docs/03-analysis/rag-optimization.analysis.md`

---

## 1. 배경 및 목적

### 현재 상태

RAG 파이프라인(rag-service 고도화 완료 후)의 **응답 시간이 ~20초**로, 사용자 경험에 심각한 영향을 미치고 있음.

4인 전문가 에이전트 팀(Profiler, Search Optimizer, LLM Optimizer, Architecture Reviewer)이 병렬 분석한 결과, 병목은 다음과 같이 분포:

| 단계 | 소요 시간 | 전체 비율 | 병목 등급 |
|------|----------|----------|----------|
| LLM 최종 응답 생성 | 2,000~5,000ms | 30~60% | 최대 병목 |
| Cross-encoder 리랭킹 | 500~800ms | 6~10% | 주요 병목 |
| 판례 RAG 검색 | 1,500~2,000ms | 18~25% | 주요 병목 |
| 쿼리 리라이팅 (follow-up) | 1,000~1,500ms | 12~18% | 조건부 병목 |
| LanceDB 벡터 검색 | 100~150ms | 1~2% | 경미 |
| 기타 (FTS, RRF, 포맷) | 100~200ms | 1~3% | 양호 |

### 목적

RAG 응답 시간을 **20초 → 5초 이하**로 줄여 사용자 체감 성능을 개선한다.

### 범위

- **포함**: RAG 파이프라인 성능 최적화 (검색, 리랭킹, LLM, 캐싱, 병렬화)
- **제외**: RAG 검색 품질(Recall/MRR) 개선, 새로운 기능 추가, UI 변경

---

## 2. 목표 성능 지표

| 지표 | 현재 | 목표 (Phase 1+2) | 최종 목표 |
|------|------|-----------------|----------|
| P50 응답 시간 | ~6초 | 3초 | 2초 |
| P99 응답 시간 | ~15초 | 7초 | 5초 |
| 콜드스타트 | ~10초 | 5초 | 3초 |
| 캐시 히트율 | 0% | 15% | 30% |
| Recall@10 | 0.80 | >= 0.78 | >= 0.78 |

> Recall 저하 허용 범위: 최대 -2% (0.80 → 0.78). 그 이상 저하 시 해당 최적화 롤백.

---

## 3. 최적화 방안 (3 Phase)

### Phase 1: 즉시 적용 (1~2일, 설정/소규모 변경)

구현 난이도 낮고 리스크 최소인 항목. 설정 변경 + 소규모 코드 수정만으로 적용 가능.

| # | 최적화 | 변경 파일 | 예상 절감 | 리스크 |
|---|--------|----------|----------|--------|
| S1 | **LLM 타이밍 로깅 추가** | `legal_search_agent.py` | 0ms (측정) | 없음 |
| S2 | **DB 연결 풀 확대** (pool_size 5→10) | `database.py` | 15~20% 안정화 | 낮음 |
| S3 | **nprobes 조정** (40→20) | `.env` | 30~50ms | 중간 |
| S4 | **판례/법령 검색 병렬화** | `legal_search_agent.py` | 250~400ms | 낮음 |
| S5 | **리랭킹 pre-filtering** (similarity>0.5) | `pipeline.py` | 250~400ms | 중간 |

**Phase 1 예상 효과**: ~15% 개선 + 정밀 병목 데이터 확보

#### S1. LLM 타이밍 로깅 추가

```python
# legal_search_agent.py - _generate_response()
import time
start = time.monotonic()
response = await self._generate_response(message, context, history)
logger.info("LLM 응답: %.0fms", (time.monotonic() - start) * 1000)
```

#### S2. DB 연결 풀 확대

```python
# database.py
pool_size=10,      # 현재: 5
max_overflow=20,   # 현재: 10
pool_recycle=3600,
```

#### S3. nprobes 조정

```bash
# backend/.env
LANCEDB_NPROBES=20  # 현재: 40
```

#### S4. 판례/법령 검색 병렬화

```python
# legal_search_agent.py - _prepare_rag_data()
# 현재: 순차 실행
precedent_result = await search_with_pipeline_async(msg, precedent_config)
law_result = await search_with_pipeline_async(msg, law_config)

# 변경: 병렬 실행
precedent_result, law_result = await asyncio.gather(
    search_with_pipeline_async(msg, precedent_config),
    search_with_pipeline_async(msg, law_config),
)
```

#### S5. 리랭킹 pre-filtering

```python
# pipeline.py - 리랭킹 전 similarity 기반 필터링
candidates = [d for d in documents if d.get("similarity", 0) > 0.5][:8]
reranked = rerank_documents(query, candidates, top_k=config.rerank_top_k)
```

---

### Phase 2: 핵심 최적화 (3~7일, 코드 수정)

캐싱 레이어 추가 및 모델 최적화로 반복 쿼리 성능 대폭 향상.

| # | 최적화 | 변경 파일 | 예상 절감 | 리스크 |
|---|--------|----------|----------|--------|
| M1 | **Cross-encoder 경량 모델** (m3→xs) | `rerank.py` | 300~500ms | 중간 |
| M2 | **쿼리 결과 LRU 캐싱** (1000개, TTL 5분) | `pipeline.py` + 신규 `cache.py` | 40~60% (히트 시) | 낮음 |
| M3 | **벡터+FTS 병렬 검색** | `retrieval.py` | 30~50ms | 중간 |
| M4 | **쿼리 임베딩 LRU 캐싱** | `embedding.py` | 40ms (재질문) | 낮음 |
| M5 | **원문 조회 배치 병렬화** | `retrieval.py` | 80~130ms | 중간 |

**Phase 2 누적 예상 효과**: ~50% 개선

#### M1. Cross-encoder 경량 모델 전환

```python
# rerank.py
# 현재
DEFAULT_RERANKER_MODEL = "dragonkue/bge-reranker-v2-m3-ko"  # 500-800ms

# 변경
DEFAULT_RERANKER_MODEL = "dragonkue/bge-reranker-v2-xs-ko"  # 150-250ms
```

> 정확도 3~5% 손실 예상. RAG eval로 검증 후 적용 결정.

#### M2. 쿼리 결과 LRU 캐싱

```python
# 신규: app/services/rag/cache.py
class QueryResultCache:
    def __init__(self, maxsize: int = 1000, ttl_seconds: int = 300):
        ...
    def get(self, query: str, doc_type: str | None) -> PipelineResult | None: ...
    def set(self, query: str, doc_type: str | None, result: PipelineResult) -> None: ...
```

```python
# pipeline.py - execute() 수정
cached = self.cache.get(query, config.doc_type)
if cached:
    return cached
result = ... # 기존 로직
self.cache.set(query, config.doc_type, result)
```

#### M3. 벡터+FTS 병렬 검색

```python
# retrieval.py - search_without_content()
vector_task = asyncio.to_thread(_search_vector_ids, query, n, doc_type)
keyword_task = asyncio.to_thread(search_by_keyword, query, n, doc_type)
vector_results, keyword_results = await asyncio.gather(vector_task, keyword_task)
```

#### M4. 쿼리 임베딩 LRU 캐싱

```python
# embedding.py
from functools import lru_cache

@lru_cache(maxsize=1000)
def create_query_embedding_cached(query: str) -> tuple[float, ...]:
    return tuple(create_query_embedding(query))
```

#### M5. 원문 조회 배치 병렬화

```python
# retrieval.py - fetch_document_contents()
# 현재: data_type별 순차 조회
# 변경: data_type별 병렬 조회
tasks = [
    _fetch_by_type(source_ids, data_type)
    for data_type, source_ids in grouped.items()
]
results = await asyncio.gather(*tasks)
```

---

### Phase 3: 고급 최적화 (2주+, 선택적)

아키텍처 수준 변경 또는 인프라 추가가 필요한 항목. Phase 1+2 효과 측정 후 선택 적용.

| # | 최적화 | 예상 절감 | 난이도 | 전제 조건 |
|---|--------|----------|--------|----------|
| L1 | **LLM 모델 분기** (간단→gpt-3.5, 복잡→gpt-4o-mini) | 500~1,000ms | 중간 | 질문 복잡도 분류기 |
| L2 | **Redis 응답 캐싱** (전체 응답 24h TTL) | 95% (히트 시) | 중간 | Redis 인프라 |
| L3 | **ONNX 리랭커** (Cross-encoder ONNX 변환) | 150ms | 중간 | onnxruntime 설치 |
| L4 | **응답 스트리밍 개선** (검색 중 부분 응답) | 체감 50% | 높음 | 프론트엔드 변경 |
| L5 | **쿼리 리라이팅 조건부 스킵** | 1,000~1,500ms | 낮음 | follow-up 판정 정밀화 |

---

## 4. 변경 범위

### 수정 파일

| 파일 | Phase | 변경 규모 | 설명 |
|------|-------|----------|------|
| `app/core/database.py` | 1 | 소 (1줄) | pool_size, max_overflow |
| `app/multi_agent/agents/legal_search_agent.py` | 1 | 중 | 타이밍 로깅, 검색 병렬화 |
| `app/services/rag/pipeline.py` | 1+2 | 중 | pre-filtering, 캐시 통합 |
| `.env` | 1 | 소 | nprobes 조정 |
| `app/services/rag/rerank.py` | 2 | 소 | 모델명 변경 |
| `app/services/rag/cache.py` | 2 | 신규 | LRU 캐시 클래스 |
| `app/services/rag/embedding.py` | 2 | 소 | 임베딩 캐싱 |
| `app/services/rag/retrieval.py` | 2 | 중 | 병렬 검색, 병렬 원문 조회 |

### 변경하지 않는 파일

| 파일 | 이유 |
|------|------|
| `keyword_search.py` | FTS 성능 이미 양호 (10~50ms) |
| `fusion.py` | RRF 알고리즘 <5ms, 최적화 불필요 |
| `query_rewrite.py` | Phase 3에서 선택 적용 (Phase 1~2 범위 밖) |
| `graph.py` | LangGraph 구조 변경 불필요 |
| `router.py` | 규칙 기반 라우팅, LLM 호출 없음 (이미 최적) |

---

## 5. 구현 순서

```
Phase 1 (Day 1-2):
  1.1 [S1] LLM 타이밍 로깅 추가 → 실측 데이터 확보
  1.2 [S2] DB 연결 풀 확대
  1.3 [S3] nprobes 조정
  1.4 [S4] 판례/법령 검색 asyncio.gather 병렬화
  1.5 [S5] 리랭킹 pre-filtering
  → 성능 측정 (Before vs After)

Phase 2 (Day 3-7):
  2.1 [M2] 쿼리 결과 LRU 캐싱 (cache.py 생성)
  2.2 [M3] 벡터+FTS 병렬 검색
  2.3 [M4] 쿼리 임베딩 캐싱
  2.4 [M5] 원문 조회 배치 병렬화
  2.5 [M1] Cross-encoder 경량 모델 (RAG eval 후 결정)
  → 성능 측정 + Recall 검증

Phase 3 (선택, Week 2+):
  3.1 [L5] 쿼리 리라이팅 조건부 스킵
  3.2 [L1] LLM 모델 분기 전략
  3.3 [L3] ONNX 리랭커
  3.4 [L2] Redis 응답 캐싱
```

---

## 6. 검증 전략

### 6.1 성능 검증

각 Phase 완료 후 벤치마크 실행:

```python
# 테스트 쿼리 세트 (10개)
test_queries = [
    "손해배상 책임 범위",
    "임금 체불 소멸시효",
    "이혼 소송 절차",
    "교통사고 과실 비율",
    "부당해고 구제신청",
    ...
]

# 측정 항목
- P50, P95, P99 응답 시간
- 콜드스타트 시간
- 캐시 히트율 (Phase 2 이후)
```

### 6.2 품질 검증

- Recall@10 >= 0.78 (현재 0.80 대비 최대 2% 저하 허용)
- 기존 RAG 평가 데이터셋으로 검증 (`backend/evaluation/`)
- 경량 리랭커 적용 시 반드시 검증

### 6.3 정적 검증

- `uv run ruff check backend/app/` 통과
- `uv run mypy backend/app/` 통과

---

## 7. 리스크 및 대응

| 리스크 | 영향 | 대응 |
|--------|------|------|
| 경량 리랭커 정확도 손실 | 답변 품질 | RAG eval 검증 후 적용, 롤백 가능 |
| LRU 캐싱 메모리 증가 | 서버 안정성 | maxsize=1000, TTL=5분으로 제한 |
| 병렬 실행 시 DB 연결 경합 | 응답 실패 | pool_size 확대 (S2)로 선제 대응 |
| asyncio.gather 예외 전파 | 전체 실패 | return_exceptions=True + 개별 fallback |

---

## 8. 성공 기준

- [ ] Phase 1 완료: P50 응답 시간 15% 이상 개선
- [ ] Phase 2 완료: P50 응답 시간 50% 이상 개선 (목표: 3초 이하)
- [ ] Recall@10 >= 0.78 유지
- [ ] 정적 검증 통과 (ruff, mypy)
- [ ] 기존 테스트 통과
- [ ] LLM 타이밍 로깅으로 실측 병목 데이터 확보
- [ ] 각 Phase 전후 벤치마크 결과 기록
