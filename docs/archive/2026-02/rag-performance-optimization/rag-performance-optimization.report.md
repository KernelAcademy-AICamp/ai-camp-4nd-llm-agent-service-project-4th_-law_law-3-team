# PDCA 완료 보고서: RAG 파이프라인 성능 최적화

> 작성일: 2026-02-24
> 피처: rag-performance-optimization
> Match Rate: **100%** (10/10)
> 반복 횟수: 0

---

## 1. 요약

RAG 응답 시간 병목을 해소하기 위해 3가지 최적화를 계획하고, 실제 구현이 필요한 2가지를 적용했다.

| # | 최적화 항목 | 결과 |
|---|-----------|------|
| 1 | LLM 동기 호출 `invoke()` → 비동기 `ainvoke()` 전환 | 구현 완료 |
| 2 | 벡터 + FTS + 원문/요약문 조회 병렬화 (`asyncio.gather`) | 구현 완료 |
| 3 | 모델 웜업 (서버 시작 시 사전 로딩) | 이미 적용됨 (변경 불필요) |

---

## 2. Plan 단계

### 2.1 목표

- `model.invoke()` 이벤트 루프 블로킹 제거
- 검색 파이프라인 내부 순차 실행 → 병렬 실행 전환 (100~200ms 절감)
- 모델 웜업으로 콜드 스타트 제거

### 2.2 현황 분석 결과

| 구분 | 분석 결과 |
|------|----------|
| 스트리밍 | `process_stream()` → `model.astream()` **이미 작동 중**. `process()` 폴백 경로만 `invoke()` 사용 |
| 검색 병렬화 | Focus+Supplementary 병렬은 완료. 파이프라인 내부(쿼리별, 벡터+FTS, 원문조회)는 순차 실행 |
| 모델 웜업 | `main.py` lifespan에서 임베딩+리랭커+인덱스 이미 구현 완료 |

### 2.3 계획 범위

- 수정 대상: 4개 파일
- 변경하지 않는 파일: `main.py`, `nodes.py`, `chat.py`, `keyword_search.py`

---

## 3. Do 단계 (구현)

### 3.1 `model.invoke()` → `await model.ainvoke()` 전환

**파일 2개 수정:**

| 파일 | 위치 | 변경 |
|------|------|------|
| `legal_search_agent.py` | `_generate_response()` :448 | `model.invoke()` → `await model.ainvoke()` |
| `base_chat.py` | `SimpleChatAgent.process()` :172 | `model.invoke()` → `await model.ainvoke()` |

`ainvoke()` 반환 타입이 `str | list[str | dict]`이므로 `isinstance` 타입 가드 추가:
```python
content = response.content
return content if isinstance(content, str) else str(content)
```

### 3.2 async 병렬 래퍼 함수 추가

**`retrieval.py`에 6개 함수 추가:**

| 함수 | 역할 |
|------|------|
| `search_without_content_async()` | 벡터 + FTS `asyncio.gather` 병렬 검색 |
| `_fetch_contents_for_type()` | 단일 data_type 원문 조회 헬퍼 |
| `fetch_document_contents_async()` | data_type별 원문 `asyncio.gather` 병렬 조회 |
| `_fetch_summaries_for_type()` | 단일 data_type 요약문 조회 헬퍼 |
| `fetch_ai_summaries_async()` | data_type별 요약문 `asyncio.gather` 병렬 조회 |
| (내부) `_search_vector_ids` / `search_by_keyword` | `asyncio.to_thread`로 래핑 |

기존 동기 함수는 하위 호환을 위해 그대로 유지.

### 3.3 `execute_async()` 진정한 async 재구현

**`pipeline.py` `RAGPipeline.execute_async()` 전면 재작성:**

이전: `await asyncio.to_thread(self.execute, ...)` (동기를 스레드에서 실행)

이후:
```
Step 1: 쿼리 리라이팅 (asyncio.to_thread)
Step 2: 다중 쿼리 병렬 검색 (asyncio.gather → 각 쿼리 내 벡터+FTS도 병렬)
Step 3: 요약문 병렬 조회 (fetch_ai_summaries_async)
Step 4: 리랭킹 (asyncio.to_thread — CPU-bound)
Step 5: 원문 병렬 조회 (fetch_document_contents_async)
```

---

## 4. Check 단계 (Gap Analysis)

### 4.1 검증 결과

| # | 검증 항목 | 상태 |
|---|----------|------|
| 1 | `_generate_response()` ainvoke 전환 | 일치 |
| 2 | `SimpleChatAgent.process()` ainvoke 전환 | 일치 |
| 3 | `search_without_content_async()` 벡터+FTS 병렬 | 일치 |
| 4 | `fetch_document_contents_async()` data_type별 병렬 | 일치 |
| 5 | `fetch_ai_summaries_async()` data_type별 병렬 | 일치 |
| 6 | `execute_async()` 다중 쿼리 병렬 검색 | 일치 |
| 7 | `execute_async()` 요약문/원문 async 조회 | 일치 |
| 8 | `execute_async()` 리랭킹 to_thread | 일치 |
| 9 | 모델 웜업 이미 존재 | 일치 |
| 10 | pipeline.py async import | 일치 |

**Match Rate: 100% (10/10)**
**Gap: 없음**

### 4.2 정적 검증

| 도구 | 결과 |
|------|------|
| `ruff check` | All checks passed |
| `mypy` | 기존 lancedb 스텁 에러만 (신규 에러 0건) |

---

## 5. 수정 파일 요약

| 파일 | 변경 내용 | 영향도 |
|------|----------|--------|
| `backend/app/multi_agent/agents/legal_search_agent.py` | `invoke()` → `ainvoke()` + 타입 가드 | 낮음 |
| `backend/app/multi_agent/agents/base_chat.py` | `invoke()` → `ainvoke()` + 타입 가드 | 낮음 |
| `backend/app/services/rag/retrieval.py` | async 병렬 래퍼 6개 함수 추가 | 중간 |
| `backend/app/services/rag/pipeline.py` | `execute_async()` 진정한 async 재구현 + import 추가 | 높음 |

변경하지 않은 파일: `main.py` (웜업 이미 존재), `nodes.py`, `chat.py`, `keyword_search.py`

---

## 6. 기대 효과

| 최적화 | 효과 |
|--------|------|
| `ainvoke()` 전환 | 이벤트 루프 블로킹 제거 → 동시 요청 처리 능력 향상 |
| 벡터+FTS 병렬화 | 순차 ~200ms → 병렬 ~100ms (약 50% 절감) |
| 다중 쿼리 병렬 검색 | 쿼리 N개 순차 → 병렬 (N배 → 1배) |
| 원문/요약문 data_type별 병렬 | data_type 수만큼 순차 → 병렬 |
| 모델 웜업 (기존) | 첫 요청 콜드 스타트 제거 |

---

## 7. 런타임 검증 가이드

아래 항목은 배포 후 LangSmith 트레이스에서 확인 가능:

- [ ] `search_without_content_async` 내 벡터/FTS span이 동시 시작
- [ ] `execute_async` 내 다중 쿼리가 동시 실행
- [ ] `fetch_ai_summaries_async` / `fetch_document_contents_async`가 data_type별 동시 실행
- [ ] `/api/chat/stream` 응답 스트리밍 정상 동작
- [ ] 전체 응답 시간 단축 확인 (Before/After 비교)
