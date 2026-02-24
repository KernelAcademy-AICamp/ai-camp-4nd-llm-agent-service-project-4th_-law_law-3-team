# Plan: RAG 파이프라인 성능 최적화

> 작성일: 2026-02-24
> 피처: rag-performance-optimization

---

## 1. 목표

RAG 응답 시간을 줄이기 위해 3가지 최적화를 적용한다.

| # | 최적화 | 예상 효과 |
|---|--------|----------|
| 1 | LLM 동기 호출(`model.invoke()`) → 비동기(`model.ainvoke()`) 전환 | 이벤트 루프 블로킹 제거 |
| 2 | 벡터 + FTS + 원문 조회 병렬화 (`asyncio.gather`) | 100~200ms 절감 |
| 3 | 모델 웜업 (서버 시작 시 사전 로딩) | 콜드 스타트 제거 |

---

## 2. 현황 분석

### 2.1 스트리밍 (이미 부분 적용됨)

| 경로 | 메서드 | LLM 호출 | 상태 |
|------|--------|---------|------|
| `/api/chat/stream` → `legal_search_node` | `process_stream()` | `model.astream()` | **이미 스트리밍** |
| 폴백 (비스트리밍) | `process()` → `_generate_response()` | `model.invoke()` | **동기 블로킹** |
| `SimpleChatAgent.process()` | `process()` | `model.invoke()` | **동기 블로킹** |

**핵심**: `legal_search_node`는 항상 `_run_streaming_node` → `process_stream()` → `model.astream()`을 호출하므로 **스트리밍은 이미 작동 중**. `_generate_response()`의 `model.invoke()`는 비스트리밍 폴백 경로에서만 사용됨.

**변경 필요**: `model.invoke()` → `await model.ainvoke()`로 전환하여 이벤트 루프 블로킹 방지.

### 2.2 검색 병렬화 (순차 실행 중)

현재 파이프라인 내부 순차 실행 지점 3곳:

```
pipeline.execute()
  └── for q in queries:              ← (A) 쿼리별 순차
        └── search_without_content()
              ├── _search_vector_ids()   ← (B) 벡터 후 FTS 순차
              └── search_by_keyword()
  └── fetch_ai_summaries()           ← 순차 테이블 쿼리
  └── fetch_document_contents()      ← (C) data_type별 순차 테이블 쿼리
```

Focus + Supplementary는 `asyncio.gather`로 병렬화됨 (이전 작업에서 완료).

### 2.3 모델 웜업 (이미 적용됨)

`main.py` `lifespan()`에서 이미 구현:
- 임베딩 모델 (`get_local_model()` + warm-up encode) ✅
- 리랭커 모델 (`_load_reranker_model()` + warm-up predict) ✅
- LanceDB 벡터 인덱스 생성 ✅

**추가 변경 불필요** — 이미 완료.

---

## 3. 구현 계획

### 3.1 `model.invoke()` → `await model.ainvoke()` 전환

**파일 2개 수정:**

#### `backend/app/multi_agent/agents/legal_search_agent.py`
- `_generate_response()` (439-449줄): `model.invoke(messages)` → `await model.ainvoke(messages)`

#### `backend/app/multi_agent/agents/base_chat.py`
- `SimpleChatAgent.process()` (172줄): `model.invoke(messages)` → `await model.ainvoke(messages)`

### 3.2 벡터 + FTS + 원문 조회 병렬화

**전략**: 기존 동기 함수는 유지하고, async 래퍼를 추가하여 `asyncio.gather`로 병렬 실행.

#### `backend/app/services/rag/retrieval.py` — async 래퍼 추가

새로 추가할 함수:
```python
async def _search_vector_ids_async(...) -> list[dict]:
    return await asyncio.to_thread(_search_vector_ids, ...)

async def _search_by_keyword_async(...) -> list[dict]:
    return await asyncio.to_thread(search_by_keyword, ...)

async def search_without_content_async(...) -> list[dict]:
    """벡터 + FTS 병렬 검색 (async)"""
    vector_results, keyword_results = await asyncio.gather(
        _search_vector_ids_async(...),
        _search_by_keyword_async(...),
    )
    # RRF 병합 (기존 로직 동일)

async def fetch_document_contents_async(...) -> dict[str, str]:
    """data_type별 병렬 원문 조회 (async)"""
    # data_type별 그룹을 asyncio.gather로 병렬 처리

async def fetch_ai_summaries_async(...) -> dict[str, str]:
    """data_type별 병렬 요약문 조회 (async)"""
```

#### `backend/app/services/rag/pipeline.py` — `execute_async()` 재구현

현재: `await asyncio.to_thread(self.execute, ...)` (동기를 스레드에서 실행)
변경: 진정한 async 구현으로 내부 병렬화 활용

```python
async def execute_async(self, query, config):
    # Step 1: 쿼리 리라이팅 (to_thread)
    queries = await asyncio.to_thread(rewrite_query, ...)

    # Step 2: 다중 쿼리 병렬 검색 (각 쿼리 내 벡터+FTS도 병렬)
    search_tasks = [search_without_content_async(q, ...) for q in queries]
    results = await asyncio.gather(*search_tasks)

    # Step 3: 요약문 조회 (병렬)
    summaries = await fetch_ai_summaries_async(...)

    # Step 4: 리랭킹 (CPU-bound → to_thread)
    reranked = await asyncio.to_thread(rerank_documents, ...)

    # Step 5: 원문 조회 (병렬)
    contents = await fetch_document_contents_async(...)
```

---

## 4. 수정 파일 요약

| # | 파일 | 변경 내용 | 영향도 |
|---|------|----------|--------|
| 1 | `backend/app/multi_agent/agents/legal_search_agent.py` | `invoke()` → `ainvoke()` | 낮음 |
| 2 | `backend/app/multi_agent/agents/base_chat.py` | `invoke()` → `ainvoke()` | 낮음 |
| 3 | `backend/app/services/rag/retrieval.py` | async 래퍼 함수 4개 추가 | 중간 |
| 4 | `backend/app/services/rag/pipeline.py` | `execute_async()` 재구현 | 높음 |

**변경하지 않는 파일:**
- `backend/app/main.py` — 모델 웜업 이미 완료
- `backend/app/multi_agent/nodes.py` — 변경 불필요
- `backend/app/api/router/chat.py` — 변경 불필요
- `backend/app/services/rag/keyword_search.py` — 기존 동기 함수 유지

---

## 5. 검증

### 정적 검증
```bash
cd backend
uv run ruff check app/
uv run mypy app/
```

### 런타임 검증
- LangSmith 트레이스에서 각 span 소요시간 비교 (Before/After)
- `search_without_content_async` 내 벡터/FTS가 병렬 실행되는지 확인
- `/api/chat/stream` 엔드포인트로 응답 스트리밍 동작 확인

---

## 6. 미포함 사항 (모델 웜업)

`main.py` lifespan에서 이미 구현 완료:
- 임베딩 모델 사전 로드 + warm-up (34-41줄) ✅
- 리랭커 모델 사전 로드 + warm-up (43-55줄) ✅
- LanceDB 벡터 인덱스 생성 (57-73줄) ✅

추가 변경 불필요.
