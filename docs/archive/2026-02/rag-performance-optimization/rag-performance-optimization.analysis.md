# Gap Analysis: RAG 파이프라인 성능 최적화

> 분석일: 2026-02-24
> Match Rate: **100%** (10/10)
> 반복 횟수: 0

---

## 항목별 결과

| # | 검증 항목 | 상태 | 비고 |
|---|----------|------|------|
| 1 | `_generate_response()` ainvoke 전환 | 일치 | `legal_search_agent.py:448` |
| 2 | `SimpleChatAgent.process()` ainvoke 전환 | 일치 | `base_chat.py:172` |
| 3 | `search_without_content_async()` 벡터+FTS 병렬 | 일치 | `retrieval.py` asyncio.gather 사용 |
| 4 | `fetch_document_contents_async()` data_type별 병렬 | 일치 | `retrieval.py` asyncio.gather 사용 |
| 5 | `fetch_ai_summaries_async()` data_type별 병렬 | 일치 | `retrieval.py` asyncio.gather 사용 |
| 6 | `execute_async()` 다중 쿼리 병렬 검색 | 일치 | `pipeline.py` asyncio.gather 사용 |
| 7 | `execute_async()` 요약문/원문 async 조회 | 일치 | `pipeline.py` async 함수 호출 |
| 8 | `execute_async()` 리랭킹 to_thread | 일치 | `pipeline.py:385` asyncio.to_thread |
| 9 | 모델 웜업 이미 존재 | 일치 | `main.py:34-55` 임베딩+리랭커 warm-up |
| 10 | pipeline.py async import | 일치 | 3개 async 함수 import 확인 |

## Gap 목록

없음.

## 정적 검증

| 도구 | 결과 |
|------|------|
| `ruff check` | All checks passed |
| `mypy` | 기존 lancedb 스텁 에러만 (신규 에러 0건) |

## 수정 파일 (4개)

| 파일 | 변경 |
|------|------|
| `backend/app/multi_agent/agents/legal_search_agent.py` | `invoke()` → `ainvoke()` |
| `backend/app/multi_agent/agents/base_chat.py` | `invoke()` → `ainvoke()` |
| `backend/app/services/rag/retrieval.py` | async 병렬 래퍼 6개 함수 추가 |
| `backend/app/services/rag/pipeline.py` | `execute_async()` 진정한 async 재구현 |
