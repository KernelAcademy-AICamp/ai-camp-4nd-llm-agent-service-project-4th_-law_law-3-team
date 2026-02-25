# Gap Analysis: RAG 다중 타입 검색 (Focus + Supplementary)

> 분석 일시: 2026-02-24
> Plan 문서: `docs/01-plan/features/rag-multi-type-search.plan.md`
> Match Rate: **100%**

## 요약

| 항목 | 상태 |
|------|------|
| 전체 Match Rate | 100% (5/5 파일) |
| Plan 요구사항 충족 | 모두 충족 |
| 린트 검증 | ruff 통과, mypy 기존 에러만 존재 |
| 미구현 항목 | 없음 |

## 파일별 분석

### 1. `backend/app/tools/vectorstore/lancedb.py` - 100%

| Plan 요구사항 | 구현 상태 |
|--------------|----------|
| `$not_in` dict 필터 지원 | `_build_filter_conditions`에 구현 완료 (L417-426) |
| SQL 인젝션 방지 | `_escape_sql()` 활용 |
| str/int/float 타입 분기 | 모두 구현 |
| NOT IN SQL 생성 | `key NOT IN ('v1', 'v2')` 형식 |

### 2. `backend/app/services/rag/pipeline.py` - 100%

| Plan 요구사항 | 구현 상태 |
|--------------|----------|
| `PipelineConfig.exclude_doc_types` 필드 | `Optional[list[str]] = None` 추가 완료 |
| `doc_type` 우선 (상호 배타) | `execute()` 내 조건 분기 구현 |
| 검색 함수에 `exclude_doc_types` 전달 | `search_fn()` 호출 시 전달 |
| LangSmith 메타데이터 기록 | `exclude_doc_types` 포함 |

### 3. `backend/app/services/rag/retrieval.py` - 100%

| Plan 요구사항 | 구현 상태 |
|--------------|----------|
| `_search_vector_ids`에 exclude 파라미터 | L181 추가 완료 |
| `search_without_content`에 exclude 파라미터 | L398 추가 완료 |
| `search_relevant_documents`에 exclude 파라미터 | L472 추가 완료 |
| `search_relevant_documents_async`에 exclude 파라미터 | L594 추가 완료 |
| NOT IN 벡터 필터 | `{"data_type": {"$not_in": exclude_doc_types}}` |
| FTS 호출 시 exclude 전달 | `search_by_keyword()` 호출에 포함 |

### 4. `backend/app/services/rag/keyword_search.py` - 100%

| Plan 요구사항 | 구현 상태 |
|--------------|----------|
| `_execute_fts_query`에 exclude 파라미터 | L128 추가 완료 |
| `search_by_keyword`에 exclude 파라미터 | L188 추가 완료 |
| WHERE NOT IN 조건 | `FtsIndex.data_type.not_in(exclude_doc_types)` |
| `doc_type` 우선 분기 | `if doc_type: ... elif exclude_doc_types:` |

### 5. `backend/app/multi_agent/agents/legal_search_agent.py` - 100%

| Plan 요구사항 | 구현 상태 |
|--------------|----------|
| `FOCUS_CONFIG` 정의 | precedent/law 각각 PipelineConfig 구현 (L33-42) |
| `SUPPLEMENTARY_CONFIG` 정의 | `exclude_doc_types` 활용 구현 (L44-53) |
| `enable_rewrite=False` 전 설정 | 4개 PipelineConfig 모두 적용 |
| 쿼리 리라이팅 1회 | `rewrite_conversational_query()` 에이전트 레벨 호출 |
| `asyncio.gather` 병렬 검색 | `_prepare_rag_data()`에서 Focus+Supplementary 병렬 (L117-120) |
| `_build_supplementary_context()` | "## 관련 법률 자료 (보충)" 제목 + data_type별 표시 (L283-300) |
| `_format_supplementary_sources()` | data_type 한국어 원본을 doc_type으로 사용 (L393-412) |
| PrecedentService 상세조회 범위 | Focus precedent만 (L127 조건 분기) |
| 기존 Law 서브파이프라인 제거 | SEARCH_CONFIG 중첩 구조 → FOCUS/SUPPLEMENTARY 분리 |

## 검증 결과

### 린트 (ruff)
```
uv run ruff check backend/app/ → All checks passed!
```

### 타입 체크 (mypy)
- 4건 에러: 모두 기존 코드의 pre-existing 에러
  - `lancedb.py:36,49,52` - `_load_decomposition_map`, `_get_thread_tokenizer` 관련
  - `legal_search_agent.py:449` - LangChain `BaseMessage.content` 반환 타입
- 이번 변경으로 인한 신규 에러 없음

## 결론

Plan 문서의 모든 요구사항이 100% 구현 완료되었습니다.

- 5개 파일 모두 계획대로 수정됨
- Focus + Supplementary 병렬 검색 구조 완성
- 쿼리 리라이팅 중복 실행 제거 (2회 → 1회)
- `exclude_doc_types` 필터 체인 (LanceDB → retrieval → keyword_search) 완성
- Supplementary 컨텍스트/소스 범용 포맷터 추가
