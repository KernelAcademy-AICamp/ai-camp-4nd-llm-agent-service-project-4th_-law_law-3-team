# rag-pipeline-normalization Gap Analysis Report

> **Feature**: rag-pipeline-normalization
> **Date**: 2026-02-24
> **Match Rate**: 89%
> **Status**: 90% 미달 → 추가 조치 필요

---

## 분석 결과 요약

7개 FR 중 5개 완전 구현, 1개 대부분 구현, 1개 부분 구현.
90% 달성까지 `law_study_agent.py`와 `small_claims_agent.py`의 파이프라인 전환 필요.

---

## 파일별 변경 요약

| 파일 | Plan 기준 줄 수 | 현재 줄 수 | 변화율 | 상태 |
|------|:-----------:|:-------:|:----:|:----:|
| `pipeline.py` | 473 | 604 | +28% | 완료 |
| `format_utils.py` | 0 (신규) | 199 | 신규 | 완료 |
| `embedding.py` | 181 | 169 | -7% | 완료 |
| `query_rewrite.py` | 257 | 121 | **-53%** | 완료 |
| `retrieval.py` | 804 | 768 | -4% | 완료 |
| `keyword_search.py` | 260 | 261 | 0% | 부분 |
| `rerank.py` | 159 | 160 | 0% | 완료 |
| `__init__.py` | - | 107 | - | 완료 |
| `legal_search_agent.py` | 458 | 244 | **-47%** | 완료 |

---

## FR별 상세 분석

### FR-01: 파이프라인 검색 타입 basic/focus — 구현됨 (100%)

- `pipeline.py` L70: `search_type: Literal["basic", "focus"] = "basic"`
- `pipeline.py` L71: `supplementary_config: Optional[PipelineConfig] = None`
- `_execute_basic_async()` L331-375, `_execute_focus_async()` L377-436
- `execute_async()` 분기 로직 L326-329
- PRESETS: focus 모드 2개 + basic 모드 4개 (L123-175)

### FR-01a: focus 모드 리라이팅 1회 공유 — 구현됨 (100%)

- `pipeline.py` L392-398: 리라이팅 1회 실행
- L404-407: 리라이팅된 `queries`를 focus/supplementary 양쪽에 전달

### FR-02: PipelineConfig 프리셋만으로 실행 — 구현됨 (100%)

- `pipeline.py` L123-175: PRESETS 6개
- `legal_search_agent.py` L39-42: `_PRESET_MAP`으로 PRESETS 참조
- L100: `search_with_pipeline_async(message, self.config)` 단일 호출

### FR-03: format_utils.py 공용 유틸리티 — 구현됨 (100%)

- `format_utils.py` 신규 생성 (199줄)
- 컨텍스트 4개: `format_precedent_context`, `format_law_context`, `format_supplementary_context`, `format_generic_context`
- 소스 3개: `format_precedent_sources`, `format_law_sources`, `format_supplementary_sources`
- `__init__.py`에서 모든 함수 export

### FR-04: 모든 에이전트 search_with_pipeline_async 통일 — 부분 구현 (33%)

| 에이전트 | 호출 방식 | 통일 여부 |
|---------|---------|:--------:|
| `LegalSearchAgent` | `search_with_pipeline_async` | 통일 |
| `SmallClaimsAgent` | `search_relevant_documents_async` 직접 호출 | **미통일** |
| `LawStudyAgent` | `search_relevant_documents` + `asyncio.to_thread` | **미통일** |

`law_study_agent.py`에 자체 `_build_study_context`, `_format_sources` 잔존 — format_utils 미사용.

### FR-05: embedding.py OpenAI 코드 제거 — 구현됨 (100%)

- OpenAI import 없음
- `USE_LOCAL_EMBEDDING` 분기 없음
- `create_query_embedding()` L131-150: 로컬 모델 직접 호출

참고: `backend/CLAUDE.md` Embedding 섹션에 `USE_LOCAL_EMBEDDING=false` OpenAI 안내 잔존 → 문서 업데이트 필요

### FR-06: conversational rewrite 삭제 — 구현됨 (100%)

| 삭제 대상 | 존재 여부 |
|----------|:--------:|
| `_FOLLOWUP_KEYWORDS` | 제거됨 |
| `_MIN_STANDALONE_LENGTH` | 제거됨 |
| `_is_followup_query()` | 제거됨 |
| `rewrite_conversational_query()` | 제거됨 |
| `_parse_rewritten_queries()` | 제거됨 |

유지: `rewrite_query()`, `extract_legal_keywords()`, `_expand_related_keywords()`, `LEGAL_KEYWORDS`

### FR-07: 불필요한 코드/중복 제거 — 대부분 구현 (80%)

| 대상 | 상태 |
|------|:----:|
| `fetch_lancedb_summaries` deprecated | 제거됨 |
| `_best_doc_per_source` vs `_unique_source_ids` | 유지 (용도 상이 확인) |
| `is_fts_available_sync` vs `is_fts_available` | 유지 (동기 경로 필요) |

---

## 전체 Match Rate

| FR | Score | Status |
|----|:-----:|:------:|
| FR-01: basic/focus 검색 타입 | 100% | 완료 |
| FR-01a: focus 리라이팅 1회 공유 | 100% | 완료 |
| FR-02: PipelineConfig 프리셋 | 100% | 완료 |
| FR-03: format_utils.py | 100% | 완료 |
| FR-04: 에이전트 통일 사용 | 33% | **부분** |
| FR-05: OpenAI 코드 제거 | 100% | 완료 |
| FR-06: conversational rewrite 삭제 | 100% | 완료 |
| FR-07: 불필요 코드 제거 | 80% | 대부분 |
| **전체** | **89%** | **90% 미달** |

---

## Definition of Done 체크리스트

| 항목 | 상태 |
|------|:----:|
| 모든 RAG 모듈 코드 리뷰 완료 | 완료 |
| OpenAI 임베딩 dead code 제거 | 완료 |
| conversational rewrite 함수 삭제 | 완료 |
| basic/focus 검색 타입 구현 | 완료 |
| format_utils.py 공용 유틸리티 생성 | 완료 |
| legal_search_agent.py 보일러플레이트 축소 | 완료 (-47%) |
| 기존 검색 기능 동작 유지 | 구조 유지 확인 |
| ruff check + mypy 통과 | 통과 |

## Quality Criteria

| 항목 | 상태 |
|------|:----:|
| config + format 호출 수준 단순화 | legal_search만 완료 |
| 에이전트 간 RAG 호출 통일 | **미완료 (1/3)** |
| 포맷팅 로직 중복 없음 | **부분 (law_study 잔존)** |

---

## 권장 조치 (90% 달성)

| 우선순위 | 항목 | 대상 | 예상 효과 |
|---------|------|------|---------|
| 1 | `law_study_agent.py` 파이프라인 전환 + format_utils | law_study_agent.py | FR-04 +33% |
| 2 | `small_claims_agent.py` 파이프라인 전환 | small_claims_agent.py | FR-04 +33% |
| 3 | `backend/CLAUDE.md` OpenAI 안내 삭제 | CLAUDE.md | 문서 동기화 |
