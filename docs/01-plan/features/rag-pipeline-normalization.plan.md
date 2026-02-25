# RAG 파이프라인 정규화 리팩토링 Planning Document

> **Summary**: RAG 파이프라인을 정규화하여 에이전트에서 최소한의 설정만으로 검색 파이프라인을 실행할 수 있도록 리팩토링
>
> **Project**: 법률 서비스 플랫폼
> **Author**: CTO Agent Team
> **Date**: 2026-02-24
> **Status**: Draft (v0.2 - 피드백 반영)

---

## 1. Overview

### 1.1 Purpose

현재 RAG 파이프라인(`backend/app/services/rag/`)은 잘 구조화되어 있으나, 에이전트마다 파이프라인 호출 방식이 다르고 컨텍스트 구성/소스 포맷팅 로직이 에이전트에 중복 존재한다. 파이프라인을 정규화하여 에이전트에서는 **설정값(PipelineConfig), 시스템 프롬프트, 특화 기능**만 정의하면 검색이 실행되도록 한다.

### 1.2 Background

- `legal_search_agent.py`(458줄)에서 Focus+Supplementary 병렬 검색, 컨텍스트 구성, 소스 포맷팅 등 **200줄 이상**이 RAG 로직에 해당
- `law_study_agent.py`는 동기 함수를 `asyncio.to_thread`로 래핑하여 호출 (비일관적)
- `small_claims_agent.py`는 `search_relevant_documents_async` 직접 호출 (파이프라인 미사용)
- 새 에이전트 추가 시마다 동일한 RAG 보일러플레이트 반복 필요

### 1.3 관련 코드

| 파일 | 줄 수 | 역할 |
|------|-------|------|
| `services/rag/pipeline.py` | 473 | 통합 RAG 파이프라인 |
| `services/rag/embedding.py` | 181 | 쿼리 임베딩 |
| `services/rag/query_rewrite.py` | 257 | 쿼리 리라이팅 |
| `services/rag/retrieval.py` | 804 | 벡터+FTS 하이브리드 검색, 원문 조회 |
| `services/rag/keyword_search.py` | 260 | FTS 검색 |
| `services/rag/rerank.py` | 159 | Cross-encoder 리랭킹 |
| `services/rag/fusion.py` | 40 | RRF 병합 |
| `services/rag/tsvector_builder.py` | 99 | tsvector 빌더 |
| `multi_agent/agents/legal_search_agent.py` | 458 | 법률 검색 에이전트 (참조 구현) |

---

## 2. Scope

### 2.1 In Scope

- [x] 각 RAG 모듈 코드 리뷰 및 불필요한 코드 정리
- [x] `pipeline.py` 정규화 - basic/focus 검색 타입 지원
- [x] 소스 포맷팅 유틸리티 추출 (`format_utils.py`)
- [x] 에이전트 호출 방식 통일 (모두 `search_with_pipeline_async` 사용)
- [x] `embedding.py` OpenAI dead code 제거 (로컬 모델 전용)
- [x] `query_rewrite.py` conversational rewrite 삭제 (법률 검색 최적화만 유지)
- [x] `legal_search_agent.py` 리팩토링 (정규화된 파이프라인 활용)

### 2.2 Out of Scope

- 새로운 에이전트 추가
- LanceDB 인덱스 구조 변경
- 임베딩 모델 교체
- 프론트엔드 변경
- 컨텍스트 구성을 파이프라인에 내재화 (별도 유틸리티로 분리)

---

## 3. Requirements

### 3.1 Functional Requirements

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| FR-01 | 파이프라인 검색 타입: `basic`(전체 검색) / `focus`(주타입+보충타입 비율 검색) | High | Pending |
| FR-01a | focus 모드에서 리라이팅 1회만 실행하고 focus/supplementary 양쪽에 공유 | High | Pending |
| FR-02 | 에이전트별 PipelineConfig 프리셋만으로 파이프라인 실행 가능 | High | Pending |
| FR-03 | 소스 포맷팅 공용 유틸리티 함수 제공 (`format_utils.py`) | Medium | Pending |
| FR-04 | 모든 에이전트가 `search_with_pipeline_async` 통일 사용 | Medium | Pending |
| FR-05 | `embedding.py` OpenAI 코드 제거 (로컬 KURE-v1 전용) | High | Pending |
| FR-06 | `query_rewrite.py` conversational rewrite 함수 삭제 | High | Pending |
| FR-07 | 각 모듈별 불필요한 코드/중복 제거 | Medium | Pending |

### 3.2 Non-Functional Requirements

| Category | Criteria | Measurement |
|----------|----------|-------------|
| 성능 | 기존 검색 응답 시간 유지 (±10%) | LangSmith 트레이싱 |
| 호환성 | 기존 에이전트 동작 변경 없음 | 기능 테스트 |
| 확장성 | 새 에이전트 추가 시 RAG 설정 최소화 | 코드 리뷰 |

---

## 4. 현재 코드 분석 (코드 리뷰)

### 4.1 `pipeline.py` (473줄) - 관리자 에이전트 담당

**강점**:
- `PipelineConfig` dataclass로 설정 깔끔하게 분리
- 동기/비동기 파이프라인 모두 지원
- LangSmith `@traceable` 트레이싱
- `PRESETS` 딕셔너리로 프리셋 관리

**개선점**:
- `execute_async()` 130줄 → 단계별 메서드 분리 필요
- Focus+Supplementary 패턴이 파이프라인에 없음 → basic/focus 타입으로 내재화
- `PRESETS`와 에이전트별 config가 분산 → 중앙화 필요

### 4.2 `embedding.py` (181줄) - 에이전트 1 담당

**강점**:
- `@lru_cache`로 모델 반복 로드 방지
- 모델 캐시 확인 로직 명확

**제거 대상 (dead code)**:
- OpenAI 임베딩 분기 (L157-163): `USE_LOCAL_EMBEDDING=false` 경로 → 로컬 모델만 사용하므로 제거
- `from openai import OpenAI` import → 제거
- `settings.USE_LOCAL_EMBEDDING` 분기 → 제거 (로컬 전용으로 단순화)
- `check_embedding_model_availability()` 내 `USE_LOCAL_EMBEDDING` 체크 (L82-84) → 항상 로컬 모델 전제

### 4.3 `query_rewrite.py` (257줄) - 에이전트 2 담당

**유지할 것**:
- `rewrite_query()`: LLM 기반 법률 검색 최적화 리라이팅
- `extract_legal_keywords()`: 법률 키워드 추출
- `_expand_related_keywords()`: 연관어 확장

**삭제할 것**:
- `rewrite_conversational_query()` (L190-257): follow-up 해소 로직 → 사용처: `legal_search_agent.py`, `law_study_agent.py`에서 import 제거
- `_is_followup_query()` (L156-187): follow-up 감지 함수
- `_FOLLOWUP_KEYWORDS` (L18-22): follow-up 키워드 셋
- `_MIN_STANDALONE_LENGTH` (L25): follow-up 관련 상수
- `_parse_rewritten_queries()` (L94-113): 사용되지 않는 함수 (rewrite_query에서 미사용)

### 4.4 `retrieval.py` (804줄) + `keyword_search.py` (260줄) - 에이전트 3 담당

**retrieval.py 개선점**:
- `fetch_lancedb_summaries` deprecated 함수 잔존 → 제거
- `_best_doc_per_source`와 `_unique_source_ids` 유사 기능 → 통합 검토

**keyword_search.py 개선점**:
- `is_fts_available_sync()`와 `is_fts_available()` 동기/비동기 중복 → 비동기만 유지 검토

### 4.5 `rerank.py` (159줄) - 에이전트 4 담당

- 깔끔한 코드. 변경 불필요
- `min_score` 기본값 0.01 적절성 확인

### 4.6 `fusion.py` (40줄), `tsvector_builder.py` (99줄)

- 유틸리티 성격으로 현재 상태 유지. 변경 불필요

---

## 5. 리팩토링 설계

### 5.1 검색 타입: basic / focus

```python
@dataclass
class PipelineConfig:
    # 기존 필드
    n_results: int = 10
    doc_type: str | None = None          # 문서 타입 필터
    exclude_doc_types: list[str] | None = None
    enable_rewrite: bool = True
    enable_rerank: bool = True
    rerank_top_k: int = 5
    use_llm_rewrite: bool = True

    # 신규: 검색 타입
    search_type: Literal["basic", "focus"] = "basic"

    # focus 전용: 보충 검색 설정
    supplementary_config: PipelineConfig | None = None
```

**basic 타입**: 전체 문서에서 검색 (현재 대부분 에이전트)
```python
# 예: law_study_agent - 전체 문서에서 3건 검색
config = PipelineConfig(n_results=3, enable_rerank=True, rerank_top_k=3)
```

**focus 타입**: 주 타입 검색 + 보충 타입 병렬 검색
```python
# 예: legal_search_agent - 판례 위주 검색 + 보충 자료
config = PipelineConfig(
    search_type="focus",
    n_results=15, doc_type="precedent",
    enable_rewrite=True,  # 리라이팅은 최상위에서 1회만 실행
    enable_rerank=True, rerank_top_k=5,
    supplementary_config=PipelineConfig(
        n_results=7, exclude_doc_types=["판례"],
        enable_rerank=True, rerank_top_k=3,
        # enable_rewrite는 무시됨 - 상위 config의 리라이팅 결과 공유
    ),
)
```

**focus 모드 실행 흐름 (리라이팅 1회 공유)**:
```
쿼리 입력
  ↓
[1] 리라이팅 1회 (상위 config.enable_rewrite=True일 때만)
  ↓
리라이팅된 쿼리
  ├──[2a] Focus 검색 (리라이팅 스킵, 리랭킹만)
  └──[2b] Supplementary 검색 (리라이팅 스킵, 리랭킹만)
  ↓ asyncio.gather 병렬
[3] PipelineResult (documents + supplementary_documents)
```

### 5.2 컨텍스트 구성: 파이프라인 밖에서 처리

**결정: Pipeline에 context를 넣지 않는다.**

이유:
1. **SRP**: Pipeline은 "검색"에만 집중. 컨텍스트 구성은 presentation 레이어 책임
2. **에이전트 다양성**: 판례 상세 vs 교육용 vs 소송 가이드 → 포맷이 다름
3. **RAG 미사용 에이전트**: 4개 에이전트가 RAG 미사용 → Pipeline에 context가 있으면 불필요한 커플링

대신 공용 포맷팅 유틸리티를 제공:

```python
# services/rag/format_utils.py (신규)
def format_precedent_context(documents, details=None) -> str: ...
def format_law_context(documents) -> str: ...
def format_generic_context(documents) -> str: ...  # data_type 기반 자동 포맷

def format_precedent_sources(documents, details=None) -> list[dict]: ...
def format_law_sources(documents) -> list[dict]: ...
def format_generic_sources(documents) -> list[dict]: ...
```

에이전트 코드:
```python
class LegalSearchAgent(BaseChatAgent):
    async def _prepare_rag_data(self, message):
        result = await search_with_pipeline_async(message, self.config)
        # 포맷팅은 유틸리티 호출
        context = format_precedent_context(result.documents, details)
        sources = format_precedent_sources(result.documents, details)
        return context, sources
```

### 5.3 PipelineResult 확장

```python
@dataclass
class PipelineResult:
    documents: list[dict]           # 검색 결과
    original_query: str
    rewritten_queries: list[str]
    total_retrieved: int
    metrics: PipelineMetrics

    # focus 타입 전용
    supplementary_documents: list[dict] | None = None  # 보충 검색 결과
```

### 5.4 embedding.py 단순화

```python
# AS-IS: 로컬/OpenAI 분기
def create_query_embedding(query: str) -> list[float]:
    if settings.USE_LOCAL_EMBEDDING:
        model = get_local_model()
        ...
    else:
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        ...

# TO-BE: 로컬 모델 전용
def create_query_embedding(query: str) -> list[float]:
    model = get_local_model()
    embedding = model.encode(query, show_progress_bar=False, normalize_embeddings=True)
    return embedding.tolist()
```

### 5.5 query_rewrite.py 정리

```python
# 삭제 대상
- _FOLLOWUP_KEYWORDS, _MIN_STANDALONE_LENGTH
- _is_followup_query()
- rewrite_conversational_query()
- _parse_rewritten_queries()  # 미사용 함수

# 유지 대상
- rewrite_query()             # LLM 법률 검색 최적화
- extract_legal_keywords()    # 키워드 추출
- _expand_related_keywords()  # 연관어 확장
- LEGAL_KEYWORDS              # 법률 키워드 목록
```

### 5.6 파일 변경 계획

| 파일 | 변경 유형 | 예상 영향 |
|------|----------|----------|
| `pipeline.py` | 확장 | basic/focus 검색 타입 추가, PipelineResult 확장, PRESETS 갱신 |
| `format_utils.py` | **신규** | 컨텍스트/소스 포맷팅 유틸리티 |
| `embedding.py` | 축소 | OpenAI dead code 제거, 로컬 전용 단순화 |
| `query_rewrite.py` | 축소 | conversational rewrite 삭제, 미사용 함수 삭제 |
| `retrieval.py` | 정리 | deprecated 함수 제거, 유사 함수 통합 |
| `keyword_search.py` | 정리 | 동기 중복 함수 제거 |
| `rerank.py` | 없음 | 현재 상태 유지 |
| `__init__.py` | 갱신 | 삭제된 export 제거, 새 export 추가 |
| `legal_search_agent.py` | 대폭 축소 | 정규화된 파이프라인 + format_utils 활용 |

---

## 6. 실행 순서 (CTO Agent Team 역할 분배)

### Phase 1: 개별 모듈 코드 리뷰 및 정리 (병렬 가능)

| 에이전트 | 담당 파일 | 작업 내용 |
|---------|----------|----------|
| Agent-1 | `embedding.py` | OpenAI dead code 제거, 로컬 전용 단순화 |
| Agent-2 | `query_rewrite.py` | conversational rewrite 삭제, 미사용 함수 삭제, `__init__.py` export 정리 |
| Agent-3 | `retrieval.py` + `keyword_search.py` | deprecated 제거, 함수 통합, 동기 중복 제거 |
| Agent-4 | `rerank.py` | 코드 리뷰 + min_score 검증 (변경 최소) |

### Phase 2: 파이프라인 정규화 (관리자 에이전트)

| 순서 | 작업 | 담당 |
|------|------|------|
| 2-1 | `PipelineConfig`에 `search_type` 추가 (basic/focus) | 관리자 |
| 2-2 | `PipelineResult`에 `supplementary_documents` 추가 | 관리자 |
| 2-3 | `execute_async()`에 focus 모드 병렬 검색 구현 | 관리자 |
| 2-4 | `format_utils.py` 신규 생성 (컨텍스트/소스 포맷팅) | 관리자 |
| 2-5 | PRESETS 확장 및 중앙화 | 관리자 |

### Phase 3: 에이전트 통합 (관리자 에이전트)

| 순서 | 작업 |
|------|------|
| 3-1 | `legal_search_agent.py` 리팩토링 (정규화된 파이프라인 + format_utils 활용) |
| 3-2 | `legal_search_agent.py`, `law_study_agent.py`에서 `rewrite_conversational_query` import 제거 |
| 3-3 | 기능 검증 (`ruff check` + `mypy` 통과) |

---

## 7. Success Criteria

### 7.1 Definition of Done

- [ ] 모든 RAG 모듈 코드 리뷰 완료
- [ ] OpenAI 임베딩 dead code 제거
- [ ] conversational rewrite 함수 삭제
- [ ] basic/focus 검색 타입 구현
- [ ] format_utils.py 공용 유틸리티 생성
- [ ] `legal_search_agent.py` RAG 보일러플레이트 대폭 축소
- [ ] 기존 검색 기능 동작 유지
- [ ] `ruff check` + `mypy` 통과

### 7.2 Quality Criteria

- [ ] 에이전트에서 RAG 설정이 config + format 호출 수준으로 단순화
- [ ] 에이전트 간 RAG 호출 방식 통일 (`search_with_pipeline_async`)
- [ ] 소스/컨텍스트 포맷팅 로직 중복 없음

---

## 8. Risks and Mitigation

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| 검색 품질 저하 | High | Low | LangSmith 트레이싱으로 전후 비교 |
| 기존 에이전트 동작 변경 | High | Medium | 단계적 리팩토링, 기능 테스트 |
| conversational rewrite 삭제 영향 | Low | Low | `query-rewrite-fix.plan.md`에서 이미 무의미 판정 |
| OpenAI 임베딩 제거 영향 | Low | Low | `USE_LOCAL_EMBEDDING=true` 고정, OpenAI 미사용 확인 |

---

## 9. 관련 Plan 문서 통합

| 문서 | 상태 | 본 Plan과 관계 |
|------|------|---------------|
| `query-rewrite-fix.plan.md` | plan 단계 | `rewrite_conversational_query` 삭제를 본 Plan에서 함께 수행 |

---

## 10. Next Steps

1. [ ] CTO Agent Team 구성 및 역할 분배
2. [ ] Phase 1 실행 (개별 모듈 코드 리뷰 및 정리)
3. [ ] Phase 2 실행 (파이프라인 정규화)
4. [ ] Phase 3 실행 (에이전트 통합)
5. [ ] 기능 검증 및 정적 분석

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-02-24 | 초안 작성 | CTO Agent Team |
| 0.2 | 2026-02-24 | 피드백 반영: basic/focus 타입, OpenAI 제거, context 분리, conversational rewrite 삭제 | CTO Agent Team |
