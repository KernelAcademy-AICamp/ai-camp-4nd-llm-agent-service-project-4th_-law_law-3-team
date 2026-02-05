# LanceDB Test Suite Planning Document

> **Summary**: LanceDB 벡터 DB의 스키마, 저장소, 벡터 검색 및 MeCab 기반 FTS 검색에 대한 포괄적 테스트 스위트 구축
>
> **Project**: law-3-team (법률 서비스 플랫폼)
> **Author**: Claude
> **Date**: 2026-02-05
> **Status**: Draft

---

## 1. Overview

### 1.1 Purpose

LanceDB 벡터 DB 구현체(`LanceDBStore`, `schema_v2`, 팩토리 함수)에 대한 체계적인 테스트를 작성하여 코드 품질과 안정성을 보장한다. 현재 기존 통합 테스트(`test_lancedb_search.py`)는 실제 데이터 의존성이 있는 스크립트 형태이며, 격리된 단위/통합 테스트가 부재하다.

### 1.2 Background

- 법령 5,841건(118,922 청크) + 판례 65,107건(134,846 청크)이 LanceDB에 임베딩 저장됨
- 스키마 v2 (단일 테이블 + NULL, 20개 컬럼)로 법령/판례를 통합 관리
- 검색 API 연동, 하이브리드 검색 등 추가 개발이 예정되어 있어 테스트 기반이 필수
- `VectorStoreBase` 인터페이스 계약 준수 여부 검증 필요
- **FTS(Full-Text Search) 구현 시 MeCab 한국어 형태소 분석기 사용 필수**

### 1.3 FTS + MeCab 기술 배경

#### LanceDB FTS 현황

LanceDB는 Tantivy 기반 FTS를 내장하며 `table.create_fts_index("content")` API를 제공한다. 그러나 **한국어/CJK 네이티브 토크나이저는 미지원** 상태다.

- LanceDB PR [#2855](https://github.com/lancedb/lancedb/pull/2855): lindera(MeCab 호환 사전 기반) CJK 토크나이저 feature flag 추가 중이나 **미머지 상태** (2026-02 기준)
- 기본 토크나이저는 공백/구두점 분리 방식으로 **교착어인 한국어에 부적합**
- 예: "손해배상청구" → 기본 토크나이저: ["손해배상청구"] (단일 토큰), MeCab: ["손해", "배상", "청구"] (형태소 분리)

#### MeCab 사전 토크나이징 전략 (Pre-tokenization)

LanceDB 네이티브 한국어 토크나이저가 없으므로, **MeCab으로 사전 토크나이징한 텍스트를 FTS 인덱싱**하는 전략을 채택한다.

```
원본 텍스트: "불법행위로 인한 손해배상청구권의 소멸시효"
    │
    ▼ MeCab 형태소 분석
토크나이징: "불법 행위 로 인하 ㄴ 손해 배상 청구권 의 소멸 시효"
    │
    ▼ LanceDB FTS 인덱싱
인덱스 생성: create_fts_index("content_tokenized")
    │
    ▼ 검색 시
쿼리 "손해배상" → MeCab → "손해 배상" → FTS 검색 → 매칭
```

#### 스키마 확장 (content_tokenized 컬럼)

```python
# schema_v2.py에 추가할 컬럼
pa.field("content_tokenized", pa.utf8()),  # MeCab 사전 토크나이징된 content
```

기존 `content` 컬럼(원본)은 유지하고, `content_tokenized` 컬럼에 MeCab 처리 결과를 저장한다.

### 1.4 Related Documents

- 설계: `docs/vectordb_design.md` (벡터 DB 설계 문서)
- 구현: `backend/app/tools/vectorstore/` (LanceDB 구현체)
- 기존 테스트: `backend/tests/integration/test_lancedb_search.py` (스크립트 형태)
- 토크나이저 유틸: `.agent/skills/embedding-pipeline/templates/tokenizer_utils.py` (KoreanTokenizer 참고)

---

## 2. Scope

### 2.1 In Scope

- [ ] **스키마 단위 테스트**: `schema_v2.py`의 `create_law_chunk()`, `create_precedent_chunk()`, `LegalChunk` 검증
- [ ] **저장소 단위 테스트**: `LanceDBStore`의 CRUD 메서드, 필터링, 메타데이터 추출
- [ ] **팩토리 테스트**: `get_vector_store()` 팩토리 함수
- [ ] **통합 테스트**: 임시 LanceDB 디렉토리를 사용한 End-to-End 검색 흐름
- [ ] **데이터 무결성 테스트**: 스키마 20컬럼, 벡터 1024차원, ID 형식 검증
- [ ] **테스트 픽스처**: 공유 conftest에 LanceDB 전용 픽스처 추가
- [ ] **MeCab 토크나이저 테스트**: MeCab 한국어 형태소 분석 및 사전 토크나이징 검증
- [ ] **FTS 인덱스 테스트**: LanceDB FTS 인덱스 생성 및 키워드 검색 테스트
- [ ] **하이브리드 검색 테스트**: 벡터 검색 + FTS 검색 결합 (RRF) 테스트

### 2.2 Out of Scope

- 실제 임베딩 모델(KURE-v1) 로딩 및 임베딩 생성 테스트 (GPU 필요)
- PostgreSQL 연동 원본 조회 테스트 (별도 DB 의존)
- 임베딩 스크립트(`create_lancedb_embeddings.py`, `runpod_lancedb_embeddings.py`) 테스트
- 프론트엔드 검색 UI 테스트

---

## 3. Requirements

### 3.1 Functional Requirements

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| FR-01 | `create_law_chunk()` 필드 매핑 정확성 테스트 | High | Pending |
| FR-02 | `create_precedent_chunk()` 필드 매핑 정확성 테스트 | High | Pending |
| FR-03 | `LegalChunk` Pydantic 모델 유효성 검증 테스트 | High | Pending |
| FR-04 | `validate_by_type()` NULL 필드 규칙 테스트 (법령→판례 필드 NULL, 판례→법령 필드 NULL) | High | Pending |
| FR-05 | `LanceDBStore` 초기화 테스트 (기본값, 커스텀 설정) | Medium | Pending |
| FR-06 | `add_law_documents()` 법령 문서 추가 테스트 | High | Pending |
| FR-07 | `add_precedent_documents()` 판례 문서 추가 테스트 | High | Pending |
| FR-08 | `search()` 벡터 유사도 검색 테스트 | High | Pending |
| FR-09 | `search_by_type()` 데이터 유형별 필터 검색 테스트 | High | Pending |
| FR-10 | `count()`, `count_by_type()` 문서 수 카운트 테스트 | Medium | Pending |
| FR-11 | `get_by_source_id()` 동일 문서 청크 조회 테스트 | High | Pending |
| FR-12 | `delete_by_source_id()` 청크 삭제 테스트 | Medium | Pending |
| FR-13 | `_escape_sql()` SQL 인젝션 방지 테스트 | Medium | Pending |
| FR-14 | `_build_filter_conditions()` WHERE 절 생성 테스트 | Medium | Pending |
| FR-15 | `reset()` 테이블 초기화 테스트 | Low | Pending |
| FR-16 | `get_vector_store()` 팩토리 함수가 LanceDBStore 반환 테스트 | Medium | Pending |
| FR-17 | E2E 검색 흐름 (문서 추가 → 검색 → source_id 추출) 테스트 | High | Pending |
| FR-18 | ID 형식 `{source_id}_{chunk_index}` 규칙 검증 테스트 | Medium | Pending |
| FR-19 | `SearchResult` 데이터클래스 동작 테스트 | Medium | Pending |
| **FTS + MeCab 요구사항** | | | |
| FR-20 | MeCab 토크나이저 초기화 및 한국어 형태소 분석 테스트 | High | Pending |
| FR-21 | MeCab 법률 용어 토크나이징 정확성 테스트 (예: "손해배상청구" → "손해 배상 청구") | High | Pending |
| FR-22 | `content_tokenized` 컬럼에 MeCab 사전 토크나이징 결과 저장 테스트 | High | Pending |
| FR-23 | LanceDB FTS 인덱스 생성 (`create_fts_index("content_tokenized")`) 테스트 | High | Pending |
| FR-24 | FTS 키워드 검색 (MeCab 토크나이징된 쿼리로 검색) 테스트 | High | Pending |
| FR-25 | FTS 검색 결과 정확성 테스트 (관련 문서가 상위 랭크) | Medium | Pending |
| FR-26 | 하이브리드 검색 (벡터 + FTS) RRF 결합 테스트 | High | Pending |
| FR-27 | MeCab 미설치 환경에서의 graceful fallback 테스트 | Medium | Pending |
| FR-28 | FTS 쿼리 타입별 테스트 (MatchQuery, PhraseQuery, BooleanQuery) | Medium | Pending |

### 3.2 Non-Functional Requirements

| Category | Criteria | Measurement Method |
|----------|----------|-------------------|
| Performance | 전체 테스트 스위트 실행 30초 이내 | `pytest --durations=10` |
| Isolation | 각 테스트가 독립적 (임시 디렉토리 사용) | `tmp_path` 픽스처 활용 |
| Reproducibility | 고정 시드 랜덤 벡터로 재현 가능 | `numpy.random.seed(42)` |
| Coverage | 핵심 메서드 80% 이상 커버리지 | `pytest --cov` |

---

## 4. Success Criteria

### 4.1 Definition of Done

- [ ] 단위 테스트 파일 3개 작성 완료 (`test_vectorstore_schema.py`, `test_lancedb_store.py`, `test_mecab_tokenizer.py`)
- [ ] 통합 테스트 파일 1개 작성 완료 (`test_lancedb_integration.py` - FTS + 하이브리드 포함)
- [ ] 공유 픽스처 conftest 업데이트 완료 (MeCab 픽스처 포함)
- [ ] 모든 테스트 통과 (`uv run pytest` 성공)
- [ ] `ruff check` 린트 통과
- [ ] `mypy` 타입 체크 통과 (또는 합리적 ignore)

### 4.2 Quality Criteria

- [ ] 테스트 커버리지 80% 이상 (vectorstore 모듈 대상)
- [ ] 린트 에러 0개
- [ ] 각 테스트 함수에 명확한 docstring
- [ ] 테스트 함수명이 테스트 의도를 명확히 표현

---

## 5. Risks and Mitigation

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| LanceDB 라이브러리 버전 호환성 문제 | Medium | Low | pyproject.toml 의존성 버전 고정 확인 |
| 임시 디렉토리 정리 실패로 디스크 공간 낭비 | Low | Low | pytest `tmp_path` 자동 정리 활용 |
| LanceDBStore 내부 `settings` 의존성으로 모킹 복잡 | Medium | Medium | `monkeypatch`로 환경변수 오버라이드 또는 직접 파라미터 주입 |
| 1024차원 벡터 생성 시 테스트 속도 저하 | Low | Low | numpy 고정 시드로 빠른 생성, 최소 필요 문서만 사용 |
| PyArrow 스키마 변경 시 테스트 깨짐 | Medium | Low | 스키마 상수(`LEGAL_CHUNKS_SCHEMA`)를 직접 참조하여 동기화 유지 |
| MeCab 설치 환경 차이 (CI/로컬) | Medium | Medium | `@pytest.mark.requires_mecab` 마커로 MeCab 필요 테스트 분리, 미설치 시 skip |
| MeCab 사전(mecab-ko-dic) 미설치 | High | Medium | 설치 스크립트 제공 + CI에서 apt/brew 설치 단계 추가 |
| LanceDB FTS 인덱스 생성 실패 | Medium | Low | `create_fts_index` 호출 전 테이블 존재 확인, 에러 핸들링 |
| MeCab 토크나이징 결과 불일치 (사전 버전) | Low | Low | 테스트에서 정확한 토큰 대신 "포함 여부"로 검증 |

---

## 6. Architecture Considerations

### 6.1 Project Level Selection

| Level | Characteristics | Recommended For | Selected |
|-------|-----------------|-----------------|:--------:|
| **Starter** | Simple structure | Static sites | |
| **Dynamic** | Feature-based modules, services layer | Web apps with backend | **V** |
| **Enterprise** | Strict layer separation, DI | High-traffic systems | |

### 6.2 Key Architectural Decisions

| Decision | Options | Selected | Rationale |
|----------|---------|----------|-----------|
| Test Framework | pytest / unittest | pytest | 프로젝트 기존 사용, asyncio 지원 |
| Mock Strategy | unittest.mock / monkeypatch / fixtures | pytest fixtures + monkeypatch | 환경변수 오버라이드 용이 |
| 벡터 생성 | 실제 임베딩 / 랜덤 벡터 | 랜덤 벡터 (np.random) | GPU 불필요, 빠른 실행 |
| DB 격리 | 실제 LanceDB (tmp) / Mock | 실제 LanceDB (tmp_path) | 실제 동작 검증 가치 높음 |
| Async 테스트 | pytest-asyncio / synchronous | synchronous | LanceDBStore는 동기 API |
| 한국어 토크나이저 | MeCab / Okt / Komoran | MeCab (mecab-python3) | 속도 최고, 법률 용어 분석 정확도 높음 |
| MeCab 사전 | mecab-ko-dic / 커스텀 사전 | mecab-ko-dic (기본) | 표준 한국어 사전, 추후 법률 커스텀 사전 확장 가능 |
| FTS 전략 | LanceDB 네이티브 / 사전 토크나이징 | 사전 토크나이징 (Pre-tokenization) | LanceDB CJK 미지원으로 MeCab 사전 처리 필수 |

### 6.3 MeCab 사전 토크나이징 아키텍처

```
┌──────────────────────────────────────────────────────┐
│                    인덱싱 시점                         │
├──────────────────────────────────────────────────────┤
│                                                       │
│  원본 content ──► MeCab 형태소 분석 ──► content_tokenized │
│  "손해배상청구"     mecab.morphs()     "손해 배상 청구"      │
│                                                       │
│  content_tokenized ──► LanceDB FTS Index              │
│                        create_fts_index()              │
│                                                       │
├──────────────────────────────────────────────────────┤
│                    검색 시점                           │
├──────────────────────────────────────────────────────┤
│                                                       │
│  사용자 쿼리 ──► MeCab 형태소 분석 ──► FTS 검색         │
│  "손해배상"       mecab.morphs()     table.search("손해 배상") │
│                                                       │
│  Vector Search ◄──┐                                   │
│  FTS Search    ◄──┤──► RRF (Reciprocal Rank Fusion)  │
│                   └──► 최종 검색 결과                   │
└──────────────────────────────────────────────────────┘
```

### 6.4 Test Architecture

```
backend/tests/
├── conftest.py                          # 공유 픽스처 (기존 + LanceDB + MeCab 추가)
├── unit/
│   ├── test_vectorstore_schema.py       # schema_v2.py 단위 테스트 (FR-01~04, 18)
│   ├── test_lancedb_store.py            # LanceDBStore 단위 테스트 (FR-05~16, 19)
│   └── test_mecab_tokenizer.py          # MeCab 토크나이저 단위 테스트 (FR-20~22, 27)
└── integration/
    ├── test_lancedb_search.py           # 기존 (유지)
    └── test_lancedb_integration.py      # E2E 검색 + FTS + 하이브리드 테스트 (FR-17, 23~26, 28)
```

---

## 7. Convention Prerequisites

### 7.1 Existing Project Conventions

- [x] `CLAUDE.md` has coding conventions section
- [x] `.claude/rules/coding-style.md` exists
- [x] `.claude/rules/code-verification.md` exists
- [x] ruff configuration (pyproject.toml)
- [x] pytest configuration (pyproject.toml)
- [x] conftest.py with shared fixtures and markers

### 7.2 Test Conventions

| Category | Convention | Example |
|----------|-----------|---------|
| **파일명** | `test_<module>.py` | `test_lancedb_store.py` |
| **클래스명** | `Test<Feature>` (선택적) | `TestSchemaCreation` |
| **함수명** | `test_<행위>_<조건>_<기대결과>` | `test_search_by_type_law_returns_only_laws` |
| **픽스처** | `snake_case`, conftest에 공유 | `lancedb_store`, `sample_law_vector` |
| **마커** | `@pytest.mark.requires_lancedb` | 기존 conftest 마커 활용 |
| **Docstring** | 각 테스트 함수에 한국어 설명 | `"""법령 청크 생성 시 필수 필드 매핑 확인"""` |

### 7.3 Environment Variables Needed

| Variable | Purpose | Scope | Default |
|----------|---------|-------|---------|
| `LANCEDB_URI` | LanceDB 저장소 경로 | Test | `tmp_path/lancedb` |
| `LANCEDB_TABLE_NAME` | 테이블명 | Test | `test_legal_chunks` |
| `VECTOR_DB` | 벡터 DB 선택 | Test | `lancedb` |

### 7.4 Dependencies (추가 필요)

| Package | Version | Purpose | Install |
|---------|---------|---------|---------|
| `mecab-python3` | >=1.0.6 | MeCab Python 바인딩 | `uv add mecab-python3` |
| `mecab-ko-dic` | latest | 한국어 MeCab 사전 | OS별 설치 필요 (아래 참조) |

**MeCab 설치 (OS별)**:

```bash
# macOS
brew install mecab mecab-ko-dic

# Ubuntu/Debian
sudo apt-get install mecab libmecab-dev mecab-ko-dic

# pip (Python 바인딩)
uv add mecab-python3
```

---

## 8. Implementation Plan

### 8.1 Test File Details

#### File 1: `backend/tests/unit/test_vectorstore_schema.py`

**대상**: `backend/app/tools/vectorstore/schema_v2.py`

| Test Function | FR | 설명 |
|---------------|-----|------|
| `test_create_law_chunk_required_fields` | FR-01 | 법령 청크 생성 시 필수 필드(id, source_id, data_type, title, content, vector, date, source_name) 매핑 |
| `test_create_law_chunk_law_specific_fields` | FR-01 | 법령 전용 필드(promulgation_date, promulgation_no, law_type, article_no) 매핑 |
| `test_create_law_chunk_precedent_fields_null` | FR-04 | 법령 청크에서 판례 전용 필드(case_number, case_type 등)가 NULL |
| `test_create_precedent_chunk_required_fields` | FR-02 | 판례 청크 생성 시 필수 필드 매핑 |
| `test_create_precedent_chunk_precedent_specific_fields` | FR-02 | 판례 전용 필드(case_number, case_type, judgment_type 등) 매핑 |
| `test_create_precedent_chunk_law_fields_null` | FR-04 | 판례 청크에서 법령 전용 필드가 NULL |
| `test_legal_chunk_validation_valid_law` | FR-03 | 유효한 법령 LegalChunk Pydantic 검증 통과 |
| `test_legal_chunk_validation_valid_precedent` | FR-03 | 유효한 판례 LegalChunk Pydantic 검증 통과 |
| `test_legal_chunk_validation_invalid_data_type` | FR-03 | 잘못된 data_type 시 검증 실패 |
| `test_legal_chunk_validation_missing_required` | FR-03 | 필수 필드 누락 시 검증 실패 |
| `test_id_format_law` | FR-18 | 법령 ID 형식: `{law_id}_{chunk_idx}` |
| `test_id_format_precedent` | FR-18 | 판례 ID 형식: `{serial_number}_{chunk_idx}` |
| `test_schema_column_count` | - | 스키마 총 20개 컬럼 확인 |
| `test_vector_dimension` | - | 벡터 차원 1024 확인 |
| `test_column_groups` | - | COMMON_COLUMNS(10), LAW_COLUMNS(4), PRECEDENT_COLUMNS(6) 확인 |

#### File 2: `backend/tests/unit/test_lancedb_store.py`

**대상**: `backend/app/tools/vectorstore/lancedb.py`

| Test Function | FR | 설명 |
|---------------|-----|------|
| `test_store_initialization_default` | FR-05 | 기본 설정으로 초기화 |
| `test_store_initialization_custom_collection` | FR-05 | 커스텀 collection_name으로 초기화 |
| `test_add_law_documents` | FR-06 | 법령 문서 추가 후 카운트 검증 |
| `test_add_precedent_documents` | FR-07 | 판례 문서 추가 후 카운트 검증 |
| `test_add_law_documents_metadata` | FR-06 | 추가된 법령 문서의 메타데이터 필드 검증 |
| `test_add_precedent_documents_metadata` | FR-07 | 추가된 판례 문서의 메타데이터 필드 검증 |
| `test_search_returns_search_result` | FR-08, FR-19 | search()가 SearchResult 타입 반환 |
| `test_search_vector_similarity` | FR-08 | 유사 벡터가 상위에 랭크되는지 확인 |
| `test_search_n_results` | FR-08 | n_results 파라미터 동작 검증 |
| `test_search_by_type_law` | FR-09 | data_type="법령" 필터 검색 |
| `test_search_by_type_precedent` | FR-09 | data_type="판례" 필터 검색 |
| `test_count_total` | FR-10 | 전체 문서 수 카운트 |
| `test_count_by_type` | FR-10 | 유형별 문서 수 카운트 |
| `test_get_by_source_id` | FR-11 | source_id로 해당 문서의 모든 청크 조회 |
| `test_get_by_source_id_not_found` | FR-11 | 존재하지 않는 source_id 조회 시 빈 결과 |
| `test_delete_by_source_id` | FR-12 | source_id로 청크 삭제 후 카운트 감소 확인 |
| `test_escape_sql` | FR-13 | SQL 특수문자 이스케이프 처리 |
| `test_build_filter_conditions` | FR-14 | WHERE 조건 생성 (단일, 복합 필터) |
| `test_reset` | FR-15 | reset() 후 count() == 0 |
| `test_get_vector_store_returns_lancedb` | FR-16 | VECTOR_DB=lancedb일 때 LanceDBStore 반환 |
| `test_search_result_dict_access` | FR-19 | SearchResult의 dict-like 접근 (get, __getitem__) |

#### File 3: `backend/tests/unit/test_mecab_tokenizer.py`

**대상**: MeCab 한국어 형태소 분석 및 사전 토크나이징

| Test Function | FR | 설명 |
|---------------|-----|------|
| `test_mecab_initialization` | FR-20 | MeCab 토크나이저 초기화 성공 |
| `test_mecab_morphs_basic` | FR-20 | 기본 한국어 문장 형태소 분석 |
| `test_mecab_legal_terms_tokenization` | FR-21 | 법률 용어 토크나이징 ("손해배상청구" → ["손해", "배상", "청구"]) |
| `test_mecab_article_reference` | FR-21 | 조문 참조 토크나이징 ("민법 제750조" 처리) |
| `test_mecab_case_number` | FR-21 | 사건번호 토크나이징 ("84나3990" 처리) |
| `test_pretokenize_content_for_fts` | FR-22 | 원본 content → 토크나이징된 문자열 변환 |
| `test_pretokenize_preserves_searchability` | FR-22 | 토크나이징 결과에 핵심 형태소가 포함되는지 검증 |
| `test_pretokenize_query` | FR-22 | 검색 쿼리도 동일한 토크나이징 적용 |
| `test_mecab_not_installed_fallback` | FR-27 | MeCab 미설치 시 공백 분리 fallback |
| `test_mecab_empty_string` | FR-20 | 빈 문자열 입력 시 에러 없이 처리 |
| `test_mecab_mixed_korean_english` | FR-20 | 한영 혼합 텍스트 처리 ("OWASP 보안 취약점") |

#### File 4: `backend/tests/integration/test_lancedb_integration.py`

**대상**: E2E 벡터 검색 + FTS + 하이브리드 검색 흐름

| Test Function | FR | 설명 |
|---------------|-----|------|
| **벡터 검색 E2E** | | |
| `test_e2e_add_and_search_law` | FR-17 | 법령 추가 → 벡터 검색 → 결과 검증 |
| `test_e2e_add_and_search_precedent` | FR-17 | 판례 추가 → 벡터 검색 → 결과 검증 |
| `test_e2e_mixed_search_filter_by_type` | FR-17 | 법령+판례 추가 → 유형별 필터 검색 |
| `test_e2e_source_id_extraction_flow` | FR-17 | 검색 → source_id 추출 → 원본 문서 ID 매핑 |
| `test_e2e_multi_chunk_document` | FR-17 | 다중 청크 문서 추가 → get_by_source_id로 전체 복원 |
| `test_e2e_add_search_delete_lifecycle` | FR-17 | 추가 → 검색 확인 → 삭제 → 검색 미발견 |
| **FTS E2E** | | |
| `test_fts_index_creation` | FR-23 | content_tokenized 컬럼에 FTS 인덱스 생성 |
| `test_fts_keyword_search` | FR-24 | MeCab 토크나이징된 쿼리로 FTS 검색 |
| `test_fts_search_accuracy` | FR-25 | 관련 법률 문서가 FTS 상위에 랭크 |
| `test_fts_match_query` | FR-28 | MatchQuery 단일 키워드 검색 |
| `test_fts_phrase_query` | FR-28 | PhraseQuery 구문 검색 |
| `test_fts_boolean_query` | FR-28 | BooleanQuery AND/OR 조합 검색 |
| **하이브리드 검색 E2E** | | |
| `test_hybrid_search_rrf` | FR-26 | 벡터 + FTS 결과를 RRF로 결합하여 최종 순위 검증 |
| `test_hybrid_search_vector_only_fallback` | FR-26 | FTS 결과 없을 때 벡터 검색만으로 동작 |
| `test_hybrid_search_fts_boost` | FR-26 | 정확한 키워드 매칭 시 FTS가 순위에 기여하는지 검증 |

### 8.2 Shared Fixtures (conftest.py 추가)

```python
# ===== LanceDB 픽스처 =====
@pytest.fixture
def lancedb_tmp_dir(tmp_path):
    """임시 LanceDB 디렉토리"""

@pytest.fixture
def sample_vector_1024():
    """고정 시드 1024차원 랜덤 벡터"""

@pytest.fixture
def sample_law_data():
    """샘플 법령 데이터 (3건, 각 2청크) - content_tokenized 포함"""

@pytest.fixture
def sample_precedent_data():
    """샘플 판례 데이터 (3건, 각 2청크) - content_tokenized 포함"""

@pytest.fixture
def lancedb_store(lancedb_tmp_dir, monkeypatch):
    """임시 디렉토리 기반 LanceDBStore 인스턴스"""

# ===== MeCab 픽스처 =====
@pytest.fixture
def mecab_tokenizer():
    """MeCab 토크나이저 인스턴스 (미설치 시 skip)"""

@pytest.fixture
def sample_legal_texts():
    """법률 도메인 샘플 텍스트 (토크나이징 테스트용)"""
    # 예: "불법행위로 인한 손해배상청구권", "민법 제750조", ...

# ===== pytest 마커 =====
# conftest.py에 추가
pytest.mark.requires_mecab  # MeCab 설치 필요 테스트
pytest.mark.requires_fts    # FTS 인덱스 필요 테스트
```

---

## 9. Next Steps

1. [ ] Design 문서 작성 (`lancedb-test.design.md`) - 각 테스트의 구체적 구현 설계
2. [ ] MeCab + mecab-ko-dic 설치 확인 및 의존성 추가
3. [ ] 리뷰 및 승인
4. [ ] 구현 시작 순서:
   - MeCab 토크나이저 유틸리티 구현
   - schema_v2에 `content_tokenized` 컬럼 추가
   - 픽스처 작성
   - 스키마 테스트 → MeCab 테스트 → 저장소 테스트 → FTS/하이브리드 통합 테스트

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-02-05 | Initial draft based on vectordb_design.md | Claude |
| 0.2 | 2026-02-05 | MeCab FTS 요구사항 추가 (FR-20~28), 사전 토크나이징 전략, test_mecab_tokenizer.py 추가 | Claude |
