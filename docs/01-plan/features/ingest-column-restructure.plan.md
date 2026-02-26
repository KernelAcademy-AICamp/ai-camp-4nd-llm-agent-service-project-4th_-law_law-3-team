# Plan: 문서별 인제스트·FTS·LLM Context 칼럼 재구성

> **Feature**: ingest-column-restructure
> **Phase**: Plan
> **Created**: 2026-02-26
> **Status**: Draft

---

## 1. 배경 및 목적

### 현재 상태

21개 문서 타입에 대해 FTS body, LLM context content_columns, context format이 설정되어 있으나:
- FTS body에 불필요한 필드가 포함되어 노이즈 증가 (예: 판례에서 reference_provisions, full_text 등)
- LLM context content_columns가 FTS body와 분리 없이 동일 구조 사용
- context format에 문서 식별 메타데이터가 일관성 없이 적용
- law, admin_rule의 content가 너무 길어 LLM 토큰 낭비

### 목적

1. **FTS body**: 검색에 핵심적인 칼럼만 선별하여 FTS 정밀도 개선
2. **LLM context content_columns**: LLM에 전달할 핵심 내용 칼럼 재정의 + 칼럼별 압축률 지정
3. **Context format**: `context format\n + LLM context` 구조로 일관 적용

### LLM Context 구조

최종 LLM context = **context format 필드** + **content_columns 필드**, 모두 `[field] value` 형태.

```
### 판례 1
[case_name] 손해배상(기)              ← context format (metadata에서 추출)
[case_number] 2020다12345             ← context format (metadata에서 추출)
[reasoning] 이 사건의 쟁점은...        ← content column (PostgreSQL 원문 조회)
[ruling] 원고의 청구를 기각한다...      ← content column (PostgreSQL 원문 조회)
```

### 범위

- **포함**:
  - 17개 타입의 FTS body 칼럼 변경 (ingest types의 `_orm_fulltext_fn`)
  - DOCUMENT_TABLE_REGISTRY content_columns 변경 (`retrieval.py`)
  - COLUMN_COMPRESSION_POLICIES 압축률 추가/변경 (`compression.py`)
  - `format_utils.py`의 CONTEXT_FORMAT 레지스트리 + 포맷 함수 변경
- **제외**:
  - Passage화 (law, admin_rule) → **별도 태스크로 분리**
  - 벡터 임베딩 재생성, FTS 인덱스 재빌드 (별도 실행)
  - 프론트엔드 변경

---

## 2. 목표

| 항목 | Before | After |
|------|--------|-------|
| FTS body 칼럼 | 과다 포함 (판례 9개 등) | 핵심만 선별 (판례 3개 등) |
| LLM context | FTS와 동일 content_columns | 별도 선별, 압축률 지정 |
| Context 구조 | `[type N] title (id)\ncontent` | `### type N\n[context_format]\n[content_columns]` |

---

## 3. 변경 명세

### 3.1 FTS Body 칼럼 변경 (ingest types의 `orm_fulltext_fn`)

| 타입 | 현재 FTS body | 변경 후 FTS body | 변경 |
|------|--------------|-----------------|------|
| **precedent** | case_name, case_number, summary, reasoning, ruling, claim, full_reason/full_text, ref_provisions, ref_cases | case_name, summary, reasoning | 축소 |
| **law** | law_name, content, supplementary | law_name, content(조문 내용, 항 내용) | supplementary 제거 |
| **admin_rule** | admin_rule_name, content, supplementary | admin_rule_name, content(내용) | supplementary 제거 |
| **interpretation_ministry** | case_name, ministry_name, inquiry, answer, reason, related_law | case_name, inquiry, answer | 축소 |
| **constitutional** | case_name, case_number, summary, reasoning, ruling, reason, ref_provisions, ref_statutes, ref_cases | case_name, summary, reasoning | 축소 |
| **administration** | case_name, case_number, ruling, claim, reason, adjudication_summary | case_name, claim, reason | 축소 |
| **legislation** | case_name, inquiry, answer, reason | case_name, inquiry, answer | reason 제거 |
| **treaty** | treaty_name_kr, treaty_name_en, counterpart_country_kr, content | treaty_name_kr, content | 축소 |
| **special_admin** | case_name, case_number, ruling, claim, reason, adjudication_summary | case_name, adjudication_summary(없으면 reason) | fallback 로직 |
| **dec_labor** | case_name, case_number, judgment_matter, judgment_summary, judgment_result, full_text | case_name, judgment_matter, judgment_summary | 축소 |
| **dec_human_rights** | case_name, case_number, decision_summary, judgment_summary, ruling, reason | case_name, decision_summary, judgment_summary | 축소 |
| **dec_privacy** | case_name, reason | case_name, reason | **변경 없음** |
| **dec_employment** | case_name, case_number, ruling, reason, claim, overview | case_name, claim, overview | 축소 |
| **dec_financial** | case_name, action_reason, action_content | case_name, action_reason, action_content | **변경 없음** |
| **dec_industrial** | title(case_label/case_number), ruling, reason, issue, claim | case_label, case_major_category, case_mid_category, case_sub_category, issue, claim | 분류 추가 |
| **dec_environment** | case_name, ruling, evaluation_opinion, party_claims, fact_investigation, case_overview | case_name, party_claims, fact_investigation, case_overview | ruling/eval 제거 |
| **dec_securities** | case_name, action_reason, action_content | case_name, action_reason | action_content 제거 |
| **dec_civil_rights** | case_name, ruling, reason, decision_summary | case_name, complaint_flag, ruling | complaint_flag 추가 |
| **dec_fair_trade** | case_name, case_number, decision_summary, ruling, reason | case_name, decision_summary, ruling | 축소 |
| **dec_media** | case_name, ruling | case_name, ruling | **변경 없음** |

### 3.2 LLM Context Content Columns 변경 (DOCUMENT_TABLE_REGISTRY)

`retrieval.py`의 `DOCUMENT_TABLE_REGISTRY`에서 PostgreSQL 원문 조회에 사용되는 칼럼.
괄호 안 숫자는 `compression.py`의 `COLUMN_COMPRESSION_POLICIES` 압축률 (없으면 기존 설정 유지).

**Context format에 필요하지만 metadata에 없는 필드**는 content_columns에 포함하여 PostgreSQL에서 함께 조회:
- `decision_number`: dec_financial, dec_environment, dec_securities, dec_civil_rights
- `counterpart_country_kr`: treaty

| 타입 | 현재 content_columns | 변경 후 content_columns |
|------|---------------------|----------------------|
| **판례** | ruling, reasoning | reasoning, ruling |
| **법령** | content | content (passage화 별도 태스크) |
| **행정규칙** | content | content (passage화 별도 태스크) |
| **부처유권해석** | answer, reason | answer, related_law(1.0) |
| **헌재결정례** | ruling, reasoning | reasoning, ruling |
| **행정심판례** | ruling, reason | ruling(1.0), adjudication_summary(0.4) |
| **법령해석례** | answer, reason | answer(1.0) |
| **조약** | content | counterpart_country_kr(1.0), content |
| **특별행정심판** | ruling, reason | ruling, adjudication_summary(0.4), reason |
| **노동위원회** | judgment_summary, judgment_result | 변경 없음 |
| **국가인권위** | ruling, judgment_summary | 변경 없음 |
| **개인정보보호** | reason | 변경 없음 |
| **고용보험심사** | ruling, reason | 변경 없음 |
| **금융위원회** | action_reason, action_content | decision_number(1.0), action_content, action_reason |
| **산업재해보상** | ruling, reason | 변경 없음 |
| **환경분쟁** | ruling, evaluation_opinion | 변경 없음 |
| **증권선물위** | action_reason, action_content | decision_number(1.0), action_reason |
| **국민권익위** | ruling, reason | decision_number(1.0), ruling, reason |
| **공정거래위** | ruling, reason | 변경 없음 |
| **방송통신위** | ruling | 변경 없음 |

**특별행정심판 fallback**: content_columns에 `adjudication_summary`, `reason` 모두 포함. format_utils.py에서 adjudication_summary가 있으면 reason 제외.

### 3.3 Context Format 변경 (format_utils.py)

LLM context 출력 시 **context format 필드가 content_columns 위에** `[field] value` 형태로 삽입.
context format 필드의 소스: **metadata** (case_name, case_number, title 등) 또는 **content_fields** (decision_number, counterpart_country_kr).

#### 구현 아키텍처

`format_utils.py`에 `CONTEXT_FORMAT` 레지스트리 추가:
- data_type(한국어 라벨) → context format 필드 목록
- 위원회결정례(11개 타입 공유 data_type) → `doc_id` 접두사(예: `dec_labor:12345`)로 구분

```python
# (출력_라벨, 소스, 소스_키)
# 소스: "meta" = metadata에서 추출, "cf" = content_fields에서 pop(중복 방지)
CONTEXT_FORMAT: dict[str, list[tuple[str, str, str]]] = {
    "판례": [("case_name", "meta", "case_name"), ("case_number", "meta", "case_number")],
    "조약": [("조약명", "meta", "title"), ("체결대상국가", "cf", "counterpart_country_kr")],
    ...
}
```

#### 타입별 Context Format 필드

| 타입 | Context Format | 필드 소스 |
|------|---------------|----------|
| **precedent** | `[case_name]` + `[case_number]` | meta, meta |
| **law** | `[law_name]` | meta(title) |
| **admin_rule** | `[admin_rule_name]` | meta(title) |
| **interpretation_ministry** | `[case_name]` | meta |
| **constitutional** | `[case_number]` | meta |
| **administration** | `[case_name]` + `[case_number]` | meta, meta |
| **legislation** | `[case_name]` + `[case_number]` | meta, meta |
| **treaty** | `[조약명]` + `[체결대상국가]` | meta(title), cf |
| **special_admin** | `[case_name]` + `[case_number]` | meta, meta |
| **dec_labor** | `[case_name]` + `[case_number]` | meta, meta |
| **dec_human_rights** | `[case_name]` + `[case_number]` | meta, meta |
| **dec_privacy** | `[case_name]` | meta |
| **dec_employment** | `[case_name]` + `[case_number]` | meta, meta |
| **dec_financial** | `[case_name]` + `[decision_number]` | meta, cf |
| **dec_industrial** | `[case_label]` + `[case_number]` | meta(title), meta |
| **dec_environment** | `[case_name]` + `[decision_number]` | meta, cf |
| **dec_securities** | `[case_name]` + `[decision_number]` | meta, cf |
| **dec_civil_rights** | `[case_name]` + `[decision_number]` | meta, cf |
| **dec_fair_trade** | `[case_name]` + `[case_number]` | meta, meta |
| **dec_media** | `[case_name]` + `[case_number]` | meta, meta |

#### 최종 출력 예시

```
## 관련 판례

### 판례 1
[case_name] 손해배상(기)
[case_number] 2020다12345
[reasoning] 이 사건의 쟁점은...
[ruling] 원고의 청구를 기각한다...

### 판례 2
[case_name] 부당이득반환
[case_number] 2021나56789
[reasoning] ...
[ruling] ...
```

### 3.4 Compression Policy 신규/변경

| 칼럼 | 현재 압축률 | 변경 압축률 | 비고 |
|------|-----------|-----------|------|
| adjudication_summary | 미등록(0.5 기본) | 0.4 | 행정심판례, 특별행정심판 |
| related_law | 미등록(0.5 기본) | 1.0 | 부처유권해석: 관련법령 보존 |
| decision_number | 미등록(0.5 기본) | 1.0 | 식별자 보존 (4개 위원회) |
| counterpart_country_kr | 미등록(0.5 기본) | 1.0 | 조약: 체결대상국가 보존 |

### 3.5 Passage화 — 별도 태스크

> **이 태스크에서 제외.** law, admin_rule, (dec_fair_trade?)의 passage화는 별도 태스크로 분리.

---

## 4. 수정 대상 파일

### 4.1 FTS Body 변경 (17개 파일)

| # | 파일 | 변경 내용 |
|---|------|----------|
| 1 | `backend/scripts/ingest/types/precedent.py` | `_orm_fulltext_fn` 축소 |
| 2 | `backend/scripts/ingest/types/law.py` | supplementary 제거 |
| 3 | `backend/scripts/ingest/types/admin_rule.py` | supplementary 제거 |
| 4 | `backend/scripts/ingest/types/interpretation_ministry.py` | 필드 축소 |
| 5 | `backend/scripts/ingest/types/constitutional.py` | 필드 축소 |
| 6 | `backend/scripts/ingest/types/administration.py` | 필드 축소 |
| 7 | `backend/scripts/ingest/types/legislation.py` | reason 제거 |
| 8 | `backend/scripts/ingest/types/treaty.py` | 필드 축소 |
| 9 | `backend/scripts/ingest/types/special_admin_appeal.py` | fallback 로직 |
| 10 | `backend/scripts/ingest/types/dec_labor.py` | 필드 축소 |
| 11 | `backend/scripts/ingest/types/dec_human_rights.py` | 필드 축소 |
| 12 | `backend/scripts/ingest/types/dec_employment.py` | 필드 축소 |
| 13 | `backend/scripts/ingest/types/dec_industrial.py` | 분류 칼럼 추가 |
| 14 | `backend/scripts/ingest/types/dec_environment.py` | ruling/eval 제거 |
| 15 | `backend/scripts/ingest/types/dec_securities.py` | action_content 제거 |
| 16 | `backend/scripts/ingest/types/dec_civil_rights.py` | complaint_flag 추가 |
| 17 | `backend/scripts/ingest/types/dec_fair_trade.py` | 축소 |

### 4.2 LLM Context Content Columns (2개 파일)

| # | 파일 | 변경 내용 |
|---|------|----------|
| 18 | `backend/app/services/rag/retrieval.py` | DOCUMENT_TABLE_REGISTRY 변경 |
| 19 | `backend/app/services/rag/compression.py` | COLUMN_COMPRESSION_POLICIES 추가 |

### 4.3 Context Format (1개 파일)

| # | 파일 | 변경 내용 |
|---|------|----------|
| 20 | `backend/app/services/rag/format_utils.py` | 문서 타입별 헤더 포맷 변경 |

---

## 5. 구현 순서

### Step 1: DOCUMENT_TABLE_REGISTRY + 압축 정책 변경
1. `retrieval.py`의 content_columns 변경 (decision_number, counterpart_country_kr 추가 포함)
2. `compression.py`의 COLUMN_COMPRESSION_POLICIES 신규 칼럼 추가
3. 린트 검증

### Step 2: Context Format 레지스트리 + 포맷 함수 변경
1. `format_utils.py`에 `CONTEXT_FORMAT` 레지스트리 추가
2. `_format_document_context()` 헬퍼 함수 추가 (context format + content fields 통합 포맷)
3. `format_precedent_context`, `format_law_context`, `format_supplementary_context` 수정
4. 린트 검증

### Step 3: FTS body 칼럼 변경 (ingest types)
1. 17개 타입 파일의 `_orm_fulltext_fn` (및 `_fulltext_fn`) 수정
2. `_fts_metadata_fn`은 변경 불필요 (메타데이터 유지)
3. 린트 검증

---

## 6. 리스크 및 고려사항

| 리스크 | 영향 | 대응 |
|--------|------|------|
| FTS 인덱스 재빌드 필요 | FTS body 변경 후 기존 인덱스와 불일치 | `scripts/ingest/cli --step fts --reset` 별도 실행 |
| DOCUMENT_TABLE_REGISTRY 변경 시 원문 조회 실패 | 존재하지 않는 칼럼 참조 | ORM 모델 칼럼 존재 확인 완료 (전수 검증) |
| 압축률 변경으로 LLM 응답 품질 영향 | 과도한 압축 시 정보 손실 | 기존 검증된 압축률(0.4, 1.0) 위주 사용 |

**ORM 칼럼 존재 확인 완료** (전수 검증):
- decision_number: dec_financial ✅, dec_environment ✅, dec_securities ✅, dec_civil_rights ✅
- counterpart_country_kr: treaty ✅
- complaint_flag: dec_civil_rights ✅
- adjudication_summary: administration ✅, special_admin_appeal ✅
- 기타 모든 변경 대상 칼럼: 전부 확인 완료

---

## 7. 검증 방법

1. **정적 검증**: `ruff check backend/app/ backend/scripts/ingest/` + `mypy backend/app/services/rag/`
2. **FTS body 검증**: FTS 재빌드는 별도 실행 (코드 변경만 이 태스크에서 수행)
3. **LLM Context 검증**: 검색 파이프라인 실행 후 content_fields 구조 + `[field] value` 출력 확인

---

## 8. 의존성

- 현재 `feature/keyword-search` 브랜치에서 작업
- FTS 재빌드는 PostgreSQL 실행 상태 필요 (별도 실행)
- Passage화는 별도 태스크로 분리
