# 데이터 타입별 PostgreSQL 전체 칼럼 매핑 + Context 활용 + 압축 정책

> **기준 파일**: `backend/app/models/` (ORM 모델), `backend/app/services/rag/retrieval.py` (`DOCUMENT_TABLE_REGISTRY`)
> **압축 정책**: `backend/app/services/rag/compression.py` (`COLUMN_COMPRESSION_POLICIES`)
> **Last Updated**: 2026-02-26

---

## 범례

| 표기 | 의미 |
|------|------|
| **굵은 글씨** | Context 활용 칼럼 (`DOCUMENT_TABLE_REGISTRY.content_columns`) |
| 1.0 | 보존 (압축 안 함) — 결론/결과 칼럼 |
| 0.5 | 중간 압축 — 혼합 칼럼 |
| 0.4 | 적극 압축 — 근거/이유 칼럼 |
| `-` | Context 미활용 (압축 대상 아님) |

> 모든 테이블의 `id`, `created_at`, `updated_at` 시스템 칼럼은 생략.

---

## 1. 기본 타입 (9개)

### 1.1 판례 (`precedent_documents`) — 16 칼럼

| 원본(한글) | 적재 칼럼 | Context | 압축률 |
|------------|----------|:-------:|:------:|
| 판례정보일련번호 | `serial_number` | | - |
| 사건명 | `case_name` | | - |
| 사건번호 | `case_number` | | - |
| 선고일자 | `decision_date` | | - |
| 법원명 | `court_name` | | - |
| 사건종류명 | `case_type` | | - |
| 판결유형 | `judgment_type` | | - |
| 판시사항 | `summary` | | - |
| **판결요지** | **`reasoning`** | **O** | **0.4** |
| **주문** | **`ruling`** | **O** | **1.0** |
| 청구취지 | `claim` | | - |
| 이유 | `full_reason` | | - |
| 판례내용 | `full_text` | | - |
| 판례요약 | `ai_summary` | | - |
| 참조조문 | `reference_provisions` | | - |
| 참조판례 | `reference_cases` | | - |

### 1.2 법령 (`law_documents`) — 10 칼럼

| 원본(한글) | 적재 칼럼 | Context | 압축률 |
|------------|----------|:-------:|:------:|
| 법령ID | `law_id` | | - |
| 법령명_한글 | `law_name` | | - |
| 법령구분 | `law_type` | | - |
| 소관부처명 | `ministry` | | - |
| 공포일자 | `promulgation_date` | | - |
| 공포번호 | `promulgation_no` | | - |
| 시행일자 | `enforcement_date` | | - |
| **조문(리스트)** | **`content`** | **O** | **0.5** |
| 부칙 | `supplementary` | | - |
| 법령 요약 | `ai_summary` | | - |

### 1.3 행정규칙 (`admin_rule_documents`) — 12 칼럼

| 원본(한글) | 적재 칼럼 | Context | 압축률 |
|------------|----------|:-------:|:------:|
| 행정규칙ID | `admin_rule_id` | | - |
| 행정규칙일련번호 | `serial_number` | | - |
| 행정규칙명 | `admin_rule_name` | | - |
| 행정규칙종류 | `admin_rule_type` | | - |
| 소관부처명 | `ministry` | | - |
| 소관부처코드 | `ministry_code` | | - |
| 상위부처명 | `parent_ministry` | | - |
| 발령일자 | `promulgation_date` | | - |
| 시행일자 | `enforcement_date` | | - |
| **조문내용(리스트)** | **`content`** | **O** | **0.5** |
| 부칙내용 | `supplementary` | | - |
| 전체요약 | `ai_summary` | | - |

### 1.4 부처유권해석 (`interpretation_ministry_documents`) — 11 칼럼

| 원본(한글) | 적재 칼럼 | Context | 압축률 |
|------------|----------|:-------:|:------:|
| 법령해석일련번호 | `serial_number` | | - |
| 안건명 | `case_name` | | - |
| 해석일자 | `interpretation_date` | | - |
| 안건번호 | `case_number` | | - |
| 질의요지 | `inquiry` | | - |
| 관련법령 | `related_law` | | - |
| **회답** | **`answer`** | **O** | **1.0** |
| **이유** | **`reason`** | **O** | **0.4** |
| 업무분야 | `business_field` | | - |
| 소관부처명 | `ministry_name` | | - |
| 해석요약 | `ai_summary` | | - |

### 1.5 헌재결정례 (`constitutional_documents`) — 16 칼럼

| 원본(한글) | 적재 칼럼 | Context | 압축률 |
|------------|----------|:-------:|:------:|
| 헌재결정례일련번호 | `serial_number` | | - |
| 사건번호 | `case_number` | | - |
| 사건명 | `case_name` | | - |
| 사건종류명 | `case_type` | | - |
| 사건종류코드 | `case_type_code` | | - |
| 종국일자 | `decision_date` | | - |
| 재판부구분코드 | `court_division_code` | | - |
| 판시사항 | `summary` | | - |
| **결정요지** | **`reasoning`** | **O** | **0.4** |
| **주문** | **`ruling`** | **O** | **1.0** |
| 전문 | `full_text` | | - |
| 이유 | `reason` | | - |
| 심판대상조문 | `reference_provisions` | | - |
| 참조조문 | `reference_statutes` | | - |
| 참조판례 | `reference_cases` | | - |
| 심판례요약 | `ai_summary` | | - |

### 1.6 행정심판례 (`administration_documents`) — 14 칼럼

| 원본(한글) | 적재 칼럼 | Context | 압축률 |
|------------|----------|:-------:|:------:|
| 행정심판례일련번호 | `serial_number` | | - |
| 사건명 | `case_name` | | - |
| 사건번호 | `case_number` | | - |
| 의결일자 | `decision_date` | | - |
| 처분일자 | `disposition_date` | | - |
| 처분청 | `disposition_agency` | | - |
| 재결청 | `adjudication_agency` | | - |
| 재결례유형명 | `case_type` | | - |
| 재결례유형코드 | `case_type_code` | | - |
| **주문** | **`ruling`** | **O** | **1.0** |
| 청구취지 | `claim` | | - |
| **이유** | **`reason`** | **O** | **0.4** |
| 재결요지 | `adjudication_summary` | | - |
| 심판례요약 | `ai_summary` | | - |

### 1.7 법령해석례 (`legislation_documents`) — 14 칼럼

| 원본(한글) | 적재 칼럼 | Context | 압축률 |
|------------|----------|:-------:|:------:|
| 법령해석례일련번호 | `serial_number` | | - |
| 안건명 | `case_name` | | - |
| 안건번호 | `case_number` | | - |
| 해석일자 | `interpretation_date` | | - |
| 등록일시 | `registration_date` | | - |
| 해석기관코드 | `interpretation_agency_code` | | - |
| 해석기관명 | `interpretation_agency` | | - |
| 질의기관코드 | `inquiry_agency_code` | | - |
| 질의기관명 | `inquiry_agency` | | - |
| 관리기관코드 | `management_agency_code` | | - |
| 질의요지 | `inquiry` | | - |
| **회답** | **`answer`** | **O** | **1.0** |
| **이유** | **`reason`** | **O** | **0.4** |
| 해석례요약 | `ai_summary` | | - |

### 1.8 조약 (`treaty_documents`) — 22 칼럼

| 원본(한글) | 적재 칼럼 | Context | 압축률 |
|------------|----------|:-------:|:------:|
| 조약일련번호 | `serial_number` | | - |
| 조약번호 | `treaty_number` | | - |
| 조약명_한글 | `treaty_name_kr` | | - |
| 조약명_영문 | `treaty_name_en` | | - |
| 조약구분코드 | `treaty_type_code` | | - |
| 체결대상국가(영문) | `counterpart_country` | | - |
| 체결대상국가(한글) | `counterpart_country_kr` | | - |
| 국가코드 | `country_code` | | - |
| 양자조약분야코드 | `bilateral_field_code` | | - |
| 양자조약분야명 | `bilateral_field` | | - |
| 서명일자 | `signing_date` | | - |
| 서명장소 | `signing_place` | | - |
| 발효일자 | `effective_date` | | - |
| 국회비준동의여부 | `parliament_approval` | | - |
| 국회비준동의일자 | `parliament_approval_date` | | - |
| 국무회의심의일자 | `cabinet_review_date` | | - |
| 국무회의심의회차 | `cabinet_review_session` | | - |
| 대통령재가일자 | `presidential_approval_date` | | - |
| 관보게재일자 | `gazette_date` | | - |
| **조약내용** | **`content`** | **O** | **0.5** |
| 비고 | `note` | | - |
| 조약요약 | `ai_summary` | | - |

### 1.9 특별행정심판 (`special_admin_appeal_documents`) — 21 칼럼

| 원본(한글) | 적재 칼럼 | Context | 압축률 |
|------------|----------|:-------:|:------:|
| 특별행정심판재결례일련번호 | `serial_number` | | - |
| 사건명 | `case_name` | | - |
| 재결번호 | `case_number` | | - |
| 의결일자 | `decision_date` | | - |
| 재결청 | `adjudication_agency` | | - |
| 재결례유형명 | `case_type` | | - |
| **주문** | **`ruling`** | **O** | **1.0** |
| 청구취지 | `claim` | | - |
| **이유** | **`reason`** | **O** | **0.4** |
| 재결요지 | `adjudication_summary` | | - |
| 참조결정 | `related_rulings` | | - |
| 따른결정 | `following_rulings` | | - |
| 세목 (조세심판원) | `tax_category` | | - |
| 관련법령 (조세심판원) | `related_law` | | - |
| 선박유형 (해양안전심판원) | `vessel_type` | | - |
| 사고유형 (해양안전심판원) | `accident_type` | | - |
| 해심위치 (해양안전심판원) | `tribunal_location` | | - |
| 해양사고관련자 (해양안전심판원) | `related_persons` | | - |
| 별지 (해양안전심판원) | `appendix` | | - |
| 재심청구안내 (해양안전심판원) | `retrial_notice` | | - |
| AI 요약 | `ai_summary` | | - |

---

## 2. 위원회결정례 (11개 테이블)

### 2.1 노동위 (`dec_labor_documents`) — 12 칼럼

| 원본(한글) | 적재 칼럼 | Context | 압축률 |
|------------|----------|:-------:|:------:|
| 결정문일련번호 | `serial_number` | | - |
| 제목 | `case_name` | | - |
| 사건번호 | `case_number` | | - |
| 등록일 | `decision_date` | | - |
| 판정사항 | `judgment_matter` | | - |
| **판정요지** | **`judgment_summary`** | **O** | **1.0** |
| **판정결과** | **`judgment_result`** | **O** | **1.0** |
| 내용 | `full_text` | | - |
| 자료구분 | `data_category` | | - |
| 담당부서 | `department` | | - |
| 기관명 | `organization_name` | | - |
| 결정문요약 | `ai_summary` | | - |

### 2.2 인권위 (`dec_human_rights_documents`) — 13 칼럼

| 원본(한글) | 적재 칼럼 | Context | 압축률 |
|------------|----------|:-------:|:------:|
| 결정문일련번호 | `serial_number` | | - |
| 사건명 | `case_name` | | - |
| 사건번호 | `case_number` | | - |
| 의결일자 | `decision_date` | | - |
| 결정요지 | `decision_summary` | | - |
| **판단요지** | **`judgment_summary`** | **O** | **1.0** |
| 분류명 | `classification_name` | | - |
| **주문** | **`ruling`** | **O** | **1.0** |
| 주문요지 | `ruling_summary` | | - |
| 이유 | `reason` | | - |
| 별지 | `appendix` | | - |
| 결정례전문 | `full_text` | | - |
| 결정문요약 | `ai_summary` | | - |

### 2.3 개인정보위 (`dec_privacy_documents`) — 5 칼럼

| 원본(한글) | 적재 칼럼 | Context | 압축률 |
|------------|----------|:-------:|:------:|
| 결정문일련번호 | `serial_number` | | - |
| 안건명 | `case_name` | | - |
| 의결일자 | `decision_date` | | - |
| **이유** | **`reason`** | **O** | **0.4** |
| 결정문요약 | `ai_summary` | | - |

### 2.4 고용보험 (`dec_employment_documents`) — 14 칼럼

| 원본(한글) | 적재 칼럼 | Context | 압축률 |
|------------|----------|:-------:|:------:|
| 결정문일련번호 | `serial_number` | | - |
| 사건명 | `case_name` | | - |
| 사건번호 | `case_number` | | - |
| 사건의분류 | `case_classification` | | - |
| 의결일자 | `decision_date` | | - |
| 의결서종류 | `resolution_type` | | - |
| **주문** | **`ruling`** | **O** | **1.0** |
| **이유** | **`reason`** | **O** | **0.4** |
| 청구취지 | `claim` | | - |
| 청구인 | `petitioner` | | - |
| 피청구인 | `respondent` | | - |
| 개요 | `overview` | | - |
| 기관명 | `organization_name` | | - |
| 결정문요약 | `ai_summary` | | - |

### 2.5 금융위 (`dec_financial_documents`) — 7 칼럼

| 원본(한글) | 적재 칼럼 | Context | 압축률 |
|------------|----------|:-------:|:------:|
| 결정문일련번호 | `serial_number` | | - |
| 안건명 | `case_name` | | - |
| 의결번호 | `decision_number` | | - |
| **조치이유** | **`action_reason`** | **O** | **0.5** |
| **조치내용** | **`action_content`** | **O** | **1.0** |
| 기관명 | `organization_name` | | - |
| 결정문요약 | `ai_summary` | | - |

### 2.6 산재심 (`dec_industrial_documents`) — 15 칼럼

| 원본(한글) | 적재 칼럼 | Context | 압축률 |
|------------|----------|:-------:|:------:|
| 결정문일련번호 | `serial_number` | | - |
| 사건번호 | `case_number` | | - |
| 사건 | `case_label` | | - |
| 사건대분류 | `case_major_category` | | - |
| 사건중분류 | `case_mid_category` | | - |
| 사건소분류 | `case_sub_category` | | - |
| 의결일자 | `decision_date` | | - |
| **주문** | **`ruling`** | **O** | **1.0** |
| **이유** | **`reason`** | **O** | **0.4** |
| 쟁점 | `issue` | | - |
| 청구취지 | `claim` | | - |
| 청구인 | `petitioner` | | - |
| 원처분기관 | `original_authority` | | - |
| 문서제공구분 | `document_provision_type` | | - |
| 결정문요약 | `ai_summary` | | - |

### 2.7 환경위 (`dec_environment_documents`) — 9 칼럼

| 원본(한글) | 적재 칼럼 | Context | 압축률 |
|------------|----------|:-------:|:------:|
| 결정문일련번호 | `serial_number` | | - |
| 사건명 | `case_name` | | - |
| 의결번호 | `decision_number` | | - |
| **주문** | **`ruling`** | **O** | **1.0** |
| **평가의견** | **`evaluation_opinion`** | **O** | **0.5** |
| 당사자주장 | `party_claims` | | - |
| 사실조사결과 | `fact_investigation` | | - |
| 사건의개요 | `case_overview` | | - |
| 결정문요약 | `ai_summary` | | - |

### 2.8 증선위 (`dec_securities_documents`) — 6 칼럼

| 원본(한글) | 적재 칼럼 | Context | 압축률 |
|------------|----------|:-------:|:------:|
| 결정문일련번호 | `serial_number` | | - |
| 안건명 | `case_name` | | - |
| 의결번호 | `decision_number` | | - |
| **조치이유** | **`action_reason`** | **O** | **0.5** |
| **조치내용** | **`action_content`** | **O** | **1.0** |
| 결정문요약 | `ai_summary` | | - |

### 2.9 국민권익위 (`dec_civil_rights_documents`) — 11 칼럼

| 원본(한글) | 적재 칼럼 | Context | 압축률 |
|------------|----------|:-------:|:------:|
| 결정문일련번호 | `serial_number` | | - |
| 제목 | `case_name` | | - |
| 의안번호 | `decision_number` | | - |
| 의결일 | `decision_date` | | - |
| 결정요지 | `decision_summary` | | - |
| **주문** | **`ruling`** | **O** | **1.0** |
| **이유** | **`reason`** | **O** | **0.4** |
| 별지 | `appendix` | | - |
| 민원표시 | `complaint_flag` | | - |
| 기관명 | `organization_name` | | - |
| 결정문요약 | `ai_summary` | | - |

### 2.10 공정위 (`dec_fair_trade_documents`) — 13 칼럼

| 원본(한글) | 적재 칼럼 | Context | 압축률 |
|------------|----------|:-------:|:------:|
| 결정문일련번호 | `serial_number` | | - |
| 사건명 | `case_name` | | - |
| 사건번호 | `case_number` | | - |
| 결정번호 | `decision_number` | | - |
| 의결일자 | `decision_date` | | - |
| 결정일자 | `decision_specific_date` | | - |
| 결정요지 | `decision_summary` | | - |
| **주문** | **`ruling`** | **O** | **1.0** |
| **이유** | **`reason`** | **O** | **0.4** |
| 별지 | `appendix` | | - |
| 의결문 | `resolution_text` | | - |
| 각주목록 | `footnotes` | | - |
| 결정문요약 | `ai_summary` | | - |

### 2.11 방통위 (`dec_media_documents`) — 6 칼럼

| 원본(한글) | 적재 칼럼 | Context | 압축률 |
|------------|----------|:-------:|:------:|
| 결정문일련번호 | `serial_number` | | - |
| 안건명 | `case_name` | | - |
| 사건번호 | `case_number` | | - |
| 의결일자 | `decision_date` | | - |
| **주문** | **`ruling`** | **O** | **1.0** |
| 결정문요약 | `ai_summary` | | - |

---

## 3. 자치법규

### 3.1 자치법규 (`local_ordinance_documents`) — 8 칼럼

| 원본(한글) | 적재 칼럼 | Context | 압축률 |
|------------|----------|:-------:|:------:|
| 자치법규ID | `ordinance_id` | | - |
| 자치법규일련번호 | `ordinance_serial` | | - |
| 자치법규명 | `ordinance_name` | | - |
| 지자체기관명 | `local_government` | | - |
| **전체요약** | **`overall_summary`** | **O** | **1.0** |
| **조(리스트)** | **`content`** | **O** | **0.5** |
| 부칙내용 | `supplementary` | | - |
| 전체요약 | `ai_summary` | | - |

---

## 4. 통계 요약

### 4.1 테이블별 칼럼 수

| # | data_type | 테이블 | 전체 칼럼 | Context 칼럼 |
|---|-----------|--------|:---------:|:----------:|
| 1 | 판례 | `precedent_documents` | 16 | 2 |
| 2 | 법령 | `law_documents` | 10 | 1 |
| 3 | 행정규칙 | `admin_rule_documents` | 12 | 1 |
| 4 | 부처유권해석 | `interpretation_ministry_documents` | 11 | 2 |
| 5 | 헌재결정례 | `constitutional_documents` | 16 | 2 |
| 6 | 행정심판례 | `administration_documents` | 14 | 2 |
| 7 | 법령해석례 | `legislation_documents` | 14 | 2 |
| 8 | 조약 | `treaty_documents` | 22 | 1 |
| 9 | 특별행정심판 | `special_admin_appeal_documents` | 21 | 2 |
| 10 | 자치법규 | `local_ordinance_documents` | 8 | 2 |
| 11 | 위원회결정례_노동위 | `dec_labor_documents` | 12 | 2 |
| 12 | 위원회결정례_인권위 | `dec_human_rights_documents` | 13 | 2 |
| 13 | 위원회결정례_개인정보위 | `dec_privacy_documents` | 5 | 1 |
| 14 | 위원회결정례_고용보험 | `dec_employment_documents` | 14 | 2 |
| 15 | 위원회결정례_금융위 | `dec_financial_documents` | 7 | 2 |
| 16 | 위원회결정례_산재심 | `dec_industrial_documents` | 15 | 2 |
| 17 | 위원회결정례_환경위 | `dec_environment_documents` | 9 | 2 |
| 18 | 위원회결정례_증선위 | `dec_securities_documents` | 6 | 2 |
| 19 | 위원회결정례_국민권익위 | `dec_civil_rights_documents` | 11 | 2 |
| 20 | 위원회결정례_공정위 | `dec_fair_trade_documents` | 13 | 2 |
| 21 | 위원회결정례_방통위 | `dec_media_documents` | 6 | 1 |
| | **합계** | **21개 테이블** | **255** | **37** |

### 4.2 압축 정책 요약

| 압축률 | 정책 | 해당 칼럼 |
|:------:|------|----------|
| **1.0** (보존) | 결론/결과 → 압축 안 함 | `ruling`, `answer`, `judgment_summary`, `judgment_result`, `action_content`, `overall_summary` |
| **0.5** (중간) | 혼합 → 중간 압축 | `content`, `action_reason`, `evaluation_opinion` |
| **0.4** (적극) | 근거/이유 → 적극 압축 | `reasoning`, `reason` |

`COLUMN_COMPRESSION_POLICIES`에 미등록 칼럼은 기본 정책 (`rate=0.5`, `min_length=500`)이 적용된다.
