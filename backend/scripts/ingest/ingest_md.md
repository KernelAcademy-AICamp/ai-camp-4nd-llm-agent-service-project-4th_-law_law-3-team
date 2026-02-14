# 인제스트 타입 저장 구조 총정리

19개 데이터 타입별 PostgreSQL / Vector DB / FTS 저장 칼럼 및 첫 행 샘플.
(위원회 결정례 10개 분리: 위원회별 스키마가 상이하므로 개별 테이블)

---

## 공통 구조

### Vector DB (LanceDB) — 모든 타입 동일: 9개 칼럼 + vector

```
id, source_id, data_type, title, content, vector(1024), source_name, chunk_index, total_chunks, date
```

### FTS (fts_index) — 모든 타입 동일: 6개 메타 + content_tsvector

```
source_id, data_type, title, date, source_name, case_number + content_tsvector
```

### PostgreSQL — 타입마다 다름 (아래 상세)

---

## 요약 테이블

| 타입 | PostgreSQL | Vector DB | FTS |
|------|-----------|-----------|-----|
| admin_rule | **8개** | 9+vector (공통) | 6+tsvector (공통) |
| constitutional | **16개** | " | " |
| administration | **14개** | " | " |
| legislation | **14개** | " | " |
| treaty | **22개** | " | " |
| dec_privacy | **5개** | " | " |
| dec_employment | **14개** | " | " |
| dec_fair_trade | **13개** | " | " |
| dec_human_rights | **13개** | " | " |
| dec_civil_rights | **11개** | " | " |
| dec_financial | **7개** | " | " |
| dec_labor | **12개** | " | " |
| dec_industrial | **15개** | " | " |
| dec_environment | **9개** | " | " |
| dec_securities | **6개** | " | " |
| interpretation_ministry | **11개** | " | " |
| special_admin_appeal | **21개** | " | " |
| law (기존) | **10개** | " | " |
| precedent (기존) | **16개** | " | " |

---

## 1. admin_rule (행정규칙) — PostgreSQL 8개 칼럼

| 칼럼 | 원래 칼럼명(한글) | 첫 행 값 |
|------|-----------------|---------|
| `admin_rule_id` | 행정규칙ID | `2138247` |
| `serial_number` | 행정규칙일련번호 | `2200000142631` |
| `admin_rule_name` | 행정규칙명 | `(한국전통문화대학교) 현장실습 운영지침` |
| `admin_rule_type` | 행정규칙종류 | `학교지침` |
| `ministry` | 소관부처명 | `국가유산청 한국전통문화대학교` |
| `content` | 조문내용 | `「현장실습 운영지침」제정 2025. 5. 13. ...` (5,770자) |
| `supplementary` | 부칙내용 | `부칙 <제39호, 2020. 11. 9.>...` |
| `ai_summary` | 행정규칙요약 | `「한국전통문화대학교 현장실습 운영지침」은 학칙 제35조...` |

- **Vector DB content** → `행정규칙요약` (ai_summary)
- **FTS fulltext** → `[제목] + 조문내용 전체`

---

## 2. constitutional (헌재결정례) — PostgreSQL 16개 칼럼

| 칼럼 | 원래 칼럼명(한글) | 첫 행 값 |
|------|-----------------|---------|
| `serial_number` | 헌재결정례일련번호 | `177507` |
| `case_number` | 사건번호 | `2022헌마1312` |
| `case_name` | 사건명 | `112신고 결과 메세지 미전송 위헌확인` |
| `case_type` | 사건종류명 | `헌마` |
| `case_type_code` | 사건종류코드 | `430105` |
| `decision_date` | 종국일자 | `2022-09-20` |
| `court_division_code` | 재판부구분코드 | `430202` |
| `summary` | 판시사항 | `수사기관의 112신고 결과 미통지가 헌법소원의 대상이...` |
| `reasoning` | 결정요지 | `공권력 불행사에 대한 헌법소원은 헌법 또는 법령에...` |
| `ruling` | 주문 | `이 사건 심판청구를 각하한다.` |
| `full_text` | 전문 | `[사건] 2022헌마1312 112신고 결과...` |
| `reason` | 이유 | `청구인은 '청구인이 2022. 9. 7.에 한 112신고에...` |
| `reference_provisions` | 심판대상조문 | (빈 문자열) |
| `reference_statutes` | 참조조문 | (빈 문자열) |
| `reference_cases` | 참조판례 | (빈 문자열) |
| `ai_summary` | 심판례요약 | `수사기관의 112신고 결과 미통지가 헌법소원 대상인지...` |

- **Vector DB content** → `심판례요약` (ai_summary)
- **FTS fulltext** → `[사건명] + 사건번호 + 판시사항 + 결정요지 + 주문 + 이유 + 전문`

---

## 3. administration (행정심판례) — PostgreSQL 14개 칼럼

| 칼럼 | 원래 칼럼명(한글) | 첫 행 값 |
|------|-----------------|---------|
| `serial_number` | 행정심판례일련번호 | `263735` |
| `case_name` | 사건명 | `0세아 어린이집 지정취소 처분 취소청구` |
| `case_number` | 사건번호 | `2017경기행심982` |
| `decision_date` | 의결일자 | `2017-07-31` |
| `disposition_date` | 처분일자 | null |
| `disposition_agency` | 처분청 | (빈 문자열) |
| `adjudication_agency` | 재결청 | `경기도행정심판위원회` |
| `case_type` | 재결례유형명 | `국민권익위원회` |
| `case_type_code` | 재결례유형코드 | `429152` |
| `ruling` | 주문 | `피청구인은 2017. 4. 20. 청구인에 대하여 한...` |
| `claim` | 청구취지 | `주문과 같다.` |
| `reason` | 이유 | `1. 사건개요 청구인은 2009. 6. ○○시...` (12,718자) |
| `adjudication_summary` | 재결요지 | `사건 2017경기행심982 0세아 어린이집...` |
| `ai_summary` | 심판례요약 | `0세아 어린이집 지정취소 처분 취소청구 사건에서...` |

- **Vector DB content** → `심판례요약` (ai_summary)
- **FTS fulltext** → `[사건명] + 사건번호 + 주문 + 청구취지 + 이유 + 재결요지`

---

## 4. legislation (법령해석례) — PostgreSQL 14개 칼럼

| 칼럼 | 원래 칼럼명(한글) | 첫 행 값 |
|------|-----------------|---------|
| `serial_number` | 법령해석례일련번호 | `323313` |
| `case_name` | 안건명 | `2009. 5. 28. 이후 지방자치단체가 「산업입지...` |
| `case_number` | 안건번호 | `10-0240` |
| `interpretation_date` | 해석일자 | null |
| `registration_date` | 등록일시 | `20210410` |
| `interpretation_agency_code` | 해석기관코드 | `1170000` |
| `interpretation_agency` | 해석기관명 | `법제처` |
| `inquiry_agency_code` | 질의기관코드 | (빈 문자열) |
| `inquiry_agency` | 질의기관명 | (빈 문자열) |
| `management_agency_code` | 관리기관코드 | (빈 문자열) |
| `inquiry` | 질의요지 | `「산업입지 및 개발에 관한 법률」에 따라 2002년...` |
| `answer` | 회답 | `「산업입지 및 개발에 관한 법률」에 따라 2002년...` |
| `reason` | 이유 | `「학교용지 확보 등에 관한 특례법」(이하 "학교용지법"...` |
| `ai_summary` | 해석례요약 | `「학교용지 확보 등에 관한 특례법」 부칙 제2조...` |

- **Vector DB content** → `해석례요약` (ai_summary)
- **FTS fulltext** → `[안건명] + 안건번호 + 질의요지 + 회답 + 이유`

---

## 5. treaty (조약) — PostgreSQL 22개 칼럼

| 칼럼 | 원래 칼럼명(한글) | 첫 행 값 |
|------|-----------------|---------|
| `serial_number` | 조약일련번호 | `1400` |
| `treaty_number` | 조약번호 | `1128` |
| `treaty_name_kr` | 조약명_한글 | `1948년 7월 1일부터 1949년 1월 31일까지의 기간중...` |
| `treaty_name_en` | 조약명_영문 | `Agreement between the Government of the Republic...` |
| `treaty_type_code` | 조약구분코드 | `440101` |
| `counterpart_country` | 체결대상국가 | `UNITED STATES` |
| `counterpart_country_kr` | 체결대상국가한글 | `미국` |
| `country_code` | 국가코드 | `195` |
| `bilateral_field_code` | 양자조약분야코드 | `440249` |
| `bilateral_field` | 양자조약분야명 | `재정` |
| `signing_date` | 서명일자 | `1949-05-27` |
| `signing_place` | 서명장소 | `서울` |
| `effective_date` | 발효일자 | `1949-12-01` |
| `parliament_approval` | 국회비준동의여부 | `X` |
| `parliament_approval_date` | 국회비준동의일자 | null |
| `cabinet_review_date` | 국무회의심의일자 | null |
| `cabinet_review_session` | 국무회의심의회차 | `0` |
| `presidential_approval_date` | 대통령재가일자 | null |
| `gazette_date` | 관보게재일자 | null |
| `content` | 조약내용 | `제1조1. 1948년7월1일부터 1949년1월31일까지...` |
| `note` | 비고 | `기타* 서명 : 김도연 재무부장관 / 로베르스트 육군준장...` |
| `ai_summary` | 조약요약 | `1948년7월1일부터1949년1월31일까지 주한미군 운영...` |

- **Vector DB content** → `조약요약` (ai_summary), source_name → 체결국(한글)
- **FTS fulltext** → `[조약명(한)] + 조약명(영) + 체결국 + 조약내용`

---

## 6. 위원회 결정례 (10개 개별 테이블)

위원회별 JSON 스키마가 크게 다르므로 (공통 필드 2개/45개) 개별 테이블로 분리.
벡터 DB/FTS는 `data_type="위원회결정례"`로 통합 검색 유지.

### 6-1. dec_privacy (개인정보보호위원회) — 5개 칼럼

| 칼럼 | 원래 칼럼명 |
|------|-----------|
| `serial_number` | 결정문일련번호 |
| `case_name` | 안건명 |
| `decision_date` | 의결일자 |
| `reason` | 이유 |
| `ai_summary` | 결정문요약 |

### 6-2. dec_employment (고용보험심사위원회) — 14개 칼럼

| 칼럼 | 원래 칼럼명 |
|------|-----------|
| `serial_number` | 결정문일련번호 |
| `case_name` | 사건명 |
| `case_number` | 사건번호 |
| `case_classification` | 사건의분류 |
| `decision_date` | 의결일자 |
| `resolution_type` | 의결서종류 |
| `ruling` | 주문 |
| `reason` | 이유 |
| `claim` | 청구취지 |
| `petitioner` | 청구인 |
| `respondent` | 피청구인 |
| `overview` | 개요 |
| `organization_name` | 기관명 |
| `ai_summary` | 결정문요약 |

### 6-3. dec_fair_trade (공정거래위원회) — 13개 칼럼

| 칼럼 | 원래 칼럼명 |
|------|-----------|
| `serial_number` | 결정문일련번호 |
| `case_name` | 사건명 |
| `case_number` | 사건번호 |
| `decision_number` | 결정번호 |
| `decision_date` | 의결일자 |
| `decision_specific_date` | 결정일자 |
| `decision_summary` | 결정요지 |
| `ruling` | 주문 |
| `reason` | 이유 |
| `appendix` | 별지 |
| `resolution_text` | 의결문 |
| `footnotes` | 각주목록 |
| `ai_summary` | 결정문요약 |

### 6-4. dec_human_rights (국가인권위원회) — 13개 칼럼

| 칼럼 | 원래 칼럼명 |
|------|-----------|
| `serial_number` | 결정문일련번호 |
| `case_name` | 사건명 |
| `case_number` | 사건번호 |
| `decision_date` | 의결일자 |
| `decision_summary` | 결정요지 |
| `judgment_summary` | 판단요지 |
| `classification_name` | 분류명 |
| `ruling` | 주문 |
| `ruling_summary` | 주문요지 |
| `reason` | 이유 |
| `appendix` | 별지 |
| `full_text` | 결정례전문 |
| `ai_summary` | 결정문요약 |

### 6-5. dec_civil_rights (국민권익위원회) — 11개 칼럼

| 칼럼 | 원래 칼럼명 |
|------|-----------|
| `serial_number` | 결정문일련번호 |
| `case_name` | 제목 |
| `decision_number` | 의안번호 |
| `decision_date` | 의결일 |
| `decision_summary` | 결정요지 |
| `ruling` | 주문 |
| `reason` | 이유 |
| `appendix` | 별지 |
| `complaint_flag` | 민원표시 |
| `organization_name` | 기관명 |
| `ai_summary` | 결정문요약 |

### 6-6. dec_financial (금융위원회) — 7개 칼럼

| 칼럼 | 원래 칼럼명 |
|------|-----------|
| `serial_number` | 결정문일련번호 |
| `case_name` | 안건명 |
| `decision_number` | 의결번호 |
| `action_reason` | 조치이유 |
| `action_content` | 조치내용 |
| `organization_name` | 기관명 |
| `ai_summary` | 결정문요약 |

### 6-7. dec_labor (노동위원회) — 12개 칼럼

| 칼럼 | 원래 칼럼명 |
|------|-----------|
| `serial_number` | 결정문일련번호 |
| `case_name` | 제목 |
| `case_number` | 사건번호 |
| `decision_date` | 등록일 |
| `judgment_matter` | 판정사항 |
| `judgment_summary` | 판정요지 |
| `judgment_result` | 판정결과 |
| `full_text` | 내용 |
| `data_category` | 자료구분 |
| `department` | 담당부서 |
| `organization_name` | 기관명 |
| `ai_summary` | 결정문요약 |

### 6-8. dec_industrial (산업재해보상보험재심사위원회) — 15개 칼럼

| 칼럼 | 원래 칼럼명 |
|------|-----------|
| `serial_number` | 결정문일련번호 |
| `case_number` | 사건번호 |
| `case_label` | 사건 |
| `case_major_category` | 사건대분류 |
| `case_mid_category` | 사건중분류 |
| `case_sub_category` | 사건소분류 |
| `decision_date` | 의결일자 |
| `ruling` | 주문 |
| `reason` | 이유 |
| `issue` | 쟁점 |
| `claim` | 청구취지 |
| `petitioner` | 청구인 |
| `original_authority` | 원처분기관 |
| `document_provision_type` | 문서제공구분 |
| `ai_summary` | 결정문요약 |

### 6-9. dec_environment (중앙환경분쟁조정위원회) — 9개 칼럼

| 칼럼 | 원래 칼럼명 |
|------|-----------|
| `serial_number` | 결정문일련번호 |
| `case_name` | 사건명 |
| `decision_number` | 의결번호 |
| `ruling` | 주문 |
| `evaluation_opinion` | 평가의견 |
| `party_claims` | 당사자주장 |
| `fact_investigation` | 사실조사결과 |
| `case_overview` | 사건의개요 |
| `ai_summary` | 결정문요약 |

### 6-10. dec_securities (증권선물위원회) — 6개 칼럼

| 칼럼 | 원래 칼럼명 |
|------|-----------|
| `serial_number` | 결정문일련번호 |
| `case_name` | 안건명 |
| `decision_number` | 의결번호 |
| `action_reason` | 조치이유 |
| `action_content` | 조치내용 |
| `ai_summary` | 결정문요약 |

**공통**: Vector DB content → `결정문요약` (ai_summary), data_type=`위원회결정례`, source_name=위원회명

---

## 7. interpretation_ministry (부처유권해석) — PostgreSQL 11개 칼럼

| 칼럼 | 원래 칼럼명(한글) | 첫 행 값 (경찰청) |
|------|-----------------|---------|
| `serial_number` | 법령해석일련번호 | `409198` |
| `case_name` | 안건명 | `112신고내역을 받아보고 싶습니다 어떻게 해야 하나요` |
| `interpretation_date` | 해석일자 | `2025-04-28` |
| `case_number` | 안건번호 | null |
| `inquiry` | 질의요지 | `112신고를 하였습니다 112신고 내역을 받아볼 수...` |
| `related_law` | 관련법령 | `공공기관의 정보공개에 관한 법률 제1조(목적)` |
| `answer` | 회답 | `안녕하세요서울금천경찰서입니다...` |
| `reason` | 이유 | null |
| `business_field` | 업무분야 | null (관세청 등 일부 부처만 보유) |
| `ministry_name` | \_\_source_group\_\_ (파일명 추출) | `경찰청` |
| `ai_summary` | 해석요약 | `112신고 내역 열람은 정보공개청구 절차를 통해...` |

- **Vector DB content** → `해석요약` (ai_summary), source_name → 부처명
- **FTS fulltext** → `[안건명] + 부처명 + 질의요지 + 관련법령 + 회답 + 이유`

---

## 8. special_admin_appeal (특별행정심판) — PostgreSQL 21개 칼럼

2개 기관(조세심판원, 해양안전심판원)의 스키마를 union으로 통합.

| 칼럼 | 원래 칼럼명(한글) | 첫 행 값 (조세심판원) |
|------|-----------------|---------|
| `serial_number` | 특별행정심판재결례일련번호 | `2040283` |
| `case_name` | 사건명 | (빈 문자열) |
| `case_number` | 재결번호 | null |
| `decision_date` | 의결일자 | `2025-02-24` |
| `adjudication_agency` | 재결청 | `조세심판원` |
| `case_type` | 재결례유형명 | null |
| `ruling` | 주문 | `심판청구를 각하한다.` |
| `claim` | 청구취지 | null |
| `reason` | 이유 | `1. 본안심리에 앞서 이 건 심판청구가 적법한지...` (3,793자) |
| `adjudication_summary` | 재결요지 | (빈 문자열) |
| `related_rulings` | 참조결정 | `조심2023중7269 / 조심2024중2839` |
| `following_rulings` | 따른결정 | (빈 문자열) |
| `tax_category` | 세목 | `지방소득` |
| `related_law` | 관련법령 | (빈 문자열) |
| `vessel_type` | 선박유형 | null (해양안전심판원 전용) |
| `accident_type` | 사고유형 | null (해양안전심판원 전용) |
| `tribunal_location` | 해심위치 | null (해양안전심판원 전용) |
| `related_persons` | 해양사고관련자 | null (해양안전심판원 전용) |
| `appendix` | 별지 | null (해양안전심판원 전용) |
| `retrial_notice` | 재심청구안내 | null (해양안전심판원 전용) |
| `ai_summary` | 심판례요약 | `쟁점: 재조사 결정 통지 전 심판청구 적법성 여부...` |

- **Vector DB content** → `심판례요약` (ai_summary), source_name → 기관명
- **FTS fulltext** → `주문 + 청구취지 + 이유 + 재결요지`

---

## 9. law (법령, 기존) — PostgreSQL 10개 칼럼

| 칼럼 | 원래 칼럼명(한글) | 첫 행 값 |
|------|-----------------|---------|
| `law_id` | 법령ID | `010719` |
| `law_name` | 법령명_한글 | `10ㆍ27법난 피해자의 명예회복 등에 관한 법률` |
| `law_type` | 법령구분 | null |
| `ministry` | 소관부처명 | null |
| `promulgation_date` | 공포일자 | null |
| `promulgation_no` | 공포번호 | null |
| `enforcement_date` | 시행일자 | null |
| `content` | 조문 | `1조 제1조(목적) 이 법은 10ㆍ27법난과 관련하여...` |
| `supplementary` | 부칙 | `부칙 <제8995호,2008.3.28>①(시행일)...` |
| `ai_summary` | 법령 요약 | `이 법은 1980년 10월 계엄사령부의 합동수사단이...` |

- **Vector DB content** → `법령 요약` (ai_summary)
- **FTS fulltext** → `[법령명] + 조문내용`

---

## 10. precedent (판례, 기존) — PostgreSQL 16개 칼럼

| 칼럼 | 원래 칼럼명(한글) | 첫 행 값 |
|------|-----------------|---------|
| `serial_number` | 판례정보일련번호 | `76396` |
| `case_name` | 사건명 | `손해배상청구사건` |
| `case_number` | 사건번호 | `84나3990` |
| `decision_date` | 선고일자 | `1986-01-15` |
| `court_name` | 법원명 | `서울고법` |
| `case_type` | 사건종류명 | `민사` |
| `judgment_type` | 판결유형 | `제11민사부판결 : 상고` |
| `summary` | 판시사항 | `수련의에게 마취를 담당케 하여 의료사고가 발생한 경우...` |
| `reasoning` | 판결요지 | `수술당일 환자측으로부터 집도의와 마취담당의를 특정한...` |
| `ruling` | 주문 | `1. 원심판결의 원고 1에게 대한 피고 패소부분중...` |
| `claim` | 청구취지 | `피고는 원고 1에게 금 62,410,194원...` |
| `full_reason` | 이유 | `1. 손해배상책임의 발생 원고 1이 선천적인...` |
| `full_text` | 판례내용 | `【원고, 피항소인】 【피고, 항소인】 대한민국...` |
| `ai_summary` | 판례요약 | `쟁점: 수련의에게 마취 및 수술회복조치를 맡겨...` |
| `reference_provisions` | 참조조문 | `민법 제750조, 제756조` |
| `reference_cases` | 참조판례 | (빈 문자열) |

- **Vector DB content** → `판례요약` (ai_summary)
- **FTS fulltext** → `[사건명] + 사건번호 + 판시사항 + 판결요지 + 주문 + 청구취지 + 이유 + 전문`
