# 인제스트 타입 저장 구조 총정리

10개 데이터 타입별 PostgreSQL / Vector DB / FTS 저장 칼럼 및 첫 행 샘플.

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
| decisions_committee | **15개** | " | " |
| interpretation_ministry | **10개** | " | " |
| special_admin_appeal | **21개** | " | " |
| law (기존) | **10개** | " | " |
| precedent (기존) | **16개** | " | " |

---

## 1. admin_rule (행정규칙) — PostgreSQL 8개 칼럼

| 칼럼 | 첫 행 값 |
|------|---------|
| `admin_rule_id` | `2138247` |
| `serial_number` | `2200000142631` |
| `admin_rule_name` | `(한국전통문화대학교) 현장실습 운영지침` |
| `admin_rule_type` | `학교지침` |
| `ministry` | `국가유산청 한국전통문화대학교` |
| `content` | `「현장실습 운영지침」제정 2025. 5. 13. ...` (5,770자) |
| `supplementary` | `부칙 <제39호, 2020. 11. 9.>...` |
| `ai_summary` | `「한국전통문화대학교 현장실습 운영지침」은 학칙 제35조...` |

- **Vector DB content** → `행정규칙요약` (ai_summary)
- **FTS fulltext** → `[제목] + 조문내용 전체`

---

## 2. constitutional (헌재결정례) — PostgreSQL 16개 칼럼

| 칼럼 | 첫 행 값 |
|------|---------|
| `serial_number` | `177507` |
| `case_number` | `2022헌마1312` |
| `case_name` | `112신고 결과 메세지 미전송 위헌확인` |
| `case_type` | `헌마` |
| `case_type_code` | `430105` |
| `decision_date` | `2022-09-20` |
| `court_division_code` | `430202` |
| `summary` | `수사기관의 112신고 결과 미통지가 헌법소원의 대상이...` |
| `reasoning` | `공권력 불행사에 대한 헌법소원은 헌법 또는 법령에...` |
| `ruling` | `이 사건 심판청구를 각하한다.` |
| `full_text` | `[사건] 2022헌마1312 112신고 결과...` |
| `reason` | `청구인은 '청구인이 2022. 9. 7.에 한 112신고에...` |
| `reference_provisions` | (빈 문자열) |
| `reference_statutes` | (빈 문자열) |
| `reference_cases` | (빈 문자열) |
| `ai_summary` | `수사기관의 112신고 결과 미통지가 헌법소원 대상인지...` |

- **Vector DB content** → `심판례요약` (ai_summary)
- **FTS fulltext** → `[사건명] + 사건번호 + 판시사항 + 결정요지 + 주문 + 이유 + 전문`

---

## 3. administration (행정심판례) — PostgreSQL 14개 칼럼

| 칼럼 | 첫 행 값 |
|------|---------|
| `serial_number` | `263735` |
| `case_name` | `0세아 어린이집 지정취소 처분 취소청구` |
| `case_number` | `2017경기행심982` |
| `decision_date` | `2017-07-31` |
| `disposition_date` | null |
| `disposition_agency` | (빈 문자열) |
| `adjudication_agency` | `경기도행정심판위원회` |
| `case_type` | `국민권익위원회` |
| `case_type_code` | `429152` |
| `ruling` | `피청구인은 2017. 4. 20. 청구인에 대하여 한...` |
| `claim` | `주문과 같다.` |
| `reason` | `1. 사건개요 청구인은 2009. 6. ○○시...` (12,718자) |
| `adjudication_summary` | `사건 2017경기행심982 0세아 어린이집...` |
| `ai_summary` | `0세아 어린이집 지정취소 처분 취소청구 사건에서...` |

- **Vector DB content** → `심판례요약` (ai_summary)
- **FTS fulltext** → `[사건명] + 사건번호 + 주문 + 청구취지 + 이유 + 재결요지`

---

## 4. legislation (법령해석례) — PostgreSQL 14개 칼럼

| 칼럼 | 첫 행 값 |
|------|---------|
| `serial_number` | `323313` |
| `case_name` | `2009. 5. 28. 이후 지방자치단체가 「산업입지...` |
| `case_number` | `10-0240` |
| `interpretation_date` | null |
| `registration_date` | `20210410` |
| `interpretation_agency_code` | `1170000` |
| `interpretation_agency` | `법제처` |
| `inquiry_agency_code` | (빈 문자열) |
| `inquiry_agency` | (빈 문자열) |
| `management_agency_code` | (빈 문자열) |
| `inquiry` | `「산업입지 및 개발에 관한 법률」에 따라 2002년...` |
| `answer` | `「산업입지 및 개발에 관한 법률」에 따라 2002년...` |
| `reason` | `「학교용지 확보 등에 관한 특례법」(이하 "학교용지법"...` |
| `ai_summary` | `「학교용지 확보 등에 관한 특례법」 부칙 제2조...` |

- **Vector DB content** → `ai_summary`
- **FTS fulltext** → `[안건명] + 안건번호 + 질의요지 + 회답 + 이유`

---

## 5. treaty (조약) — PostgreSQL 22개 칼럼

| 칼럼 | 첫 행 값 |
|------|---------|
| `serial_number` | `1400` |
| `treaty_number` | `1128` |
| `treaty_name_kr` | `1948년 7월 1일부터 1949년 1월 31일까지의 기간중...` |
| `treaty_name_en` | `Agreement between the Government of the Republic...` |
| `treaty_type_code` | `440101` |
| `counterpart_country` | `UNITED STATES` |
| `counterpart_country_kr` | `미국` |
| `country_code` | `195` |
| `bilateral_field_code` | `440249` |
| `bilateral_field` | `재정` |
| `signing_date` | `1949-05-27` |
| `signing_place` | `서울` |
| `effective_date` | `1949-12-01` |
| `parliament_approval` | `X` |
| `parliament_approval_date` | null |
| `cabinet_review_date` | null |
| `cabinet_review_session` | `0` |
| `presidential_approval_date` | null |
| `gazette_date` | null |
| `content` | `제1조1. 1948년7월1일부터 1949년1월31일까지...` |
| `note` | `기타* 서명 : 김도연 재무부장관 / 로베르스트 육군준장...` |
| `ai_summary` | `1948년7월1일부터1949년1월31일까지 주한미군 운영...` |

- **Vector DB content** → `조약요약` (ai_summary), source_name → 체결국(한글)
- **FTS fulltext** → `[조약명(한)] + 조약명(영) + 체결국 + 조약내용`

---

## 6. decisions_committee (위원회결정례) — PostgreSQL 15개 칼럼

10개 위원회별 JSON 스키마가 상이하므로 다중 필드명 매핑 적용:
- `사건명/안건명/제목` → `case_name`
- `의결일자/의결일/결정일자/등록일` → `decision_date`
- `결정번호/의결번호/의안번호` → `decision_number`
- `결정요지/판단요지/판정요지/판정사항` → `decision_summary`
- `의결문/결정례전문/내용` → `full_text`

| 칼럼 | 첫 행 값 (개인정보보호위원회) |
|------|---------|
| `serial_number` | `6091` |
| `ai_summary` | `원주시가 방범용 CCTV 영상정보를...` |
| `committee_name` | `개인정보보호위원회` |
| `case_name` | null (안건명 필드 없는 위원회) |
| `decision_date` | `2024.4.24.` |
| `case_number` | null |
| `decision_number` | null |
| `ruling` | null |
| `reason` | `1. 질의배경○ 신청인/신청일 : 원주시장/2023...` |
| `decision_summary` | null |
| `claim` | null |
| `appendix` | null |
| `action_reason` | null |
| `action_content` | null |
| `full_text` | null |

- **Vector DB content** → `결정문요약` (ai_summary)
- **FTS fulltext** → `위원회명 + 주문 + 이유 + 결정요지 + 청구취지 + 조치이유 + 조치내용`

---

## 7. interpretation_ministry (부처유권해석) — PostgreSQL 10개 칼럼

| 칼럼 | 첫 행 값 (경찰청) |
|------|---------|
| `serial_number` | `409198` |
| `case_name` | `112신고내역을 받아보고 싶습니다 어떻게 해야 하나요` |
| `interpretation_date` | `2025-04-28` |
| `case_number` | null |
| `inquiry` | `112신고를 하였습니다 112신고 내역을 받아볼 수...` |
| `related_law` | `공공기관의 정보공개에 관한 법률 제1조(목적)` |
| `answer` | `안녕하세요서울금천경찰서입니다...` |
| `reason` | null |
| `ministry_name` | `경찰청` |
| `ai_summary` | `112신고 내역 열람은 정보공개청구 절차를 통해...` |

- **Vector DB content** → `ai_summary`, source_name → 부처명
- **FTS fulltext** → `[안건명] + 부처명 + 질의요지 + 관련법령 + 회답 + 이유`

---

## 8. special_admin_appeal (특별행정심판) — PostgreSQL 21개 칼럼

2개 기관(조세심판원, 해양안전심판원)의 스키마를 union으로 통합.

| 칼럼 | 첫 행 값 (조세심판원) |
|------|---------|
| `serial_number` | `2040283` |
| `case_name` | (빈 문자열) |
| `case_number` | null |
| `decision_date` | `2025-02-24` |
| `adjudication_agency` | `조세심판원` |
| `case_type` | null |
| `ruling` | `심판청구를 각하한다.` |
| `claim` | null |
| `reason` | `1. 본안심리에 앞서 이 건 심판청구가 적법한지...` (3,793자) |
| `adjudication_summary` | (빈 문자열) |
| `related_rulings` | `조심2023중7269 / 조심2024중2839` |
| `following_rulings` | (빈 문자열) |
| `tax_category` | `지방소득` |
| `related_law` | (빈 문자열) |
| `vessel_type` | null (해양안전심판원 전용) |
| `accident_type` | null (해양안전심판원 전용) |
| `tribunal_location` | null (해양안전심판원 전용) |
| `related_persons` | null (해양안전심판원 전용) |
| `appendix` | null (해양안전심판원 전용) |
| `retrial_notice` | null (해양안전심판원 전용) |
| `ai_summary` | `쟁점: 재조사 결정 통지 전 심판청구 적법성 여부...` |

- **Vector DB content** → `심판례요약` (ai_summary), source_name → 기관명
- **FTS fulltext** → `주문 + 청구취지 + 이유 + 재결요지`

---

## 9. law (법령, 기존) — PostgreSQL 10개 칼럼

| 칼럼 | 첫 행 값 |
|------|---------|
| `law_id` | `010719` |
| `law_name` | `10ㆍ27법난 피해자의 명예회복 등에 관한 법률` |
| `law_type` | null |
| `ministry` | null |
| `promulgation_date` | null |
| `promulgation_no` | null |
| `enforcement_date` | null |
| `content` | `1조 제1조(목적) 이 법은 10ㆍ27법난과 관련하여...` |
| `supplementary` | `부칙 <제8995호,2008.3.28>①(시행일)...` |
| `ai_summary` | `이 법은 1980년 10월 계엄사령부의 합동수사단이...` |

- **Vector DB content** → `법령요약` (ai_summary)
- **FTS fulltext** → `[법령명] + 조문내용`

---

## 10. precedent (판례, 기존) — PostgreSQL 16개 칼럼

| 칼럼 | 첫 행 값 |
|------|---------|
| `serial_number` | `76396` |
| `case_name` | `손해배상청구사건` |
| `case_number` | `84나3990` |
| `decision_date` | `1986-01-15` |
| `court_name` | `서울고법` |
| `case_type` | `민사` |
| `judgment_type` | `제11민사부판결 : 상고` |
| `summary` | `수련의에게 마취를 담당케 하여 의료사고가 발생한 경우...` |
| `reasoning` | `수술당일 환자측으로부터 집도의와 마취담당의를 특정한...` |
| `ruling` | `1. 원심판결의 원고 1에게 대한 피고 패소부분중...` |
| `claim` | `피고는 원고 1에게 금 62,410,194원...` |
| `full_reason` | `1. 손해배상책임의 발생 원고 1이 선천적인...` |
| `full_text` | `【원고, 피항소인】 【피고, 항소인】 대한민국...` |
| `ai_summary` | `쟁점: 수련의에게 마취 및 수술회복조치를 맡겨...` |
| `reference_provisions` | `민법 제750조, 제756조` |
| `reference_cases` | (빈 문자열) |

- **Vector DB content** → `판례요약` (ai_summary)
- **FTS fulltext** → `[사건명] + 사건번호 + 판시사항 + 판결요지 + 주문 + 청구취지 + 이유 + 전문`
