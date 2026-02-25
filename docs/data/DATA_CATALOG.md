# 법률 데이터 카탈로그

이 문서는 프로젝트에서 사용되는 법률 데이터의 소스 파일 정보와 분류를 정리한 것입니다.

## 데이터 현황 요약

| 분류 | 주요 데이터 타입 | 소스 파일 예시 | 비고 |
|------|-----------------|---------------|------|
| **기본 법률 데이터** | 판례, 행정심판례, 헌재결정례, 법령해석례 | `precedents_full.json`, `administration_full.json` 등 | 핵심 RAG 데이터 |
| **법령 및 규정** | 법령(일반/변호사), 행정규칙, 자치법규, 조약 | `law_full-30.son`, `administrative_rules_full.json` 등 | 조문 참조용 |
| **위원회 결정문** | 12개 주요 위원회 결정례 | `ftc-full.json`, `nlrc-full.json` 등 | 행정적 판단 사례 |
| **부처별 법령해석** | 28개+ 정부 부처 법령 해석 사례 | `moelCgmExpc-full.json` 등 | 실무 지침 및 유권해석 |
| **특별 행정심판례** | 조세, 해양안전, 인사 등 특수 분야 | `ttSpecialDecc_list.json` 등 | 전문 분야 심판례 |
| **기타 및 메타데이터** | 법률 용어, 체계도, 신구법, 약칭 등 | `lawterms_full.json`, `law_hierarchy.json` 등 | 보조 데이터 및 시스템용 |

---

## 1. 기본 법률 데이터 (Basic Legal Data)

| 데이터 구분 | EDA 파일명 | 원본 파일명 | 비고 |
|-----------|-----------|-----------|------|
| 판례 (Full) | `precedents_v1.json` | `precedents_full.json (1~5)` | [cleaned] 포함 |
| 판례 (Partial) | - | `precedents_partial.json` | |
| 행정심판례 | `administration_v1.json` | `administration_full.json` | |
| 헌재결정례 | `constitutional_v1.json` | `constitutional_full.json` | |
| 법령해석례 | `legislation_v1.json` | `legislation_full.json` | [cleaned] 포함 |

## 2. 법령 및 규정 (Laws & Regulations)

| 데이터 구분 | EDA 파일명 | 원본 파일명 | 비고 |
|-----------|-----------|-----------|------|
| 법령 데이터 | `law_v1.json` | `[DONE]law-2.json` | |
| 법령 데이터 (일반인용) | - | `law_full-30.json` | 추후 사용 |
| 법령 데이터 (변호사용) | - | `law_full-29.json` | 추후 사용 |
| 행정규칙 | `admin_rule_v1.json` | `administrative_rules_full.json` | |
| 자치법규 | - | `local_rules-full.json` | |
| 조약 | `treaty_v1.json` | `treaty-full.json` | |

## 3. 위원회 결정문 (Committee Decisions)

**EDA 파일 네이밍**: `dec_comm_{위원회명}_v1.json`

| 위원회명 | EDA 파일명 | 원본 파일명 | 레코드 수 |
|---------|-----------|-----------|----------|
| 노동위원회 | `dec_comm_노동위원회_v1.json` | `nlrc-full.json` | 41,445개 |
| 공정거래위원회 | `dec_comm_공정거래위원회_v1.json` | `ftc-full.json` | 8,042개 |
| 국가인권위원회 | `dec_comm_국가인권위원회_v1.json` | `nhrck-full.json` | 4,026개 |
| 개인정보보호위원회 | `dec_comm_개인정보보호위원회_v1.json` | `ppc-full.json` | 3,898개 |
| 산업재해보상위험재심사위원회 | `dec_comm_산업재해보상위험재심사위원회_v1.json` | `iaciac-full.json` | 934개 |
| 금융위원회 | `dec_comm_금융위원회_v1.json` | `fsc-full.json` | 663개 |
| 증권선물위원회 | `dec_comm_증권선물위원회_v1.json` | `sfc-full.json` | 636개 |
| 국민권익위원회 | `dec_comm_국민권익위원회_v1.json` | `acr-full.json` | 635개 |
| 중앙환경분쟁조정위원회 | `dec_comm_중앙환경분쟁조정위원회_v1.json` | `ecc-full.json` | 358개 |
| 고용보험심사위원회 | `dec_comm_고용보험심사위원회_v1.json` | `eiac-full.json` | 118개 |

## 4. 부처별 법령해석 (Ministry Legal Interpretations)

**EDA 파일 네이밍**: `intp_min_{기관명}_v1.json`

| 기관명 | EDA 파일명 | 원본 파일명 | 레코드 수 |
|-------|-----------|-----------|----------|
| 고용노동부 | `intp_min_고용노동부_v1.json` | `moelCgmExpc-full.json` | 9,563개 |
| 국토교통부 | `intp_min_국토교통부_v1.json` | `molitCgmExpc-full.json` | 5,660개 |
| 식품의약품안전처 | `intp_min_식품의약품안전처_v1.json` | `mfdsCgmExpc-full.json` | 4,341개 |
| 행정안전부 | `intp_min_행정안전부_v1.json` | `moisCgmExpc-full.json` | 4,055개 |
| 기후에너지환경부 | `intp_min_기후에너지환경부_v1.json` | `meCgmExpc-full.json` | 2,291개 |
| 보건복지부 | `intp_min_보건복지부_v1.json` | `mohwCgmExpc-full.json` | 1,417개 |
| 산림청 | `intp_min_산림청_v1.json` | `kfsCgmExpc-full.json` | 1,412개 |
| 관세청 | `intp_min_관세청_v1.json` | `kcsCgmExpc-full.json` | 1,261개 |
| 지식재산처 | `intp_min_지식재산처_v1.json` | `kipoCgmExpc-full.json` | 1,013개 |
| 산업통상자원부 | `intp_min_산업통상자원부_v1.json` | `motieCgmExpc-full.json` | 941개 |
| 조달청 | - | `ppsCgmExpc-full.json` | 864개 |
| 소방청 | `intp_min_소방청_v1.json` | `nfaCgmExpc-full.json` | 731개 |
| 국가보훈부 | `intp_min_국가보훈부_v1.json` | `mpvaCgmExpc-full.json` | 701개 |
| 국가유산청 | - | `khsCgmExpc-full.json` | 580개 |
| 국방부 | `intp_min_국방부_v1.json` | `mndCgmExpc-full.json` | 569개 |
| 해양수산부 | `intp_min_해양수산부_v1.json` | `mofCgmExpc-full.json` | 547개 |
| 방위사업청 | `intp_min_방위사업청_v1.json` | `dapaCgmExpc-full.json` | 528개 |
| 경찰청 | `intp_min_경찰청_v1.json` | `npaCgmExpc-full.json` | 487개 |
| 법무부 | `intp_min_법무부_v1.json` | `mojCgmExpc-full.json` | 378개 |
| 과학기술정보통신부 | `intp_min_과학기술정보통신부_v1.json` | `msitCgmExpc-full.json` | 331개 |
| 교육부 | `intp_min_교육부_v1.json` | `moeCgmExpc-full.json` | 330개 |
| 농림축산식품부 | `intp_min_농림축산식품부_v1.json` | `mafraCgmExpc-full.json` | 286개 |
| 외교부 | `intp_min_외교부_v1.json` | `mofaCgmExpc-full.json` | 89개 |
| 통일부 | `intp_min_통일부_v1.json` | `mouCgmExpc-full.json` | 76개 |
| 인사혁신처 | `intp_min_인사혁신처_v1.json` | `mpmCgmExpc-full.json` | 76개 |
| 기상청 | `intp_min_기상청_v1.json` | `kmaCgmExpc-full.json` | 71개 |
| 문화체육관광부 | `intp_min_문화체육관광부_v1.json` | `mcstCgmExpc-full.json` | 50개 |
| 병무청 | - | `mmaCgmExpc-full.json` | 44개 |
| 법제처 | - | `molegCgmExpc-full.json` | 39개 |
| 행정중심복합도시건설청 | - | `naaccCgmExpc-full.json` | 36개 |
| 성평등가족부 | `intp_min_성평등가족부_v1.json` | `mogefCgmExpc-full.json` | 28개 |
| 농촌진흥청 | `intp_min_농촌진흥청_v1.json` | `rdaCgmExpc-full.json` | 22개 |
| 해양경찰청 | - | `kcgCgmExpc-full.json` | 22개 |
| 중소벤처기업부 | `intp_min_중소벤처기업부_v1.json` | `mssCgmExpc-full.json` | 10개 |
| 국가데이터처 (통계청) | - | `kostatCgmExpc-full.json` | 7개 |
| 질병관리청 | - | `kdcaCgmExpc-full.json` | 4개 |
| 재외동포청 | - | `okaCgmExpc-full.json` | 3개 |

## 5. 특별 행정심판례 (Special Administrative Trials)

| 기관/분야 | EDA 파일명 | 원본 파일명 | 레코드 수 |
|---------|-----------|-----------|----------|
| 조세심판원 | `sadm_case_조세심판원_v1.json` | `ttSpecialDecc_list.json` | 138,614개 |
| 해양안전심판원 | `sadm_case_해양안전심판원_v1.json` | `kmstSpecialDecc_list.json` | 13,846개 |
| 인사혁신처 소청심사위원회 | - | `adapSpecialDecc_list.json` | 210개 |
| 국민권익위원회 (특별) | - | `acrSpecialDecc_list.json` | 85개 |

## 6. 기타 및 메타데이터 (Others & Metadata)

| 데이터 구분 | EDA 파일명 | 원본 파일명 | 비고 |
|-----------|-----------|-----------|------|
| 법정 용어 | `lawterms_v1.json` | `lawterms_full.json` | |
| 법령체계도 | - | `law_hierarchy.json` | [cleaned] 포함 |
| 신구법 목록 | - | `oldAndNew.json` | |
| 법령명 약칭 (줄임말) | - | `law_abbreviations.json` | |
| 학칙공단 | - | `school-full.json` | |

---

*최종 업데이트: 2026-02-24 (data/ingest_source/ 구조 변경, 신규 데이터 추가 반영)*