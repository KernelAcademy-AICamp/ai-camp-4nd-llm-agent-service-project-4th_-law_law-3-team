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
| **기타 및 메타데이터** | 법률 용어, 체계도, 신구법, 약칭 등 | `lawterms_full.json`, `lsStmd-full.json` 등 | 보조 데이터 및 시스템용 |

---

## 1. 기본 법률 데이터 (Basic Legal Data)

| 데이터 구분 | 파일명 | 비고 |
|-----------|-------|------|
| 판례 (Full) | `precedents_full.json (1~5)` | [cleaned] 포함 |
| 판례 (Partial) | `precedents_partial.json` | |
| 행정심판례 | `administration_full.json` | |
| 헌재결정례 | `constitutional_full.json` | |
| 법령해석례 | `legislation_full.json` | [cleaned] 포함 |

## 2. 법령 및 규정 (Laws & Regulations)

| 데이터 구분 | 파일명 | 비고 |
|-----------|-------|------|
| 법령 데이터 (일반인용) | `law_full-30.son` | |
| 법령 데이터 (변호사용) | `law_full-29.son` | |
| 행정규칙 | `administrative_rules_full.json` | |
| 자치법규 | `local_rules-full.json` | |
| 조약 | `treaty-full.json` | |

## 3. 위원회 결정문 (Committee Decisions)

| 위원회명 | 파일명 | 레코드 수 |
|---------|-------|----------|
| 노동위원회 | `nlrc-full.json` | 41,445개 |
| 공정거래위원회 | `ftc-full.json` | 8,042개 |
| 국가인권위원회 | `nhrck-full.json` | 4,026개 |
| 개인정보보호위원회 | `ppc-full.json` | 3,898개 |
| 산업재해보상위험재심사위원회 | `iaciac-full.json` | 934개 |
| 방송미디어통신위원회 | `kcc-full.json` | 811개 |
| 금융위원회 | `fsc-full.json` | 663개 |
| 증권선물위원회 | `sfc-full.json` | 636개 |
| 국민권익위원회 | `acr-full.json` | 635개 |
| 중앙환경분쟁조정위원회 | `ecc-full.json` | 358개 |
| 고용보험심사위원회 | `eiac-full.json` | 118개 |
| 중앙토지수용위원회 | `oclt-full.json` | 23개 |

## 4. 부처별 법령해석 (Ministry Legal Interpretations)

| 기관명 | 파일명 | 레코드 수 |
|-------|-------|----------|
| 고용노동부 | `moelCgmExpc-full.json` | 9,563개 |
| 국토교통부 | `molitCgmExpc-full.json` | 5,660개 |
| 식품의약품안전처 | `mfdsCgmExpc-full.json` | 4,341개 |
| 행정안전부 | `moisCgmExpc-full.json` | 4,055개 |
| 기후에너지환경부 | `meCgmExpc-full.json` | 2,291개 |
| 보건복지부 | `mohwCgmExpc-full.json` | 1,417개 |
| 산림청 | `kfsCgmExpc-full.json` | 1,412개 |
| 관세청 | `kcsCgmExpc-full.json` | 1,261개 |
| 지식재산처 (특허청) | `kipoCgmExpc-full.json` | 1,013개 |
| 산업통상자원부 | `motieCgmExpc-full.json` | 941개 |
| 조달청 | `ppsCgmExpc-full.json` | 864개 |
| 소방청 | `nfaCgmExpc-full.json` | 731개 |
| 국가보훈부 | `mpvaCgmExpc-full.json` | 701개 |
| 국가유산청 | `khsCgmExpc-full.json` | 580개 |
| 국방부 | `mndCgmExpc-full.json` | 569개 |
| 해양수산부 | `mofCgmExpc-full.json` | 547개 |
| 경찰청 | `npaCgmExpc-full.json` | 487개 |
| 방위사업청 | `dapaCgmExpc-full.json` | 528개 |
| 법무부 | `mojCgmExpc-full.json` | 378개 |
| 과학기술정보통신부 | `msitCgmExpc-full.json` | 331개 |
| 교육부 | `moeCgmExpc-full.json` | 330개 |
| 농림축산식품부 | `mafraCgmExpc-full.json` | 286개 |
| 외교부 | `mofaCgmExpc-full.json` | 89개 |
| 통일부 | `mouCgmExpc-full.json` | 76개 |
| 인사혁신처 | `mpmCgmExpc-full.json` | 76개 |
| 기상청 | `kmaCgmExpc-full.json` | 71개 |
| 문화체육관광부 | `mcstCgmExpc-full.json` | 50개 |
| 병무청 | `mmaCgmExpc-full.json` | 44개 |
| 법제처 | `molegCgmExpc-full.json` | 39개 |
| 행정중심복합도시건설청 | `naaccCgmExpc-full.json` | 36개 |
| 성평등가족부 (여가부) | `mogefCgmExpc-full.json` | 28개 |
| 농촌진흥청 | `rdaCgmExpc-full.json` | 22개 |
| 해양경찰청 | `kcgCgmExpc-full.json` | 22개 |
| 중소벤처기업부 | `mssCgmExpc-full.json` | 10개 |
| 국가데이터처 (통계청) | `kostatCgmExpc-full.json` | 7개 |
| 질병관리청 | `kdcaCgmExpc-full.json` | 4개 |
| 재외동포청 | `okaCgmExpc-full.json` | 3개 |

## 5. 특별 행정심판례 (Special Administrative Trials)

| 기관/분야 | 파일명 | 레코드 수 |
|---------|-------|----------|
| 조세심판원 | `ttSpecialDecc_list.json` | 138,614개 |
| 해양안전심판원 | `kmstSpecialDecc_list.json` | 13,846개 |
| 인사혁신처 소청심사위원회 | `adapSpecialDecc_list.json` | 210개 |
| 국민권익위원회 (특별) | `acrSpecialDecc_list.json` | 85개 |

## 6. 기타 및 메타데이터 (Others & Metadata)

| 데이터 구분 | 파일명 | 비고 |
|-----------|-------|------|
| 법정 용어 | `lawterms_full.json` | |
| 법령체계도 | `lsStmd-full.json` | [cleaned] 포함 |
| 신구법 목록 | `oldAndNew.json` | |
| 법령명 약칭 (줄임말) | `lsAbrv.json` | |
| 학칙공단 | `school-full.json` | |

---

*최종 업데이트: 2026-02-04 (사용자 제공 데이터 기반 현행화)*