# LLM 요약 필드 클리닝 기록 (2026-02-15)

## 개요

19개 데이터 타입, 643,242건의 LLM 요약 필드에 대해 규칙 기반 후처리(클리닝)를 수행했다.
클리닝된 JSON 파일은 버전을 올려 `data/`에 배치하고, 기존 원본은 삭제했다.

## 클리닝 규칙

| Phase | 규칙 | 대상 |
|-------|------|------|
| Phase 1 | 마크다운/HTML 잔여 제거 (`**`, `##`, `<p>`, `<br>` 등) | 전체 |
| Phase 2 | 불완전 문장 보정 (트레일링 `...`, 열린 괄호 등) | 전체 |
| Phase 3 | null/empty → 제목 기반 fallback (`"{제목}에 관한 내용"`) | 전체 |

## 클리닝 결과 (전체 19개 타입)

| 타입 | 데이터명 | 대상 건수 | 수정 건수 | 수정율 | 마크다운 | 불완전 | null |
|------|----------|-----------|-----------|--------|----------|--------|------|
| law | 법령 | 5,548 | 3,482 | 62.8% | 3,482 | 0 | 0 |
| law (조문요약) | 법령 조문별 | 219,318 | 20 | 0.009% | 0 | 0 | 20 |
| precedent | 판례 | 92,055 | 1 | 0.001% | 1 | 0 | 0 |
| constitutional | 헌재결정례 | 31,718 | 31,679 | 99.9% | 31,679 | 0 | 0 |
| administration | 행정심판례 | 34,254 | 34,199 | 99.8% | 34,199 | 0 | 0 |
| admin_rule | 행정규칙 | 5,258 | 4,781 | 90.9% | 4,781 | 0 | 0 |
| legislation | 법령해석례 | 8,597 | 18 | 0.2% | 0 | 18 | 0 |
| treaty | 조약 | 3,589 | 0 | 0% | 0 | 0 | 0 |
| dec_privacy | 개인정보보호위원회 | 1,448 | 0 | 0% | 0 | 0 | 0 |
| dec_employment | 고용보험심사위원회 | 118 | 0 | 0% | 0 | 0 | 0 |
| dec_fair_trade | 공정거래위원회 | 7,728 | 0 | 0% | 0 | 0 | 0 |
| dec_human_rights | 국가인권위원회 | 3,721 | 3 | 0.08% | 3 | 0 | 0 |
| dec_civil_rights | 국민권익위원회 | 635 | 0 | 0% | 0 | 0 | 0 |
| dec_financial | 금융위원회 | 662 | 1 | 0.15% | 0 | 1 | 0 |
| dec_labor | 노동위원회 | 40,714 | 0 | 0% | 0 | 0 | 0 |
| dec_industrial | 산재보상재심사 | 782 | 0 | 0% | 0 | 0 | 0 |
| dec_environment | 환경분쟁조정 | 358 | 0 | 0% | 0 | 0 | 0 |
| dec_securities | 증권선물위원회 | 636 | 0 | 0% | 0 | 0 | 0 |
| interpretation_ministry | 부처해석례 (28개) | 37,325 | 1,009 | 2.7% | 1,009 | 0 | 0 |
| special_admin_appeal | 특별행정심판 (2개) | 148,778 | 1 | 0.001% | 1 | 0 | 1 |
| **합계** | | **643,242** | **75,194** | **11.7%** | **75,155** | **19** | **21** |

### 주요 발견

- **법령(law)**: 3,482건(62.8%)에서 `**볼드**` 마크다운 패턴 다수 발견. 클리닝 효과 가장 큼.
- **헌재결정례**: 31,679건(99.9%)이 마크다운 포함. 거의 전 건에 `**` 패턴 존재.
- **행정심판례**: 34,199건(99.8%)이 마크다운 포함. 헌재와 유사한 패턴.
- **행정규칙**: 4,781건(90.9%) 마크다운 포함.
- **법령해석례**: 18건 불완전 문장 보정 (트레일링 `...` 제거).
- **판례/조약/대부분 위원회**: 수정 0건 또는 극소수 (요약 품질 양호).

## 파일 버전 매핑

### 단일 JSON 파일 (7개)

| 타입 | 이전 파일명 | 새 파일명 |
|------|------------|-----------|
| law | `law_v2.json` | `law_v3.json` |
| precedent | `precedents_v1.json` | `precedents_v2.json` |
| admin_rule | `admin_rule_v1.json` | `admin_rule_v2.json` |
| administration | `administration_v1.json` | `administration_v2.json` |
| constitutional | `constitutional_v1.json` | `constitutional_v2.json` |
| legislation | `legislation_v1.json` | `legislation_v2.json` |
| treaty | `treaty_v1.json` | `treaty_v2.json` |

### 위원회 결정례 (10개, `decisions_committee/`)

| 타입 | 이전 파일명 | 새 파일명 |
|------|------------|-----------|
| dec_fair_trade | `dec_comm_공정거래위원회_v2.json` | `dec_comm_공정거래위원회_v3.json` |
| dec_privacy | `dec_comm_개인정보보호위원회_v1.json` | `dec_comm_개인정보보호위원회_v2.json` |
| dec_employment | `dec_comm_고용보험심사위원회_v1.json` | `dec_comm_고용보험심사위원회_v2.json` |
| dec_human_rights | `dec_comm_국가인권위원회_v1.json` | `dec_comm_국가인권위원회_v2.json` |
| dec_civil_rights | `dec_comm_국민권익위원회_v1.json` | `dec_comm_국민권익위원회_v2.json` |
| dec_financial | `dec_comm_금융위원회_v1.json` | `dec_comm_금융위원회_v2.json` |
| dec_labor | `dec_comm_노동위원회_v1.json` | `dec_comm_노동위원회_v2.json` |
| dec_industrial | `dec_comm_산업재해보상위험재심사위원회_v1.json` | `dec_comm_산업재해보상위험재심사위원회_v2.json` |
| dec_environment | `dec_comm_중앙환경분쟁조정위원회_v1.json` | `dec_comm_중앙환경분쟁조정위원회_v2.json` |
| dec_securities | `dec_comm_증권선물위원회_v1.json` | `dec_comm_증권선물위원회_v2.json` |

### 부처해석례 (28개, `interpretation_ministry/`)

모든 파일 `_v1.json` → `_v2.json`:

| 이전 | 새 |
|------|-----|
| `intp_min_경찰청_v1.json` | `intp_min_경찰청_v2.json` |
| `intp_min_고용노동부_v1.json` | `intp_min_고용노동부_v2.json` |
| `intp_min_과학기술정보통신부_v1.json` | `intp_min_과학기술정보통신부_v2.json` |
| `intp_min_관세청_v1.json` | `intp_min_관세청_v2.json` |
| `intp_min_교육부_v1.json` | `intp_min_교육부_v2.json` |
| `intp_min_국가보훈부_v1.json` | `intp_min_국가보훈부_v2.json` |
| `intp_min_국방부_v1.json` | `intp_min_국방부_v2.json` |
| `intp_min_국토교통부_v1.json` | `intp_min_국토교통부_v2.json` |
| `intp_min_기상청_v1.json` | `intp_min_기상청_v2.json` |
| `intp_min_기후에너지환경부_v1.json` | `intp_min_기후에너지환경부_v2.json` |
| `intp_min_농림축산식품부_v1.json` | `intp_min_농림축산식품부_v2.json` |
| `intp_min_농촌진흥청_v1.json` | `intp_min_농촌진흥청_v2.json` |
| `intp_min_문화체육관광부_v1.json` | `intp_min_문화체육관광부_v2.json` |
| `intp_min_방위사업청_v1.json` | `intp_min_방위사업청_v2.json` |
| `intp_min_법무부_v1.json` | `intp_min_법무부_v2.json` |
| `intp_min_보건복지부_v1.json` | `intp_min_보건복지부_v2.json` |
| `intp_min_산림청_v1.json` | `intp_min_산림청_v2.json` |
| `intp_min_산업통상자원부_v1.json` | `intp_min_산업통상자원부_v2.json` |
| `intp_min_성평등가족부_v1.json` | `intp_min_성평등가족부_v2.json` |
| `intp_min_소방청_v1.json` | `intp_min_소방청_v2.json` |
| `intp_min_식품의약품안전처_v1.json` | `intp_min_식품의약품안전처_v2.json` |
| `intp_min_외교부_v1.json` | `intp_min_외교부_v2.json` |
| `intp_min_인사혁신처_v1.json` | `intp_min_인사혁신처_v2.json` |
| `intp_min_중소벤처기업부_v1.json` | `intp_min_중소벤처기업부_v2.json` |
| `intp_min_지식재산처_v1.json` | `intp_min_지식재산처_v2.json` |
| `intp_min_통일부_v1.json` | `intp_min_통일부_v2.json` |
| `intp_min_해양수산부_v1.json` | `intp_min_해양수산부_v2.json` |
| `intp_min_행정안전부_v1.json` | `intp_min_행정안전부_v2.json` |

### 특별행정심판례 (2개, `special_admin_appeal/`)

| 이전 | 새 |
|------|-----|
| `sadm_case_조세심판원_v1.json` | `sadm_case_조세심판원_v2.json` |
| `sadm_case_해양안전심판원_v1.json` | `sadm_case_해양안전심판원_v2.json` |

## 수정된 코드 파일

인제스트 config의 `_DEFAULT_SOURCE` / `source_filename`을 새 버전으로 업데이트:

| 파일 | 변경 내용 |
|------|----------|
| `backend/scripts/ingest/types/law.py` | `law_v1.json` → `law_v3.json` |
| `backend/scripts/ingest/types/precedent.py` | `precedents_v1.json` → `precedents_v2.json` |
| `backend/scripts/ingest/types/admin_rule.py` | `admin_rule_v1.json` → `admin_rule_v2.json` |
| `backend/scripts/ingest/types/administration.py` | `administration_v1.json` → `administration_v2.json` |
| `backend/scripts/ingest/types/constitutional.py` | `constitutional_v1.json` → `constitutional_v2.json` |
| `backend/scripts/ingest/types/legislation.py` | `legislation_v1.json` → `legislation_v2.json` |
| `backend/scripts/ingest/types/treaty.py` | `treaty_v1.json` → `treaty_v2.json` |
| `backend/scripts/ingest/types/dec_privacy.py` | `_v1` → `_v2` |
| `backend/scripts/ingest/types/dec_employment.py` | `_v1` → `_v2` |
| `backend/scripts/ingest/types/dec_fair_trade.py` | `_v2` → `_v3` |
| `backend/scripts/ingest/types/dec_human_rights.py` | `_v1` → `_v2` |
| `backend/scripts/ingest/types/dec_civil_rights.py` | `_v1` → `_v2` |
| `backend/scripts/ingest/types/dec_financial.py` | `_v1` → `_v2` |
| `backend/scripts/ingest/types/dec_labor.py` | `_v1` → `_v2` |
| `backend/scripts/ingest/types/dec_industrial.py` | `_v1` → `_v2` |
| `backend/scripts/ingest/types/dec_environment.py` | `_v1` → `_v2` |
| `backend/scripts/ingest/types/dec_securities.py` | `_v1` → `_v2` |

디렉토리 소스 타입(interpretation_ministry, special_admin_appeal)은 디렉토리 경로만 참조하므로 config 변경 불필요. 내부 파일명이 자동으로 새 버전을 참조한다.

## 클리닝 스크립트

| 파일 | 설명 |
|------|------|
| `backend/scripts/clean_summaries.py` | 클리닝 실행 스크립트 (신규 생성) |

### 사용법

```bash
cd backend

# 드라이런 (파일 수정 없이 통계만)
uv run python scripts/clean_summaries.py --data-dir ../data --dry-run

# 전체 클리닝
uv run python scripts/clean_summaries.py --data-dir ../data

# 특정 타입만
uv run python scripts/clean_summaries.py --data-dir ../data --type law
```

## 원본 파일 처리

- 원본 JSON 파일(19개 타입, 구버전)은 모두 삭제됨 (디스크 약 4.7GB 회수)
- `data/`에 클리닝된 새 버전 파일만 남아있음
- 비데이터 파일(`lawyers.json`, `lawterms_v1.json`, `population.json` 등)은 유지
