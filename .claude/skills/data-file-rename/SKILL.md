---
name: data-file-rename
description: 외부에서 들어온 법률 데이터 JSON 파일을 프로젝트 네이밍 규칙에 맞게 변환. 데이터 파일 이름 변환, 파일 정리, 새 데이터 수신 처리, incoming 디렉토리 관리, 데이터 파일 버전 관리 시 사용. ingest-pipeline 스킬의 전처리 단계로 동작.
---

# 데이터 파일 리네임 스킬

외부 소스(법제처 API, 크롤링 등)에서 받은 법률 데이터 JSON 파일을 프로젝트 네이밍 규칙에 맞게 변환합니다.

이 스킬은 **데이터 라이프사이클의 첫 단계**입니다:

```
외부 데이터 도착 → [data-file-rename] → data/ 배치 → [ingest-pipeline] → DB/벡터 적재
```

## 스크립트 위치

`backend/scripts/rename_incoming_data.py`

## 네이밍 규칙

| 데이터 유형 | 패턴 | 예시 |
|------------|------|------|
| 단일 파일 (법령, 판례 등) | `<타입>_v<N>.json` | `law_v3.json`, `precedents_v2.json` |
| 부처 해석례 | `intp_min_<부처명>_v<N>.json` | `intp_min_고용노동부_v3.json` |
| 위원회 결정문 | `dec_comm_<위원회명>_v<N>.json` | `dec_comm_금융위원회_v3.json` |
| 특별행정심판 | `sadm_case_<기관명>_v<N>.json` | `sadm_case_조세심판원_v2.json` |

## 사용법

### 기본 워크플로우

```bash
cd backend

# 1. 외부 파일을 data/incoming/ 에 복사
cp ~/downloads/ppc.json ../data/incoming/
cp ~/downloads/law_full-30.json ../data/incoming/

# 2. 미리보기 (dry-run, 기본 동작)
uv run python scripts/rename_incoming_data.py

# 3. 실제 복사 실행
uv run python scripts/rename_incoming_data.py --execute

# 4. (선택) 복사 대신 이동
uv run python scripts/rename_incoming_data.py --execute --move
```

### CLI 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| (기본) | dry-run 미리보기 | ✅ |
| `--execute` | 실제 복사/이동 수행 | `false` |
| `--move` | 복사 대신 이동 (`--execute`와 함께) | `false` |
| `--file <이름>` | 특정 파일만 처리 | 전체 |
| `--show-mapping` | 63개 매핑 테이블 출력 | - |
| `--incoming-dir <경로>` | incoming 디렉토리 변경 | `data/incoming/` |

### 매핑 테이블 확인

```bash
# 지원하는 모든 파일명과 대상 경로 확인
uv run python scripts/rename_incoming_data.py --show-mapping
```

## 버전 관리 동작

스크립트는 대상 디렉토리를 스캔하여 **자동으로 다음 버전 번호**를 결정합니다:

- `data/law_v3.json` 존재 → 새 파일은 `law_v4.json`
- `data/decisions_committee/dec_comm_금융위원회_v2.json` 존재 → `dec_comm_금융위원회_v3.json`
- 대상 디렉토리에 해당 파일 없음 → `v1` (신규)

macOS APFS의 Unicode NFD/NFC 차이를 자동 처리합니다.

## 지원 파일 (63개 매핑)

| 카테고리 | 예시 incoming 파일명 | 변환 결과 |
|----------|---------------------|-----------|
| 판례 | `precedents-5.json` | `precedents_v<N>.json` |
| 법령 | `law_full-30.json` | `law_v<N>.json` |
| 행정규칙 | `administrative_rules-28.json` | `admin_rule_v<N>.json` |
| 헌재결정례 | `constitutionalDecc.json` | `constitutional_v<N>.json` |
| 행정심판례 | `administrationDecc.json` | `administration_v<N>.json` |
| 법령해석례 | `legislationItpt.json` | `legislation_v<N>.json` |
| 조약 | `treaties.json` | `treaty_v<N>.json` |
| 위원회 결정문 | `ppc.json`, `fsc-full.json` 등 | `dec_comm_<위원회>_v<N>.json` |
| 부처 해석례 | `moelCgmExpc.json` 등 | `intp_min_<부처>_v<N>.json` |
| 특별행정심판 | `ttSpecialDecc.json` 등 | `sadm_case_<기관>_v<N>.json` |

전체 매핑은 `--show-mapping` 옵션으로 확인하세요.

## 연계 스킬

### 전처리 → 인제스트 파이프라인

리네임 완료 후 `ingest-pipeline` 스킬로 DB/벡터 적재를 진행합니다:

```bash
# 1. 리네임 (이 스킬)
uv run python scripts/rename_incoming_data.py --execute

# 2. 인제스트 (ingest-pipeline 스킬)
uv run python -m scripts.ingest.cli --type all --step all --reset
```

### 데이터 구조 참조

각 데이터 타입의 JSON 필드 구조는 `korean-legal-domain` 스킬을 참조하세요.
법령 XML 계층, 판례 필드, 참조 관계 유형 등의 도메인 지식이 정리되어 있습니다.

### 새 데이터 타입 추가

매핑에 없는 새로운 데이터 유형이 들어올 경우:

1. `rename_incoming_data.py`의 `MAPPING` dict에 항목 추가
2. `ingest-pipeline` 스킬의 새 타입 추가 패턴 참조 (ORM, migration, sources.yaml 등)

## 디렉토리 구조

```
data/
├── incoming/              ← 외부 파일 임시 저장 (이 스킬이 처리)
│   ├── .gitkeep
│   └── (외부에서 받은 JSON 파일들)
│
├── law_v3.json            ← 리네임 후 배치 위치
├── precedents_v2.json
├── decisions_committee/
│   └── dec_comm_*_v<N>.json
├── interpretation_ministry/
│   └── intp_min_*_v<N>.json
└── special_admin_appeal/
    └── sadm_case_*_v<N>.json
```

## 에러 처리

- **매핑 없는 파일**: 경고 출력 후 건너뜀 (exit code 1)
- **서브디렉토리 자동 생성**: 대상 디렉토리가 없으면 자동 생성
- **파일 충돌**: 대상 경로에 이미 동일 버전 파일이 있으면 건너뜀
