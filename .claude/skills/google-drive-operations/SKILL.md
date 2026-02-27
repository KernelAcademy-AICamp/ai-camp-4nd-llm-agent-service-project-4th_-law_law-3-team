---
name: google-drive-operations
description: Google Drive를 통한 DB 백업/복원, 원본 데이터(data/) 동기화, 새 환경 세팅 가이드. 백업, 복원, Google Drive, rclone, 데이터 동기화, 새 환경 세팅, 데이터 업로드, 데이터 다운로드 관련 작업 시 반드시 사용. DB 백업이 필요하거나 다른 기기에서 프로젝트를 세팅할 때도 이 스킬을 참조.
---

# Google Drive 운영 스킬

rclone을 통해 Google Drive와 2개 DB(PostgreSQL, LanceDB) 백업/복원, 원본 JSON 데이터 동기화를 수행합니다.

## 아키텍처 개요

```
Google Drive (gdrive:)
├── <TIMESTAMP>/              ← DB 백업 (scripts/backup_to_gdrive.sh)
│   ├── postgres.dump
│   └── lancedb_data.tar.gz
│
└── data/                     ← 원본 데이터 (~7GB)
    ├── ingest_source/        ← 인제스트 파이프라인 전용 데이터
    │   ├── law_v3.json
    │   ├── precedents_v2.json
    │   ├── local_rules_v1.json
    │   ├── decisions_committee/
    │   ├── interpretation_ministry/
    │   └── special_admin_appeal/
    ├── lawyers.json           ← 인제스트 외 데이터
    ├── lawterms_v1.json
    ├── population.json
    ├── trial_statistics_data/
    └── incoming/
```

## 사전 준비 (1회)

### 1. rclone 설치

```bash
# macOS
brew install rclone

# Linux/WSL
curl https://rclone.org/install.sh | sudo bash
```

### 2. 인증 파일 배치

팀에서 받은 파일을 프로젝트 루트에 배치합니다. 두 가지 인증 방식 중 하나를 사용합니다:

| 방식 | 필요 파일 | 특징 |
|------|----------|------|
| Service Account | `rclone.conf` + `secrets/<SA>.json` | 서버/CI 환경 권장 |
| OAuth token | `rclone.conf` (token 포함) | 개인 개발 환경 권장 |

두 파일 모두 `.gitignore`에 포함되어 있으므로 git 외부(Slack, DM 등)로 공유합니다.

### 3. 연결 확인

```bash
# Google Drive 접근 테스트
rclone ls --config rclone.conf gdrive: --max-depth 1
```

## 작업별 명령어

### A. DB 백업 (→ Google Drive)

```bash
# 전체 백업 + 업로드
./scripts/backup_to_gdrive.sh

# 로컬 덤프만 (업로드 안 함)
./scripts/backup_to_gdrive.sh --skip-upload

# 특정 DB 건너뛰기
./scripts/backup_to_gdrive.sh --skip-postgres
./scripts/backup_to_gdrive.sh --skip-lancedb

# 미리보기 (실제 실행 없이 계획만)
./scripts/backup_to_gdrive.sh --dry-run
```

**스크립트 동작**: Docker 컨테이너에서 덤프 → 로컬 `backups/<TIMESTAMP>/` 저장 → rclone으로 업로드 → 오래된 로컬 백업 자동 정리 (기본 5개 유지)

### B. DB 복원 (Google Drive →)

```bash
# 최신 백업 복원
./scripts/restore_from_gdrive.sh latest

# 특정 타임스탬프 복원
./scripts/restore_from_gdrive.sh 20260211_153000

# 다운로드만 (DB 복원 안 함)
./scripts/restore_from_gdrive.sh latest --download-only

# 특정 DB 건너뛰기
./scripts/restore_from_gdrive.sh latest --skip-postgres
```

**스크립트 동작**: 백업 목록 조회 → 다운로드 → 사용자 확인 프롬프트 → DB별 복원

**LanceDB 복원 후 필수 작업** (인덱스 재생성):

```bash
cd backend

# 벡터 인덱스 재생성 (LANCEDB_INDEX_TYPE=IVF_FLAT 설정 시)
uv run python -c \
  "from app.tools.vectorstore import get_lancedb_store; s = get_lancedb_store(); s.create_index()"

# FTS content_tokenized 인덱스 재생성
uv run --no-sync python scripts/update_content_tokenized.py --userdic
```

### C. 원본 JSON 데이터 동기화 (data/)

DB 백업과는 별개로, 법령/판례 등 원본 JSON 파일(~3.5GB)을 Google Drive `data/` 폴더와 동기화합니다.

```bash
# ── 다운로드 (다른 기기에서 복원) ──
# 전체 다운로드
rclone copy --config rclone.conf gdrive:data/ data/ --progress

# 특정 파일만 다운로드
rclone copy --config rclone.conf gdrive:data/precedents_v2.json data/ --progress

# ── 업로드 (로컬 변경 후 동기화) ──
# sync: 로컬에 없는 파일은 드라이브에서도 삭제
rclone sync data/ --config rclone.conf gdrive:data/ --progress

# ── 현재 Drive 내용 확인 ──
rclone ls --config rclone.conf gdrive:data/

# ── 로컬 vs Drive 차이 비교 (dry-run) ──
rclone sync data/ --config rclone.conf gdrive:data/ --dry-run
```

**`copy` vs `sync` 사용 기준**:

| 명령 | 동작 | 사용 시점 |
|------|------|----------|
| `copy` | 소스에 있는 파일만 추가/덮어쓰기 | 다운로드(복원) 시 |
| `sync` | 소스와 동일하게 만듦 (소스에 없으면 삭제) | 업로드(동기화) 시 |

파일명 변경(`v2→v3`) 후 `copy`를 쓰면 이전 버전이 드라이브에 잔류하므로 `sync` 권장.

## 새 환경 세팅 (전체 워크플로우)

새 기기에서 프로젝트를 처음 세팅할 때의 전체 흐름:

```bash
# 1. 코드 클론
git clone <repo-url> && cd law-3-team

# 2. 팀에서 받은 인증 파일 배치
cp ~/받은파일/rclone.conf .
cp ~/받은파일/gdrive-service-account.json secrets/  # SA 방식 시

# 3. 원본 JSON 데이터 복원
rclone copy --config rclone.conf gdrive:data/ data/ --progress

# 4. DB 백업 복원 (Docker 컨테이너 먼저 시작)
docker compose up -d
./scripts/restore_from_gdrive.sh latest

# 5. LanceDB 인덱스 재생성
cd backend
uv run --no-sync python scripts/update_content_tokenized.py --userdic

# 6. 임베딩/리랭커 모델 다운로드
uv run python scripts/download_models.py
```

## 환경변수

`.env`에서 설정 가능한 변수 (기본값이 있으므로 보통 변경 불필요):

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `POSTGRES_CONTAINER` | PostgreSQL 컨테이너명 | `law-platform-db` |
| `LANCEDB_DATA_DIR` | LanceDB 데이터 경로 | `backend/lancedb_data` |
| `BACKUP_KEEP_LOCAL` | 로컬 백업 보관 개수 | `5` |
| `RCLONE_CONF` | rclone 설정 파일 경로 | `rclone.conf` |
| `RCLONE_REMOTE` | rclone 리모트 이름 | `gdrive` |

## 관련 파일

| 파일 | 설명 |
|------|------|
| `scripts/backup_to_gdrive.sh` | DB 백업 + Google Drive 업로드 |
| `scripts/restore_from_gdrive.sh` | Google Drive → DB 복원 |
| `rclone.conf` | rclone 설정 (.gitignored) |
| `secrets/` | 서비스 계정 키 (.gitignored) |
| `backups/` | 로컬 백업 저장 (.gitignored) |
| `data/` | 원본 JSON 데이터 (.gitignored) |
| `docs/operations/backup-restore.md` | 운영 문서 (사람용) |

## 트러블슈팅

### rclone 인증 만료 (OAuth token 방식)

OAuth token은 일정 기간 후 만료됩니다. `403 Forbidden` 또는 `token expired` 에러 시:

```bash
# 토큰 재발급
rclone config reconnect gdrive: --config rclone.conf
```

### Docker 컨테이너 미실행

백업/복원 스크립트는 사전 검증(preflight check)에서 컨테이너 상태를 확인합니다. 실행 중이 아니면:

```bash
docker compose up -d
# WSL2 환경에서는
docker.exe compose up -d
```

### rclone 속도 개선

대용량 전송 시 `--transfers` 옵션으로 병렬 수를 조정합니다 (기본 4):

```bash
rclone copy --config rclone.conf gdrive:data/ data/ --progress --transfers 8
```

### Drive 용량 확인

```bash
rclone about --config rclone.conf gdrive:
```

## 연계 스킬

| 스킬 | 연계 시점 |
|------|----------|
| `data-file-rename` | 외부 데이터 수신 → 리네임 → data/ 배치 → Drive 업로드 |
| `ingest-pipeline` | data/ 복원 후 → DB/벡터 적재 |
| `korean-legal-domain` | 데이터 타입별 JSON 필드 구조 참조 |

**데이터 라이프사이클**: 외부 수신 → `data-file-rename` → data/ingest_source/ 배치 → `rclone sync` (Drive 동기화) → `ingest-pipeline` (DB 적재) → 서비스 운영 → `backup_to_gdrive.sh` (DB 백업)
