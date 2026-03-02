# DB 백업 / 복원 (Google Drive)

PostgreSQL, Neo4j, LanceDB 3개 DB를 Google Drive에 백업/복원합니다.
인증 방식은 **Service Account** 또는 **OAuth token** 둘 다 지원합니다 (rclone.conf에 따라 자동 감지).

## 사전 준비 (1회)

1. **rclone 설치**: `brew install rclone`
2. **팀에서 받은 `rclone.conf`를 프로젝트 루트에 배치**
   - Service Account 방식: `rclone.conf` + `secrets/<service-account>.json`
   - OAuth token 방식: `rclone.conf` (token 포함, 별도 키 파일 불필요)
3. 두 파일 모두 `.gitignore`에 포함되어 있으므로 git 외부로 공유

## 백업

```bash
# 전체 백업 + Google Drive 업로드
./scripts/backup_to_gdrive.sh

# 로컬 덤프만 (업로드 안 함)
./scripts/backup_to_gdrive.sh --skip-upload

# 특정 DB 건너뛰기
./scripts/backup_to_gdrive.sh --skip-neo4j

# 미리보기
./scripts/backup_to_gdrive.sh --dry-run
```

## 복원

```bash
# 최신 백업 복원
./scripts/restore_from_gdrive.sh latest

# 특정 백업 복원
./scripts/restore_from_gdrive.sh 20260211_153000

# 다운로드만 (복원 안 함)
./scripts/restore_from_gdrive.sh latest --download-only
```

## 환경변수

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `POSTGRES_CONTAINER` | PostgreSQL 컨테이너명 | `law-platform-db` |
| `NEO4J_CONTAINER` | Neo4j 컨테이너명 | `neo4j-law-graph` |
| `LANCEDB_DATA_DIR` | LanceDB 데이터 경로 | `backend/lancedb_data` |
| `BACKUP_KEEP_LOCAL` | 로컬 백업 보관 개수 | `5` |
| `RCLONE_CONF` | rclone 설정 파일 경로 | `rclone.conf` |
| `RCLONE_REMOTE` | rclone 리모트 이름 | `gdrive` |

## 관련 파일

| 파일 | 설명 |
|------|------|
| `scripts/backup_to_gdrive.sh` | 백업 + 업로드 (LanceDB는 data/ only, 인덱스 제외) |
| `scripts/restore_from_gdrive.sh` | 다운로드 + 복원 |
| `secrets/` | 서비스 계정 키 등 (.gitignored) |
| `rclone.conf` | rclone 설정 (.gitignored) |

## 복원 후 BM25 인덱스 + law_articles 재생성

PostgreSQL 백업을 복원하면 테이블 데이터는 복원되지만, **BM25 인덱스**는 별도로 재생성해야 할 수 있습니다.

```bash
cd backend

# 1. BM25 인덱스 상태 확인
uv run python scripts/create_bm25_index.py --check

# 2-a. 인덱스가 없는 경우 → 생성
uv run python scripts/create_bm25_index.py

# 2-b. 인덱스가 손상된 경우 → 삭제 후 재생성
uv run python scripts/create_bm25_index.py --drop

# 3. search_text가 비어 있는 경우 (복원 데이터에 search_text가 NULL일 때)
uv run python -m scripts.ingest.cli --type all --step fts
uv run python scripts/create_bm25_index.py

# 4. law_articles 테이블이 비어 있는 경우
uv run python scripts/load_law_articles_data.py
uv run python scripts/load_law_articles_data.py --verify
```

> **참고**: `search_text` 컬럼 데이터가 있어야 BM25 인덱스 생성이 가능합니다.
> `--step fts`는 ORM 원본 테이블에서 `search_text`만 재빌드하므로 JSON 재처리 없이 빠르게 완료됩니다.

## data/ 동기화 (원본 데이터 + backend 데이터)

`gdrive:data/`에는 원본 JSON 파일과 backend 전용 데이터가 함께 저장되어 있습니다.
로컬에서는 경로가 다르므로 **반드시 전용 스크립트**를 사용하세요.

> **주의**: `rclone copy gdrive:data/ data/`를 직접 실행하면 `lancedb_data`, `mecab_userdic`, `models`가
> 잘못된 경로(`data/`)에 들어갑니다.

### 경로 매핑

| Google Drive | 로컬 경로 |
|---|---|
| `gdrive:data/lancedb_data/` | `backend/lancedb_data/` |
| `gdrive:data/mecab_userdic/` | `backend/data/mecab_userdic/` |
| `gdrive:data/models/` | `backend/data/models/` |
| `gdrive:data/*.json` 등 | `data/` |

### 다운로드

```bash
# 전체 다운로드 (경로 자동 매핑)
./scripts/sync_data_from_gdrive.sh

# 미리보기
./scripts/sync_data_from_gdrive.sh --dry-run

# JSON/원본 데이터만 (data/)
./scripts/sync_data_from_gdrive.sh --only-json

# backend 데이터만 (lancedb, mecab, models)
./scripts/sync_data_from_gdrive.sh --only-backend
```

### 업로드 (수동)

업로드는 기존 rclone 명령을 사용합니다.

```bash
# 원본 JSON 업로드
rclone sync data/ --config rclone.conf gdrive:data/ --progress \
  --exclude "lancedb_data/**" --exclude "mecab_userdic/**" --exclude "models/**"

# 현재 Google Drive 내용 확인
rclone ls --config rclone.conf gdrive:data/
```

> **`copy` vs `sync`**: 복원 시에는 `copy` (추가만), 업로드 시에는 `sync` (삭제 반영) 사용.
>
> **참고**: `data/`는 `.gitignore`에 포함되어 있어 git clone만으로는 받을 수 없습니다.
> 새 환경 세팅 시 DB 복원(`restore_from_gdrive.sh`)과 함께 이 단계를 수행하세요.

## ONNX 모델 백업 / 복원

ONNX 최적화 모델(리랭커/임베딩, FP32 및 QDQ INT8)을 Google Drive에 백업/복원합니다.
DB 백업과 별도 스크립트로 관리합니다 (Docker 불필요, 빈도가 다름).

### Google Drive 구조

```
gdrive:onnx-models/
  reranker-ort-opt/           # 리랭커 FP32 Fusion
  reranker-ort-opt-qdq/       # 리랭커 QDQ INT8 (주 사용 대상)
  kure-v1-ort-opt/            # 임베딩 FP32
  kure-v1-ort-opt-qdq/        # 임베딩 QDQ INT8
  model_versions.json         # 빌드 메타데이터 (검증 결과 포함)
```

### 빌드 + 백업 (WSL2/Linux)

```bash
# 리랭커만 빌드 → 검증 → Google Drive 업로드
./scripts/onnx_model_gdrive.sh build-and-backup --reranker-only

# 전체 모델 빌드 + 백업 (기존 모델 덮어쓰기)
./scripts/onnx_model_gdrive.sh build-and-backup --overwrite

# 빌드 없이 기존 모델만 백업
./scripts/onnx_model_gdrive.sh backup --reranker-only

# 미리보기
./scripts/onnx_model_gdrive.sh backup --dry-run
```

### 복원 (Mac ARM 등 다른 환경)

```bash
# 리랭커 모델만 복원
./scripts/onnx_model_gdrive.sh restore --reranker-only

# 전체 모델 복원
./scripts/onnx_model_gdrive.sh restore

# 원격 모델 목록 확인
./scripts/onnx_model_gdrive.sh list
```

### .env 설정 (복원 후)

```bash
# 리랭커 ONNX 사용
USE_ONNX_RERANKER=true
ONNX_RERANKER_VARIANT=ort-opt-qdq

# 임베딩 ONNX 사용
USE_ONNX_EMBEDDING=true
ONNX_EMBEDDING_VARIANT=ort-opt-qdq
```

### 워크플로우

```
[WSL2/Linux]                              [Mac ARM]
1. ./scripts/onnx_model_gdrive.sh \
     build-and-backup --reranker-only
   (빌드 → 검증 → gdrive 업로드)
                                          2. ./scripts/onnx_model_gdrive.sh restore --reranker-only
                                          3. .env 설정 (위 참조)
                                          4. cd backend && uv run uvicorn app.main:app --reload
```

### 관련 파일

| 파일 | 설명 |
|------|------|
| `scripts/onnx_model_gdrive.sh` | ONNX 모델 백업/복원/빌드 |
| `backend/scripts/build_optimized_onnx.py` | ONNX 모델 빌드 (--reranker-only) |
| `backend/app/services/rag/onnx_session.py` | 런타임 모델 로딩, 파일 검증 |
| `backend/data/models/model_versions.json` | 빌드 메타데이터 |
