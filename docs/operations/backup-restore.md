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

## data/ JSON 파일 (원본 데이터)

법령/판례 등 원본 JSON 데이터(약 3.5GB, 63개 파일)는 `gdrive:data/`에 저장되어 있습니다.
DB 백업/복원 스크립트와는 별개이며, 항상 최신 작업 데이터를 유지합니다.

> **버전 아카이브**(`v1`, `v2` 등 과거 버전)는 별도 Google Drive에서 관리합니다.
> `gdrive:data/`는 "현재 작업 세트"이며, 버전 히스토리 용도가 아닙니다.

```bash
# 사전 조건: rclone 설치 + rclone.conf 배치 (위 "사전 준비" 참조)

# ── 다른 기기에서 복원 ──
rclone copy --config rclone.conf gdrive:data/ data/ --progress

# 특정 파일만 복원
rclone copy --config rclone.conf gdrive:data/precedents_v2.json data/ --progress

# ── 로컬 변경 후 업로드 (동기화) ──
# sync: 로컬에 없는 파일은 드라이브에서도 삭제 (항상 로컬과 동일하게 유지)
rclone sync data/ --config rclone.conf gdrive:data/ --progress

# 현재 Google Drive 내용 확인
rclone ls --config rclone.conf gdrive:data/
```

> **`copy` vs `sync`**: 복원 시에는 `copy` (추가만), 업로드 시에는 `sync` (삭제 반영) 사용.
> 파일명 변경(`v2→v3`) 시 `copy`를 쓰면 이전 버전이 드라이브에 잔류하므로 `sync` 권장.
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
