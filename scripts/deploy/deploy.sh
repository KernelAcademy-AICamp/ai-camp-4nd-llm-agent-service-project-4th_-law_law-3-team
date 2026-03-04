#!/usr/bin/env bash
# 프로덕션 배포 스크립트 (반복 배포용)
# Usage: bash scripts/deploy/deploy.sh [OPTIONS]
#
# 초기 서버 세팅은 setup-server.sh를 먼저 실행하세요.
set -euo pipefail

COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.prod.yml}"
ENV_FILE="${ENV_FILE:-.env.prod}"
BRANCH="${BRANCH:-deploy/aws-arm}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-900}"
BACKEND_CONTAINER="law-platform-backend"

SKIP_PULL=false
SKIP_BUILD=false
SYNC_DATA=false
LOAD_LANCEDB=false

for arg in "$@"; do
    case "$arg" in
        --skip-pull)   SKIP_PULL=true ;;
        --skip-build)  SKIP_BUILD=true ;;
        --sync-data)   SYNC_DATA=true ;;
        --load-lancedb) LOAD_LANCEDB=true ;;
        --help|-h)
            cat <<'USAGE'
Usage: bash scripts/deploy/deploy.sh [OPTIONS]

코드 배포:
  (기본)            pull → build → up --wait → health check
  --skip-pull       git pull 건너뛰기 (로컬 변경 테스트 시)
  --skip-build      Docker 빌드 건너뛰기 (설정만 변경 시)

데이터 업데이트:
  --sync-data       S3에서 data/, MeCab 사전 다운로드 (bind mount → 재시작으로 반영)
  --load-lancedb    S3에서 LanceDB 다운로드 + Docker named volume에 복사

조합 예시:
  bash deploy.sh                          # 코드만 배포
  bash deploy.sh --sync-data              # 코드 + data/MeCab 동기화
  bash deploy.sh --load-lancedb           # 코드 + LanceDB 벡터 교체
  bash deploy.sh --sync-data --load-lancedb  # 전체 데이터 갱신
  bash deploy.sh --skip-pull --skip-build --sync-data  # 데이터만 갱신

환경변수 오버라이드:
  BRANCH          배포 브랜치 (기본: deploy/aws-arm)
  WAIT_TIMEOUT    healthcheck 대기 시간 초 (기본: 300)
  COMPOSE_FILE    compose 파일 (기본: docker-compose.prod.yml)
  ENV_FILE        환경 파일 (기본: .env.prod)
USAGE
            exit 0
            ;;
        *)
            echo "ERROR: 알 수 없는 옵션: $arg"
            echo "  bash scripts/deploy/deploy.sh --help"
            exit 1
            ;;
    esac
done

DC="docker compose --env-file $ENV_FILE -f $COMPOSE_FILE"

cd "$(dirname "$0")/../.."
echo "=== 배포 시작: $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "  브랜치: $BRANCH"
echo "  Compose: $COMPOSE_FILE"
echo "  데이터: sync-data=$SYNC_DATA, load-lancedb=$LOAD_LANCEDB"
echo ""

# ──────────────────────────────────────────────
# 사전 검증
# ──────────────────────────────────────────────
if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: $ENV_FILE 파일이 없습니다."
    echo "  cp .env.prod.example .env.prod 후 설정하세요."
    exit 1
fi

if [ "$SYNC_DATA" = true ] || [ "$LOAD_LANCEDB" = true ]; then
    if ! command -v aws &>/dev/null; then
        echo "ERROR: aws CLI가 설치되어 있지 않습니다."
        echo "  sudo apt-get install -y awscli"
        exit 1
    fi
fi

# ──────────────────────────────────────────────
# 1. 코드 업데이트
# ──────────────────────────────────────────────
if [ "$SKIP_PULL" = false ]; then
    echo "=== 1. 코드 업데이트 ==="
    git fetch origin
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
else
    echo "=== 1. 코드 업데이트 (건너뜀) ==="
fi

# 볼륨 마운트 디렉토리 보장
mkdir -p backend/data/models backend/data/mecab_userdic

# ──────────────────────────────────────────────
# 2. S3 데이터 동기화 (bind mount 대상)
#    data/, MeCab 사전 → 호스트에 내려받으면 재시작 시 바로 반영
# ──────────────────────────────────────────────
if [ "$SYNC_DATA" = true ]; then
    echo "=== 2. S3 데이터 동기화 (data/, MeCab) ==="
    bash scripts/deploy/s3-download.sh

    # MeCab userdic .dic 빌드 (ARM native)
    # S3에서 CSV/JSON만 내려받고 .dic은 제외되므로, 호스트에서 빌드 필요
    MECAB_CSV="backend/data/mecab_userdic/legal_terms.csv"
    MECAB_DIC="backend/data/mecab_userdic/legal_terms.dic"
    if [ -f "$MECAB_CSV" ] && [ ! -f "$MECAB_DIC" ]; then
        echo "  MeCab userdic .dic 빌드 (ARM native)..."
        # mecab-builder stage를 활용하여 아키텍처 호환 .dic 생성
        docker build -q --target mecab-builder \
            -f docker/backend/Dockerfile.prod \
            -t law-mecab-builder backend/ > /dev/null 2>&1
        docker run --rm \
            -v "$(pwd)/backend/data/mecab_userdic:/data" \
            law-mecab-builder \
            /usr/local/libexec/mecab/mecab-dict-index \
            -d /usr/local/lib/mecab/dic/mecab-ko-dic \
            -u /data/legal_terms.dic \
            -f utf-8 -t utf-8 \
            /data/legal_terms.csv
        echo "  MeCab userdic .dic 빌드 완료: $MECAB_DIC"
    elif [ -f "$MECAB_DIC" ]; then
        echo "  MeCab userdic .dic 이미 존재: $MECAB_DIC"
    else
        echo "  WARNING: MeCab CSV 없음, .dic 빌드 건너뜀 (S3에 mecab_userdic 데이터 확인 필요)"
    fi
else
    echo "=== 2. S3 데이터 동기화 (건너뜀) ==="
fi

# ──────────────────────────────────────────────
# 3. Docker 빌드
# ──────────────────────────────────────────────
if [ "$SKIP_BUILD" = false ]; then
    echo "=== 3. Docker 빌드 ==="
    $DC build
else
    echo "=== 3. Docker 빌드 (건너뜀) ==="
fi

# ──────────────────────────────────────────────
# 3.5. Named volume 권한 수정
#      비루트(appuser) 컨테이너용. 빈 volume은 root 소유로 생성되므로 수정 필요
# ──────────────────────────────────────────────
echo "=== 3.5. Named volume 권한 수정 ==="
for vol in law-platform_media_data law-platform_lancedb_data; do
    if docker volume inspect "$vol" &>/dev/null; then
        docker run --rm -v "$vol:/mnt/vol" alpine \
            chmod -R 777 /mnt/vol 2>/dev/null || true
    fi
done

# ──────────────────────────────────────────────
# 4. 서비스 기동
#    - entrypoint: download_models.py + alembic upgrade head 자동 실행
#    - --wait: 모든 컨테이너 healthy 될 때까지 대기
# ──────────────────────────────────────────────
echo "=== 4. 서비스 기동 (healthcheck 대기, 최대 ${WAIT_TIMEOUT}s) ==="
$DC up -d --wait --wait-timeout "$WAIT_TIMEOUT"

# ──────────────────────────────────────────────
# 5. LanceDB 벡터 데이터 교체 (named volume)
#    s3-download.sh가 호스트 backend/lancedb_data/에 내려받은 데이터를
#    Docker named volume에 복사 후 backend 재시작
# ──────────────────────────────────────────────
if [ "$LOAD_LANCEDB" = true ]; then
    echo "=== 5. LanceDB 데이터 → Docker volume 복사 ==="

    # s3-download.sh가 아직 안 돌았으면 LanceDB 부분만 다운로드
    if [ ! -d "backend/lancedb_data" ] || [ -z "$(ls -A backend/lancedb_data 2>/dev/null)" ]; then
        echo "  호스트에 LanceDB 데이터 없음 → S3에서 다운로드..."
        mkdir -p backend/lancedb_data
        aws s3 sync \
            "s3://law-3/deploy/lancedb_data/" \
            "backend/lancedb_data/" \
            --region ap-northeast-2
    fi

    echo "  docker cp → $BACKEND_CONTAINER:/app/lancedb_data/ ..."
    docker cp backend/lancedb_data/. "$BACKEND_CONTAINER":/app/lancedb_data/
    echo "  backend 재시작..."
    $DC restart backend

    # 재시작 후 healthy 대기
    echo "  healthcheck 재대기..."
    $DC up -d --wait --wait-timeout "$WAIT_TIMEOUT"
else
    echo "=== 5. LanceDB 데이터 교체 (건너뜀) ==="
fi

# ──────────────────────────────────────────────
# 6. 최종 헬스 체크
# ──────────────────────────────────────────────
echo "=== 6. 헬스 체크 ==="
if $DC exec -T backend curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo "Health check PASSED (backend direct)"
else
    echo "WARNING: Backend health check failed. 로그 확인:"
    $DC logs --tail=30 backend
    exit 1
fi

echo ""
echo "=== 배포 완료: $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "  상태: $DC ps"
echo "  로그: $DC logs -f backend"
