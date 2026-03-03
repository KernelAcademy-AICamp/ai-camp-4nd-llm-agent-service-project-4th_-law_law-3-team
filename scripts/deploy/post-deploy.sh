#!/usr/bin/env bash
# 초기 배포 후 데이터 프로비저닝 스크립트
# deploy.sh로 서비스를 기동한 뒤, 데이터·모델·SSL을 한번에 설정합니다.
#
# Usage: bash scripts/deploy/post-deploy.sh [OPTIONS]
#
# 옵션:
#   --ssl            SSL 인증서 발급 (init-letsencrypt.sh 호출)
#   --skip-s3        S3 다운로드 건너뛰기 (이미 데이터가 있을 때)
#   --skip-lancedb   LanceDB 볼륨 복사 건너뛰기
#   --skip-db-load   PostgreSQL 데이터 적재 건너뛰기 (이미 적재된 경우)
#   --clean-models   불완전한 모델 캐시 삭제 (재시작 시 재다운로드)
#   --dry-run        S3 다운로드 dry-run (실제 전송 없이 확인만)
#   --help|-h        도움말
#
# 전체 실행 예시 (첫 배포 후):
#   bash scripts/deploy/post-deploy.sh --ssl --clean-models
#
# 데이터만 갱신:
#   bash scripts/deploy/post-deploy.sh --skip-s3  # LanceDB 볼륨만 재복사
#
# DB 적재만 재실행:
#   bash scripts/deploy/post-deploy.sh --skip-s3 --skip-lancedb
set -euo pipefail

cd "$(dirname "$0")/../.."

COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.prod.yml}"
ENV_FILE="${ENV_FILE:-.env.prod}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-900}"

DO_SSL=false
SKIP_S3=false
SKIP_LANCEDB=false
SKIP_DB_LOAD=false
CLEAN_MODELS=false
DRY_RUN=""

for arg in "$@"; do
    case "$arg" in
        --ssl)           DO_SSL=true ;;
        --skip-s3)       SKIP_S3=true ;;
        --skip-lancedb)  SKIP_LANCEDB=true ;;
        --skip-db-load)  SKIP_DB_LOAD=true ;;
        --clean-models)  CLEAN_MODELS=true ;;
        --dry-run)       DRY_RUN="--dry-run" ;;
        --help|-h)
            sed -n '2,/^set -euo/{ /^set -euo/d; s/^# \?//; p }' "$0"
            exit 0
            ;;
        *)
            echo "ERROR: 알 수 없는 옵션: $arg"
            echo "  bash scripts/deploy/post-deploy.sh --help"
            exit 1
            ;;
    esac
done

DC="docker compose --env-file $ENV_FILE -f $COMPOSE_FILE"

echo "============================================"
echo "  포스트 배포 데이터 프로비저닝"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
echo ""

# ──────────────────────────────────────────────
# 사전 검증
# ──────────────────────────────────────────────
if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: $ENV_FILE 파일이 없습니다."
    exit 1
fi

if ! $DC ps backend 2>/dev/null | grep -qE "running|Up|healthy"; then
    echo "ERROR: backend 컨테이너가 실행 중이 아닙니다."
    echo "  먼저 deploy.sh를 실행하세요: bash scripts/deploy/deploy.sh"
    exit 1
fi

# 컨테이너 ID를 동적 조회 (이름 충돌로 접두사가 붙는 경우 대비)
BACKEND_CONTAINER=$($DC ps -q backend)
if [ -z "$BACKEND_CONTAINER" ]; then
    echo "ERROR: backend 컨테이너 ID를 조회할 수 없습니다."
    exit 1
fi
echo "Backend 컨테이너: $BACKEND_CONTAINER"

# ──────────────────────────────────────────────
# 1. S3 데이터 다운로드 (data/, MeCab CSV/JSON)
# ──────────────────────────────────────────────
if [ "$SKIP_S3" = false ]; then
    echo "=== 1. S3 데이터 다운로드 ==="
    if ! command -v aws &>/dev/null; then
        echo "ERROR: aws CLI가 설치되어 있지 않습니다."
        exit 1
    fi
    bash scripts/deploy/s3-download.sh $DRY_RUN
else
    echo "=== 1. S3 데이터 다운로드 (건너뜀) ==="
fi

# ──────────────────────────────────────────────
# 2. MeCab userdic .dic 빌드 (ARM native)
#    S3에서 CSV/JSON만 내려받고 .dic은 아키텍처별 빌드 필요
# ──────────────────────────────────────────────
echo ""
echo "=== 2. MeCab userdic .dic 빌드 ==="
MECAB_CSV="backend/data/mecab_userdic/legal_terms.csv"
MECAB_DIC="backend/data/mecab_userdic/legal_terms.dic"

if [ -f "$MECAB_CSV" ] && [ ! -f "$MECAB_DIC" ]; then
    echo "  .dic 빌드 시작 (ARM native)..."
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
    echo "  완료: $MECAB_DIC"
elif [ -f "$MECAB_DIC" ]; then
    echo "  .dic 이미 존재: $MECAB_DIC (건너뜀)"
else
    echo "  WARNING: MeCab CSV 없음, .dic 빌드 건너뜀"
fi

# ──────────────────────────────────────────────
# 3. LanceDB → Docker named volume 복사
# ──────────────────────────────────────────────
echo ""
if [ "$SKIP_LANCEDB" = false ]; then
    echo "=== 3. LanceDB → Docker volume 복사 ==="
    if [ -d "backend/lancedb_data" ] && [ -n "$(ls -A backend/lancedb_data 2>/dev/null)" ]; then
        echo "  docker cp → $BACKEND_CONTAINER:/app/lancedb_data/ ..."
        docker cp backend/lancedb_data/. "$BACKEND_CONTAINER":/app/lancedb_data/
        echo "  완료"
    else
        echo "  WARNING: backend/lancedb_data/ 가 비어있음 (S3 다운로드 확인 필요)"
    fi
else
    echo "=== 3. LanceDB 볼륨 복사 (건너뜀) ==="
fi

# ──────────────────────────────────────────────
# 4. 불완전한 모델 캐시 정리
#    entrypoint의 download_models.py가 재다운로드하도록 유도
# ──────────────────────────────────────────────
echo ""
if [ "$CLEAN_MODELS" = true ]; then
    echo "=== 4. 불완전한 모델 캐시 정리 ==="
    # 호스트 bind mount 경로에서 직접 정리
    MODEL_DIR="backend/data/models"
    cleaned=0
    if [ -d "$MODEL_DIR" ]; then
        for model_dir in "$MODEL_DIR"/models--*; do
            [ -d "$model_dir" ] || continue
            # 100MB 미만이면 불완전한 캐시로 판단
            dir_size=$(du -sb "$model_dir" 2>/dev/null | cut -f1)
            if [ "${dir_size:-0}" -lt 104857600 ]; then
                echo "  삭제 (불완전, ${dir_size}B): $(basename "$model_dir")"
                rm -rf "$model_dir"
                cleaned=$((cleaned + 1))
            fi
        done
    fi
    if [ "$cleaned" -eq 0 ]; then
        echo "  불완전한 캐시 없음 (정상)"
    else
        echo "  ${cleaned}개 불완전 캐시 삭제 완료"
        echo "  → 재시작 시 entrypoint에서 자동 재다운로드됩니다"
    fi
else
    echo "=== 4. 모델 캐시 정리 (건너뜀) ==="
fi

# ──────────────────────────────────────────────
# 5. 서비스 재시작 + 헬스체크
# ──────────────────────────────────────────────
echo ""
echo "=== 5. 서비스 재시작 + 헬스체크 ==="
$DC restart backend
echo "  healthcheck 대기 (최대 ${WAIT_TIMEOUT}s)..."
$DC up -d --wait --wait-timeout "$WAIT_TIMEOUT"

# 직접 헬스 체크
if $DC exec -T backend curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo "  Health check PASSED"
else
    echo "  WARNING: Health check failed. 로그 확인:"
    $DC logs --tail=20 backend
fi

# ──────────────────────────────────────────────
# 6. PostgreSQL 데이터 적재
#    S3에서 받은 data/ 파일 → PostgreSQL 테이블
#    alembic upgrade head는 entrypoint에서 자동 실행됨
# ──────────────────────────────────────────────
echo ""
if [ "$SKIP_DB_LOAD" = false ]; then
    echo "=== 6. PostgreSQL 데이터 적재 ==="

    db_load_ok=0
    db_load_fail=0

    # 헬퍼: 컨테이너 내부에서 Python 스크립트 실행
    run_in_backend() {
        local desc="$1"
        shift
        echo "  [$desc] 시작..."
        if docker exec "$BACKEND_CONTAINER" "$@" 2>&1 | tail -5; then
            echo "  [$desc] 완료"
            db_load_ok=$((db_load_ok + 1))
        else
            echo "  WARNING: [$desc] 실패 (계속 진행)"
            db_load_fail=$((db_load_fail + 1))
        fi
        echo ""
    }

    # 6-1. 변호사 데이터 (lawyer_finder + lawyer_stats)
    #      소스: data/lawyers.json (S3 다운로드 포함)
    run_in_backend "변호사 데이터 (17,326건)" \
        uv run python scripts/load_lawyers_data.py

    # 6-2. 재판 통계 (lawyer_stats 수요 분석)
    #      소스: data/trial_statistics_data/*.csv (S3 다운로드 포함)
    run_in_backend "재판 통계" \
        uv run python scripts/load_trial_statistics_data.py

    # 6-3. 법령 체계도 그래프 (statute_hierarchy + 관련 테이블)
    #      소스: data/law_hierarchy.json, data/law_abbreviations.json (S3 다운로드 포함)
    run_in_backend "법령 체계도 그래프" \
        uv run python -m scripts.ingest.cli --step graph

    # 6-4. 법률 용어 사전 (USE_LEGAL_TERM_DICT=true인 경우만 필요)
    #      소스: data/lawterms_v1.json (S3 다운로드 포함)
    if docker exec "$BACKEND_CONTAINER" \
        python -c "from app.core.config import settings; exit(0 if settings.USE_LEGAL_TERM_DICT else 1)" 2>/dev/null; then
        run_in_backend "법률 용어 사전 (72,700건)" \
            uv run python scripts/load_legal_terms_data.py
    else
        echo "  [법률 용어 사전] USE_LEGAL_TERM_DICT=false → 건너뜀"
    fi

    echo "  적재 결과: 성공 ${db_load_ok}건, 실패 ${db_load_fail}건"

    if [ "$db_load_fail" -gt 0 ]; then
        echo "  WARNING: 일부 적재 실패. 로그를 확인하세요."
        echo "    $DC logs --tail=50 backend"
    fi

    echo ""
    echo "  참고: 법령/판례 전문 DB 적재 (BM25 키워드 검색용)는 별도 실행 필요:"
    echo "    docker exec $BACKEND_CONTAINER uv run python -m scripts.ingest.cli --type all --step db"
    echo "    docker exec $BACKEND_CONTAINER uv run python scripts/create_bm25_index.py"
    echo "    (ingest_source/ 디렉토리가 컨테이너 내 /app/data/에 있어야 합니다)"
else
    echo "=== 6. PostgreSQL 데이터 적재 (건너뜀, --skip-db-load로 비활성화) ==="
fi

# ──────────────────────────────────────────────
# 7. SSL 인증서 발급 (선택)
# ──────────────────────────────────────────────
echo ""
if [ "$DO_SSL" = true ]; then
    echo "=== 7. SSL 인증서 발급 ==="
    bash scripts/deploy/init-letsencrypt.sh
else
    echo "=== 7. SSL 인증서 (건너뜀, --ssl로 활성화) ==="
fi

# ──────────────────────────────────────────────
# 완료 요약
# ──────────────────────────────────────────────
echo ""
echo "============================================"
echo "  포스트 배포 완료: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
echo ""
echo "상태 확인:"
echo "  $DC ps"
echo "  $DC logs -f backend"
echo ""
echo "검색 API 테스트:"
echo "  curl -s http://localhost:8000/health | python3 -m json.tool"
echo ""
