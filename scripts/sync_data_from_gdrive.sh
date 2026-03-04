#!/usr/bin/env bash
#
# Google Drive data/ → 로컬 올바른 경로로 동기화
#
# 문제: gdrive:data/ 안에 lancedb_data, mecab_userdic, models가 섞여 있어서
#       단순 `rclone copy gdrive:data/ data/`하면 잘못된 경로에 들어감.
#
# 경로 매핑:
#   gdrive:data/lancedb_data/   → backend/lancedb_data/
#   gdrive:data/mecab_userdic/  → backend/data/mecab_userdic/
#   gdrive:data/models/         → backend/data/models/
#   gdrive:data/*.json 등       → data/
#
# 사용법:
#   ./scripts/sync_data_from_gdrive.sh              # 전체 다운로드
#   ./scripts/sync_data_from_gdrive.sh --dry-run    # 미리보기
#   ./scripts/sync_data_from_gdrive.sh --only-json  # JSON 파일만
#   ./scripts/sync_data_from_gdrive.sh --only-backend  # backend 데이터만
#
# 필수 조건:
#   - rclone 설치 (brew install rclone)
#   - rclone.conf 배치 (프로젝트 루트)

set -euo pipefail

# ─────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# .env 로드 (존재하면)
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${PROJECT_ROOT}/.env"
    set +a
fi

RCLONE_CONF="${RCLONE_CONF:-${PROJECT_ROOT}/rclone.conf}"
RCLONE_REMOTE="${RCLONE_REMOTE:-gdrive}"

# backend 데이터 경로 매핑 (gdrive:data/ 하위 → 로컬 경로)
# 형식: "gdrive_subdir:local_path"
BACKEND_DIRS=(
    "lancedb_data:${PROJECT_ROOT}/backend/lancedb_data"
    "mecab_userdic:${PROJECT_ROOT}/backend/data/mecab_userdic"
    "models:${PROJECT_ROOT}/backend/data/models"
)

# 위 디렉토리들을 제외한 나머지는 data/로 다운로드
DATA_DIR="${PROJECT_ROOT}/data"

# 옵션
DRY_RUN=false
ONLY_JSON=false
ONLY_BACKEND=false

# ─────────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ─────────────────────────────────────────────────
# 인자 파싱
# ─────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)       DRY_RUN=true;       shift ;;
        --only-json)     ONLY_JSON=true;      shift ;;
        --only-backend)  ONLY_BACKEND=true;   shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run        실제 다운로드 없이 미리보기"
            echo "  --only-json      JSON/원본 데이터만 (data/)"
            echo "  --only-backend   backend 데이터만 (lancedb, mecab, models)"
            echo "  -h, --help       도움말"
            echo ""
            echo "경로 매핑:"
            echo "  gdrive:data/lancedb_data/   → backend/lancedb_data/"
            echo "  gdrive:data/mecab_userdic/  → backend/data/mecab_userdic/"
            echo "  gdrive:data/models/         → backend/data/models/"
            echo "  gdrive:data/*.json 등       → data/"
            exit 0
            ;;
        *)
            log_error "알 수 없는 옵션: $1"
            exit 1
            ;;
    esac
done

if [[ "${ONLY_JSON}" == "true" && "${ONLY_BACKEND}" == "true" ]]; then
    log_error "--only-json과 --only-backend는 동시 사용 불가"
    exit 1
fi

# ─────────────────────────────────────────────────
# 사전 검증
# ─────────────────────────────────────────────────

log_info "사전 검증..."
errors=0

if ! command -v rclone &>/dev/null; then
    log_error "rclone이 설치되어 있지 않습니다 (brew install rclone)"
    errors=$((errors + 1))
fi

if [[ ! -f "${RCLONE_CONF}" ]]; then
    log_error "rclone 설정 파일이 없습니다: ${RCLONE_CONF}"
    errors=$((errors + 1))
fi

if (( errors > 0 )); then
    log_error "${errors}개 검증 실패"
    exit 1
fi

log_ok "사전 검증 완료"

# ─────────────────────────────────────────────────
# rclone 공통 옵션
# ─────────────────────────────────────────────────

RCLONE_OPTS=(--config "${RCLONE_CONF}" --progress --transfers 4)

if [[ "${DRY_RUN}" == "true" ]]; then
    RCLONE_OPTS+=(--dry-run)
    log_warn "Dry Run 모드 (실제 다운로드 없음)"
fi

TOTAL_START=$(date +%s)

# ─────────────────────────────────────────────────
# 1. Backend 데이터 (lancedb_data, mecab_userdic, models)
# ─────────────────────────────────────────────────

if [[ "${ONLY_JSON}" == "false" ]]; then
    echo ""
    log_info "━━━ [1] Backend 데이터 다운로드 ━━━"

    for mapping in "${BACKEND_DIRS[@]}"; do
        gdrive_subdir="${mapping%%:*}"
        local_path="${mapping#*:}"

        echo ""
        log_info "gdrive:data/${gdrive_subdir}/ → ${local_path#"${PROJECT_ROOT}"/}/"

        mkdir -p "${local_path}"

        rclone copy \
            "${RCLONE_REMOTE}:data/${gdrive_subdir}/" \
            "${local_path}/" \
            "${RCLONE_OPTS[@]}"

        log_ok "${gdrive_subdir} 완료"
    done
fi

# ─────────────────────────────────────────────────
# 2. JSON/원본 데이터 (backend 디렉토리 제외)
# ─────────────────────────────────────────────────

if [[ "${ONLY_BACKEND}" == "false" ]]; then
    echo ""
    log_info "━━━ [2] JSON/원본 데이터 다운로드 ━━━"
    log_info "gdrive:data/ → data/ (backend 디렉토리 제외)"

    mkdir -p "${DATA_DIR}"

    # backend 디렉토리들을 exclude
    EXCLUDE_OPTS=()
    for mapping in "${BACKEND_DIRS[@]}"; do
        gdrive_subdir="${mapping%%:*}"
        EXCLUDE_OPTS+=(--exclude "${gdrive_subdir}/**")
    done

    rclone copy \
        "${RCLONE_REMOTE}:data/" \
        "${DATA_DIR}/" \
        "${RCLONE_OPTS[@]}" \
        "${EXCLUDE_OPTS[@]}"

    log_ok "JSON/원본 데이터 완료"
fi

# ─────────────────────────────────────────────────
# 결과 요약
# ─────────────────────────────────────────────────

TOTAL_END=$(date +%s)
DURATION=$(( TOTAL_END - TOTAL_START ))

echo ""
echo "=========================================="
echo "  데이터 동기화 완료"
echo "=========================================="
echo ""
echo "경로 매핑:"
for mapping in "${BACKEND_DIRS[@]}"; do
    gdrive_subdir="${mapping%%:*}"
    local_path="${mapping#*:}"
    echo "  gdrive:data/${gdrive_subdir}/ → ${local_path#"${PROJECT_ROOT}"/}/"
done
echo "  gdrive:data/*.json 등       → data/"
echo ""
if (( DURATION < 60 )); then
    echo "소요 시간: ${DURATION}s"
else
    echo "소요 시간: $((DURATION / 60))m $((DURATION % 60))s"
fi
echo "=========================================="
