#!/usr/bin/env bash
#
# Google Drive → DB 복원 스크립트
#
# 사용법:
#   ./scripts/restore_from_gdrive.sh latest              # 최신 백업 복원
#   ./scripts/restore_from_gdrive.sh 20260211_153000      # 특정 백업 복원
#   ./scripts/restore_from_gdrive.sh latest --download-only  # 다운로드만
#   ./scripts/restore_from_gdrive.sh latest --skip-neo4j     # Neo4j 제외
#
# 필수 조건:
#   - rclone 설치 (brew install rclone)
#   - secrets/gdrive-service-account.json 배치
#   - rclone.conf 배치 (프로젝트 루트)
#   - Docker 컨테이너 실행 중

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

# 환경변수 (기본값)
POSTGRES_CONTAINER="${POSTGRES_CONTAINER:-law-platform-db}"
POSTGRES_USER="${POSTGRES_USER:-lawuser}"
POSTGRES_DB="${POSTGRES_DB:-lawdb}"
NEO4J_CONTAINER="${NEO4J_CONTAINER:-neo4j-law-graph}"
LANCEDB_DATA_DIR="${LANCEDB_DATA_DIR:-${PROJECT_ROOT}/backend/lancedb_data}"
BACKUP_BASE_DIR="${BACKUP_BASE_DIR:-${PROJECT_ROOT}/backups}"
RCLONE_CONF="${RCLONE_CONF:-${PROJECT_ROOT}/rclone.conf}"
RCLONE_REMOTE="${RCLONE_REMOTE:-gdrive}"

# 옵션 플래그
SKIP_POSTGRES=false
SKIP_NEO4J=false
SKIP_LANCEDB=false
DOWNLOAD_ONLY=false
TARGET=""

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

format_duration() {
    local seconds=$1
    if (( seconds < 60 )); then
        echo "${seconds}s"
    else
        echo "$((seconds / 60))m $((seconds % 60))s"
    fi
}

# ─────────────────────────────────────────────────
# 인자 파싱
# ─────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-postgres)  SKIP_POSTGRES=true;  shift ;;
        --skip-neo4j)     SKIP_NEO4J=true;    shift ;;
        --skip-lancedb)   SKIP_LANCEDB=true;   shift ;;
        --download-only)  DOWNLOAD_ONLY=true;  shift ;;
        -h|--help)
            echo "Usage: $0 <TARGET> [OPTIONS]"
            echo ""
            echo "TARGET:"
            echo "  latest             최신 백업 복원"
            echo "  <TIMESTAMP>        특정 타임스탬프 복원 (예: 20260211_153000)"
            echo ""
            echo "Options:"
            echo "  --skip-postgres    PostgreSQL 복원 건너뛰기"
            echo "  --skip-neo4j       Neo4j 복원 건너뛰기"
            echo "  --skip-lancedb     LanceDB 복원 건너뛰기"
            echo "  --download-only    다운로드만 (복원 안 함)"
            echo "  -h, --help         도움말"
            exit 0
            ;;
        -*)
            log_error "알 수 없는 옵션: $1"
            exit 1
            ;;
        *)
            if [[ -z "${TARGET}" ]]; then
                TARGET="$1"
            else
                log_error "인자가 너무 많습니다: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

if [[ -z "${TARGET}" ]]; then
    log_error "복원 대상을 지정하세요: latest 또는 타임스탬프"
    echo "  사용법: $0 latest"
    echo "  사용법: $0 20260211_153000"
    exit 1
fi

# ─────────────────────────────────────────────────
# 사전 검증
# ─────────────────────────────────────────────────

log_info "사전 검증 시작..."
errors=0

if ! command -v rclone &>/dev/null; then
    log_error "rclone이 설치되어 있지 않습니다 (brew install rclone)"
    errors=$((errors + 1))
fi

if [[ ! -f "${RCLONE_CONF}" ]]; then
    log_error "rclone 설정 파일이 없습니다: ${RCLONE_CONF}"
    errors=$((errors + 1))
fi

SA_FILE=$(grep -i 'service_account_file' "${RCLONE_CONF}" 2>/dev/null | head -1 | sed 's/.*=[[:space:]]*//')
HAS_TOKEN=$(grep -c 'token' "${RCLONE_CONF}" 2>/dev/null || echo 0)
if [[ -n "${SA_FILE}" ]]; then
    if [[ ! -f "${PROJECT_ROOT}/${SA_FILE}" ]]; then
        log_error "서비스 계정 키가 없습니다: ${SA_FILE}"
        errors=$((errors + 1))
    else
        log_ok "인증: Service Account (${SA_FILE})"
    fi
elif (( HAS_TOKEN > 0 )); then
    log_ok "인증: OAuth token"
else
    log_error "rclone.conf에 인증 설정이 없습니다 (service_account_file 또는 token)"
    errors=$((errors + 1))
fi

if [[ "${DOWNLOAD_ONLY}" == "false" ]]; then
    if ! command -v docker &>/dev/null; then
        log_error "docker가 설치되어 있지 않습니다"
        errors=$((errors + 1))
    fi
fi

if (( errors > 0 )); then
    log_error "${errors}개 검증 실패"
    exit 1
fi

log_ok "사전 검증 완료"

# ─────────────────────────────────────────────────
# 백업 목록 조회 및 대상 결정
# ─────────────────────────────────────────────────

echo ""
log_info "Google Drive 백업 목록 조회 중..."

BACKUP_LIST=$(rclone --config "${RCLONE_CONF}" lsd "${RCLONE_REMOTE}:" 2>/dev/null | awk '{print $NF}' | sort)

if [[ -z "${BACKUP_LIST}" ]]; then
    log_error "Google Drive에 백업이 없습니다"
    exit 1
fi

echo ""
echo "사용 가능한 백업:"
echo "${BACKUP_LIST}" | while read -r name; do
    echo "  - ${name}"
done
echo ""

if [[ "${TARGET}" == "latest" ]]; then
    TARGET=$(echo "${BACKUP_LIST}" | tail -n 1)
    log_info "최신 백업 선택: ${TARGET}"
else
    if ! echo "${BACKUP_LIST}" | grep -q "^${TARGET}$"; then
        log_error "백업을 찾을 수 없습니다: ${TARGET}"
        log_info "사용 가능한 백업:"
        echo "${BACKUP_LIST}"
        exit 1
    fi
fi

# ─────────────────────────────────────────────────
# 다운로드
# ─────────────────────────────────────────────────

RESTORE_DIR="${BACKUP_BASE_DIR}/${TARGET}"

echo ""
log_info "━━━ 다운로드: ${TARGET} ━━━"
DOWNLOAD_START=$(date +%s)

mkdir -p "${RESTORE_DIR}"

rclone --config "${RCLONE_CONF}" \
    copy "${RCLONE_REMOTE}:${TARGET}" "${RESTORE_DIR}" \
    --progress \
    --transfers 4

DOWNLOAD_END=$(date +%s)
log_ok "다운로드 완료 ($(format_duration $((DOWNLOAD_END - DOWNLOAD_START))))"

echo ""
echo "다운로드된 파일:"
ls -lh "${RESTORE_DIR}/"
echo ""

if [[ "${DOWNLOAD_ONLY}" == "true" ]]; then
    log_info "다운로드 전용 모드. 복원은 수행하지 않습니다."
    log_info "다운로드 경로: ${RESTORE_DIR}"
    exit 0
fi

# ─────────────────────────────────────────────────
# 복원 확인
# ─────────────────────────────────────────────────

echo ""
echo -e "${YELLOW}주의: 복원하면 현재 데이터가 덮어씌워집니다!${NC}"
echo ""
read -rp "계속하시겠습니까? [y/N] " confirm
if [[ "${confirm}" != "y" && "${confirm}" != "Y" ]]; then
    log_info "복원을 취소합니다."
    exit 0
fi

TOTAL_START=$(date +%s)

# ─────────────────────────────────────────────────
# 1. PostgreSQL 복원
# ─────────────────────────────────────────────────

if [[ "${SKIP_POSTGRES}" == "false" && -f "${RESTORE_DIR}/postgres.dump" ]]; then
    echo ""
    log_info "━━━ [1/3] PostgreSQL 복원 ━━━"
    PG_START=$(date +%s)

    # 컨테이너 실행 확인
    if ! docker ps --format '{{.Names}}' | grep -q "^${POSTGRES_CONTAINER}$"; then
        log_error "PostgreSQL 컨테이너가 실행 중이 아닙니다"
        log_info "  실행: docker compose up -d postgres"
        exit 1
    fi

    # pg_restore 실행
    docker exec -i "${POSTGRES_CONTAINER}" \
        pg_restore \
        --clean \
        --if-exists \
        --username="${POSTGRES_USER}" \
        --dbname="${POSTGRES_DB}" \
        --no-owner \
        --no-privileges \
        < "${RESTORE_DIR}/postgres.dump" \
        2>&1 | grep -v "does not exist, skipping" || true

    PG_END=$(date +%s)
    log_ok "PostgreSQL 복원 완료 ($(format_duration $((PG_END - PG_START))))"
elif [[ "${SKIP_POSTGRES}" == "false" ]]; then
    log_warn "postgres.dump 파일 없음, PostgreSQL 복원 건너뛰기"
fi

# ─────────────────────────────────────────────────
# 2. Neo4j 복원
# ─────────────────────────────────────────────────

if [[ "${SKIP_NEO4J}" == "false" ]]; then
    NEO_DUMP=""
    if [[ -f "${RESTORE_DIR}/neo4j.dump" ]]; then
        NEO_DUMP="neo4j.dump"
    elif [[ -f "${RESTORE_DIR}/neo4j_data.tar.gz" ]]; then
        NEO_DUMP="neo4j_data.tar.gz"
    fi

    if [[ -n "${NEO_DUMP}" ]]; then
        echo ""
        log_info "━━━ [2/3] Neo4j 복원 ━━━"
        NEO_START=$(date +%s)

        # 컨테이너 실행 확인
        if ! docker ps --format '{{.Names}}' | grep -q "^${NEO4J_CONTAINER}$"; then
            log_error "Neo4j 컨테이너가 실행 중이 아닙니다"
            log_info "  실행: docker compose up -d neo4j"
            exit 1
        fi

        # Neo4j 정지
        log_info "Neo4j 컨테이너 정지 중..."
        docker stop "${NEO4J_CONTAINER}" >/dev/null

        if [[ "${NEO_DUMP}" == "neo4j.dump" ]]; then
            # neo4j-admin database load
            log_info "Neo4j 데이터베이스 로드 중..."
            docker run --rm \
                --volumes-from "${NEO4J_CONTAINER}" \
                -v "${RESTORE_DIR}:/backup" \
                neo4j:5.15.0 \
                neo4j-admin database load neo4j --from-path=/backup --overwrite-destination=true \
                2>/dev/null || {
                    log_warn "neo4j-admin load 실패"
                }
        else
            # tar.gz fallback
            log_info "Neo4j data 디렉토리 복원 중..."
            docker run --rm \
                --volumes-from "${NEO4J_CONTAINER}" \
                -v "${RESTORE_DIR}:/backup" \
                alpine \
                sh -c "rm -rf /data/* && tar xzf /backup/neo4j_data.tar.gz -C /data"
        fi

        # Neo4j 재시작
        log_info "Neo4j 컨테이너 재시작 중..."
        docker start "${NEO4J_CONTAINER}" >/dev/null

        # 재시작 대기
        for i in $(seq 1 15); do
            if docker exec "${NEO4J_CONTAINER}" wget --no-verbose --tries=1 --spider localhost:7474 2>/dev/null; then
                break
            fi
            sleep 2
        done

        NEO_END=$(date +%s)
        log_ok "Neo4j 복원 완료 ($(format_duration $((NEO_END - NEO_START))))"
    else
        log_warn "Neo4j 덤프 파일 없음, 건너뛰기"
    fi
elif [[ "${SKIP_NEO4J}" == "true" ]]; then
    log_warn "Neo4j 건너뛰기"
fi

# ─────────────────────────────────────────────────
# 3. LanceDB 복원
# ─────────────────────────────────────────────────

if [[ "${SKIP_LANCEDB}" == "false" && -f "${RESTORE_DIR}/lancedb_data.tar.gz" ]]; then
    echo ""
    log_info "━━━ [3/3] LanceDB 복원 ━━━"
    LANCE_START=$(date +%s)

    # 기존 데이터 백업 (안전 장치)
    if [[ -d "${LANCEDB_DATA_DIR}" ]]; then
        log_info "기존 LanceDB 데이터를 임시 백업..."
        mv "${LANCEDB_DATA_DIR}" "${LANCEDB_DATA_DIR}.bak.$$"
    fi

    # 새 디렉토리 생성 후 압축 해제
    mkdir -p "${LANCEDB_DATA_DIR}"
    tar xzf "${RESTORE_DIR}/lancedb_data.tar.gz" -C "${LANCEDB_DATA_DIR}"

    # 임시 백업 삭제
    if [[ -d "${LANCEDB_DATA_DIR}.bak.$$" ]]; then
        rm -rf "${LANCEDB_DATA_DIR}.bak.$$"
    fi

    LANCE_END=$(date +%s)
    log_ok "LanceDB 복원 완료 ($(format_duration $((LANCE_END - LANCE_START))))"

    echo ""
    log_warn "━━━ 인덱스 재생성 필요 ━━━"
    echo ""
    echo "LanceDB 벡터 인덱스와 FTS 인덱스는 백업에 포함되지 않습니다."
    echo "검색 성능을 위해 아래 명령으로 인덱스를 재생성하세요:"
    echo ""
    echo "  # 벡터 인덱스 재생성 (LANCEDB_INDEX_TYPE=IVF_FLAT 설정 시)"
    echo "  cd backend && uv run python -c \\"
    echo "    \"from app.tools.vectorstore import get_lancedb_store; s = get_lancedb_store(); s.create_index()\""
    echo ""
    echo "  # FTS content_tokenized 인덱스 재생성"
    echo "  cd backend && uv run --no-sync python scripts/update_content_tokenized.py --userdic"
    echo ""
elif [[ "${SKIP_LANCEDB}" == "false" ]]; then
    log_warn "lancedb_data.tar.gz 파일 없음, LanceDB 복원 건너뛰기"
fi

# ─────────────────────────────────────────────────
# 결과 요약
# ─────────────────────────────────────────────────

TOTAL_END=$(date +%s)

echo ""
echo "=========================================="
echo "  복원 완료"
echo "=========================================="
echo ""
echo "복원 대상:   ${TARGET}"
echo "소스 경로:   ${RESTORE_DIR}/"
echo ""
echo "총 소요 시간: $(format_duration $((TOTAL_END - TOTAL_START)))"
echo "=========================================="
