#!/usr/bin/env bash
#
# DB 백업 → Google Drive 업로드 스크립트
#
# 사용법:
#   ./scripts/backup_to_gdrive.sh                # 전체 백업 + 업로드
#   ./scripts/backup_to_gdrive.sh --skip-upload   # 로컬 덤프만
#   ./scripts/backup_to_gdrive.sh --skip-neo4j    # Neo4j 제외
#   ./scripts/backup_to_gdrive.sh --dry-run       # 미리보기
#
# 필수 조건:
#   - rclone 설치 (brew install rclone)
#   - rclone.conf 배치 (프로젝트 루트, Service Account 또는 OAuth token)
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
BACKUP_KEEP_LOCAL="${BACKUP_KEEP_LOCAL:-5}"
RCLONE_CONF="${RCLONE_CONF:-${PROJECT_ROOT}/rclone.conf}"
RCLONE_REMOTE="${RCLONE_REMOTE:-gdrive}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="${BACKUP_BASE_DIR}/${TIMESTAMP}"

# 옵션 플래그
SKIP_POSTGRES=false
SKIP_NEO4J=false
SKIP_LANCEDB=false
SKIP_UPLOAD=false
DRY_RUN=false

# ─────────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# Docker 명령어 자동 감지 (WSL2 호환)
detect_docker_cmd() {
    if command -v docker &>/dev/null; then
        echo "docker"
    elif command -v docker.exe &>/dev/null; then
        echo "docker.exe"
    else
        echo ""
    fi
}

DOCKER_CMD=$(detect_docker_cmd)

# 소요 시간 포맷
format_duration() {
    local seconds=$1
    if (( seconds < 60 )); then
        echo "${seconds}s"
    else
        echo "$((seconds / 60))m $((seconds % 60))s"
    fi
}

# 파일 크기 포맷 (bytes → human readable)
format_size() {
    local bytes=$1
    if (( bytes >= 1073741824 )); then
        echo "$(echo "scale=1; ${bytes}/1073741824" | bc)GB"
    elif (( bytes >= 1048576 )); then
        echo "$(echo "scale=1; ${bytes}/1048576" | bc)MB"
    elif (( bytes >= 1024 )); then
        echo "$(echo "scale=1; ${bytes}/1024" | bc)KB"
    else
        echo "${bytes}B"
    fi
}

# ─────────────────────────────────────────────────
# 인자 파싱
# ─────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-postgres) SKIP_POSTGRES=true; shift ;;
        --skip-neo4j)    SKIP_NEO4J=true;   shift ;;
        --skip-lancedb)  SKIP_LANCEDB=true;  shift ;;
        --skip-upload)   SKIP_UPLOAD=true;   shift ;;
        --dry-run)       DRY_RUN=true;       shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-postgres  PostgreSQL 건너뛰기"
            echo "  --skip-neo4j     Neo4j 건너뛰기"
            echo "  --skip-lancedb   LanceDB 건너뛰기"
            echo "  --skip-upload    로컬 덤프만 (Google Drive 업로드 안 함)"
            echo "  --dry-run        실제 실행 없이 계획만 출력"
            echo "  -h, --help       도움말"
            exit 0
            ;;
        *)
            log_error "알 수 없는 옵션: $1"
            exit 1
            ;;
    esac
done

# ─────────────────────────────────────────────────
# 사전 검증
# ─────────────────────────────────────────────────

preflight_check() {
    log_info "사전 검증 시작..."
    local errors=0

    # Docker 확인
    if [[ -z "${DOCKER_CMD}" ]]; then
        log_error "docker가 설치되어 있지 않습니다 (docker 또는 docker.exe)"
        errors=$((errors + 1))
    fi

    # PostgreSQL 컨테이너 확인
    if [[ "${SKIP_POSTGRES}" == "false" ]]; then
        if ! ${DOCKER_CMD} ps --format '{{.Names}}' | grep -q "^${POSTGRES_CONTAINER}$"; then
            log_error "PostgreSQL 컨테이너 '${POSTGRES_CONTAINER}'가 실행 중이 아닙니다"
            log_info "  실행: docker compose up -d postgres"
            errors=$((errors + 1))
        else
            log_ok "PostgreSQL 컨테이너 실행 중"
        fi
    fi

    # Neo4j 컨테이너 확인
    if [[ "${SKIP_NEO4J}" == "false" ]]; then
        if ! ${DOCKER_CMD} ps --format '{{.Names}}' | grep -q "^${NEO4J_CONTAINER}$"; then
            log_error "Neo4j 컨테이너 '${NEO4J_CONTAINER}'가 실행 중이 아닙니다"
            log_info "  실행: docker compose up -d neo4j"
            errors=$((errors + 1))
        else
            log_ok "Neo4j 컨테이너 실행 중"
        fi
    fi

    # LanceDB 데이터 디렉토리 확인
    if [[ "${SKIP_LANCEDB}" == "false" ]]; then
        if [[ ! -d "${LANCEDB_DATA_DIR}" ]]; then
            log_error "LanceDB 데이터 디렉토리가 없습니다: ${LANCEDB_DATA_DIR}"
            errors=$((errors + 1))
        else
            log_ok "LanceDB 데이터 디렉토리 존재"
        fi
    fi

    # rclone 확인 (업로드 시)
    if [[ "${SKIP_UPLOAD}" == "false" ]]; then
        if ! command -v rclone &>/dev/null; then
            log_error "rclone이 설치되어 있지 않습니다"
            log_info "  설치: brew install rclone"
            errors=$((errors + 1))
        else
            log_ok "rclone 설치됨"
        fi

        if [[ ! -f "${RCLONE_CONF}" ]]; then
            log_error "rclone 설정 파일이 없습니다: ${RCLONE_CONF}"
            log_info "  팀에서 rclone.conf 파일을 받아 프로젝트 루트에 배치하세요"
            errors=$((errors + 1))
        else
            log_ok "rclone 설정 파일 존재"
        fi

        # rclone.conf 인증 방식 확인 (Service Account 또는 OAuth token)
        SA_FILE=$(grep -i 'service_account_file' "${RCLONE_CONF}" 2>/dev/null | head -1 | sed 's/.*=[[:space:]]*//' || true)
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
    fi

    if (( errors > 0 )); then
        log_error "${errors}개 검증 실패. 위 문제를 해결한 후 다시 실행하세요."
        exit 1
    fi

    log_ok "사전 검증 완료"
}

# ─────────────────────────────────────────────────
# Dry Run
# ─────────────────────────────────────────────────

if [[ "${DRY_RUN}" == "true" ]]; then
    echo ""
    echo "=========================================="
    echo "  백업 계획 (Dry Run)"
    echo "=========================================="
    echo ""
    echo "타임스탬프:  ${TIMESTAMP}"
    echo "백업 경로:   ${BACKUP_DIR}/"
    echo ""
    echo "대상 DB:"
    [[ "${SKIP_POSTGRES}" == "false" ]] && echo "  - PostgreSQL (${POSTGRES_CONTAINER})" || echo "  - PostgreSQL (건너뛰기)"
    [[ "${SKIP_NEO4J}" == "false" ]]    && echo "  - Neo4j (${NEO4J_CONTAINER})"         || echo "  - Neo4j (건너뛰기)"
    [[ "${SKIP_LANCEDB}" == "false" ]]  && echo "  - LanceDB (${LANCEDB_DATA_DIR})"      || echo "  - LanceDB (건너뛰기)"
    echo ""
    [[ "${SKIP_UPLOAD}" == "false" ]] && echo "업로드:      ${RCLONE_REMOTE}:${TIMESTAMP}/" || echo "업로드:      건너뛰기"
    echo ""
    echo "로컬 보관:   최근 ${BACKUP_KEEP_LOCAL}개"
    echo ""
    echo "=========================================="
    echo "  실제 실행하려면 --dry-run 없이 실행하세요"
    echo "=========================================="
    exit 0
fi

# ─────────────────────────────────────────────────
# 사전 검증 실행
# ─────────────────────────────────────────────────

preflight_check

# ─────────────────────────────────────────────────
# 백업 디렉토리 생성
# ─────────────────────────────────────────────────

mkdir -p "${BACKUP_DIR}"
log_info "백업 디렉토리: ${BACKUP_DIR}"

TOTAL_START=$(date +%s)

# ─────────────────────────────────────────────────
# 1. PostgreSQL 백업
# ─────────────────────────────────────────────────

if [[ "${SKIP_POSTGRES}" == "false" ]]; then
    echo ""
    log_info "━━━ [1/3] PostgreSQL 백업 ━━━"
    PG_START=$(date +%s)

    PG_DUMP_FILE="${BACKUP_DIR}/postgres.dump"

    ${DOCKER_CMD} exec "${POSTGRES_CONTAINER}" \
        pg_dump \
        --format=custom \
        --compress=6 \
        --username="${POSTGRES_USER}" \
        --dbname="${POSTGRES_DB}" \
        > "${PG_DUMP_FILE}"

    PG_END=$(date +%s)
    PG_SIZE=$(stat -f%z "${PG_DUMP_FILE}" 2>/dev/null || stat --printf="%s" "${PG_DUMP_FILE}" 2>/dev/null || echo 0)
    log_ok "PostgreSQL 백업 완료: $(format_size "${PG_SIZE}") ($(format_duration $((PG_END - PG_START))))"
else
    log_warn "PostgreSQL 건너뛰기"
fi

# ─────────────────────────────────────────────────
# 2. Neo4j 백업
# ─────────────────────────────────────────────────

if [[ "${SKIP_NEO4J}" == "false" ]]; then
    echo ""
    log_info "━━━ [2/3] Neo4j 백업 ━━━"
    NEO_START=$(date +%s)

    NEO_DUMP_FILE="${BACKUP_DIR}/neo4j.dump"

    # Neo4j 컨테이너 정지 (dump는 오프라인 필요)
    log_info "Neo4j 컨테이너 정지 중..."
    ${DOCKER_CMD} stop "${NEO4J_CONTAINER}" >/dev/null

    # neo4j-admin database dump 실행
    log_info "Neo4j 데이터베이스 덤프 중..."
    ${DOCKER_CMD} run --rm \
        --volumes-from "${NEO4J_CONTAINER}" \
        -v "${BACKUP_DIR}:/backup" \
        neo4j:5.15.0 \
        neo4j-admin database dump neo4j --to-path=/backup --overwrite-destination=true \
        2>/dev/null || {
            # 5.x dump 명령이 다를 수 있으므로 fallback
            log_warn "neo4j-admin dump 실패, data 디렉토리 직접 복사..."
            ${DOCKER_CMD} run --rm \
                --volumes-from "${NEO4J_CONTAINER}" \
                -v "${BACKUP_DIR}:/backup" \
                alpine \
                tar czf /backup/neo4j_data.tar.gz -C /data .
        }

    # Neo4j 컨테이너 재시작
    log_info "Neo4j 컨테이너 재시작 중..."
    ${DOCKER_CMD} start "${NEO4J_CONTAINER}" >/dev/null

    # 재시작 대기
    for i in $(seq 1 15); do
        if ${DOCKER_CMD} exec "${NEO4J_CONTAINER}" wget --no-verbose --tries=1 --spider localhost:7474 2>/dev/null; then
            break
        fi
        sleep 2
    done

    NEO_END=$(date +%s)

    # 덤프 파일 크기 확인 (dump 또는 tar.gz)
    if [[ -f "${BACKUP_DIR}/neo4j.dump" ]]; then
        NEO_SIZE=$(stat -f%z "${BACKUP_DIR}/neo4j.dump" 2>/dev/null || stat --printf="%s" "${BACKUP_DIR}/neo4j.dump" 2>/dev/null || echo 0)
    elif [[ -f "${BACKUP_DIR}/neo4j_data.tar.gz" ]]; then
        NEO_SIZE=$(stat -f%z "${BACKUP_DIR}/neo4j_data.tar.gz" 2>/dev/null || stat --printf="%s" "${BACKUP_DIR}/neo4j_data.tar.gz" 2>/dev/null || echo 0)
    else
        NEO_SIZE=0
    fi

    log_ok "Neo4j 백업 완료: $(format_size "${NEO_SIZE}") ($(format_duration $((NEO_END - NEO_START))))"
    log_ok "Neo4j 컨테이너 재시작 완료"
else
    log_warn "Neo4j 건너뛰기"
fi

# ─────────────────────────────────────────────────
# 3. LanceDB 백업
# ─────────────────────────────────────────────────

if [[ "${SKIP_LANCEDB}" == "false" ]]; then
    echo ""
    log_info "━━━ [3/3] LanceDB 백업 ━━━"
    LANCE_START=$(date +%s)

    LANCE_ARCHIVE="${BACKUP_DIR}/lancedb_data.tar.gz"

    # data/ 폴더만 압축 (인덱스/_transactions/_versions 제외 → 용량 ~1.3GB)
    # 인덱스는 복원 후 재생성
    log_info "LanceDB data 디렉토리 압축 중 (data/ only)..."
    tar czf "${LANCE_ARCHIVE}" \
        -C "${LANCEDB_DATA_DIR}" \
        --exclude='.DS_Store' \
        --exclude='_indices' \
        --exclude='_transactions' \
        --exclude='_versions' \
        .

    LANCE_END=$(date +%s)
    LANCE_SIZE=$(stat -f%z "${LANCE_ARCHIVE}" 2>/dev/null || stat --printf="%s" "${LANCE_ARCHIVE}" 2>/dev/null || echo 0)
    log_ok "LanceDB 백업 완료: $(format_size "${LANCE_SIZE}") ($(format_duration $((LANCE_END - LANCE_START))))"
else
    log_warn "LanceDB 건너뛰기"
fi

# ─────────────────────────────────────────────────
# 4. Google Drive 업로드
# ─────────────────────────────────────────────────

if [[ "${SKIP_UPLOAD}" == "false" ]]; then
    echo ""
    log_info "━━━ Google Drive 업로드 ━━━"
    UPLOAD_START=$(date +%s)

    rclone --config "${RCLONE_CONF}" \
        copy "${BACKUP_DIR}" "${RCLONE_REMOTE}:${TIMESTAMP}" \
        --progress \
        --transfers 4

    UPLOAD_END=$(date +%s)
    log_ok "Google Drive 업로드 완료 ($(format_duration $((UPLOAD_END - UPLOAD_START))))"
else
    log_warn "Google Drive 업로드 건너뛰기"
fi

# ─────────────────────────────────────────────────
# 5. 오래된 로컬 백업 정리
# ─────────────────────────────────────────────────

if [[ -d "${BACKUP_BASE_DIR}" ]]; then
    BACKUP_COUNT=$(find "${BACKUP_BASE_DIR}" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')

    if (( BACKUP_COUNT > BACKUP_KEEP_LOCAL )); then
        echo ""
        log_info "━━━ 오래된 백업 정리 ━━━"
        DELETE_COUNT=$((BACKUP_COUNT - BACKUP_KEEP_LOCAL))

        # 가장 오래된 것부터 삭제
        find "${BACKUP_BASE_DIR}" -mindepth 1 -maxdepth 1 -type d | sort | head -n "${DELETE_COUNT}" | while read -r old_dir; do
            log_info "삭제: $(basename "${old_dir}")"
            rm -rf "${old_dir}"
        done

        log_ok "${DELETE_COUNT}개 오래된 백업 삭제 (최근 ${BACKUP_KEEP_LOCAL}개 유지)"
    fi
fi

# ─────────────────────────────────────────────────
# 결과 요약
# ─────────────────────────────────────────────────

TOTAL_END=$(date +%s)

echo ""
echo "=========================================="
echo "  백업 완료"
echo "=========================================="
echo ""
echo "타임스탬프:  ${TIMESTAMP}"
echo "로컬 경로:   ${BACKUP_DIR}/"
[[ "${SKIP_UPLOAD}" == "false" ]] && echo "Drive 경로:  ${RCLONE_REMOTE}:${TIMESTAMP}/"
echo ""
echo "백업 파일:"
if [[ -d "${BACKUP_DIR}" ]]; then
    ls -lh "${BACKUP_DIR}/" | tail -n +2
fi
echo ""
echo "총 소요 시간: $(format_duration $((TOTAL_END - TOTAL_START)))"
echo "=========================================="
