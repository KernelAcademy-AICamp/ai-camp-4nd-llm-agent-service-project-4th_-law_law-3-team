#!/usr/bin/env bash
#
# ONNX 모델 전용 Google Drive 백업/복원 스크립트
#
# 사용법:
#   ./scripts/onnx_model_gdrive.sh backup [--reranker-only|--embedding-only] [--dry-run]
#   ./scripts/onnx_model_gdrive.sh restore [--reranker-only|--embedding-only] [--dry-run]
#   ./scripts/onnx_model_gdrive.sh build-and-backup [--reranker-only] [--overwrite]
#   ./scripts/onnx_model_gdrive.sh list
#
# 필수 조건:
#   - rclone 설치 (brew install rclone)
#   - rclone.conf 배치 (프로젝트 루트, Service Account 또는 OAuth token)
#   - build-and-backup: uv, Python 3.11+

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
RCLONE_CONF="${RCLONE_CONF:-${PROJECT_ROOT}/rclone.conf}"
RCLONE_REMOTE="${RCLONE_REMOTE:-gdrive}"
GDRIVE_ONNX_DIR="${GDRIVE_ONNX_DIR:-onnx-models}"
LOCAL_MODELS_DIR="${PROJECT_ROOT}/backend/data/models"

# ONNX 모델 디렉토리 목록
RERANKER_DIRS=("reranker-ort-opt" "reranker-ort-opt-qdq")
EMBEDDING_DIRS=("kure-v1-ort-opt" "kure-v1-ort-opt-qdq")
ALL_DIRS=("${RERANKER_DIRS[@]}" "${EMBEDDING_DIRS[@]}")

# 최소 모델 파일 크기 (1MB, onnx_session.py 기준)
MIN_MODEL_SIZE=1000000

# 옵션 플래그
SUBCOMMAND=""
RERANKER_ONLY=false
EMBEDDING_ONLY=false
DRY_RUN=false
OVERWRITE=false

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

# 파일 크기 (WSL2 호환: stat -f%z → stat --printf)
get_file_size() {
    stat -f%z "$1" 2>/dev/null || stat --printf="%s" "$1" 2>/dev/null || echo 0
}

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

# 대상 디렉토리 목록 결정
get_target_dirs() {
    if [[ "${RERANKER_ONLY}" == "true" ]]; then
        echo "${RERANKER_DIRS[@]}"
    elif [[ "${EMBEDDING_ONLY}" == "true" ]]; then
        echo "${EMBEDDING_DIRS[@]}"
    else
        echo "${ALL_DIRS[@]}"
    fi
}

# ─────────────────────────────────────────────────
# 인자 파싱
# ─────────────────────────────────────────────────

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <COMMAND> [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  backup            ONNX 모델을 Google Drive에 백업"
    echo "  restore           Google Drive에서 ONNX 모델 복원"
    echo "  build-and-backup  ONNX 모델 빌드 후 백업"
    echo "  list              원격 모델 목록 확인"
    echo ""
    echo "Options:"
    echo "  --reranker-only   리랭커 모델만 대상"
    echo "  --embedding-only  임베딩 모델만 대상"
    echo "  --dry-run         실제 실행 없이 계획만 출력"
    echo "  --overwrite       빌드 시 기존 모델 덮어쓰기"
    echo "  -h, --help        도움말"
    exit 0
fi

SUBCOMMAND="$1"
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --reranker-only)  RERANKER_ONLY=true;  shift ;;
        --embedding-only) EMBEDDING_ONLY=true; shift ;;
        --dry-run)        DRY_RUN=true;        shift ;;
        --overwrite)      OVERWRITE=true;      shift ;;
        -h|--help)
            echo "Usage: $0 <backup|restore|build-and-backup|list> [OPTIONS]"
            exit 0
            ;;
        *)
            log_error "알 수 없는 옵션: $1"
            exit 1
            ;;
    esac
done

if [[ "${RERANKER_ONLY}" == "true" && "${EMBEDDING_ONLY}" == "true" ]]; then
    log_error "--reranker-only와 --embedding-only는 동시에 사용할 수 없습니다"
    exit 1
fi

# ─────────────────────────────────────────────────
# rclone 사전 검증
# ─────────────────────────────────────────────────

check_rclone() {
    local errors=0

    if ! command -v rclone &>/dev/null; then
        log_error "rclone이 설치되어 있지 않습니다 (brew install rclone)"
        errors=$((errors + 1))
    else
        log_ok "rclone 설치됨"
    fi

    if [[ ! -f "${RCLONE_CONF}" ]]; then
        log_error "rclone 설정 파일이 없습니다: ${RCLONE_CONF}"
        errors=$((errors + 1))
    else
        log_ok "rclone 설정 파일 존재"

        # 인증 방식 확인
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
            log_error "rclone.conf에 인증 설정이 없습니다"
            errors=$((errors + 1))
        fi
    fi

    if (( errors > 0 )); then
        log_error "${errors}개 검증 실패"
        exit 1
    fi
}

# ─────────────────────────────────────────────────
# backup: 로컬 → Google Drive
# ─────────────────────────────────────────────────

do_backup() {
    log_info "사전 검증 시작..."
    check_rclone

    local target_dirs
    read -ra target_dirs <<< "$(get_target_dirs)"

    # 검증: model_versions.json 확인
    local versions_file="${LOCAL_MODELS_DIR}/model_versions.json"
    if [[ -f "${versions_file}" ]]; then
        log_ok "model_versions.json 존재"
    else
        log_warn "model_versions.json 없음 (검증 정보 없이 업로드)"
    fi

    # Dry run
    if [[ "${DRY_RUN}" == "true" ]]; then
        echo ""
        echo "=========================================="
        echo "  백업 계획 (Dry Run)"
        echo "=========================================="
        echo ""
        echo "소스:    ${LOCAL_MODELS_DIR}/"
        echo "대상:    ${RCLONE_REMOTE}:${GDRIVE_ONNX_DIR}/"
        echo ""
        echo "대상 모델:"
        for dir in "${target_dirs[@]}"; do
            local local_dir="${LOCAL_MODELS_DIR}/${dir}"
            if [[ -d "${local_dir}" ]]; then
                echo "  - ${dir}/ (존재)"
            else
                echo "  - ${dir}/ (없음 - 건너뜀)"
            fi
        done
        echo ""
        echo "=========================================="
        exit 0
    fi

    log_ok "사전 검증 완료"

    echo ""
    log_info "━━━ ONNX 모델 백업 시작 ━━━"
    local total_start
    total_start=$(date +%s)
    local uploaded=0

    for dir in "${target_dirs[@]}"; do
        local local_dir="${LOCAL_MODELS_DIR}/${dir}"
        if [[ ! -d "${local_dir}" ]]; then
            log_warn "${dir}/ 없음, 건너뛰기"
            continue
        fi

        # verification_passed 확인 (model_versions.json)
        if [[ -f "${versions_file}" ]]; then
            local passed
            passed=$(python3 -c "
import json, sys
try:
    v = json.load(open('${versions_file}'))
    meta = v.get('${dir}', {})
    print('true' if meta.get('verification_passed', True) else 'false')
except: print('true')
" 2>/dev/null || echo "true")
            if [[ "${passed}" == "false" ]]; then
                log_error "${dir}: verification_passed=false, 건너뛰기"
                continue
            fi
        fi

        log_info "${dir}/ 업로드 중..."
        local upload_start
        upload_start=$(date +%s)

        rclone --config "${RCLONE_CONF}" \
            sync "${local_dir}" "${RCLONE_REMOTE}:${GDRIVE_ONNX_DIR}/${dir}" \
            --progress \
            --transfers 4

        local upload_end
        upload_end=$(date +%s)
        log_ok "${dir}/ 업로드 완료 ($(format_duration $((upload_end - upload_start))))"
        uploaded=$((uploaded + 1))
    done

    # model_versions.json도 업로드
    if [[ -f "${versions_file}" ]]; then
        log_info "model_versions.json 업로드..."
        rclone --config "${RCLONE_CONF}" \
            copy "${versions_file}" "${RCLONE_REMOTE}:${GDRIVE_ONNX_DIR}/" \
            --transfers 1
        log_ok "model_versions.json 업로드 완료"
    fi

    local total_end
    total_end=$(date +%s)
    echo ""
    echo "=========================================="
    echo "  백업 완료: ${uploaded}개 모델"
    echo "  Drive 경로: ${RCLONE_REMOTE}:${GDRIVE_ONNX_DIR}/"
    echo "  소요 시간: $(format_duration $((total_end - total_start)))"
    echo "=========================================="
}

# ─────────────────────────────────────────────────
# restore: Google Drive → 로컬
# ─────────────────────────────────────────────────

do_restore() {
    log_info "사전 검증 시작..."
    check_rclone
    log_ok "사전 검증 완료"

    local target_dirs
    read -ra target_dirs <<< "$(get_target_dirs)"

    # Dry run
    if [[ "${DRY_RUN}" == "true" ]]; then
        echo ""
        echo "=========================================="
        echo "  복원 계획 (Dry Run)"
        echo "=========================================="
        echo ""
        echo "소스:    ${RCLONE_REMOTE}:${GDRIVE_ONNX_DIR}/"
        echo "대상:    ${LOCAL_MODELS_DIR}/"
        echo ""
        echo "대상 모델:"
        for dir in "${target_dirs[@]}"; do
            echo "  - ${dir}/"
        done
        echo ""
        echo "=========================================="
        exit 0
    fi

    echo ""
    log_info "━━━ ONNX 모델 복원 시작 ━━━"
    local total_start
    total_start=$(date +%s)
    local restored=0

    mkdir -p "${LOCAL_MODELS_DIR}"

    for dir in "${target_dirs[@]}"; do
        log_info "${dir}/ 다운로드 중..."
        local dl_start
        dl_start=$(date +%s)

        local local_dir="${LOCAL_MODELS_DIR}/${dir}"
        mkdir -p "${local_dir}"

        rclone --config "${RCLONE_CONF}" \
            copy "${RCLONE_REMOTE}:${GDRIVE_ONNX_DIR}/${dir}" "${local_dir}" \
            --progress \
            --transfers 4

        local dl_end
        dl_end=$(date +%s)

        # 모델 파일 크기 검증
        local model_file=""
        for candidate in "model_optimized.onnx" "model.onnx"; do
            if [[ -f "${local_dir}/${candidate}" ]]; then
                model_file="${local_dir}/${candidate}"
                break
            fi
        done

        if [[ -z "${model_file}" ]]; then
            log_error "${dir}: ONNX 모델 파일 없음 (model_optimized.onnx 또는 model.onnx)"
            continue
        fi

        local file_size
        file_size=$(get_file_size "${model_file}")
        if (( file_size < MIN_MODEL_SIZE )); then
            log_error "${dir}: 모델 파일이 너무 작음 ($(format_size "${file_size}"), 최소 $(format_size ${MIN_MODEL_SIZE}))"
            continue
        fi

        log_ok "${dir}/ 복원 완료: $(format_size "${file_size}") ($(format_duration $((dl_end - dl_start))))"
        restored=$((restored + 1))
    done

    # model_versions.json도 다운로드
    log_info "model_versions.json 다운로드..."
    rclone --config "${RCLONE_CONF}" \
        copy "${RCLONE_REMOTE}:${GDRIVE_ONNX_DIR}/model_versions.json" "${LOCAL_MODELS_DIR}/" \
        --transfers 1 2>/dev/null || log_warn "model_versions.json 없음 (정상 동작에 필수 아님)"

    local total_end
    total_end=$(date +%s)
    echo ""
    echo "=========================================="
    echo "  복원 완료: ${restored}개 모델"
    echo "  로컬 경로: ${LOCAL_MODELS_DIR}/"
    echo "  소요 시간: $(format_duration $((total_end - total_start)))"
    echo "=========================================="
    echo ""
    echo "다음 단계:"
    echo "  .env에 아래 설정 추가 후 서버 재시작:"
    echo "    USE_ONNX_RERANKER=true"
    echo "    ONNX_RERANKER_VARIANT=ort-opt-qdq"
    echo "    USE_ONNX_EMBEDDING=true"
    echo "    ONNX_EMBEDDING_VARIANT=ort-opt-qdq"
}

# ─────────────────────────────────────────────────
# build-and-backup: 빌드 → 검증 → 백업
# ─────────────────────────────────────────────────

do_build_and_backup() {
    log_info "━━━ ONNX 모델 빌드 + 백업 ━━━"

    # 빌드 인자 구성
    local build_args=("--verify")
    if [[ "${RERANKER_ONLY}" == "true" ]]; then
        build_args+=("--reranker-only")
    fi
    if [[ "${OVERWRITE}" == "true" ]]; then
        build_args+=("--overwrite")
    fi

    log_info "빌드 명령: cd backend && uv run python scripts/build_optimized_onnx.py ${build_args[*]}"

    if [[ "${DRY_RUN}" == "true" ]]; then
        echo ""
        echo "=========================================="
        echo "  빌드+백업 계획 (Dry Run)"
        echo "=========================================="
        echo ""
        echo "1. cd backend && uv run python scripts/build_optimized_onnx.py ${build_args[*]}"
        echo "2. 검증 통과 확인"
        echo "3. rclone sync → ${RCLONE_REMOTE}:${GDRIVE_ONNX_DIR}/"
        echo ""
        echo "=========================================="
        exit 0
    fi

    # 빌드 실행
    local build_start
    build_start=$(date +%s)

    (cd "${PROJECT_ROOT}/backend" && uv run python scripts/build_optimized_onnx.py "${build_args[@]}")

    local build_end
    build_end=$(date +%s)
    log_ok "빌드 완료 ($(format_duration $((build_end - build_start))))"

    # 백업 실행
    do_backup
}

# ─────────────────────────────────────────────────
# list: 원격 모델 목록
# ─────────────────────────────────────────────────

do_list() {
    check_rclone

    echo ""
    log_info "━━━ Google Drive ONNX 모델 목록 ━━━"
    echo ""
    echo "경로: ${RCLONE_REMOTE}:${GDRIVE_ONNX_DIR}/"
    echo ""

    rclone --config "${RCLONE_CONF}" \
        lsd "${RCLONE_REMOTE}:${GDRIVE_ONNX_DIR}/" 2>/dev/null || {
        log_warn "원격 디렉토리가 비어있거나 존재하지 않습니다"
        exit 0
    }

    echo ""

    # model_versions.json 내용 표시
    local tmp_versions
    tmp_versions=$(mktemp)
    if rclone --config "${RCLONE_CONF}" \
        cat "${RCLONE_REMOTE}:${GDRIVE_ONNX_DIR}/model_versions.json" > "${tmp_versions}" 2>/dev/null; then
        echo "model_versions.json:"
        python3 -c "
import json, sys
v = json.load(open('${tmp_versions}'))
for key, meta in sorted(v.items()):
    passed = meta.get('verification_passed', 'N/A')
    ts = meta.get('build_timestamp', 'N/A')
    print(f'  {key}: verified={passed}, built={ts}')
" 2>/dev/null || cat "${tmp_versions}"
    fi
    rm -f "${tmp_versions}"
}

# ─────────────────────────────────────────────────
# 서브커맨드 실행
# ─────────────────────────────────────────────────

case "${SUBCOMMAND}" in
    backup)           do_backup ;;
    restore)          do_restore ;;
    build-and-backup) do_build_and_backup ;;
    list)             do_list ;;
    -h|--help)
        echo "Usage: $0 <backup|restore|build-and-backup|list> [OPTIONS]"
        exit 0
        ;;
    *)
        log_error "알 수 없는 명령: ${SUBCOMMAND}"
        echo "사용법: $0 <backup|restore|build-and-backup|list>"
        exit 1
        ;;
esac
