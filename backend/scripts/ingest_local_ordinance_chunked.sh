#!/bin/bash
# 자치법규 벡터 분할 실행 스크립트
#
# WSL2 CUDA 장시간 사용 시 프리즈 우회를 위해
# max_vectors 단위로 프로세스를 재시작하며 이어쓰기합니다.
#
# Usage:
#   cd backend
#   bash scripts/ingest_local_ordinance_chunked.sh [--reset] [--chunk-size N]

set -euo pipefail

CHUNK_SIZE=20000  # 1회 실행당 최대 벡터 수
RESET_FIRST=false

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case "$1" in
        --reset) RESET_FIRST=true; shift ;;
        --chunk-size) CHUNK_SIZE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "============================================"
echo "  자치법규 벡터 분할 실행"
echo "  chunk_size: ${CHUNK_SIZE} vectors/run"
echo "  reset: ${RESET_FIRST}"
echo "============================================"

ROUND=0
TOTAL_EMBEDDED=0

while true; do
    ROUND=$((ROUND + 1))

    # 첫 라운드만 --reset 적용
    RESET_FLAG=""
    if [[ "$ROUND" -eq 1 && "$RESET_FIRST" == "true" ]]; then
        RESET_FLAG="--reset"
    fi

    echo ""
    echo "--- Round ${ROUND} (누적: ${TOTAL_EMBEDDED} vectors) ---"

    # 인제스트 실행 (--no-cache: WSL2 디스크 캐시 I/O 병목 방지)
    OUTPUT=$(uv run --no-sync python -m scripts.ingest.cli \
        --type local_ordinance \
        --step vector \
        --max-vectors "${CHUNK_SIZE}" \
        --no-cache \
        ${RESET_FLAG} 2>&1) || true

    echo "$OUTPUT" | tail -5

    # embedded 수 추출 (CLI 출력: "vector.embedded: 20,000")
    EMBEDDED=$(echo "$OUTPUT" | grep -oP 'vector\.embedded: \K[0-9,]+' | tail -1 | tr -d ',')

    if [[ -z "$EMBEDDED" || "$EMBEDDED" -eq 0 ]]; then
        echo ""
        echo "============================================"
        echo "  완료! 더 이상 임베딩할 데이터 없음"
        echo "  총 라운드: ${ROUND}"
        echo "  총 벡터: ${TOTAL_EMBEDDED}"
        echo "============================================"
        break
    fi

    TOTAL_EMBEDDED=$((TOTAL_EMBEDDED + EMBEDDED))
    echo "  이번 라운드: +${EMBEDDED} → 누적: ${TOTAL_EMBEDDED}"

    # CUDA 상태 초기화를 위한 짧은 대기
    echo "  CUDA 상태 초기화 대기 (3초)..."
    sleep 3
done
