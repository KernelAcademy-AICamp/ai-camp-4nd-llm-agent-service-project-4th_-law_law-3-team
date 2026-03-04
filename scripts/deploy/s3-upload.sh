#!/bin/bash
# S3 데이터 업로드 스크립트 (로컬 → S3)
# Usage: bash scripts/deploy/s3-upload.sh [--dry-run]
set -euo pipefail

S3_BUCKET="s3://law-3"
S3_PREFIX="deploy"
PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DRY_RUN=""

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN="--dryrun"
    echo "[DRY-RUN] 실제 업로드 없이 확인만 합니다."
fi

echo "============================================"
echo "  S3 데이터 업로드 (law-3 배포용)"
echo "============================================"
echo "버킷: ${S3_BUCKET}/${S3_PREFIX}/"
echo ""

# 1. LanceDB 벡터 데이터 (7.7GB)
echo "[1/3] LanceDB 데이터 업로드 중..."
aws s3 sync \
    "${PROJECT_DIR}/backend/lancedb_data/" \
    "${S3_BUCKET}/${S3_PREFIX}/lancedb_data/" \
    --exclude "*.tmp" \
    --exclude "*.lock" \
    ${DRY_RUN} \
    --region ap-northeast-2

# 2. data/ (ingest_source 제외 — 런타임에 불필요)
echo ""
echo "[2/3] 런타임 데이터 업로드 중..."
aws s3 sync \
    "${PROJECT_DIR}/data/" \
    "${S3_BUCKET}/${S3_PREFIX}/data/" \
    --exclude "ingest_source/*" \
    --exclude "bar_exam_raw/*" \
    ${DRY_RUN} \
    --region ap-northeast-2

# 3. MeCab 사전 (csv + json만 — .dic는 플랫폼별 바이너리이므로 업로드 제외)
echo ""
echo "[3/3] MeCab 사전 업로드 중..."
aws s3 sync \
    "${PROJECT_DIR}/backend/data/mecab_userdic/" \
    "${S3_BUCKET}/${S3_PREFIX}/mecab_userdic/" \
    --exclude "*.dic" \
    ${DRY_RUN} \
    --region ap-northeast-2

echo ""
echo "============================================"
echo "  업로드 완료!"
echo "============================================"
echo ""
echo "S3 경로 확인:"
echo "  aws s3 ls ${S3_BUCKET}/${S3_PREFIX}/ --region ap-northeast-2"
echo ""
echo "EC2에서 다운로드:"
echo "  bash scripts/deploy/s3-download.sh"
