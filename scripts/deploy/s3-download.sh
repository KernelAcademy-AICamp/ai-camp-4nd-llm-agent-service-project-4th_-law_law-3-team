#!/bin/bash
# S3 데이터 다운로드 스크립트 (S3 → EC2 서버)
# Usage: bash scripts/deploy/s3-download.sh [--dry-run]
set -euo pipefail

S3_BUCKET="s3://law-3"
S3_PREFIX="deploy"
PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DRY_RUN=""

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN="--dryrun"
    echo "[DRY-RUN] 실제 다운로드 없이 확인만 합니다."
fi

echo "============================================"
echo "  S3 데이터 다운로드 (EC2 배포용)"
echo "============================================"
echo "소스: ${S3_BUCKET}/${S3_PREFIX}/"
echo ""

# 디렉토리 생성
mkdir -p "${PROJECT_DIR}/backend/lancedb_data"
mkdir -p "${PROJECT_DIR}/data"
mkdir -p "${PROJECT_DIR}/backend/data/mecab_userdic"
mkdir -p "${PROJECT_DIR}/backend/data/models"

# 1. LanceDB 벡터 데이터
echo "[1/4] LanceDB 데이터 다운로드 중..."
aws s3 sync \
    "${S3_BUCKET}/${S3_PREFIX}/lancedb_data/" \
    "${PROJECT_DIR}/backend/lancedb_data/" \
    ${DRY_RUN} \
    --region ap-northeast-2

# 2. 런타임 데이터
echo ""
echo "[2/4] 런타임 데이터 다운로드 중..."
aws s3 sync \
    "${S3_BUCKET}/${S3_PREFIX}/data/" \
    "${PROJECT_DIR}/data/" \
    ${DRY_RUN} \
    --region ap-northeast-2

# 3. MeCab 사전 (csv + json만 — .dic는 EC2에서 빌드하거나 scp 전송)
echo ""
echo "[3/4] MeCab 사전 다운로드 중..."
aws s3 sync \
    "${S3_BUCKET}/${S3_PREFIX}/mecab_userdic/" \
    "${PROJECT_DIR}/backend/data/mecab_userdic/" \
    --exclude "*.dic" \
    ${DRY_RUN} \
    --region ap-northeast-2

# 4. ONNX 모델 (임베딩: kure-v1-ort-opt, 리랭커: reranker-ort-opt-qdq)
echo ""
echo "[4/4] ONNX 모델 다운로드 중..."
aws s3 sync \
    "${S3_BUCKET}/${S3_PREFIX}/models/" \
    "${PROJECT_DIR}/backend/data/models/" \
    --exclude "models--*" \
    ${DRY_RUN} \
    --region ap-northeast-2

echo ""
echo "============================================"
echo "  다운로드 완료!"
echo "============================================"
echo ""
echo "LanceDB 데이터를 Docker 볼륨에 복사하려면:"
echo "  docker compose --env-file .env.prod -f docker-compose.prod.yml up -d backend"
echo "  docker cp backend/lancedb_data/. law-platform-backend:/app/lancedb_data/"
echo "  docker compose --env-file .env.prod -f docker-compose.prod.yml restart backend"
