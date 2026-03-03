#!/bin/bash
# scripts/news_pipeline/cron_setup.sh
# KST 06:00 자동 실행 cron 설정

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
UV_PATH="$(which uv)"
LOG_DIR="${PROJECT_DIR}/data/news_pipeline/reports"

mkdir -p "$LOG_DIR"

CRON_LINE="0 6 * * * cd ${PROJECT_DIR}/backend && ${UV_PATH} run python -m scripts.news_pipeline --date yesterday >> ${LOG_DIR}/cron.log 2>&1"

echo "등록할 cron 라인:"
echo "$CRON_LINE"
echo ""
echo "crontab에 추가하려면:"
echo "(crontab -l 2>/dev/null; echo '$CRON_LINE') | crontab -"
