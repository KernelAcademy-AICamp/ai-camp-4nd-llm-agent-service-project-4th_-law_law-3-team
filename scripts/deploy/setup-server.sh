#!/usr/bin/env bash
# EC2 ARM(Graviton) 초기 서버 설정 스크립트
# Usage: bash scripts/deploy/setup-server.sh
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/opt/law-platform}"
BRANCH="${BRANCH:-deploy/aws-arm}"
REPO_URL="${REPO_URL:-git@github.com:your-org/law-3.git}"

echo "=== 1/6. 시스템 패키지 업데이트 ==="
sudo apt-get update -y
sudo apt-get install -y curl git

echo "=== 2/6. Docker 설치 ==="
if ! command -v docker &>/dev/null; then
    curl -fsSL https://get.docker.com | sudo sh
    sudo usermod -aG docker "$USER"
    echo "Docker 설치 완료. 현재 세션에서 docker 사용을 위해 재로그인이 필요할 수 있습니다."
else
    echo "Docker 이미 설치됨: $(docker --version)"
fi

echo "=== 3/6. Docker Compose 플러그인 확인 ==="
if ! docker compose version &>/dev/null; then
    sudo apt-get install -y docker-compose-plugin
fi
echo "Docker Compose: $(docker compose version)"

echo "=== 4/6. 프로젝트 클론/업데이트 ==="
if [ -d "$PROJECT_DIR" ]; then
    cd "$PROJECT_DIR"
    git fetch origin
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
else
    sudo mkdir -p "$PROJECT_DIR"
    sudo chown "$USER:$USER" "$PROJECT_DIR"
    git clone -b "$BRANCH" "$REPO_URL" "$PROJECT_DIR"
    cd "$PROJECT_DIR"
fi

echo "=== 5/6. 환경 설정 ==="
if [ ! -f .env.prod ]; then
    cp .env.prod.example .env.prod
    echo ""
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "  .env.prod 파일을 생성했습니다."
    echo "  반드시 비밀번호/API 키를 설정한 후 계속하세요."
    echo "  편집: nano $PROJECT_DIR/.env.prod"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo ""
    read -rp ".env.prod 설정을 완료했으면 Enter를 누르세요..."
fi

echo "=== 6/6. 서비스 빌드 및 시작 ==="
# 볼륨 마운트 디렉토리 사전 생성
mkdir -p backend/data/models backend/data/mecab_userdic

docker compose --env-file .env.prod -f docker-compose.prod.yml build
docker compose --env-file .env.prod -f docker-compose.prod.yml up -d

echo ""
echo "=== 서비스 시작 완료 ==="
echo "상태 확인: docker compose -f docker-compose.prod.yml ps"
echo "로그 확인: docker compose -f docker-compose.prod.yml logs -f backend"
echo ""

# DB 마이그레이션 대기
echo "=== PostgreSQL 준비 대기 ==="
for i in $(seq 1 15); do
    if docker compose -f docker-compose.prod.yml exec -T postgres pg_isready -U lawuser -d lawdb 2>&1 | grep -q "accepting"; then
        echo "PostgreSQL 준비 완료"
        break
    fi
    echo "대기 중... ($i/15)"
    sleep 2
done

echo "=== DB 마이그레이션 ==="
docker compose -f docker-compose.prod.yml exec -T backend python -m alembic upgrade head

echo ""
echo "=== 초기 설정 완료 ==="
echo ""
echo "다음 단계 (수동):"
echo "  1. ML 모델 다운로드:"
echo "     docker compose -f docker-compose.prod.yml exec backend python scripts/download_models.py"
echo "  2. 데이터 로드 (선택):"
echo "     docker compose -f docker-compose.prod.yml exec backend python -m scripts.ingest.cli --type all --step db"
echo "  3. 헬스 체크:"
echo "     curl http://localhost/health"
echo ""
