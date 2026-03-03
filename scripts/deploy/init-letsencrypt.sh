#!/usr/bin/env bash
# Let's Encrypt 초기 인증서 발급 스크립트
# Usage: bash scripts/deploy/init-letsencrypt.sh
#
# 사전 조건:
#   1. DNS가 서버 IP를 가리키고 있어야 함
#   2. .env.prod에 API_DOMAIN, CERTBOT_EMAIL 설정 완료
#   3. docker compose 서비스가 실행 중이어야 함
#      (nginx는 wrapper가 자체서명 인증서를 자동 생성하므로 항상 정상 기동)
set -euo pipefail

# .env.prod에서 도메인 읽기
ENV_FILE="${ENV_FILE:-.env.prod}"
if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: $ENV_FILE 파일이 없습니다. .env.prod.example에서 복사 후 설정하세요."
    exit 1
fi

DOMAIN=$(grep -E "^API_DOMAIN=" "$ENV_FILE" | cut -d= -f2- | tr -d '"' | tr -d "'")
EMAIL=$(grep -E "^CERTBOT_EMAIL=" "$ENV_FILE" | cut -d= -f2- | tr -d '"' | tr -d "'" || echo "")

if [ -z "$DOMAIN" ] || [ "$DOMAIN" = "api.example.com" ] || [ "$DOMAIN" = "api.your-free-domain.example" ]; then
    echo "ERROR: .env.prod의 API_DOMAIN을 실제 도메인으로 설정하세요."
    echo "  현재 값: ${DOMAIN:-<비어있음>}"
    exit 1
fi

if [ -z "$EMAIL" ]; then
    echo ""
    echo "CERTBOT_EMAIL이 .env.prod에 없습니다."
    read -rp "Let's Encrypt 알림 이메일 주소: " EMAIL
    if [ -z "$EMAIL" ]; then
        echo "ERROR: 이메일 주소가 필요합니다."
        exit 1
    fi
fi

COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.prod.yml}"

echo "=== Let's Encrypt 인증서 발급 ==="
echo "  도메인: $DOMAIN"
echo "  이메일: $EMAIL"
echo ""

# 1. nginx가 실행 중인지 확인 (wrapper가 자체서명 인증서를 자동 생성하므로 정상 기동됨)
if ! docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" ps nginx 2>/dev/null | grep -qE "running|Up|healthy"; then
    echo "nginx가 실행 중이 아닙니다. 먼저 서비스를 시작하세요:"
    echo "  docker compose --env-file $ENV_FILE -f $COMPOSE_FILE up -d"
    exit 1
fi

# 2. Let's Encrypt 인증서 발급 (ACME webroot challenge)
# nginx wrapper가 이미 자체서명 인증서로 443을 리스닝 중이므로 바로 발급 가능
echo "=== 1/2. Let's Encrypt 인증서 발급 ==="
docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" run --rm certbot \
    certonly --webroot \
    --webroot-path=/var/www/certbot \
    --email "$EMAIL" \
    --agree-tos \
    --no-eff-email \
    -d "$DOMAIN"

# 3. nginx 재시작 (실제 인증서 적용)
echo "=== 2/2. nginx 재시작 (실제 인증서 적용) ==="
docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" restart nginx

echo ""
echo "=== 인증서 발급 완료 ==="
echo "  도메인: https://$DOMAIN"
echo "  헬스체크: curl https://$DOMAIN/health"
echo ""
echo "자동 갱신은 certbot 컨테이너가 12시간마다 실행합니다."
echo ""
