#!/bin/sh
# Nginx entrypoint wrapper — SSL 인증서 없으면 자체서명 인증서 자동 생성
# nginx가 443 포트를 리스닝하려면 인증서 파일이 반드시 존재해야 함

CERT_DIR="/etc/nginx/certs/live/${API_DOMAIN:-api.example.com}"
CERT_FILE="$CERT_DIR/fullchain.pem"
KEY_FILE="$CERT_DIR/privkey.pem"

if [ ! -f "$CERT_FILE" ] || [ ! -f "$KEY_FILE" ]; then
    echo "[nginx-wrapper] SSL 인증서 미발견 — 자체서명 인증서 생성 중..."
    mkdir -p "$CERT_DIR"
    openssl req -x509 -nodes -newkey rsa:2048 -days 30 \
        -keyout "$KEY_FILE" \
        -out "$CERT_FILE" \
        -subj "/CN=${API_DOMAIN:-localhost}" \
        2>/dev/null
    echo "[nginx-wrapper] 자체서명 인증서 생성 완료. Let's Encrypt 발급 후 교체하세요."
    echo "[nginx-wrapper]   bash scripts/deploy/init-letsencrypt.sh"
fi

# 원래 nginx entrypoint 실행
exec /docker-entrypoint.sh "$@"
