"""
API 인증 미들웨어

API_KEY 환경변수가 설정되면 X-API-Key 헤더 검증을 수행.
비어있으면 인증을 건너뜀 (개발 모드).
"""

import hmac
import logging

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from app.core.config import settings

logger = logging.getLogger(__name__)

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    request: Request,
    api_key: str | None = Security(_API_KEY_HEADER),
) -> None:
    """API Key 인증 의존성.

    settings.API_KEY가 비어있으면 인증을 건너뜀.
    /health, /docs, /openapi.json 경로는 항상 허용.
    """
    if not settings.API_KEY:
        return

    # 공개 경로는 인증 생략
    public_paths = {"/health", "/docs", "/openapi.json", "/redoc"}
    if request.url.path in public_paths:
        return

    # 인증 엔드포인트는 API 키 불필요 (사용자 인증으로 보호)
    if request.url.path.startswith("/api/auth/"):
        return

    if not api_key or not hmac.compare_digest(
        api_key.encode("utf-8"),
        settings.API_KEY.encode("utf-8"),
    ):
        raise HTTPException(
            status_code=401,
            detail="유효하지 않은 API 키입니다.",
        )
