"""
세션 미들웨어 — HttpOnly 쿠키 기반 세션 토큰 관리

session_token 쿠키가 없으면 자동 생성하여 Set-Cookie 응답.
request.state.session_token에 세션 토큰 주입.
"""

import secrets

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.core.config import settings

SESSION_COOKIE_NAME = "session_token"
SESSION_MAX_AGE = 30 * 24 * 60 * 60  # 30일


class SessionMiddleware(BaseHTTPMiddleware):
    """HttpOnly 세션 쿠키 미들웨어

    모든 요청에 session_token을 보장:
    - 쿠키가 있으면 기존 토큰 사용
    - 쿠키가 없으면 신규 토큰 생성 + Set-Cookie
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        token = request.cookies.get(SESSION_COOKIE_NAME)
        is_new = False

        if not token:
            token = secrets.token_hex(32)
            is_new = True

        request.state.session_token = token

        response = await call_next(request)

        if is_new:
            response.set_cookie(
                key=SESSION_COOKIE_NAME,
                value=token,
                httponly=True,
                secure=not settings.DEBUG,
                samesite="lax",
                max_age=SESSION_MAX_AGE,
                path="/",
            )

        return response
