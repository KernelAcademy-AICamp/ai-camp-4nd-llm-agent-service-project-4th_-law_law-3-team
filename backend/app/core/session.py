"""
세션 미들웨어 — HttpOnly 쿠키 기반 세션 토큰 관리

session_token 쿠키가 없으면 자동 생성하여 Set-Cookie 응답.
request.state.session_token에 세션 토큰 주입.
access_token 쿠키가 있으면 user_id/user_email도 주입.
"""

import logging
import secrets
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.core.config import settings
from app.core.jwt import verify_access_token

logger = logging.getLogger(__name__)

SESSION_COOKIE_NAME = "session_token"
SESSION_MAX_AGE = 7 * 24 * 60 * 60  # 7일
_VALID_TOKEN_LENGTH = 64  # secrets.token_hex(32) → 64자 hex


class SessionMiddleware(BaseHTTPMiddleware):
    """HttpOnly 세션 쿠키 미들웨어

    모든 요청에 session_token을 보장:
    - 쿠키가 있으면 기존 토큰 사용
    - 쿠키가 없으면 신규 토큰 생성 + Set-Cookie

    access_token 쿠키가 있으면 user_id/user_email 파싱 추가:
    - 유효하면 request.state.user_id, request.state.user_email 설정
    - 실패 시 None (기존 동작 변경 없음)
    """

    @staticmethod
    def _is_valid_token(token: str) -> bool:
        """토큰이 64자 hex 형식인지 검증"""
        if len(token) != _VALID_TOKEN_LENGTH:
            return False
        try:
            int(token, 16)
        except ValueError:
            return False
        return True

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        token = request.cookies.get(SESSION_COOKIE_NAME)
        is_new = False

        if not token or not self._is_valid_token(token):
            token = secrets.token_hex(32)
            is_new = True

        request.state.session_token = token

        # access_token 파싱 (실패해도 기존 동작에 영향 없음)
        request.state.user_id = None
        request.state.user_email = None
        access_token = request.cookies.get("access_token")
        if access_token:
            payload = verify_access_token(access_token)
            if payload:
                try:
                    request.state.user_id = uuid.UUID(str(payload["sub"]))
                    request.state.user_email = payload.get("email")
                except (KeyError, ValueError):
                    pass

        response = await call_next(request)

        if is_new:
            response.set_cookie(
                key=SESSION_COOKIE_NAME,
                value=token,
                httponly=True,
                secure=settings.ENVIRONMENT != "development",
                samesite="lax",
                max_age=SESSION_MAX_AGE,
                path="/",
            )

        return response
