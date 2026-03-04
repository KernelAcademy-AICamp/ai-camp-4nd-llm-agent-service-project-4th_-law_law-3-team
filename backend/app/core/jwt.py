"""
JWT 토큰 생성 및 검증 유틸리티
"""

import hashlib
import logging
import secrets
import uuid
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt  # type: ignore[import-untyped]

from app.core.config import settings

logger = logging.getLogger(__name__)


def _get_secret_key() -> str:
    """JWT 시크릿 키 반환. 미설정 시 개발용 임시 키 사용."""
    if settings.JWT_SECRET_KEY:
        return settings.JWT_SECRET_KEY
    if settings.ENVIRONMENT == "production":
        raise ValueError("프로덕션 환경에서 JWT_SECRET_KEY가 설정되지 않았습니다.")
    return "dev-jwt-secret-key-do-not-use-in-production"


def create_access_token(user_id: uuid.UUID, email: str) -> str:
    """Access Token 생성 (JWT)"""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "email": email,
        "iat": now,
        "exp": now + timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES),
        "type": "access",
    }
    return str(jwt.encode(payload, _get_secret_key(), algorithm=settings.JWT_ALGORITHM))


def create_refresh_token() -> tuple[str, str]:
    """Refresh Token 생성. (raw_token, token_hash) 튜플 반환."""
    raw_token = secrets.token_urlsafe(48)
    token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
    return raw_token, token_hash


def verify_access_token(token: str) -> dict[str, object] | None:
    """Access Token 검증. 유효하면 payload dict 반환, 실패 시 None."""
    try:
        payload = jwt.decode(
            token, _get_secret_key(), algorithms=[settings.JWT_ALGORITHM]
        )
        if payload.get("type") != "access":
            return None
        return dict(payload)
    except JWTError:
        return None


def hash_token(raw_token: str) -> str:
    """토큰을 SHA-256 해시로 변환"""
    return hashlib.sha256(raw_token.encode()).hexdigest()
