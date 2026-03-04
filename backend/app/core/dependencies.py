"""
인증 의존성 — FastAPI Depends로 사용
"""

import uuid

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.jwt import verify_access_token
from app.models.user import User


async def get_current_user(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> User:
    """인증 필수 의존성. 미인증 시 401."""
    user = await _extract_user(request, db)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="인증이 필요합니다.",
        )
    return user


async def get_optional_user(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> User | None:
    """인증 선택 의존성. 미인증 시 None."""
    return await _extract_user(request, db)


async def _extract_user(request: Request, db: AsyncSession) -> User | None:
    """쿠키에서 access_token을 읽어 사용자 조회"""
    token = request.cookies.get("access_token")
    if not token:
        return None

    payload = verify_access_token(token)
    if payload is None:
        return None

    try:
        user_id = uuid.UUID(str(payload["sub"]))
    except (KeyError, ValueError):
        return None

    result = await db.execute(select(User).where(User.id == user_id, User.is_active.is_(True)))
    return result.scalar_one_or_none()
