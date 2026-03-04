"""
인증 모듈 라우터 — 13개 API 엔드포인트
"""

import logging
import secrets
from typing import Literal
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.core.dependencies import get_current_user
from app.core.jwt import create_access_token, hash_token
from app.models.user import User
from app.modules.auth.schema import (
    AuthResponse,
    LoginRequest,
    MessageResponse,
    PasswordResetConfirm,
    PasswordResetRequest,
    ProfileUpdateRequest,
    RegisterRequest,
    RoleUpdateRequest,
    UserResponse,
)
from app.modules.auth.service import AuthService

logger = logging.getLogger(__name__)

router = APIRouter()

# 쿠키 공통 설정
_COOKIE_SECURE = settings.ENVIRONMENT != "development"
_COOKIE_SAMESITE: Literal["lax", "strict", "none"] = "lax"


def _user_to_response(user: User) -> UserResponse:
    return UserResponse(
        user_id=str(user.id),
        email=user.email,
        display_name=user.display_name,
        avatar_url=user.avatar_url,
        role=user.role,
        email_verified=user.email_verified,
    )


def _set_auth_cookies(
    response: Response,
    access_token: str,
    refresh_token: str,
) -> None:
    """인증 쿠키 설정"""
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        secure=_COOKIE_SECURE,
        samesite=_COOKIE_SAMESITE,
        max_age=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        path="/",
    )
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        secure=_COOKIE_SECURE,
        samesite=_COOKIE_SAMESITE,
        max_age=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
        path="/api/auth/refresh",
    )


def _clear_auth_cookies(response: Response) -> None:
    """인증 쿠키 삭제"""
    response.delete_cookie("access_token", path="/")
    response.delete_cookie("refresh_token", path="/api/auth/refresh")


# --- 이메일 인증 ---


@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def register(
    body: RegisterRequest,
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
) -> AuthResponse:
    """이메일 회원가입"""
    try:
        user = await AuthService.register(db, body.email, body.password, body.display_name)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) from e

    access = create_access_token(user.id, user.email)
    raw_refresh, _ = await AuthService.create_refresh_token_record(
        db,
        user.id,
        user_agent=request.headers.get("user-agent"),
        ip_address=request.client.host if request.client else None,
    )

    # 세션 마이그레이션
    session_token = request.cookies.get("session_token")
    if session_token:
        await AuthService.migrate_session(db, session_token, user.id)

    _set_auth_cookies(response, access, raw_refresh)
    return AuthResponse(user=_user_to_response(user))


@router.post("/login", response_model=AuthResponse)
async def login(
    body: LoginRequest,
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
) -> AuthResponse:
    """이메일 로그인"""
    user = await AuthService.login(db, body.email, body.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="이메일 또는 비밀번호가 올바르지 않습니다.",
        )

    access = create_access_token(user.id, user.email)
    raw_refresh, _ = await AuthService.create_refresh_token_record(
        db,
        user.id,
        user_agent=request.headers.get("user-agent"),
        ip_address=request.client.host if request.client else None,
    )

    # 세션 마이그레이션
    session_token = request.cookies.get("session_token")
    if session_token:
        await AuthService.migrate_session(db, session_token, user.id)

    _set_auth_cookies(response, access, raw_refresh)
    return AuthResponse(user=_user_to_response(user))


@router.post("/logout", response_model=MessageResponse)
async def logout(
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
) -> MessageResponse:
    """로그아웃 — refresh_token 폐기"""
    raw_refresh = request.cookies.get("refresh_token")
    if raw_refresh:
        await AuthService.revoke_refresh_token(db, hash_token(raw_refresh))

    _clear_auth_cookies(response)
    return MessageResponse(message="로그아웃 되었습니다.")


@router.post("/refresh", response_model=AuthResponse)
async def refresh(
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
) -> AuthResponse:
    """Access Token 갱신"""
    raw_refresh = request.cookies.get("refresh_token")
    if not raw_refresh:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token이 없습니다.",
        )

    result = await AuthService.rotate_refresh_token(
        db,
        hash_token(raw_refresh),
        user_agent=request.headers.get("user-agent"),
        ip_address=request.client.host if request.client else None,
    )
    if not result:
        _clear_auth_cookies(response)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="유효하지 않은 Refresh token입니다.",
        )

    new_raw_refresh, new_record = result
    user = await AuthService.get_user_by_id(db, new_record.user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="사용자를 찾을 수 없습니다.")

    access = create_access_token(user.id, user.email)
    _set_auth_cookies(response, access, new_raw_refresh)
    return AuthResponse(user=_user_to_response(user))


# --- 프로필 ---


@router.get("/me", response_model=UserResponse)
async def get_me(user: User = Depends(get_current_user)) -> UserResponse:
    """현재 사용자 정보"""
    return _user_to_response(user)


@router.patch("/me", response_model=UserResponse)
async def update_me(
    body: ProfileUpdateRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> UserResponse:
    """프로필 수정"""
    user = await AuthService.update_profile(db, user, body.display_name, body.avatar_url)
    return _user_to_response(user)


@router.patch("/me/role", response_model=UserResponse)
async def update_role(
    body: RoleUpdateRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> UserResponse:
    """역할 변경 (user↔lawyer)"""
    user = await AuthService.update_role(db, user, body.role)
    return _user_to_response(user)


# --- 비밀번호 재설정 ---


@router.post("/password/reset-request", response_model=MessageResponse)
async def password_reset_request(
    body: PasswordResetRequest,
    db: AsyncSession = Depends(get_db),
) -> MessageResponse:
    """비밀번호 재설정 요청 (이메일 전송은 미구현, 토큰만 로깅)"""
    user = await AuthService.get_user_by_email(db, body.email)
    if user:
        token = AuthService.generate_password_reset_token()
        logger.info("비밀번호 재설정 토큰 (개발용): %s for %s", token, body.email)
    return MessageResponse(message="비밀번호 재설정 링크가 이메일로 발송되었습니다.")


@router.post("/password/reset-confirm", response_model=MessageResponse)
async def password_reset_confirm(
    body: PasswordResetConfirm,
    db: AsyncSession = Depends(get_db),
) -> MessageResponse:
    """비밀번호 재설정 확인 (토큰 검증은 추후 구현)"""
    return MessageResponse(message="비밀번호가 변경되었습니다.")


# --- OAuth ---


@router.get("/oauth/google")
async def oauth_google() -> dict[str, str]:
    """Google OAuth 인증 URL 반환"""
    if not settings.GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Google OAuth가 설정되지 않았습니다.")

    state = secrets.token_urlsafe(32)
    params = {
        "client_id": settings.GOOGLE_CLIENT_ID,
        "redirect_uri": _google_redirect_uri(),
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "access_type": "offline",
        "prompt": "consent",
    }
    return {"url": f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}", "state": state}


@router.get("/oauth/google/callback")
async def oauth_google_callback(
    code: str,
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
) -> AuthResponse:
    """Google OAuth 콜백"""
    async with httpx.AsyncClient() as client:
        token_resp = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": settings.GOOGLE_CLIENT_ID,
                "client_secret": settings.GOOGLE_CLIENT_SECRET,
                "redirect_uri": _google_redirect_uri(),
                "grant_type": "authorization_code",
            },
        )
        if token_resp.status_code != 200:
            raise HTTPException(status_code=400, detail="Google 인증에 실패했습니다.")

        tokens = token_resp.json()
        userinfo_resp = await client.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
        if userinfo_resp.status_code != 200:
            raise HTTPException(status_code=400, detail="Google 사용자 정보를 가져올 수 없습니다.")

        userinfo = userinfo_resp.json()

    user = await AuthService.oauth_login_or_register(
        db,
        provider="google",
        provider_user_id=userinfo["id"],
        email=userinfo["email"],
        display_name=userinfo.get("name"),
        avatar_url=userinfo.get("picture"),
        provider_data=userinfo,
    )

    access = create_access_token(user.id, user.email)
    raw_refresh, _ = await AuthService.create_refresh_token_record(
        db,
        user.id,
        user_agent=request.headers.get("user-agent"),
        ip_address=request.client.host if request.client else None,
    )

    session_token = request.cookies.get("session_token")
    if session_token:
        await AuthService.migrate_session(db, session_token, user.id)

    _set_auth_cookies(response, access, raw_refresh)
    return AuthResponse(user=_user_to_response(user))


@router.get("/oauth/kakao")
async def oauth_kakao() -> dict[str, str]:
    """Kakao OAuth 인증 URL 반환"""
    if not settings.KAKAO_OAUTH_CLIENT_ID:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Kakao OAuth가 설정되지 않았습니다.")

    state = secrets.token_urlsafe(32)
    params = {
        "client_id": settings.KAKAO_OAUTH_CLIENT_ID,
        "redirect_uri": _kakao_redirect_uri(),
        "response_type": "code",
        "state": state,
    }
    return {"url": f"https://kauth.kakao.com/oauth/authorize?{urlencode(params)}", "state": state}


@router.get("/oauth/kakao/callback")
async def oauth_kakao_callback(
    code: str,
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
) -> AuthResponse:
    """Kakao OAuth 콜백"""
    async with httpx.AsyncClient() as client:
        token_resp = await client.post(
            "https://kauth.kakao.com/oauth/token",
            data={
                "grant_type": "authorization_code",
                "client_id": settings.KAKAO_OAUTH_CLIENT_ID,
                "client_secret": settings.KAKAO_OAUTH_CLIENT_SECRET,
                "redirect_uri": _kakao_redirect_uri(),
                "code": code,
            },
        )
        if token_resp.status_code != 200:
            raise HTTPException(status_code=400, detail="Kakao 인증에 실패했습니다.")

        tokens = token_resp.json()
        userinfo_resp = await client.get(
            "https://kapi.kakao.com/v2/user/me",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
        if userinfo_resp.status_code != 200:
            raise HTTPException(status_code=400, detail="Kakao 사용자 정보를 가져올 수 없습니다.")

        userinfo = userinfo_resp.json()

    kakao_account = userinfo.get("kakao_account", {})
    profile = kakao_account.get("profile", {})

    email = kakao_account.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="Kakao 계정에서 이메일을 가져올 수 없습니다.")

    user = await AuthService.oauth_login_or_register(
        db,
        provider="kakao",
        provider_user_id=str(userinfo["id"]),
        email=email,
        display_name=profile.get("nickname"),
        avatar_url=profile.get("profile_image_url"),
        provider_data=userinfo,
    )

    access = create_access_token(user.id, user.email)
    raw_refresh, _ = await AuthService.create_refresh_token_record(
        db,
        user.id,
        user_agent=request.headers.get("user-agent"),
        ip_address=request.client.host if request.client else None,
    )

    session_token = request.cookies.get("session_token")
    if session_token:
        await AuthService.migrate_session(db, session_token, user.id)

    _set_auth_cookies(response, access, raw_refresh)
    return AuthResponse(user=_user_to_response(user))


# --- 헬퍼 ---


def _google_redirect_uri() -> str:
    if settings.ENVIRONMENT == "production":
        return f"{settings.CORS_ORIGINS[0]}/api/auth/oauth/google/callback"
    return "http://localhost:3000/api/auth/oauth/google/callback"


def _kakao_redirect_uri() -> str:
    if settings.ENVIRONMENT == "production":
        return f"{settings.CORS_ORIGINS[0]}/api/auth/oauth/kakao/callback"
    return "http://localhost:3000/api/auth/oauth/kakao/callback"
