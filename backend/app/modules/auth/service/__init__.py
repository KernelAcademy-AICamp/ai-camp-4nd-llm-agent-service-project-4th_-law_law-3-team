"""
인증 서비스 — 비즈니스 로직
"""

import logging
import secrets
import uuid
from datetime import datetime, timedelta, timezone

from passlib.context import CryptContext  # type: ignore[import-untyped]
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.jwt import create_refresh_token
from app.models.user import RefreshToken, SocialAccount, User
from app.models.workspace_case import IdentityLink

logger = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthService:
    """인증 비즈니스 로직"""

    @staticmethod
    def hash_password(password: str) -> str:
        return str(pwd_context.hash(password))

    @staticmethod
    def verify_password(plain: str, hashed: str) -> bool:
        return bool(pwd_context.verify(plain, hashed))

    @staticmethod
    async def get_user_by_email(db: AsyncSession, email: str) -> User | None:
        result = await db.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()

    @staticmethod
    async def get_user_by_id(db: AsyncSession, user_id: uuid.UUID) -> User | None:
        result = await db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    @classmethod
    async def register(
        cls,
        db: AsyncSession,
        email: str,
        password: str,
        display_name: str | None = None,
    ) -> User:
        """이메일 회원가입"""
        existing = await cls.get_user_by_email(db, email)
        if existing:
            raise ValueError("이미 등록된 이메일입니다.")

        user = User(
            email=email,
            hashed_password=cls.hash_password(password),
            display_name=display_name or email.split("@")[0],
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
        logger.info("새 사용자 등록: %s", email)
        return user

    @classmethod
    async def login(
        cls, db: AsyncSession, email: str, password: str
    ) -> User | None:
        """이메일 로그인. 성공 시 User 반환, 실패 시 None."""
        user = await cls.get_user_by_email(db, email)
        if not user or not user.hashed_password:
            return None
        if not cls.verify_password(password, user.hashed_password):
            return None
        if not user.is_active:
            return None

        user.last_login_at = datetime.now(timezone.utc)
        await db.commit()
        return user

    @staticmethod
    async def create_refresh_token_record(
        db: AsyncSession,
        user_id: uuid.UUID,
        user_agent: str | None = None,
        ip_address: str | None = None,
    ) -> tuple[str, RefreshToken]:
        """Refresh Token 생성 및 DB 저장. (raw_token, record) 반환."""
        raw_token, token_hash = create_refresh_token()
        record = RefreshToken(
            user_id=user_id,
            token_hash=token_hash,
            expires_at=datetime.now(timezone.utc)
            + timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS),
            user_agent=user_agent,
            ip_address=ip_address,
        )
        db.add(record)
        await db.commit()
        return raw_token, record

    @staticmethod
    async def rotate_refresh_token(
        db: AsyncSession,
        old_token_hash: str,
        user_agent: str | None = None,
        ip_address: str | None = None,
    ) -> tuple[str, RefreshToken] | None:
        """Refresh Token 로테이션. 기존 토큰 폐기 + 새 토큰 발급."""
        result = await db.execute(
            select(RefreshToken).where(
                RefreshToken.token_hash == old_token_hash,
                RefreshToken.revoked_at.is_(None),
                RefreshToken.expires_at > datetime.now(timezone.utc),
            )
        )
        old_record = result.scalar_one_or_none()
        if not old_record:
            return None

        old_record.revoked_at = datetime.now(timezone.utc)

        raw_token, new_hash = create_refresh_token()
        new_record = RefreshToken(
            user_id=old_record.user_id,
            token_hash=new_hash,
            expires_at=datetime.now(timezone.utc)
            + timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS),
            user_agent=user_agent,
            ip_address=ip_address,
        )
        db.add(new_record)
        await db.commit()
        return raw_token, new_record

    @staticmethod
    async def revoke_refresh_token(db: AsyncSession, token_hash: str) -> None:
        """Refresh Token 폐기"""
        await db.execute(
            update(RefreshToken)
            .where(RefreshToken.token_hash == token_hash, RefreshToken.revoked_at.is_(None))
            .values(revoked_at=datetime.now(timezone.utc))
        )
        await db.commit()

    @staticmethod
    async def revoke_all_user_tokens(db: AsyncSession, user_id: uuid.UUID) -> None:
        """사용자의 모든 Refresh Token 폐기"""
        await db.execute(
            update(RefreshToken)
            .where(RefreshToken.user_id == user_id, RefreshToken.revoked_at.is_(None))
            .values(revoked_at=datetime.now(timezone.utc))
        )
        await db.commit()

    @classmethod
    async def oauth_login_or_register(
        cls,
        db: AsyncSession,
        provider: str,
        provider_user_id: str,
        email: str,
        display_name: str | None = None,
        avatar_url: str | None = None,
        provider_data: dict[str, object] | None = None,
    ) -> User:
        """OAuth 로그인 또는 자동 회원가입"""
        # 기존 소셜 계정으로 조회
        result = await db.execute(
            select(SocialAccount).where(
                SocialAccount.provider == provider,
                SocialAccount.provider_user_id == provider_user_id,
            )
        )
        social = result.scalar_one_or_none()

        if social:
            user = await cls.get_user_by_id(db, social.user_id)
            if user:
                user.last_login_at = datetime.now(timezone.utc)
                await db.commit()
                return user

        # 이메일로 기존 사용자 조회
        user = await cls.get_user_by_email(db, email)
        if not user:
            user = User(
                email=email,
                email_verified=True,
                display_name=display_name or email.split("@")[0],
                avatar_url=avatar_url,
            )
            db.add(user)
            await db.flush()
            logger.info("OAuth 자동 회원가입: %s (%s)", email, provider)

        # 소셜 계정 연결
        social = SocialAccount(
            user_id=user.id,
            provider=provider,
            provider_user_id=provider_user_id,
            provider_email=email,
            provider_data=provider_data,
        )
        db.add(social)

        user.last_login_at = datetime.now(timezone.utc)
        await db.commit()
        await db.refresh(user)
        return user

    @staticmethod
    async def migrate_session(
        db: AsyncSession, session_token: str, user_id: uuid.UUID
    ) -> None:
        """익명 세션 데이터를 사용자 계정에 연결"""
        result = await db.execute(
            select(IdentityLink).where(
                IdentityLink.session_token == session_token,
                IdentityLink.user_id == user_id,
            )
        )
        existing = result.scalar_one_or_none()
        if existing:
            return

        link = IdentityLink(
            session_token=session_token,
            user_id=user_id,
            provider="session_migration",
        )
        db.add(link)
        await db.commit()
        logger.info("세션 마이그레이션: session=%s → user=%s", session_token[:8], user_id)

    @staticmethod
    async def update_profile(
        db: AsyncSession,
        user: User,
        display_name: str | None = None,
        avatar_url: str | None = None,
    ) -> User:
        """프로필 수정"""
        if display_name is not None:
            user.display_name = display_name
        if avatar_url is not None:
            user.avatar_url = avatar_url
        await db.commit()
        await db.refresh(user)
        return user

    @staticmethod
    async def update_role(db: AsyncSession, user: User, role: str) -> User:
        """역할 변경"""
        user.role = role
        await db.commit()
        await db.refresh(user)
        return user

    @staticmethod
    def generate_password_reset_token() -> str:
        """비밀번호 재설정 토큰 생성"""
        return secrets.token_urlsafe(32)

    @classmethod
    async def reset_password(
        cls, db: AsyncSession, user: User, new_password: str
    ) -> None:
        """비밀번호 재설정"""
        user.hashed_password = cls.hash_password(new_password)
        await cls.revoke_all_user_tokens(db, user.id)
        await db.commit()
