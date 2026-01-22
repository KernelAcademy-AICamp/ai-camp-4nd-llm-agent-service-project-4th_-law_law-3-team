"""
데이터베이스 설정 및 세션 관리

SQLAlchemy 2.0 async 패턴 사용
"""

from typing import AsyncGenerator
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from app.core.config import settings

# Async Engine 생성
engine = create_async_engine(
    settings.DATABASE_URL_ASYNC,
    echo=settings.DEBUG,  # SQL 로깅 (개발 시)
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,  # 연결 유효성 검사
)

# Session Factory
async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Sync Engine (for non-async contexts like chat_service)
sync_engine = create_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
)

# Sync Session Factory
sync_session_factory = sessionmaker(
    sync_engine,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Base class for models
Base = declarative_base()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI Dependency Injection용 DB 세션 제공

    Note:
        - 읽기 전용 요청은 commit 불필요
        - 쓰기 작업 시 호출자가 명시적으로 await db.commit() 호출 필요
        - 세션 종료는 async context manager가 자동 처리

    Usage:
        @router.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db)):
            ...

        @router.post("/items")
        async def create_item(db: AsyncSession = Depends(get_db)):
            db.add(item)
            await db.commit()  # 명시적 commit 필요
    """
    async with async_session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """
    데이터베이스 테이블 생성 (개발용)
    프로덕션에서는 Alembic 마이그레이션 사용
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """데이터베이스 연결 종료"""
    await engine.dispose()
