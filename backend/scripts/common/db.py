"""동기 DB 세션 유틸리티 (스크립트용).

스크립트에서 사용하는 동기 SQLAlchemy 엔진과 세션 팩토리를 통합합니다.

Usage:
    from scripts.common.db import create_sync_session_factory
    Session = create_sync_session_factory()
    with Session() as db:
        db.execute(...)
"""

from __future__ import annotations

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker


def create_sync_engine(echo: bool = False) -> Engine:
    """settings.DATABASE_URL 기반 동기 엔진 생성.

    Args:
        echo: SQL 로깅 활성화 여부

    Returns:
        SQLAlchemy Engine 인스턴스
    """
    from app.core.config import settings

    return create_engine(
        settings.DATABASE_URL,
        echo=echo,
        pool_size=5,
        pool_pre_ping=True,
    )


def create_sync_session_factory(echo: bool = False) -> sessionmaker[Session]:
    """settings.DATABASE_URL 기반 동기 세션 팩토리 생성.

    Args:
        echo: SQL 로깅 활성화 여부

    Returns:
        sessionmaker 인스턴스
    """
    engine = create_sync_engine(echo=echo)
    return sessionmaker(engine)
