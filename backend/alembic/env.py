"""
Alembic 마이그레이션 환경 설정
"""

import os
import sys
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 설정 및 모델 임포트
from app.core.config import settings
from app.common.database import Base
from app.models import LegalDocument, Law, LegalReference  # 모델 임포트 (테이블 등록)

# Alembic Config 객체
config = context.config

# 환경변수에서 DATABASE_URL 가져와서 설정
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

# Python 로깅 설정
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 모델 메타데이터 (autogenerate 지원)
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """
    오프라인 모드 마이그레이션

    DB 연결 없이 SQL 스크립트만 생성
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    온라인 모드 마이그레이션

    실제 DB에 연결하여 마이그레이션 실행
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,  # 컬럼 타입 변경 감지
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
