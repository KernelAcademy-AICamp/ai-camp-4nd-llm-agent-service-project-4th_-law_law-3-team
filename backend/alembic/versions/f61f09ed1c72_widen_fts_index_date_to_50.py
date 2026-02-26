"""widen_fts_index_date_to_50

Revision ID: f61f09ed1c72
Revises: 016
Create Date: 2026-02-25 00:52:00.523992

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f61f09ed1c72'
down_revision: Union[str, Sequence[str], None] = '016'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """fts_index.date: VARCHAR(20) → VARCHAR(50) 확장 (복수 날짜 대응)"""
    op.alter_column(
        "fts_index",
        "date",
        type_=sa.String(50),
        existing_type=sa.String(20),
        existing_nullable=True,
    )


def downgrade() -> None:
    """fts_index.date: VARCHAR(50) → VARCHAR(20) 복원"""
    op.alter_column(
        "fts_index",
        "date",
        type_=sa.String(20),
        existing_type=sa.String(50),
        existing_nullable=True,
    )
