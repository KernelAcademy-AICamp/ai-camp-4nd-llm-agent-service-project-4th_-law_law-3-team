"""add source_count to legal_terms

Revision ID: 007
Revises: 006
Create Date: 2026-02-09

법률 용어 사전 source_count 컬럼 추가
- 동일 용어가 여러 사전에 등재된 경우 출처 수 저장
"""
from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "007"
down_revision: Union[str, None] = "006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "legal_terms",
        sa.Column(
            "source_count",
            sa.Integer(),
            nullable=False,
            server_default="1",
            comment="동일 용어 출처 법령 수",
        ),
    )


def downgrade() -> None:
    op.drop_column("legal_terms", "source_count")
