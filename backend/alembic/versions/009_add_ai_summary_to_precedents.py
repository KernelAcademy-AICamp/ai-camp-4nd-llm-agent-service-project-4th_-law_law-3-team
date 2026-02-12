"""add ai_summary to precedent_documents

Revision ID: 009
Revises: 008
Create Date: 2026-02-11

요약 기반 RAG 아키텍처 전환을 위해 precedent_documents 테이블에
ai_summary 컬럼 추가. 판례요약 텍스트를 저장하여 1문서=1벡터 임베딩에 사용.
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "009"
down_revision: Union[str, None] = "008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "precedent_documents",
        sa.Column(
            "ai_summary",
            sa.Text(),
            nullable=True,
            comment="AI 생성 판례요약",
        ),
    )


def downgrade() -> None:
    op.drop_column("precedent_documents", "ai_summary")
