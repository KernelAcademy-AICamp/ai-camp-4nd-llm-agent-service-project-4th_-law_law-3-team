"""law_articles 테이블 생성

법령 조문 단위 저장 테이블. LanceDB 벡터 검색에서 매칭된 조문만
선별적으로 LLM 컨텍스트에 포함하기 위한 구조.

Revision ID: 021
Revises: 020
Create Date: 2026-03-01
"""

import sqlalchemy as sa
from alembic import op

revision = "021"
down_revision = "020"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "law_articles",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "law_id",
            sa.String(50),
            nullable=False,
            comment="법령 ID (law_documents.law_id 대응)",
        ),
        sa.Column(
            "article_number",
            sa.String(50),
            nullable=False,
            comment="조문번호 (LanceDB article_number과 동일 형식)",
        ),
        sa.Column(
            "article_title",
            sa.String(500),
            nullable=True,
            comment="조문제목",
        ),
        sa.Column(
            "article_content",
            sa.Text(),
            nullable=False,
            comment="조문 본문 (항+호 포함)",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=True,
            comment="레코드 생성일시",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "law_id", "article_number", name="uq_law_articles_law_article"
        ),
    )
    op.create_index("idx_law_articles_law_id", "law_articles", ["law_id"])


def downgrade() -> None:
    op.drop_index("idx_law_articles_law_id", table_name="law_articles")
    op.drop_table("law_articles")
