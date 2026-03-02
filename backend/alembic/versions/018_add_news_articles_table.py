"""뉴스 기사 테이블 생성

news_articles + news_article_dlq 테이블 생성.
법률 뉴스 파이프라인 v0.3.0.

Revision ID: 018
Revises: 017
Create Date: 2026-02-26

"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "018"
down_revision: Union[str, None] = "017"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # news_articles 테이블
    op.create_table(
        "news_articles",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("source", sa.String(20), nullable=False),
        sa.Column("publisher", sa.String(200), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("author", sa.String(100), nullable=True),
        sa.Column(
            "published_at", sa.DateTime(timezone=True), nullable=True,
        ),
        sa.Column(
            "collected_at", sa.DateTime(timezone=True), nullable=False,
        ),
        sa.Column("url", sa.Text(), nullable=False, unique=True),
        sa.Column("section", sa.String(50), nullable=True),
        sa.Column("tags", postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column("cleaned_text", sa.Text(), nullable=False),
        sa.Column("summary_one_liner", sa.Text(), nullable=False),
        sa.Column(
            "summary_issues", postgresql.ARRAY(sa.String()), nullable=True,
        ),
        sa.Column(
            "summary_laws", postgresql.ARRAY(sa.String()), nullable=True,
        ),
        sa.Column(
            "summary_cases", postgresql.ARRAY(sa.String()), nullable=True,
        ),
        sa.Column(
            "summary_institutions",
            postgresql.ARRAY(sa.String()),
            nullable=True,
        ),
        sa.Column(
            "summary_implications",
            postgresql.ARRAY(sa.String()),
            nullable=True,
        ),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("disclaimer", sa.Text(), nullable=False),
        sa.Column(
            "schema_version",
            sa.String(10),
            nullable=False,
            server_default="1.0",
        ),
        sa.Column(
            "is_indexed",
            sa.Boolean(),
            nullable=False,
            server_default="false",
        ),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
    )

    # 단일 인덱스
    op.create_index("idx_news_source", "news_articles", ["source"])
    op.create_index("idx_news_published_at", "news_articles", ["published_at"])
    op.create_index("idx_news_content_hash", "news_articles", ["content_hash"])

    # 복합 인덱스
    op.create_index(
        "idx_news_source_published",
        "news_articles",
        ["source", "published_at"],
    )

    # GIN 인덱스 (ARRAY 검색)
    op.create_index(
        "idx_news_tags",
        "news_articles",
        ["tags"],
        postgresql_using="gin",
    )

    # news_article_dlq 테이블
    op.create_table(
        "news_article_dlq",
        sa.Column(
            "id", sa.Integer(), primary_key=True, autoincrement=True,
        ),
        sa.Column("article_url", sa.Text(), nullable=False, unique=True),
        sa.Column("stage", sa.String(30), nullable=False),
        sa.Column("error_type", sa.String(200), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=False),
        sa.Column("raw_payload", sa.Text(), nullable=True),
        sa.Column(
            "retry_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "is_resolved",
            sa.Boolean(),
            nullable=False,
            server_default="false",
        ),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("news_article_dlq")
    op.drop_index("idx_news_tags", table_name="news_articles")
    op.drop_index("idx_news_source_published", table_name="news_articles")
    op.drop_index("idx_news_content_hash", table_name="news_articles")
    op.drop_index("idx_news_published_at", table_name="news_articles")
    op.drop_index("idx_news_source", table_name="news_articles")
    op.drop_table("news_articles")
