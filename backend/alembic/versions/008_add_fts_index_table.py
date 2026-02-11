"""add fts_index table

Revision ID: 008
Revises: 007
Create Date: 2026-02-10

하이브리드 검색용 FTS 인덱스 테이블 (문서 단위)
- fts_index 테이블: 원문 기반 tsvector (검색 인덱스 전용, content 미저장)
- GIN 인덱스: content_tsvector (키워드 검색)
- B-tree 인덱스: data_type
"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import TSVECTOR

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "008"
down_revision: Union[str, None] = "007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "fts_index",
        sa.Column(
            "source_id",
            sa.String(length=100),
            nullable=False,
            comment="원본 문서 ID",
        ),
        sa.Column(
            "data_type",
            sa.String(length=20),
            nullable=False,
            comment="문서 유형 (법령/판례)",
        ),
        sa.Column(
            "title",
            sa.Text(),
            nullable=False,
            server_default="",
            comment="법령명 또는 사건명",
        ),
        sa.Column(
            "date",
            sa.String(length=20),
            nullable=True,
            comment="시행일 또는 선고일",
        ),
        sa.Column(
            "source_name",
            sa.String(length=200),
            nullable=True,
            comment="소관부처 또는 법원명",
        ),
        sa.Column(
            "case_number",
            sa.String(length=100),
            nullable=True,
            comment="판례 사건번호",
        ),
        sa.Column(
            "content_tsvector",
            TSVECTOR(),
            nullable=True,
            comment="MeCab 기반 tsvector",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=True,
            comment="레코드 생성일시",
        ),
        sa.PrimaryKeyConstraint("source_id"),
        comment="하이브리드 검색용 FTS 인덱스 (문서 단위, content 미저장)",
    )

    # GIN 인덱스 (FTS 검색)
    op.create_index(
        "idx_fts_index_content_tsvector",
        "fts_index",
        ["content_tsvector"],
        postgresql_using="gin",
    )

    # B-tree 인덱스
    op.create_index("idx_fts_index_data_type", "fts_index", ["data_type"])


def downgrade() -> None:
    op.drop_index("idx_fts_index_data_type", table_name="fts_index")
    op.drop_index(
        "idx_fts_index_content_tsvector",
        table_name="fts_index",
        postgresql_using="gin",
    )
    op.drop_table("fts_index")
