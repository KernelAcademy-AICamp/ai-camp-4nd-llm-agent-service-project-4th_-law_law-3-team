"""fts_index에 search_text 컬럼 추가 + GIN 인덱스 제거

BM25 마이그레이션 Layer 2: pg_textsearch BM25 인덱스 대상 컬럼 추가.
- search_text(Text): MeCab 전처리된 공백 구분 토큰 텍스트
- 기존 content_tsvector GIN 인덱스 제거 (BM25 인덱스로 대체)
- content_tsvector 컬럼은 롤백 안전용으로 유지
- BM25 인덱스는 데이터 적재 후 Layer 6에서 생성 (INSERT 성능 보호)

Revision ID: 020
Revises: 019
Create Date: 2026-03-01
"""

import sqlalchemy as sa
from alembic import op

revision = "020"
down_revision = "019"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. search_text 컬럼 추가
    op.add_column(
        "fts_index",
        sa.Column(
            "search_text",
            sa.Text(),
            nullable=True,
            comment="MeCab 전처리된 공백 구분 토큰 텍스트 (BM25 인덱스 대상)",
        ),
    )

    # 2. 기존 GIN 인덱스 제거 (BM25 인덱스로 대체)
    op.drop_index("idx_fts_index_content_tsvector", table_name="fts_index")


def downgrade() -> None:
    # 1. GIN 인덱스 복원
    op.create_index(
        "idx_fts_index_content_tsvector",
        "fts_index",
        ["content_tsvector"],
        postgresql_using="gin",
    )

    # 2. search_text 컬럼 제거
    op.drop_column("fts_index", "search_text")
