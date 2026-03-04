"""news_articles에 search_text 컬럼 + BM25 인덱스 추가

뉴스 검색을 LanceDB → PostgreSQL BM25로 전환하기 위한 마이그레이션.
search_text 컬럼에 MeCab 토크나이징된 텍스트를 저장하고
pg_textsearch BM25 인덱스를 생성한다.

Revision ID: 024
Revises: 023
Create Date: 2026-03-04

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "024"
down_revision: Union[str, None] = "023"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. search_text 컬럼 추가
    op.add_column(
        "news_articles",
        sa.Column(
            "search_text",
            sa.Text(),
            nullable=True,
            comment="MeCab 토크나이징된 검색용 텍스트 (BM25)",
        ),
    )

    # 2. BM25 인덱스 생성 (pg_textsearch)
    # 기존 fts_index 테이블의 idx_fts_bm25 패턴과 동일
    op.execute(
        "CREATE INDEX idx_news_bm25 ON news_articles "
        "USING bm25(search_text) WITH (text_config='simple')"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_news_bm25")
    op.drop_column("news_articles", "search_text")
