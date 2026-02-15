"""refactor document tables: raw_data 제거, ai_summary 추가, 중복 인덱스 제거

Revision ID: 009
Revises: 008
Create Date: 2026-02-13

인제스트 파이프라인 리팩토링:
- precedent_documents: ai_summary 추가, raw_data 제거
- law_documents: ai_summary 추가, raw_data 제거, 중복 인덱스 제거
"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "009"
down_revision: Union[str, None] = "008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. precedent_documents: ai_summary 추가
    op.add_column(
        "precedent_documents",
        sa.Column(
            "ai_summary",
            sa.Text(),
            nullable=True,
            comment="AI 생성 판례요약",
        ),
    )

    # 2. precedent_documents: raw_data 삭제
    op.drop_column("precedent_documents", "raw_data")

    # 3. law_documents: ai_summary 추가
    op.add_column(
        "law_documents",
        sa.Column(
            "ai_summary",
            sa.Text(),
            nullable=True,
            comment="AI 생성 법령요약",
        ),
    )

    # 4. law_documents: raw_data 삭제
    op.drop_column("law_documents", "raw_data")

    # 5. law_documents: 중복 인덱스 제거
    #    law_name, law_type은 컬럼 정의에 index=True가 있어 자동 인덱스 보유
    #    ministry, enforcement_date는 별도 인덱스 불필요 (조회 패턴 변경)
    op.drop_index("idx_law_docs_name", table_name="law_documents")
    op.drop_index("idx_law_docs_type", table_name="law_documents")
    op.drop_index("idx_law_docs_ministry", table_name="law_documents")
    op.drop_index("idx_law_docs_enforcement", table_name="law_documents")


def downgrade() -> None:
    # 5. law_documents: 인덱스 복원
    op.create_index(
        "idx_law_docs_enforcement",
        "law_documents",
        ["enforcement_date"],
    )
    op.create_index(
        "idx_law_docs_ministry",
        "law_documents",
        ["ministry"],
    )
    op.create_index(
        "idx_law_docs_type",
        "law_documents",
        ["law_type"],
    )
    op.create_index(
        "idx_law_docs_name",
        "law_documents",
        ["law_name"],
    )

    # 4. law_documents: raw_data 복원
    op.add_column(
        "law_documents",
        sa.Column(
            "raw_data",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
            comment="원본 JSON 데이터 전체",
        ),
    )

    # 3. law_documents: ai_summary 제거
    op.drop_column("law_documents", "ai_summary")

    # 2. precedent_documents: raw_data 복원
    op.add_column(
        "precedent_documents",
        sa.Column(
            "raw_data",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
            comment="원본 JSON 데이터 전체",
        ),
    )

    # 1. precedent_documents: ai_summary 제거
    op.drop_column("precedent_documents", "ai_summary")
