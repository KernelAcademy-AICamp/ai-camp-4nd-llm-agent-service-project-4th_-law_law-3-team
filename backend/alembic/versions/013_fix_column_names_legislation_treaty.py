"""fix column names: legislation rename + treaty drop/recreate

Revision ID: 013
Revises: 012
Create Date: 2026-02-19

수정 내용:
- legislation_documents: interpretation_agency_name → interpretation_agency,
                         inquiry_agency_name → inquiry_agency
- treaty_documents: ORM과 불일치하는 컬럼이 많아 drop 후 ORM 기준으로 재생성
  (데이터 0건이므로 안전)
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "013"
down_revision: Union[str, None] = "012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── legislation_documents: 컬럼명 rename ──
    op.alter_column(
        "legislation_documents",
        "interpretation_agency_name",
        new_column_name="interpretation_agency",
    )
    op.alter_column(
        "legislation_documents",
        "inquiry_agency_name",
        new_column_name="inquiry_agency",
    )

    # ── treaty_documents: drop 후 ORM 기준 재생성 (0건) ──
    op.drop_table("treaty_documents")

    op.create_table(
        "treaty_documents",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("serial_number", sa.String(100), nullable=False),
        sa.Column("treaty_number", sa.String(50), nullable=True),
        sa.Column("treaty_name_kr", sa.Text(), nullable=False),
        sa.Column("treaty_name_en", sa.Text(), nullable=True),
        sa.Column("treaty_type_code", sa.String(20), nullable=True),
        # 국가 정보
        sa.Column("counterpart_country", sa.String(200), nullable=True),
        sa.Column("counterpart_country_kr", sa.String(200), nullable=True),
        sa.Column("country_code", sa.String(20), nullable=True),
        # 분야
        sa.Column("bilateral_field_code", sa.String(20), nullable=True),
        sa.Column("bilateral_field", sa.String(100), nullable=True),
        # 일자 정보
        sa.Column("signing_date", sa.Date(), nullable=True),
        sa.Column("signing_place", sa.String(200), nullable=True),
        sa.Column("effective_date", sa.Date(), nullable=True),
        sa.Column("parliament_approval", sa.String(10), nullable=True),
        sa.Column("parliament_approval_date", sa.Date(), nullable=True),
        sa.Column("cabinet_review_date", sa.Date(), nullable=True),
        sa.Column("cabinet_review_session", sa.String(20), nullable=True),
        sa.Column("presidential_approval_date", sa.Date(), nullable=True),
        sa.Column("gazette_date", sa.Date(), nullable=True),
        # 내용
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("ai_summary", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_treaty_documents_serial_number",
        "treaty_documents",
        ["serial_number"],
        unique=True,
    )
    op.create_index(
        "ix_treaty_documents_treaty_number",
        "treaty_documents",
        ["treaty_number"],
    )
    op.create_index(
        "ix_treaty_documents_counterpart_country_kr",
        "treaty_documents",
        ["counterpart_country_kr"],
    )
    op.create_index(
        "ix_treaty_documents_bilateral_field",
        "treaty_documents",
        ["bilateral_field"],
    )
    op.create_index(
        "ix_treaty_documents_signing_date",
        "treaty_documents",
        ["signing_date"],
    )


def downgrade() -> None:
    # legislation: revert rename
    op.alter_column(
        "legislation_documents",
        "interpretation_agency",
        new_column_name="interpretation_agency_name",
    )
    op.alter_column(
        "legislation_documents",
        "inquiry_agency",
        new_column_name="inquiry_agency_name",
    )

    # treaty: drop recreated table and restore old schema
    op.drop_table("treaty_documents")
    op.create_table(
        "treaty_documents",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("serial_number", sa.String(100), nullable=False),
        sa.Column("treaty_number", sa.String(50), nullable=True),
        sa.Column("treaty_name_kr", sa.Text(), nullable=False),
        sa.Column("treaty_name_en", sa.Text(), nullable=True),
        sa.Column("treaty_type_code", sa.String(20), nullable=True),
        sa.Column("country_code", sa.String(20), nullable=True),
        sa.Column("country_name", sa.String(200), nullable=True),
        sa.Column("bilateral_type_code", sa.String(20), nullable=True),
        sa.Column("bilateral_type_name", sa.String(100), nullable=True),
        sa.Column("signing_date", sa.Date(), nullable=True),
        sa.Column("signing_place", sa.String(200), nullable=True),
        sa.Column("effective_date", sa.Date(), nullable=True),
        sa.Column("proclamation_number", sa.String(50), nullable=True),
        sa.Column("proclamation_date", sa.Date(), nullable=True),
        sa.Column("parliament_consent", sa.String(10), nullable=True),
        sa.Column("parliament_consent_date", sa.Date(), nullable=True),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("ai_summary", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_treaty_documents_serial_number",
        "treaty_documents",
        ["serial_number"],
        unique=True,
    )
