"""admin_rule_documents: v3 신규 컬럼 4개 추가

Revision ID: 014
Revises: 013
Create Date: 2026-02-19

수정 내용:
- admin_rule_documents 테이블에 v3 신규 필드 4개 추가
  - promulgation_date (발령일자)
  - enforcement_date (시행일자)
  - ministry_code (소관부처코드)
  - parent_ministry (상위부처명)
"""

import sqlalchemy as sa

from alembic import op

revision = "014"
down_revision = "013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "admin_rule_documents",
        sa.Column(
            "promulgation_date",
            sa.String(20),
            nullable=True,
            comment="발령일자",
        ),
    )
    op.add_column(
        "admin_rule_documents",
        sa.Column(
            "enforcement_date",
            sa.String(20),
            nullable=True,
            comment="시행일자",
        ),
    )
    op.add_column(
        "admin_rule_documents",
        sa.Column(
            "ministry_code",
            sa.String(50),
            nullable=True,
            comment="소관부처코드",
        ),
    )
    op.add_column(
        "admin_rule_documents",
        sa.Column(
            "parent_ministry",
            sa.String(200),
            nullable=True,
            comment="상위부처명",
        ),
    )


def downgrade() -> None:
    op.drop_column("admin_rule_documents", "parent_ministry")
    op.drop_column("admin_rule_documents", "ministry_code")
    op.drop_column("admin_rule_documents", "enforcement_date")
    op.drop_column("admin_rule_documents", "promulgation_date")
