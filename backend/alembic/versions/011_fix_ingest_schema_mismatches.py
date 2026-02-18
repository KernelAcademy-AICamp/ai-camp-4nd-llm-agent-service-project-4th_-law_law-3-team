"""fix ingest schema mismatches: ORM-마이그레이션 칼럼 불일치 수정

Revision ID: 011
Revises: 010
Create Date: 2026-02-18

수정 내용:
- administration_documents: case_type_name → case_type (ORM 칼럼명에 맞춤)
- constitutional_documents: case_type_name → case_type (ORM 칼럼명에 맞춤)
- dec_civil_rights_documents: complaint_flag varchar(100) → varchar(200) (ORM 정의에 맞춤)
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "011"
down_revision: Union[str, None] = "010"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. administration_documents: case_type_name → case_type
    op.alter_column(
        "administration_documents",
        "case_type_name",
        new_column_name="case_type",
    )

    # 2. constitutional_documents: case_type_name → case_type
    op.alter_column(
        "constitutional_documents",
        "case_type_name",
        new_column_name="case_type",
    )

    # 3. dec_civil_rights_documents: complaint_flag varchar(100) → varchar(200)
    op.alter_column(
        "dec_civil_rights_documents",
        "complaint_flag",
        type_=sa.String(200),
        existing_type=sa.String(100),
    )


def downgrade() -> None:
    # 3. dec_civil_rights_documents: complaint_flag varchar(200) → varchar(100)
    op.alter_column(
        "dec_civil_rights_documents",
        "complaint_flag",
        type_=sa.String(100),
        existing_type=sa.String(200),
    )

    # 2. constitutional_documents: case_type → case_type_name
    op.alter_column(
        "constitutional_documents",
        "case_type",
        new_column_name="case_type_name",
    )

    # 1. administration_documents: case_type → case_type_name
    op.alter_column(
        "administration_documents",
        "case_type",
        new_column_name="case_type_name",
    )
