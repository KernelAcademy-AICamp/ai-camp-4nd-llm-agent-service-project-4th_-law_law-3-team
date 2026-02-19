"""alter case_number varchar(100) → text (precedent_documents + fts_index)

Revision ID: 012
Revises: 011
Create Date: 2026-02-19

수정 내용:
- precedent_documents.case_number: varchar(100) → text
- fts_index.case_number: varchar(100) → text
  병합사건의 경우 사건번호가 수백 자에 달하므로 text로 변경
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "012"
down_revision: Union[str, None] = "011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # precedent_documents.case_number
    op.execute(
        "DROP INDEX IF EXISTS ix_precedent_documents_case_number"
    )
    op.alter_column(
        "precedent_documents",
        "case_number",
        existing_type=sa.String(100),
        type_=sa.Text(),
        existing_nullable=True,
    )

    # fts_index.case_number
    op.alter_column(
        "fts_index",
        "case_number",
        existing_type=sa.String(100),
        type_=sa.Text(),
        existing_nullable=True,
    )


def downgrade() -> None:
    op.alter_column(
        "precedent_documents",
        "case_number",
        existing_type=sa.Text(),
        type_=sa.String(100),
        existing_nullable=True,
    )
    op.create_index(
        "ix_precedent_documents_case_number",
        "precedent_documents",
        ["case_number"],
    )
    op.alter_column(
        "fts_index",
        "case_number",
        existing_type=sa.Text(),
        type_=sa.String(100),
        existing_nullable=True,
    )
