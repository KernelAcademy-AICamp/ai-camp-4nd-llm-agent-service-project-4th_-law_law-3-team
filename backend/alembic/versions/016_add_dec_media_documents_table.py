"""dec_media_documents 테이블 추가

Revision ID: 016
Revises: 015
Create Date: 2026-02-24

수정 내용:
- dec_media_documents 테이블 생성 (방송미디어통신위원회 결정례 811건)
- serial_number UNIQUE 인덱스
"""

import sqlalchemy as sa

from alembic import op

revision = "016"
down_revision = "015"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "dec_media_documents",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "serial_number",
            sa.String(100),
            nullable=False,
            comment="결정문 일련번호",
        ),
        sa.Column("case_name", sa.Text(), nullable=True, comment="안건명"),
        sa.Column(
            "case_number", sa.String(200), nullable=True, comment="사건번호"
        ),
        sa.Column(
            "decision_date", sa.String(50), nullable=True, comment="의결일자"
        ),
        sa.Column("ruling", sa.Text(), nullable=True, comment="주문"),
        sa.Column("ai_summary", sa.Text(), nullable=True, comment="AI 생성 요약"),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=True,
            comment="레코드 생성일시",
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            nullable=True,
            comment="레코드 수정일시",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_dec_media_documents_serial_number",
        "dec_media_documents",
        ["serial_number"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_dec_media_documents_serial_number",
        table_name="dec_media_documents",
    )
    op.drop_table("dec_media_documents")
