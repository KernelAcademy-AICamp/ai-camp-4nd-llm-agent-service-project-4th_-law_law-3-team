"""local_ordinance_documents 테이블 추가

Revision ID: 015
Revises: 014
Create Date: 2026-02-20

수정 내용:
- local_ordinance_documents 테이블 생성 (자치법규 160,276건)
- ordinance_id UNIQUE 인덱스
"""

import sqlalchemy as sa

from alembic import op

revision = "015"
down_revision = "014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "local_ordinance_documents",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "ordinance_id",
            sa.String(length=20),
            nullable=False,
            comment="자치법규ID",
        ),
        sa.Column(
            "ordinance_serial",
            sa.String(length=20),
            nullable=True,
            comment="자치법규일련번호",
        ),
        sa.Column(
            "ordinance_name",
            sa.Text(),
            nullable=False,
            comment="자치법규명",
        ),
        sa.Column(
            "local_government",
            sa.String(length=100),
            nullable=True,
            comment="지자체기관명",
        ),
        sa.Column(
            "overall_summary",
            sa.Text(),
            nullable=True,
            comment="전체요약 (Basic)",
        ),
        sa.Column(
            "content",
            sa.Text(),
            nullable=True,
            comment="조문 전체 텍스트 (concat)",
        ),
        sa.Column(
            "supplementary",
            sa.Text(),
            nullable=True,
            comment="부칙내용",
        ),
        sa.Column(
            "ai_summary",
            sa.Text(),
            nullable=True,
            comment="AI 생성 요약 (= overall_summary, 인제스트 호환)",
        ),
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
        sa.UniqueConstraint("ordinance_id"),
    )
    op.create_index(
        "ix_local_ordinance_documents_ordinance_id",
        "local_ordinance_documents",
        ["ordinance_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_local_ordinance_documents_ordinance_id",
        table_name="local_ordinance_documents",
    )
    op.drop_table("local_ordinance_documents")
