"""lawyer_personas + lawyer_persona_feedback 테이블 추가

Revision ID: 019
Revises: 018
Create Date: 2026-02-22

수정 내용:
- lawyer_personas 테이블 생성 (변호사 페르소나 v2.0)
- lawyer_persona_feedback 테이블 생성 (대본 피드백)
- 인덱스: user_id, persona_id
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = "019"
down_revision = "018"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # lawyer_personas 테이블
    op.create_table(
        "lawyer_personas",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(255), nullable=False, unique=True),
        sa.Column("specialty_areas", postgresql.JSONB(), nullable=False),
        sa.Column(
            "focus_topics",
            postgresql.JSONB(),
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "preferred_tone",
            sa.String(50),
            nullable=False,
            server_default="professional",
        ),
        sa.Column(
            "target_audience",
            sa.String(50),
            nullable=False,
            server_default="general_public",
        ),
        sa.Column("channel_style", sa.String(50), nullable=True),
        sa.Column("source", sa.String(10), nullable=False),
        sa.Column("confidence", sa.Float(), server_default=sa.text("1.0")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "idx_lawyer_personas_user_id", "lawyer_personas", ["user_id"],
    )

    # lawyer_persona_feedback 테이블
    op.create_table(
        "lawyer_persona_feedback",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("persona_id", sa.String(36), nullable=False),
        sa.Column("script_id", sa.String(36), nullable=True),
        sa.Column("rating", sa.Float(), nullable=False),
        sa.Column("feedback_type", sa.String(50), nullable=True),
        sa.Column("feedback_text", sa.String(1000), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "idx_persona_feedback_persona_id",
        "lawyer_persona_feedback",
        ["persona_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "idx_persona_feedback_persona_id",
        table_name="lawyer_persona_feedback",
    )
    op.drop_table("lawyer_persona_feedback")
    op.drop_index(
        "idx_lawyer_personas_user_id", table_name="lawyer_personas",
    )
    op.drop_table("lawyer_personas")
