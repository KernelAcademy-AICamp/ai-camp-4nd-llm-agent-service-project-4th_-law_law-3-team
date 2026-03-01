"""workspace_cases, chat_conversations, chat_messages 등 6개 테이블 추가

Revision ID: 020
Revises: 019
Create Date: 2026-03-01

수정 내용:
- workspace_cases 테이블 (사건 워크스페이스)
- chat_conversations 테이블 (대화 메타데이터)
- chat_messages 테이블 (대화 메시지)
- workspace_case_timeline_items 테이블 (타임라인)
- workspace_activity_logs 테이블 (활동 로그)
- identity_links 테이블 (세션-계정 연결)
- 관련 인덱스 10개
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "020"
down_revision = "019"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. workspace_cases (독립)
    op.create_table(
        "workspace_cases",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("session_token", sa.String(100), nullable=False),
        sa.Column("case_name", sa.String(200), nullable=False),
        sa.Column("case_type", sa.String(50), nullable=True),
        sa.Column("status", sa.String(20), server_default="active", nullable=False),
        sa.Column("summary", postgresql.JSONB(), nullable=True),
        sa.Column(
            "tagged_items", postgresql.JSONB(), server_default="[]", nullable=False
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "idx_workspace_cases_session", "workspace_cases", ["session_token"]
    )
    op.create_index(
        "idx_workspace_cases_tagged_items",
        "workspace_cases",
        ["tagged_items"],
        postgresql_using="gin",
    )

    # 2. chat_conversations (FK → workspace_cases)
    op.create_table(
        "chat_conversations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("thread_id", sa.String(100), nullable=False),
        sa.Column("session_token", sa.String(100), nullable=False),
        sa.Column(
            "case_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("workspace_cases.id"),
            nullable=True,
        ),
        sa.Column("title", sa.String(200), nullable=True),
        sa.Column("case_type", sa.String(50), nullable=True),
        sa.Column("customer_name", sa.String(100), nullable=True),
        sa.Column(
            "is_title_manual", sa.Boolean(), server_default="false", nullable=False
        ),
        sa.Column("summary", postgresql.JSONB(), nullable=True),
        sa.Column(
            "tagged_items", postgresql.JSONB(), server_default="[]", nullable=False
        ),
        sa.Column("last_agent", sa.String(50), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "idx_chat_conversations_session", "chat_conversations", ["session_token"]
    )
    op.create_index(
        "idx_chat_conversations_tagged_items",
        "chat_conversations",
        ["tagged_items"],
        postgresql_using="gin",
    )
    op.create_index(
        "idx_chat_conversations_thread",
        "chat_conversations",
        ["thread_id"],
        unique=True,
    )

    # 3. chat_messages (FK → chat_conversations)
    op.create_table(
        "chat_messages",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "conversation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("chat_conversations.id"),
            nullable=False,
        ),
        sa.Column("role", sa.String(10), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("agent_type", sa.String(50), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "idx_chat_messages_conversation",
        "chat_messages",
        ["conversation_id", "created_at"],
    )

    # 4. workspace_case_timeline_items (FK → workspace_cases)
    op.create_table(
        "workspace_case_timeline_items",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "case_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("workspace_cases.id"),
            nullable=False,
        ),
        sa.Column("date_text", sa.String(50), nullable=True),
        sa.Column("date_normalized", sa.Date(), nullable=True),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("category", sa.String(50), nullable=True),
        sa.Column("source_type", sa.String(20), nullable=True),
        sa.Column("source_ref", postgresql.JSONB(), nullable=True),
        sa.Column("sort_order", sa.Integer(), server_default="0", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "idx_timeline_items_case",
        "workspace_case_timeline_items",
        ["case_id", "sort_order"],
    )

    # 5. workspace_activity_logs (FK → workspace_cases, chat_conversations)
    op.create_table(
        "workspace_activity_logs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "case_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("workspace_cases.id"),
            nullable=True,
        ),
        sa.Column(
            "conversation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("chat_conversations.id"),
            nullable=True,
        ),
        sa.Column("session_token", sa.String(100), nullable=False),
        sa.Column("action", sa.String(50), nullable=False),
        sa.Column("detail", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "idx_activity_logs_session",
        "workspace_activity_logs",
        ["session_token"],
    )

    # 6. identity_links (독립)
    op.create_table(
        "identity_links",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("session_token", sa.String(100), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("provider", sa.String(50), nullable=True),
        sa.Column(
            "linked_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "uq_identity_links_session_provider",
        "identity_links",
        ["session_token", "provider"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_table("identity_links")
    op.drop_table("workspace_activity_logs")
    op.drop_table("workspace_case_timeline_items")
    op.drop_table("chat_messages")
    op.drop_table("chat_conversations")
    op.drop_table("workspace_cases")
