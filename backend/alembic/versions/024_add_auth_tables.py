"""인증 테이블 추가 (users, social_accounts, refresh_tokens)

Revision ID: 024_add_auth_tables
Revises: 024
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "024_add_auth_tables"
down_revision = "024"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # --- users ---
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("email", sa.String(255), unique=True, nullable=False),
        sa.Column("email_verified", sa.Boolean(), server_default="false"),
        sa.Column("hashed_password", sa.Text(), nullable=True),
        sa.Column("display_name", sa.String(100), nullable=True),
        sa.Column("avatar_url", sa.String(500), nullable=True),
        sa.Column("role", sa.String(20), server_default="user"),
        sa.Column("is_active", sa.Boolean(), server_default="true"),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
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
    op.create_index("idx_users_email", "users", ["email"])
    op.create_index("idx_users_role", "users", ["role"])

    # --- social_accounts ---
    op.create_table(
        "social_accounts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("provider", sa.String(50), nullable=False),
        sa.Column("provider_user_id", sa.String(255), nullable=False),
        sa.Column("provider_email", sa.String(255), nullable=True),
        sa.Column("provider_data", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "provider", "provider_user_id", name="uq_social_provider_user"
        ),
    )
    op.create_index("idx_social_accounts_user", "social_accounts", ["user_id"])

    # --- refresh_tokens ---
    op.create_table(
        "refresh_tokens",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("token_hash", sa.String(64), unique=True, nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("user_agent", sa.String(500), nullable=True),
        sa.Column("ip_address", sa.String(45), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )
    op.create_index("idx_refresh_tokens_user", "refresh_tokens", ["user_id"])
    op.create_index("idx_refresh_tokens_hash", "refresh_tokens", ["token_hash"])

    # --- identity_links에 user_id FK 추가 ---
    op.create_foreign_key(
        "fk_identity_links_user",
        "identity_links",
        "users",
        ["user_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint("fk_identity_links_user", "identity_links", type_="foreignkey")
    op.drop_table("refresh_tokens")
    op.drop_table("social_accounts")
    op.drop_table("users")
