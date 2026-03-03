"""
워크스페이스 모델 — workspace_cases + timeline_items + activity_logs + identity_links
"""

import uuid
from datetime import date, datetime

from sqlalchemy import (
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class WorkspaceCase(Base):
    """사건 워크스페이스"""

    __tablename__ = "workspace_cases"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_token: Mapped[str] = mapped_column(String(100), nullable=False)
    case_name: Mapped[str] = mapped_column(String(200), nullable=False)
    case_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="active")
    summary: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    tagged_items: Mapped[list] = mapped_column(JSONB, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    timeline_items: Mapped[list["WorkspaceCaseTimelineItem"]] = relationship(
        back_populates="case", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_workspace_cases_session", "session_token"),
        Index(
            "idx_workspace_cases_tagged_items",
            "tagged_items",
            postgresql_using="gin",
        ),
    )


class WorkspaceCaseTimelineItem(Base):
    """타임라인 항목"""

    __tablename__ = "workspace_case_timeline_items"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    case_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("workspace_cases.id"), nullable=False
    )
    date_text: Mapped[str | None] = mapped_column(String(50), nullable=True)
    date_normalized: Mapped[date | None] = mapped_column(Date, nullable=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    category: Mapped[str | None] = mapped_column(String(50), nullable=True)
    source_type: Mapped[str | None] = mapped_column(String(20), nullable=True)
    source_ref: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    sort_order: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    case: Mapped["WorkspaceCase"] = relationship(back_populates="timeline_items")

    __table_args__ = (
        Index("idx_timeline_items_case", "case_id", "sort_order"),
    )


class WorkspaceActivityLog(Base):
    """활동 로그"""

    __tablename__ = "workspace_activity_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    case_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("workspace_cases.id"), nullable=True
    )
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chat_conversations.id"), nullable=True
    )
    session_token: Mapped[str] = mapped_column(String(100), nullable=False)
    action: Mapped[str] = mapped_column(String(50), nullable=False)
    detail: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("idx_activity_logs_session", "session_token"),
    )


class IdentityLink(Base):
    """세션-계정 연결 (향후 로그인 연동 대비)"""

    __tablename__ = "identity_links"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_token: Mapped[str] = mapped_column(String(100), nullable=False)
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    provider: Mapped[str | None] = mapped_column(String(50), nullable=True)
    linked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index(
            "uq_identity_links_session_provider",
            "session_token",
            "provider",
            unique=True,
        ),
    )
