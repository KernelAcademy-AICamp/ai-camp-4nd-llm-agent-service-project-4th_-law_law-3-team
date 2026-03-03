"""
대화 영속화 모델 — chat_conversations + chat_messages
"""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class ChatConversation(Base):  # type: ignore[misc]
    """대화 세션 메타데이터 (state 정본은 LangGraph Checkpointer)"""

    __tablename__ = "chat_conversations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    thread_id: Mapped[str] = mapped_column(String(100), nullable=False)
    session_token: Mapped[str] = mapped_column(String(100), nullable=False)
    case_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("workspace_cases.id"), nullable=True
    )
    title: Mapped[str | None] = mapped_column(String(200), nullable=True)
    case_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    customer_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    is_title_manual: Mapped[bool] = mapped_column(Boolean, default=False)
    summary: Mapped[dict | None] = mapped_column(JSONB, nullable=True)  # type: ignore[type-arg]
    tagged_items: Mapped[list] = mapped_column(JSONB, default=list)  # type: ignore[type-arg]
    last_agent: Mapped[str | None] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_chat_conversations_session", "session_token"),
        Index(
            "idx_chat_conversations_tagged_items",
            "tagged_items",
            postgresql_using="gin",
        ),
        Index("idx_chat_conversations_thread", "thread_id", unique=True),
    )


class ChatMessage(Base):  # type: ignore[misc]
    """대화 메시지 (사용자/어시스턴트)"""

    __tablename__ = "chat_messages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chat_conversations.id"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(10), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    agent_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB, nullable=True)  # type: ignore[type-arg]
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    conversation: Mapped["ChatConversation"] = relationship(back_populates="messages")

    __table_args__ = (
        Index("idx_chat_messages_conversation", "conversation_id", "created_at"),
    )
