"""
대화 영속화 서비스

대화 생성/조회, 메시지 저장, 태그 서버사이드 누적.
"""

import uuid
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat_conversation import ChatConversation, ChatMessage

MAX_TAGGED_ITEMS = 50


def _merge_tags_dict(
    existing: list[dict[str, Any]],
    new_tags: list[dict[str, Any]],
    max_items: int = MAX_TAGGED_ITEMS,
) -> list[dict[str, Any]]:
    """dict 리스트 기반 태그 병합 (중복 제거 + 상한 적용)"""
    merged = list(existing)
    for tag in new_tags:
        is_duplicate = any(
            item.get("content") == tag.get("content")
            and item.get("tag_type") == tag.get("tag_type")
            for item in merged
        )
        if not is_duplicate:
            merged.append(tag)

    if len(merged) > max_items:
        merged.sort(key=lambda x: x.get("turn_index", 0))
        merged = merged[-max_items:]

    return merged


class ChatPersistenceService:
    """대화 영속화 서비스 — stateless static 메서드 집합"""

    @staticmethod
    async def get_or_create_conversation(
        db: AsyncSession,
        session_token: str,
        thread_id: str,
        conversation_id: str | None = None,
    ) -> ChatConversation:
        """conversation_id가 있으면 소유권 검증 후 조회, 없으면 생성"""
        if conversation_id:
            result = await db.execute(
                select(ChatConversation).where(
                    ChatConversation.id == uuid.UUID(conversation_id),
                    ChatConversation.session_token == session_token,
                )
            )
            conversation = result.scalar_one_or_none()
            if conversation:
                return conversation

        # thread_id로 기존 대화 조회
        result = await db.execute(
            select(ChatConversation).where(
                ChatConversation.thread_id == thread_id,
                ChatConversation.session_token == session_token,
            )
        )
        conversation = result.scalar_one_or_none()
        if conversation:
            return conversation

        # 신규 생성
        conversation = ChatConversation(
            session_token=session_token,
            thread_id=thread_id,
        )
        db.add(conversation)
        await db.flush()
        return conversation

    @staticmethod
    async def save_message(
        db: AsyncSession,
        conversation_id: uuid.UUID,
        role: str,
        content: str,
        agent_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage:
        """메시지 저장"""
        message = ChatMessage(
            conversation_id=conversation_id,
            role=role,
            content=content,
            agent_type=agent_type,
            metadata_=metadata,
        )
        db.add(message)
        await db.flush()
        return message

    @staticmethod
    async def append_tags(
        db: AsyncSession,
        conversation_id: uuid.UUID,
        new_tags: list[dict[str, Any]],
    ) -> None:
        """태그를 DB에 직접 누적 (서버사이드, merge_tags로 중복 제거)"""
        if not new_tags:
            return

        result = await db.execute(
            select(ChatConversation.tagged_items).where(
                ChatConversation.id == conversation_id,
            )
        )
        existing_raw = result.scalar_one_or_none() or []

        merged = _merge_tags_dict(existing_raw, new_tags)

        await db.execute(
            update(ChatConversation)
            .where(ChatConversation.id == conversation_id)
            .values(tagged_items=merged, updated_at=func.now())
        )

    @staticmethod
    async def update_summary(
        db: AsyncSession,
        conversation_id: uuid.UUID,
        summary: dict[str, Any],
    ) -> None:
        """구조화 요약 업데이트"""
        await db.execute(
            update(ChatConversation)
            .where(ChatConversation.id == conversation_id)
            .values(summary=summary, updated_at=func.now())
        )

    @staticmethod
    async def update_last_agent(
        db: AsyncSession,
        conversation_id: uuid.UUID,
        agent_type: str,
    ) -> None:
        """마지막 사용 에이전트 업데이트"""
        await db.execute(
            update(ChatConversation)
            .where(ChatConversation.id == conversation_id)
            .values(last_agent=agent_type, updated_at=func.now())
        )

    @staticmethod
    async def list_conversations(
        db: AsyncSession,
        session_token: str,
        case_id: uuid.UUID | None = None,
        search: str | None = None,
        page: int = 1,
        page_size: int = 20,
        agent: str | None = None,
        session_tokens: list[str] | None = None,
    ) -> tuple[list[ChatConversation], int]:
        """대화 목록 조회 (페이지네이션)"""
        tokens = session_tokens or [session_token]
        query = select(ChatConversation).where(
            ChatConversation.session_token.in_(tokens),
        )

        if case_id:
            query = query.where(ChatConversation.case_id == case_id)

        if search:
            query = query.where(ChatConversation.title.ilike(f"%{search}%"))

        if agent:
            query = query.where(ChatConversation.last_agent == agent)

        # 전체 수
        count_result = await db.execute(
            select(func.count()).select_from(query.subquery())
        )
        total = count_result.scalar_one()

        # 페이지네이션
        query = (
            query.order_by(ChatConversation.updated_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        result = await db.execute(query)
        conversations = list(result.scalars().all())

        return conversations, total

    @staticmethod
    async def get_conversation_with_messages(
        db: AsyncSession,
        conversation_id: uuid.UUID,
        session_token: str,
    ) -> ChatConversation | None:
        """대화 + 메시지 조회 (소유권 검증)"""
        result = await db.execute(
            select(ChatConversation).where(
                ChatConversation.id == conversation_id,
                ChatConversation.session_token == session_token,
            )
        )
        conversation = result.scalar_one_or_none()
        if not conversation:
            return None

        # 메시지 조회
        msg_result = await db.execute(
            select(ChatMessage)
            .where(ChatMessage.conversation_id == conversation_id)
            .order_by(ChatMessage.created_at)
        )
        conversation.messages = list(msg_result.scalars().all())
        return conversation

    @staticmethod
    async def get_tags(
        db: AsyncSession,
        conversation_id: uuid.UUID,
    ) -> list[dict[str, Any]]:
        """대화 태그 조회"""
        result = await db.execute(
            select(ChatConversation.tagged_items).where(
                ChatConversation.id == conversation_id,
            )
        )
        return result.scalar_one_or_none() or []
