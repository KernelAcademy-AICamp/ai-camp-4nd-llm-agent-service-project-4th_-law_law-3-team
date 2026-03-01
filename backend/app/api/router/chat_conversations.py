"""
대화 관리 API 라우터

대화 목록 조회, 상세 조회, 수정, 내보내기.
"""

import json
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import func, update

from app.core.database import async_session_factory
from app.models.chat_conversation import ChatConversation
from app.services.workspace.chat_persistence import ChatPersistenceService

router = APIRouter(prefix="/chat/conversations", tags=["chat-conversations"])


class ConversationListItem(BaseModel):
    """대화 목록 항목"""

    id: str
    title: str | None = None
    case_type: str | None = None
    customer_name: str | None = None
    last_agent: str | None = None
    tag_count: int = 0
    message_count: int = 0
    created_at: str | None = None
    updated_at: str | None = None


class ConversationListResponse(BaseModel):
    """대화 목록 응답"""

    items: list[ConversationListItem]
    total: int
    page: int
    page_size: int


class ConversationUpdateRequest(BaseModel):
    """대화 수정 요청"""

    title: str | None = None
    case_id: str | None = None
    customer_name: str | None = None


@router.get("", response_model=ConversationListResponse)
async def list_conversations(
    request: Request,
    case_id: str | None = Query(None),
    search: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=50),
) -> ConversationListResponse:
    """대화 목록 조회"""
    session_token: str = request.state.session_token

    case_uuid = uuid.UUID(case_id) if case_id else None

    async with async_session_factory() as db:
        conversations, total = await ChatPersistenceService.list_conversations(
            db, session_token, case_uuid, search, page, page_size
        )

        items: list[ConversationListItem] = []
        for conv in conversations:
            # 메시지 수 조회
            from sqlalchemy import select

            from app.models.chat_conversation import ChatMessage

            msg_count_result = await db.execute(
                select(func.count()).where(
                    ChatMessage.conversation_id == conv.id,
                )
            )
            message_count = msg_count_result.scalar_one()

            items.append(
                ConversationListItem(
                    id=str(conv.id),
                    title=conv.title,
                    case_type=conv.case_type,
                    customer_name=conv.customer_name,
                    last_agent=conv.last_agent,
                    tag_count=len(conv.tagged_items) if conv.tagged_items else 0,
                    message_count=message_count,
                    created_at=conv.created_at.isoformat() if conv.created_at else None,
                    updated_at=conv.updated_at.isoformat() if conv.updated_at else None,
                )
            )

    return ConversationListResponse(
        items=items, total=total, page=page, page_size=page_size
    )


@router.get("/{conversation_id}")
async def get_conversation(
    request: Request,
    conversation_id: str,
) -> dict[str, Any]:
    """대화 상세 조회 (메시지 포함)"""
    session_token: str = request.state.session_token

    async with async_session_factory() as db:
        conv = await ChatPersistenceService.get_conversation_with_messages(
            db, uuid.UUID(conversation_id), session_token
        )
        if not conv:
            raise HTTPException(status_code=404, detail="대화를 찾을 수 없습니다.")

        messages = [
            {
                "role": m.role,
                "content": m.content,
                "agent_type": m.agent_type,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in conv.messages
        ]

        return {
            "id": str(conv.id),
            "thread_id": conv.thread_id,
            "title": conv.title,
            "case_id": str(conv.case_id) if conv.case_id else None,
            "case_type": conv.case_type,
            "customer_name": conv.customer_name,
            "is_title_manual": conv.is_title_manual,
            "summary": conv.summary,
            "tagged_items": conv.tagged_items,
            "last_agent": conv.last_agent,
            "messages": messages,
            "created_at": conv.created_at.isoformat() if conv.created_at else None,
            "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
        }


@router.patch("/{conversation_id}")
async def update_conversation(
    request: Request,
    conversation_id: str,
    body: ConversationUpdateRequest,
) -> dict[str, Any]:
    """대화 수정 (제목, 사건 연결, 고객명)"""
    session_token: str = request.state.session_token

    async with async_session_factory() as db:
        from sqlalchemy import select

        result = await db.execute(
            select(ChatConversation).where(
                ChatConversation.id == uuid.UUID(conversation_id),
                ChatConversation.session_token == session_token,
            )
        )
        conv = result.scalar_one_or_none()
        if not conv:
            raise HTTPException(status_code=404, detail="대화를 찾을 수 없습니다.")

        update_values: dict[str, Any] = {}

        if body.title is not None:
            update_values["title"] = body.title
            update_values["is_title_manual"] = True

        if body.case_id is not None:
            update_values["case_id"] = uuid.UUID(body.case_id)

        if body.customer_name is not None:
            update_values["customer_name"] = body.customer_name

        if update_values:
            update_values["updated_at"] = func.now()
            await db.execute(
                update(ChatConversation)
                .where(ChatConversation.id == uuid.UUID(conversation_id))
                .values(**update_values)
            )
            await db.commit()

    # 최신 상태 반환
    return await get_conversation(request, conversation_id)


@router.get("/{conversation_id}/export")
async def export_conversation(
    request: Request,
    conversation_id: str,
    format: str = Query("json", regex="^(json|txt)$"),
) -> StreamingResponse:
    """대화 내보내기"""
    session_token: str = request.state.session_token

    async with async_session_factory() as db:
        conv = await ChatPersistenceService.get_conversation_with_messages(
            db, uuid.UUID(conversation_id), session_token
        )
        if not conv:
            raise HTTPException(status_code=404, detail="대화를 찾을 수 없습니다.")

        title_safe = (conv.title or "대화")[:50].replace("/", "_")

        if format == "json":
            data = {
                "title": conv.title,
                "case_type": conv.case_type,
                "summary": conv.summary,
                "tagged_items": conv.tagged_items,
                "messages": [
                    {
                        "role": m.role,
                        "content": m.content,
                        "agent_type": m.agent_type,
                        "created_at": m.created_at.isoformat() if m.created_at else None,
                    }
                    for m in conv.messages
                ],
            }
            content = json.dumps(data, ensure_ascii=False, indent=2)
            media_type = "application/json"
            filename = f"{title_safe}.json"
        else:
            lines: list[str] = [f"# {conv.title or '법률 상담'}\n"]
            if conv.case_type:
                lines.append(f"사건 유형: {conv.case_type}\n")
            lines.append("")
            for m in conv.messages:
                role_label = "사용자" if m.role == "user" else "상담원"
                lines.append(f"[{role_label}] {m.content}\n")
            content = "\n".join(lines)
            media_type = "text/plain; charset=utf-8"
            filename = f"{title_safe}.txt"

    return StreamingResponse(
        iter([content]),
        media_type=media_type,
        headers={
            "Content-Disposition": f"attachment; filename=\"{filename}\"",
        },
    )
