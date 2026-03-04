"""
워크스페이스 사건 서비스

사건 CRUD, 대화 연결, 태그 병합.
"""

import uuid
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat_conversation import ChatConversation
from app.models.workspace_case import (
    IdentityLink,
    WorkspaceCase,
    WorkspaceCaseTimelineItem,
)
from app.services.workspace.activity_logger import ActivityLogger
from app.services.workspace.chat_persistence import _merge_tags_dict


class WorkspaceCaseService:
    """사건 워크스페이스 서비스"""

    @staticmethod
    async def _get_session_tokens(
        db: AsyncSession,
        session_token: str,
        user_id: uuid.UUID | None = None,
    ) -> list[str]:
        """인증 사용자는 모든 연결된 세션 토큰 목록, 익명이면 단일 토큰 반환"""
        if not user_id:
            return [session_token]

        result = await db.execute(
            select(IdentityLink.session_token).where(
                IdentityLink.user_id == user_id,
            )
        )
        tokens = [row for row in result.scalars().all()]
        # 현재 세션도 포함 (아직 마이그레이션 안 된 경우)
        if session_token not in tokens:
            tokens.append(session_token)
        return tokens

    @staticmethod
    async def create_case(
        db: AsyncSession,
        session_token: str,
        case_name: str,
        case_type: str | None = None,
        conversation_ids: list[str] | None = None,
    ) -> WorkspaceCase:
        """사건 생성 + 대화 연결 + 태그 병합"""
        case = WorkspaceCase(
            session_token=session_token,
            case_name=case_name,
            case_type=case_type,
        )
        db.add(case)
        await db.flush()

        # 대화 연결 + 태그 병합
        if conversation_ids:
            merged_tags: list[dict[str, Any]] = []
            for cid in conversation_ids:
                result = await db.execute(
                    select(ChatConversation).where(
                        ChatConversation.id == uuid.UUID(cid),
                        ChatConversation.session_token == session_token,
                    )
                )
                conv = result.scalar_one_or_none()
                if conv:
                    conv.case_id = case.id
                    if conv.tagged_items:
                        merged_tags = _merge_tags_dict(merged_tags, conv.tagged_items)

            if merged_tags:
                case.tagged_items = merged_tags

        # 활동 로그
        await ActivityLogger.log(
            db, session_token, "create_case",
            case_id=case.id,
            detail={"case_name": case_name},
        )

        return case

    @staticmethod
    async def list_cases(
        db: AsyncSession,
        session_token: str,
        status: str = "active",
        search: str | None = None,
        page: int = 1,
        page_size: int = 20,
        user_id: uuid.UUID | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """사건 목록 조회 (카운트 포함)"""
        tokens = await WorkspaceCaseService._get_session_tokens(
            db, session_token, user_id
        )
        query = select(WorkspaceCase).where(
            WorkspaceCase.session_token.in_(tokens),
        )
        if status:
            query = query.where(WorkspaceCase.status == status)
        if search:
            query = query.where(WorkspaceCase.case_name.ilike(f"%{search}%"))

        # 전체 수
        count_result = await db.execute(
            select(func.count()).select_from(query.subquery())
        )
        total = count_result.scalar_one()

        # 페이지네이션
        query = (
            query.order_by(WorkspaceCase.updated_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        result = await db.execute(query)
        cases = list(result.scalars().all())

        # 각 사건에 대한 집계 정보 추가
        items: list[dict[str, Any]] = []
        for case in cases:
            conv_count_result = await db.execute(
                select(func.count()).where(
                    ChatConversation.case_id == case.id,
                )
            )
            conversation_count = conv_count_result.scalar_one()

            timeline_count_result = await db.execute(
                select(func.count()).where(
                    WorkspaceCaseTimelineItem.case_id == case.id,
                )
            )
            timeline_item_count = timeline_count_result.scalar_one()

            items.append({
                "id": str(case.id),
                "case_name": case.case_name,
                "case_type": case.case_type,
                "status": case.status,
                "conversation_count": conversation_count,
                "timeline_item_count": timeline_item_count,
                "tag_count": len(case.tagged_items) if case.tagged_items else 0,
                "created_at": case.created_at.isoformat() if case.created_at else None,
                "updated_at": case.updated_at.isoformat() if case.updated_at else None,
            })

        return items, total

    @staticmethod
    async def get_case_detail(
        db: AsyncSession,
        case_id: uuid.UUID,
        session_token: str,
        user_id: uuid.UUID | None = None,
    ) -> dict[str, Any] | None:
        """사건 상세 조회 (대화 + 타임라인 포함)"""
        tokens = await WorkspaceCaseService._get_session_tokens(
            db, session_token, user_id
        )
        result = await db.execute(
            select(WorkspaceCase).where(
                WorkspaceCase.id == case_id,
                WorkspaceCase.session_token.in_(tokens),
            )
        )
        case = result.scalar_one_or_none()
        if not case:
            return None

        # 연결된 대화 목록
        conv_result = await db.execute(
            select(ChatConversation)
            .where(ChatConversation.case_id == case_id)
            .order_by(ChatConversation.updated_at.desc())
        )
        conversations = [
            {
                "id": str(c.id),
                "title": c.title,
                "case_type": c.case_type,
                "last_agent": c.last_agent,
                "created_at": c.created_at.isoformat() if c.created_at else None,
                "updated_at": c.updated_at.isoformat() if c.updated_at else None,
            }
            for c in conv_result.scalars().all()
        ]

        # 타임라인 항목
        timeline_result = await db.execute(
            select(WorkspaceCaseTimelineItem)
            .where(WorkspaceCaseTimelineItem.case_id == case_id)
            .order_by(WorkspaceCaseTimelineItem.sort_order)
        )
        timeline = [
            {
                "id": str(t.id),
                "date_text": t.date_text,
                "date_normalized": t.date_normalized.isoformat() if t.date_normalized else None,
                "title": t.title,
                "description": t.description,
                "category": t.category,
                "source_type": t.source_type,
                "sort_order": t.sort_order,
            }
            for t in timeline_result.scalars().all()
        ]

        return {
            "id": str(case.id),
            "case_name": case.case_name,
            "case_type": case.case_type,
            "status": case.status,
            "summary": case.summary,
            "tagged_items": case.tagged_items,
            "conversations": conversations,
            "timeline": timeline,
            "created_at": case.created_at.isoformat() if case.created_at else None,
            "updated_at": case.updated_at.isoformat() if case.updated_at else None,
        }

    @staticmethod
    async def update_case(
        db: AsyncSession,
        case_id: uuid.UUID,
        session_token: str,
        updates: dict[str, Any],
        user_id: uuid.UUID | None = None,
    ) -> dict[str, Any] | None:
        """사건 업데이트"""
        tokens = await WorkspaceCaseService._get_session_tokens(
            db, session_token, user_id
        )
        result = await db.execute(
            select(WorkspaceCase).where(
                WorkspaceCase.id == case_id,
                WorkspaceCase.session_token.in_(tokens),
            )
        )
        case = result.scalar_one_or_none()
        if not case:
            return None

        allowed_fields = {"case_name", "case_type", "status"}
        update_values: dict[str, Any] = {}
        for field in allowed_fields:
            if field in updates and updates[field] is not None:
                update_values[field] = updates[field]

        if update_values:
            update_values["updated_at"] = func.now()
            await db.execute(
                update(WorkspaceCase)
                .where(WorkspaceCase.id == case_id)
                .values(**update_values)
            )

        # 최신 상태 반환
        return await WorkspaceCaseService.get_case_detail(
            db, case_id, session_token, user_id
        )

    @staticmethod
    async def delete_case(
        db: AsyncSession,
        case_id: uuid.UUID,
        session_token: str,
        user_id: uuid.UUID | None = None,
    ) -> bool:
        """사건 삭제 (연결된 대화의 case_id null 처리)"""
        tokens = await WorkspaceCaseService._get_session_tokens(
            db, session_token, user_id
        )
        result = await db.execute(
            select(WorkspaceCase).where(
                WorkspaceCase.id == case_id,
                WorkspaceCase.session_token.in_(tokens),
            )
        )
        case = result.scalar_one_or_none()
        if not case:
            return False

        # 연결된 대화의 case_id를 null로 변경
        await db.execute(
            update(ChatConversation)
            .where(ChatConversation.case_id == case_id)
            .values(case_id=None)
        )

        # 활동 로그
        await ActivityLogger.log(
            db, session_token, "delete_case",
            case_id=case_id,
            detail={"case_name": case.case_name},
        )

        # 사건 삭제 (cascade로 타임라인도 삭제됨)
        await db.delete(case)
        return True
