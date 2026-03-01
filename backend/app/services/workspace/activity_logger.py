"""
활동 로그 서비스

워크스페이스 활동(사건 생성, 타임라인 재생성 등)을 기록한다.
"""

import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.workspace_case import WorkspaceActivityLog


class ActivityLogger:
    """활동 로그 기록"""

    @staticmethod
    async def log(
        db: AsyncSession,
        session_token: str,
        action: str,
        case_id: uuid.UUID | None = None,
        conversation_id: uuid.UUID | None = None,
        detail: dict[str, Any] | None = None,
    ) -> WorkspaceActivityLog:
        """활동 로그 기록"""
        entry = WorkspaceActivityLog(
            session_token=session_token,
            action=action,
            case_id=case_id,
            conversation_id=conversation_id,
            detail=detail,
        )
        db.add(entry)
        await db.flush()
        return entry
