"""
chat_messages 30일 정리 배치

오래된 메시지를 삭제하여 PostgreSQL 저장소를 효율화한다.
대화 메타데이터(chat_conversations)와 요약은 영구 보존된다.
"""

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, func, select

from app.core.database import async_session_factory
from app.models.chat_conversation import ChatMessage

logger = logging.getLogger(__name__)

# 메시지 보존 기간 (일)
MESSAGE_RETENTION_DAYS = 30
# 배치 삭제 크기
BATCH_SIZE = 1000


async def cleanup_old_messages(
    retention_days: int = MESSAGE_RETENTION_DAYS,
) -> dict[str, int]:
    """보존 기간이 지난 chat_messages 삭제

    Args:
        retention_days: 보존 기간 (일). 기본 30일.

    Returns:
        {"deleted": 삭제 건수, "remaining": 잔여 건수}
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    total_deleted = 0

    async with async_session_factory() as db:
        # 배치 삭제 (대량 DELETE 시 lock 시간 최소화)
        while True:
            # 삭제 대상 ID 조회
            target_ids = await db.execute(
                select(ChatMessage.id)
                .where(ChatMessage.created_at < cutoff)
                .limit(BATCH_SIZE)
            )
            ids = [row[0] for row in target_ids.all()]

            if not ids:
                break

            await db.execute(
                delete(ChatMessage).where(ChatMessage.id.in_(ids))
            )
            await db.commit()
            total_deleted += len(ids)

            logger.info(
                "chat_messages 정리: %d건 삭제 (누적 %d건)",
                len(ids), total_deleted,
            )

            if len(ids) < BATCH_SIZE:
                break

        # 잔여 건수 확인
        remaining_result = await db.execute(
            select(func.count()).select_from(ChatMessage)
        )
        remaining = remaining_result.scalar() or 0

    logger.info(
        "chat_messages 정리 완료: 총 %d건 삭제, %d건 잔여 (기준: %d일)",
        total_deleted, remaining, retention_days,
    )

    return {"deleted": total_deleted, "remaining": remaining}
