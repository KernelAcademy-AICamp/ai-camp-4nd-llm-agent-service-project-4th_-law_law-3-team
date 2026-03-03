"""Dead-Letter Queue: 실패 기사 재처리 서비스 (v0.3.0)

Consultant 피드백: 에러 로그+알림만으로는 실패 기사 추적/재처리 어려움.
PostgreSQL 테이블 기반 DLQ로 실패 기사를 저장하고 자동 재시도.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# 최대 재시도 횟수
MAX_RETRY_COUNT = 3


class DeadLetterService:
    """Dead-Letter Queue 관리 서비스

    실패한 기사를 news_article_dlq 테이블에 저장하고,
    재시도 가능한 기사를 조회하여 파이프라인에 재투입.
    """

    async def enqueue(
        self,
        db: AsyncSession,
        *,
        article_url: str,
        stage: str,
        error_type: str,
        error_message: str,
        raw_payload: str | None = None,
    ) -> None:
        """실패 기사를 DLQ에 등록

        이미 등록된 URL이면 retry_count 증가.
        """
        from sqlalchemy.dialects.postgresql import insert

        from app.models.news_article_dlq import NewsArticleDLQ

        stmt = insert(NewsArticleDLQ).values(
            article_url=article_url,
            stage=stage,
            error_type=error_type,
            error_message=error_message[:2000],
            raw_payload=raw_payload,
            retry_count=0,
        ).on_conflict_do_update(
            index_elements=["article_url"],
            set_={
                "retry_count": NewsArticleDLQ.retry_count + 1,
                "error_type": error_type,
                "error_message": error_message[:2000],
                "stage": stage,
                "updated_at": datetime.now(timezone.utc),
            },
        )
        await db.execute(stmt)
        await db.flush()
        logger.info("DLQ 등록: [%s] %s — %s", stage, article_url, error_type)

    async def get_retryable(
        self,
        db: AsyncSession,
        *,
        limit: int = 50,
    ) -> list[dict[str, str | int | None]]:
        """재시도 가능한 기사 목록 조회

        조건: retry_count < MAX_RETRY_COUNT, is_resolved=False
        """
        from app.models.news_article_dlq import NewsArticleDLQ

        result = await db.execute(
            select(NewsArticleDLQ)
            .where(
                NewsArticleDLQ.retry_count < MAX_RETRY_COUNT,
                NewsArticleDLQ.is_resolved == False,  # noqa: E712
            )
            .order_by(NewsArticleDLQ.created_at)
            .limit(limit),
        )
        rows = result.scalars().all()

        return [
            {
                "id": int(row.id),
                "article_url": str(row.article_url),
                "stage": str(row.stage),
                "retry_count": int(row.retry_count),
                "raw_payload": str(row.raw_payload) if row.raw_payload else None,
            }
            for row in rows
        ]

    async def mark_resolved(
        self,
        db: AsyncSession,
        article_url: str,
    ) -> None:
        """재처리 성공 시 해결 완료 표시"""
        from app.models.news_article_dlq import NewsArticleDLQ

        await db.execute(
            update(NewsArticleDLQ)
            .where(NewsArticleDLQ.article_url == article_url)
            .values(is_resolved=True, updated_at=datetime.now(timezone.utc)),
        )
        await db.flush()
        logger.info("DLQ 해결 완료: %s", article_url)
