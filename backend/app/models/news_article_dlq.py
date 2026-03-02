"""뉴스 기사 Dead-Letter Queue ORM 모델 (v0.3.0)"""

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Integer,
    String,
    Text,
)

from app.core.database import Base


class NewsArticleDLQ(Base):  # type: ignore[misc]
    """뉴스 기사 Dead-Letter Queue 테이블

    파이프라인에서 처리 실패한 기사를 저장하고 자동 재시도 관리.
    retry_count < MAX_RETRY_COUNT 인 기사만 재시도 대상.
    """

    __tablename__ = "news_article_dlq"

    id = Column(Integer, primary_key=True, autoincrement=True)
    article_url = Column(
        Text,
        nullable=False,
        unique=True,
        comment="실패한 기사 URL",
    )
    stage = Column(
        String(30),
        nullable=False,
        comment="실패 단계 (collect|clean|summarize|store|chunk)",
    )
    error_type = Column(String(200), nullable=False, comment="예외 클래스명")
    error_message = Column(Text, nullable=False, comment="에러 메시지 (최대 2000자)")
    raw_payload = Column(Text, nullable=True, comment="실패 시점 원시 데이터 (JSON)")

    retry_count = Column(
        Integer,
        nullable=False,
        default=0,
        comment="재시도 횟수 (3회 초과 시 재시도 중단)",
    )
    is_resolved = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="재처리 성공 여부",
    )

    created_at = Column(DateTime, default=datetime.utcnow, comment="최초 실패 일시")
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        comment="최종 업데이트 일시",
    )

    def __repr__(self) -> str:
        return f"<NewsArticleDLQ(id={self.id}, url={self.article_url[:50]}, retries={self.retry_count})>"
