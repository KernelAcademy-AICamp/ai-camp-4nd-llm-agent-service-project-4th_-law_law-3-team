"""뉴스 기사 ORM 모델"""

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Index,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import ARRAY

from app.core.database import Base


class NewsArticle(Base):  # type: ignore[misc]
    """뉴스 기사 테이블

    법률 뉴스 파이프라인에서 수집/정제/요약된 기사를 저장.
    LanceDB news_chunks 테이블과 doc_id로 연결.
    """

    __tablename__ = "news_articles"

    # PK: SHA256(url)
    id = Column(
        String(64),
        primary_key=True,
        comment="문서 ID (SHA256(url))",
    )

    # 소스 정보
    source = Column(
        String(20),
        nullable=False,
        index=True,
        comment="소스 유형 (lawtimes | naver)",
    )
    publisher = Column(
        String(200),
        nullable=False,
        comment="매체명",
    )

    # 기사 기본 정보
    title = Column(Text, nullable=False, comment="기사 제목")
    author = Column(String(100), nullable=True, comment="기자명")
    published_at = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="기사 발행일시 (KST)",
    )
    collected_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        comment="수집 일시",
    )
    url = Column(
        Text,
        nullable=False,
        unique=True,
        comment="기사 원문 URL",
    )
    section = Column(String(50), nullable=True, comment="기사 섹션")
    tags: Column[list[str] | None] = Column(
        ARRAY(String),
        nullable=True,
        comment="기사 태그/키워드",
    )

    # 정제된 본문
    cleaned_text = Column(Text, nullable=False, comment="정제된 본문")

    # 구조화 요약
    summary_one_liner = Column(Text, nullable=False, comment="한줄 요지")
    summary_issues: Column[list[str] | None] = Column(ARRAY(String), nullable=True, comment="주요 쟁점")
    summary_laws: Column[list[str] | None] = Column(ARRAY(String), nullable=True, comment="언급 법령")
    summary_cases: Column[list[str] | None] = Column(ARRAY(String), nullable=True, comment="언급 판례")
    summary_institutions: Column[list[str] | None] = Column(ARRAY(String), nullable=True, comment="언급 기관")
    summary_implications: Column[list[str] | None] = Column(ARRAY(String), nullable=True, comment="시사점")

    # 메타데이터
    content_hash = Column(
        String(64),
        nullable=False,
        index=True,
        comment="정제 본문 SHA256 해시",
    )
    disclaimer = Column(
        Text,
        nullable=False,
        default="본 문서는 기사 요약이며 법령/판례 원문이 아닙니다",
        comment="면책 고지",
    )
    schema_version = Column(
        String(10),
        nullable=False,
        default="1.0",
        comment="스키마 버전",
    )
    is_indexed = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="LanceDB 임베딩 완료 여부",
    )

    # 타임스탬프
    created_at = Column(DateTime, default=datetime.utcnow, comment="레코드 생성일시")
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        comment="수정일시",
    )

    # 인덱스
    __table_args__ = (
        Index("idx_news_source_published", "source", "published_at"),
        Index("idx_news_tags", "tags", postgresql_using="gin"),
    )

    def __repr__(self) -> str:
        return f"<NewsArticle(id={self.id}, source={self.source}, title={self.title[:30]})>"
