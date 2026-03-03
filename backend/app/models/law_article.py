"""
법령 조문 모델

법령 문서(law_documents)의 content를 조문 단위로 분리 저장.
LanceDB 벡터 검색에서 매칭된 조문만 선별적으로 LLM 컨텍스트에 포함하기 위해 사용.

키 정합성:
    law_articles.law_id = LanceDB source_id
    law_articles.article_number = LanceDB article_number (JSON 조문번호 원본값)
"""

from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)

from app.core.database import Base


class LawArticle(Base):  # type: ignore[misc]
    """법령 조문 테이블

    law_documents.content(전체 텍스트)를 조문 단위로 분리 저장.
    벡터 검색 결과의 article_number로 해당 조문만 조회.

    사용 예시:
        result = await session.execute(
            select(LawArticle).where(
                LawArticle.law_id == source_id,
                LawArticle.article_number == article_number,
            )
        )
        article = result.scalar_one_or_none()
    """

    __tablename__ = "law_articles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    law_id = Column(
        String(50),
        nullable=False,
        comment="법령 ID (law_documents.law_id 대응)",
    )
    article_number = Column(
        String(50),
        nullable=False,
        comment="조문번호 (LanceDB article_number과 동일 형식, 예: '1', '2')",
    )
    article_title = Column(
        String(500),
        nullable=True,
        comment="조문제목",
    )
    article_content = Column(
        Text,
        nullable=False,
        comment="조문 본문 (항+호 포함)",
    )
    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        comment="레코드 생성일시",
    )

    __table_args__ = (
        UniqueConstraint(
            "law_id", "article_number", name="uq_law_articles_law_article"
        ),
        Index("idx_law_articles_law_id", "law_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<LawArticle(law_id={self.law_id}, "
            f"article_number={self.article_number})>"
        )
