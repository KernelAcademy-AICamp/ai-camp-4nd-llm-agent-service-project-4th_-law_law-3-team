"""
법령 비공식 약칭 모델

Neo4j Alias 노드를 PostgreSQL로 마이그레이션
"""

from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, String

from app.core.database import Base


class StatuteAlias(Base):
    """
    법령 비공식 약칭 테이블

    Neo4j의 (Alias)-[:ALIAS_OF]->(Statute) 관계를 대체
    예: 민소법 → 민사소송법, 특가법 → 특정범죄가중법
    """

    __tablename__ = "statute_aliases"

    id = Column(Integer, primary_key=True, autoincrement=True)

    law_doc_id = Column(
        Integer,
        ForeignKey("law_documents.id", ondelete="CASCADE"),
        nullable=False,
        comment="법령 문서 FK",
    )
    alias_name = Column(
        String(200),
        nullable=False,
        comment="비공식 약칭 (예: 민소법)",
    )
    category = Column(
        String(50),
        nullable=True,
        comment="약칭 카테고리 (비공식, 관용 등)",
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        comment="레코드 생성일시",
    )

    __table_args__ = (
        Index("idx_sa_alias_name", "alias_name", unique=True),
        Index("idx_sa_law_doc_id", "law_doc_id"),
    )

    def __repr__(self) -> str:
        return f"<StatuteAlias(id={self.id}, alias={self.alias_name})>"
