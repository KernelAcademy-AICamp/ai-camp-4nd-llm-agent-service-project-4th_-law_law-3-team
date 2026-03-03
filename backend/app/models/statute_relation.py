"""
법령 관련 관계 모델

Neo4j (Statute)-[:RELATED_TO]->(Statute) 관계를 PostgreSQL로 마이그레이션
"""

from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    UniqueConstraint,
)

from app.core.database import Base


class StatuteRelation(Base):  # type: ignore[misc]
    """
    법령 관련 관계 테이블

    양방향 관계 (법령 A ↔ 법령 B)
    law_doc_id_1 < law_doc_id_2 제약으로 중복 방지
    """

    __tablename__ = "statute_relations"

    id = Column(Integer, primary_key=True, autoincrement=True)

    law_doc_id_1 = Column(
        Integer,
        ForeignKey("law_documents.id", ondelete="CASCADE"),
        nullable=False,
        comment="법령 1 FK (작은 ID)",
    )
    law_doc_id_2 = Column(
        Integer,
        ForeignKey("law_documents.id", ondelete="CASCADE"),
        nullable=False,
        comment="법령 2 FK (큰 ID)",
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        comment="레코드 생성일시",
    )

    __table_args__ = (
        CheckConstraint(
            "law_doc_id_1 < law_doc_id_2",
            name="ck_statute_relations_order",
        ),
        UniqueConstraint("law_doc_id_1", "law_doc_id_2", name="uq_statute_relations"),
        Index("idx_sr_id_1", "law_doc_id_1"),
        Index("idx_sr_id_2", "law_doc_id_2"),
    )

    def __repr__(self) -> str:
        return f"<StatuteRelation(id1={self.law_doc_id_1}, id2={self.law_doc_id_2})>"
