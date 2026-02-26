"""
판례→판례 인용 관계 모델

Neo4j (Case)-[:CITES_CASE]->(Case) 관계를 PostgreSQL로 마이그레이션
"""

from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, UniqueConstraint

from app.core.database import Base


class CaseCaseCitation(Base):
    """
    판례→판례 인용 관계 테이블

    참조판례 필드에서 추출한 판례 간 인용 관계
    """

    __tablename__ = "case_case_citations"

    id = Column(Integer, primary_key=True, autoincrement=True)

    citing_case_id = Column(
        Integer,
        ForeignKey("precedent_documents.id", ondelete="CASCADE"),
        nullable=False,
        comment="인용하는 판례 FK",
    )
    cited_case_id = Column(
        Integer,
        ForeignKey("precedent_documents.id", ondelete="CASCADE"),
        nullable=False,
        comment="인용되는 판례 FK",
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        comment="레코드 생성일시",
    )

    __table_args__ = (
        UniqueConstraint("citing_case_id", "cited_case_id", name="uq_case_case_citation"),
        Index("idx_ccc_citing", "citing_case_id"),
        Index("idx_ccc_cited", "cited_case_id"),
    )

    def __repr__(self) -> str:
        return f"<CaseCaseCitation(citing={self.citing_case_id}, cited={self.cited_case_id})>"
