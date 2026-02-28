"""
판례→법령 인용 관계 모델

Neo4j (Case)-[:CITES]->(Statute) 관계를 PostgreSQL로 마이그레이션
"""

from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, UniqueConstraint

from app.core.database import Base


class CaseStatuteCitation(Base):
    """
    판례→법령 인용 관계 테이블

    판례가 참조조문에서 인용한 법령을 저장
    유사 판례 검색의 핵심 테이블 (self-join)
    """

    __tablename__ = "case_statute_citations"

    id = Column(Integer, primary_key=True, autoincrement=True)

    case_doc_id = Column(
        Integer,
        ForeignKey("precedent_documents.id", ondelete="CASCADE"),
        nullable=False,
        comment="판례 문서 FK",
    )
    law_doc_id = Column(
        Integer,
        ForeignKey("law_documents.id", ondelete="CASCADE"),
        nullable=False,
        comment="인용된 법령 FK",
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        comment="레코드 생성일시",
    )

    __table_args__ = (
        UniqueConstraint("case_doc_id", "law_doc_id", name="uq_case_statute_citation"),
        Index("idx_csc_case", "case_doc_id"),
        Index("idx_csc_statute", "law_doc_id"),
    )

    def __repr__(self) -> str:
        return f"<CaseStatuteCitation(case={self.case_doc_id}, law={self.law_doc_id})>"
