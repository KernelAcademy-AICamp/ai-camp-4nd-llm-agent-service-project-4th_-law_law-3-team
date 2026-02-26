"""
법령 계급 관계 모델

Neo4j (Statute)-[:HIERARCHY_OF]->(Statute) 관계를 PostgreSQL로 마이그레이션
시행령 → 법률 (child → parent) 방향
"""

from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, UniqueConstraint

from app.core.database import Base


class StatuteHierarchy(Base):
    """
    법령 계급 관계 테이블

    child_id → parent_id: 하위법령 → 상위법령
    예: 도로교통법 시행령(child) → 도로교통법(parent)
    """

    __tablename__ = "statute_hierarchy"

    id = Column(Integer, primary_key=True, autoincrement=True)

    child_id = Column(
        Integer,
        ForeignKey("law_documents.id", ondelete="CASCADE"),
        nullable=False,
        comment="하위 법령 FK (시행령)",
    )
    parent_id = Column(
        Integer,
        ForeignKey("law_documents.id", ondelete="CASCADE"),
        nullable=False,
        comment="상위 법령 FK (법률)",
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        comment="레코드 생성일시",
    )

    __table_args__ = (
        UniqueConstraint("child_id", "parent_id", name="uq_statute_hierarchy"),
        Index("idx_sh_child", "child_id"),
        Index("idx_sh_parent", "parent_id"),
    )

    def __repr__(self) -> str:
        return f"<StatuteHierarchy(child={self.child_id}, parent={self.parent_id})>"
