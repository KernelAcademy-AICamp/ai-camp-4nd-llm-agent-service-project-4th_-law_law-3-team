"""
재판 통계 ORM 모델

trial_statistics 테이블: 법원별/카테고리별/연도별 사건 처리 건수
"""

from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Index,
    Integer,
    String,
    UniqueConstraint,
)

from app.core.database import Base


class TrialStatistics(Base):
    """재판 통계 테이블"""

    __tablename__ = "trial_statistics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    category = Column(
        String(50), nullable=False,
        comment="사건 카테고리 (민사_본안_단독, 형사_공판 등)",
    )
    court_name = Column(
        String(100), nullable=False,
        comment="법원명 (서울중앙지방법원, 고양지원 등)",
    )
    court_type = Column(
        String(20), nullable=False,
        comment="법원 유형 (main: 본원, branch: 지원)",
    )
    parent_court = Column(
        String(100), nullable=True,
        comment="지원의 상위 본원명 (본원은 NULL)",
    )
    year = Column(
        Integer, nullable=False,
        comment="연도 (2015~2024)",
    )
    case_count = Column(
        Integer, nullable=False,
        comment="사건 처리 건수",
    )

    # 타임스탬프
    created_at = Column(
        DateTime, default=datetime.utcnow,
        comment="레코드 생성일시",
    )
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow,
        comment="레코드 수정일시",
    )

    __table_args__ = (
        UniqueConstraint(
            "category", "court_name", "year",
            name="uq_trial_stats_cat_court_year",
        ),
        Index("idx_trial_stats_category", "category"),
        Index("idx_trial_stats_court", "court_name"),
        Index("idx_trial_stats_year", "year"),
        Index("idx_trial_stats_cat_year", "category", "year"),
        Index("idx_trial_stats_parent", "parent_court"),
        {"comment": "재판 통계 (법원별/카테고리별/연도별 사건 처리 건수)"},
    )

    def __repr__(self) -> str:
        return (
            f"<TrialStatistics(id={self.id}, category='{self.category}', "
            f"court='{self.court_name}', year={self.year}, count={self.case_count})>"
        )
