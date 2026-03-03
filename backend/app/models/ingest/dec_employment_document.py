"""
고용보험심사위원회 결정례 문서 모델 (순수 테이블 정의)

data/decisions_committee/ 데이터를 PostgreSQL에 저장.
적재 로직(JSON->ORM 변환)은 scripts/ingest/types/dec_employment.py 에 위치.
"""

from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    String,
    Text,
)

from app.core.database import Base


class DecEmploymentDocument(Base):  # type: ignore[misc]
    """고용보험심사위원회 결정례 테이블"""

    __tablename__ = "dec_employment_documents"

    id = Column(Integer, primary_key=True, autoincrement=True)

    serial_number = Column(
        String(100),
        unique=True,
        index=True,
        nullable=False,
        comment="결정문 일련번호",
    )
    case_name = Column(
        Text,
        nullable=True,
        comment="사건명",
    )
    case_number = Column(
        String(200),
        nullable=True,
        comment="사건번호",
    )
    case_classification = Column(
        String(200),
        nullable=True,
        comment="사건의분류",
    )
    decision_date = Column(
        String(50),
        index=True,
        nullable=True,
        comment="의결일자",
    )
    resolution_type = Column(
        String(100),
        nullable=True,
        comment="의결서종류",
    )
    ruling = Column(
        Text,
        nullable=True,
        comment="주문",
    )
    reason = Column(
        Text,
        nullable=True,
        comment="이유",
    )
    claim = Column(
        Text,
        nullable=True,
        comment="청구취지",
    )
    petitioner = Column(
        Text,
        nullable=True,
        comment="청구인",
    )
    respondent = Column(
        Text,
        nullable=True,
        comment="피청구인",
    )
    overview = Column(
        Text,
        nullable=True,
        comment="개요",
    )
    organization_name = Column(
        String(200),
        nullable=True,
        comment="기관명",
    )
    ai_summary = Column(
        Text,
        nullable=True,
        comment="AI 생성 요약",
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        comment="레코드 생성일시",
    )
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        comment="레코드 수정일시",
    )

    def __repr__(self) -> str:
        return (
            f"<DecEmploymentDocument(id={self.id}, "
            f"serial={self.serial_number}, "
            f"case_name={self.case_name})>"
        )
