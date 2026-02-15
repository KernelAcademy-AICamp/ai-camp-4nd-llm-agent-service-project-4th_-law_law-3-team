"""
행정심판례 문서 모델 (순수 테이블 정의)

data/ingest_source/administration_v1.json 데이터를 PostgreSQL에 저장하기 위한 테이블.
적재 로직(JSON→ORM 변환)은 scripts/ingest/types/administration.py 에 위치.
"""

from datetime import datetime

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Integer,
    String,
    Text,
)

from app.core.database import Base


class AdministrationDocument(Base):
    """
    행정심판례 테이블

    data/ingest_source/administration_v1.json (34,254건)
    """

    __tablename__ = "administration_documents"

    id = Column(Integer, primary_key=True, autoincrement=True)

    serial_number = Column(
        String(100),
        unique=True,
        index=True,
        nullable=False,
        comment="행정심판례 일련번호",
    )
    case_name = Column(
        Text,
        nullable=True,
        comment="사건명",
    )
    case_number = Column(
        String(100),
        index=True,
        nullable=True,
        comment="사건번호",
    )
    decision_date = Column(
        Date,
        index=True,
        nullable=True,
        comment="의결일자",
    )
    disposition_date = Column(
        Date,
        nullable=True,
        comment="처분일자",
    )
    disposition_agency = Column(
        String(200),
        nullable=True,
        comment="처분청",
    )
    adjudication_agency = Column(
        String(200),
        index=True,
        nullable=True,
        comment="재결청",
    )
    case_type = Column(
        String(50),
        index=True,
        nullable=True,
        comment="재결례유형명",
    )
    case_type_code = Column(
        String(20),
        nullable=True,
        comment="재결례유형코드",
    )

    # 주요 내용
    ruling = Column(
        Text,
        nullable=True,
        comment="주문",
    )
    claim = Column(
        Text,
        nullable=True,
        comment="청구취지",
    )
    reason = Column(
        Text,
        nullable=True,
        comment="이유",
    )
    adjudication_summary = Column(
        Text,
        nullable=True,
        comment="재결요지",
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
            f"<AdministrationDocument(id={self.id}, "
            f"serial={self.serial_number}, "
            f"case_number={self.case_number})>"
        )
