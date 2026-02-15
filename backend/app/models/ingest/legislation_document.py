"""
법령해석례 문서 모델 (순수 테이블 정의)

data/legislation_v1.json 데이터를 PostgreSQL에 저장하기 위한 테이블.
적재 로직(JSON→ORM 변환)은 scripts/ingest/types/legislation.py 에 위치.
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


class LegislationDocument(Base):
    """
    법령해석례 테이블

    data/legislation_v1.json (8,597건)
    """

    __tablename__ = "legislation_documents"

    id = Column(Integer, primary_key=True, autoincrement=True)

    serial_number = Column(
        String(100),
        unique=True,
        index=True,
        nullable=False,
        comment="법령해석례 일련번호",
    )
    case_name = Column(
        Text,
        nullable=True,
        comment="안건명",
    )
    case_number = Column(
        String(50),
        index=True,
        nullable=True,
        comment="안건번호",
    )
    interpretation_date = Column(
        Date,
        index=True,
        nullable=True,
        comment="해석일자",
    )
    registration_date = Column(
        String(20),
        nullable=True,
        comment="등록일시 (YYYYMMDD)",
    )

    # 기관 정보
    interpretation_agency_code = Column(
        String(20),
        nullable=True,
        comment="해석기관코드",
    )
    interpretation_agency = Column(
        String(200),
        index=True,
        nullable=True,
        comment="해석기관명",
    )
    inquiry_agency_code = Column(
        String(20),
        nullable=True,
        comment="질의기관코드",
    )
    inquiry_agency = Column(
        String(200),
        nullable=True,
        comment="질의기관명",
    )
    management_agency_code = Column(
        String(20),
        nullable=True,
        comment="관리기관코드",
    )

    # 주요 내용
    inquiry = Column(
        Text,
        nullable=True,
        comment="질의요지",
    )
    answer = Column(
        Text,
        nullable=True,
        comment="회답",
    )
    reason = Column(
        Text,
        nullable=True,
        comment="이유",
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
            f"<LegislationDocument(id={self.id}, "
            f"serial={self.serial_number}, "
            f"case_number={self.case_number})>"
        )
