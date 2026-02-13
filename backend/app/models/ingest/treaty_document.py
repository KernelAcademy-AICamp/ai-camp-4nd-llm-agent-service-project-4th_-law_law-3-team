"""
조약 문서 모델 (순수 테이블 정의)

data/ingest_source/treaty_v1.json 데이터를 PostgreSQL에 저장하기 위한 테이블.
적재 로직(JSON→ORM 변환)은 scripts/ingest/types/treaty.py 에 위치.
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


class TreatyDocument(Base):
    """
    조약 테이블

    data/ingest_source/treaty_v1.json (3,589건)
    """

    __tablename__ = "treaty_documents"

    id = Column(Integer, primary_key=True, autoincrement=True)

    serial_number = Column(
        String(100),
        unique=True,
        index=True,
        nullable=False,
        comment="조약 일련번호",
    )
    treaty_number = Column(
        String(50),
        index=True,
        nullable=True,
        comment="조약번호",
    )
    treaty_name_kr = Column(
        Text,
        nullable=False,
        comment="조약명 (한글)",
    )
    treaty_name_en = Column(
        Text,
        nullable=True,
        comment="조약명 (영문)",
    )
    treaty_type_code = Column(
        String(20),
        nullable=True,
        comment="조약구분코드",
    )

    # 국가 정보
    counterpart_country = Column(
        String(200),
        nullable=True,
        comment="체결대상국가 (영문)",
    )
    counterpart_country_kr = Column(
        String(200),
        index=True,
        nullable=True,
        comment="체결대상국가 (한글)",
    )
    country_code = Column(
        String(20),
        nullable=True,
        comment="국가코드",
    )

    # 분야
    bilateral_field_code = Column(
        String(20),
        nullable=True,
        comment="양자조약분야코드",
    )
    bilateral_field = Column(
        String(100),
        index=True,
        nullable=True,
        comment="양자조약분야명",
    )

    # 일자 정보
    signing_date = Column(
        Date,
        index=True,
        nullable=True,
        comment="서명일자",
    )
    signing_place = Column(
        String(200),
        nullable=True,
        comment="서명장소",
    )
    effective_date = Column(
        Date,
        nullable=True,
        comment="발효일자",
    )
    parliament_approval = Column(
        String(10),
        nullable=True,
        comment="국회비준동의여부",
    )
    parliament_approval_date = Column(
        Date,
        nullable=True,
        comment="국회비준동의일자",
    )
    cabinet_review_date = Column(
        Date,
        nullable=True,
        comment="국무회의심의일자",
    )
    cabinet_review_session = Column(
        String(20),
        nullable=True,
        comment="국무회의심의회차",
    )
    presidential_approval_date = Column(
        Date,
        nullable=True,
        comment="대통령재가일자",
    )
    gazette_date = Column(
        Date,
        nullable=True,
        comment="관보게재일자",
    )

    # 내용
    content = Column(
        Text,
        nullable=True,
        comment="조약내용",
    )
    note = Column(
        Text,
        nullable=True,
        comment="비고",
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
            f"<TreatyDocument(id={self.id}, "
            f"serial={self.serial_number}, "
            f"name={self.treaty_name_kr[:30] if self.treaty_name_kr else ''})>"
        )
