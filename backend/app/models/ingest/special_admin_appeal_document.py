"""
특별행정심판 재결례 문서 모델 (순수 테이블 정의)

data/special_admin_appeal/ 디렉토리의 JSON 파일들을 PostgreSQL에 저장.
적재 로직(JSON→ORM 변환)은 scripts/ingest/types/special_admin_appeal.py 에 위치.
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


class SpecialAdminAppealDocument(Base):
    """
    특별행정심판 재결례 테이블

    data/special_admin_appeal/ (2파일)
    조세심판원, 해양안전심판원
    """

    __tablename__ = "special_admin_appeal_documents"

    id = Column(Integer, primary_key=True, autoincrement=True)

    serial_number = Column(
        String(100),
        unique=True,
        index=True,
        nullable=False,
        comment="특별행정심판 재결례 일련번호",
    )
    case_name = Column(
        Text,
        nullable=True,
        comment="사건명",
    )
    case_number = Column(
        String(200),
        index=True,
        nullable=True,
        comment="재결번호",
    )
    decision_date = Column(
        Date,
        index=True,
        nullable=True,
        comment="의결일자",
    )

    # 기관 정보
    adjudication_agency = Column(
        String(200),
        index=True,
        nullable=True,
        comment="재결청",
    )
    case_type = Column(
        String(100),
        index=True,
        nullable=True,
        comment="재결례유형명",
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

    # 조세심판원 특화 필드
    adjudication_summary = Column(
        Text,
        nullable=True,
        comment="재결요지 (조세심판원)",
    )
    related_rulings = Column(
        Text,
        nullable=True,
        comment="참조결정 (조세심판원)",
    )
    following_rulings = Column(
        Text,
        nullable=True,
        comment="따른결정 (조세심판원)",
    )
    tax_category = Column(
        String(200),
        nullable=True,
        comment="세목 (조세심판원)",
    )
    related_law = Column(
        Text,
        nullable=True,
        comment="관련법령 (조세심판원)",
    )

    # 해양안전심판원 특화 필드
    vessel_type = Column(
        String(200),
        nullable=True,
        comment="선박유형 (해양안전심판원)",
    )
    accident_type = Column(
        String(100),
        nullable=True,
        comment="사고유형 (해양안전심판원)",
    )
    tribunal_location = Column(
        String(100),
        nullable=True,
        comment="해심위치 (해양안전심판원)",
    )
    related_persons = Column(
        Text,
        nullable=True,
        comment="해양사고관련자 (해양안전심판원)",
    )
    appendix = Column(
        Text,
        nullable=True,
        comment="별지 (해양안전심판원)",
    )
    retrial_notice = Column(
        Text,
        nullable=True,
        comment="재심청구안내 (해양안전심판원)",
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
            f"<SpecialAdminAppealDocument(id={self.id}, "
            f"serial={self.serial_number}, "
            f"agency={self.adjudication_agency})>"
        )
