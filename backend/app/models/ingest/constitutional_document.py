"""
헌법재판소 결정례 문서 모델 (순수 테이블 정의)

data/ingest_source/constitutional_v1.json 데이터를 PostgreSQL에 저장하기 위한 테이블.
적재 로직(JSON→ORM 변환)은 scripts/ingest/types/constitutional.py 에 위치.
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


class ConstitutionalDocument(Base):
    """
    헌법재판소 결정례 테이블

    data/ingest_source/constitutional_v1.json (31,718건)
    """

    __tablename__ = "constitutional_documents"

    id = Column(Integer, primary_key=True, autoincrement=True)

    serial_number = Column(
        String(100),
        unique=True,
        index=True,
        nullable=False,
        comment="헌재결정례 일련번호",
    )
    case_number = Column(
        String(100),
        index=True,
        nullable=True,
        comment="사건번호 (예: 2022헌마1312)",
    )
    case_name = Column(
        Text,
        nullable=True,
        comment="사건명",
    )
    case_type = Column(
        String(50),
        index=True,
        nullable=True,
        comment="사건종류명 (헌마/헌바/헌가 등)",
    )
    case_type_code = Column(
        String(20),
        nullable=True,
        comment="사건종류코드",
    )
    decision_date = Column(
        Date,
        index=True,
        nullable=True,
        comment="종국일자",
    )
    court_division_code = Column(
        String(20),
        nullable=True,
        comment="재판부구분코드",
    )

    # 주요 내용
    summary = Column(
        Text,
        nullable=True,
        comment="판시사항",
    )
    reasoning = Column(
        Text,
        nullable=True,
        comment="결정요지",
    )
    ruling = Column(
        Text,
        nullable=True,
        comment="주문",
    )
    full_text = Column(
        Text,
        nullable=True,
        comment="전문",
    )
    reason = Column(
        Text,
        nullable=True,
        comment="이유",
    )

    # 참조 정보
    reference_provisions = Column(
        Text,
        nullable=True,
        comment="심판대상조문",
    )
    reference_statutes = Column(
        Text,
        nullable=True,
        comment="참조조문",
    )
    reference_cases = Column(
        Text,
        nullable=True,
        comment="참조판례",
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
            f"<ConstitutionalDocument(id={self.id}, "
            f"serial={self.serial_number}, "
            f"case_number={self.case_number})>"
        )
