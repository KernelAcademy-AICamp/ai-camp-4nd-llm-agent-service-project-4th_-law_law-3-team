"""
금융위원회 결정례 문서 모델 (순수 테이블 정의)

data/decisions_committee/ 데이터를 PostgreSQL에 저장.
적재 로직(JSON->ORM 변환)은 scripts/ingest/types/dec_financial.py 에 위치.
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


class DecFinancialDocument(Base):  # type: ignore[misc]
    """금융위원회 결정례 테이블"""

    __tablename__ = "dec_financial_documents"

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
        comment="안건명",
    )
    decision_number = Column(
        String(200),
        nullable=True,
        comment="의결번호",
    )
    action_reason = Column(
        Text,
        nullable=True,
        comment="조치이유",
    )
    action_content = Column(
        Text,
        nullable=True,
        comment="조치내용",
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
            f"<DecFinancialDocument(id={self.id}, "
            f"serial={self.serial_number}, "
            f"case_name={self.case_name})>"
        )
