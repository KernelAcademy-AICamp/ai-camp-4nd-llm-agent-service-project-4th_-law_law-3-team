"""
부처 유권해석 문서 모델 (순수 테이블 정의)

data/ingest_source/interpretation_ministry/ 디렉토리의 JSON 파일들을 PostgreSQL에 저장.
적재 로직(JSON→ORM 변환)은 scripts/ingest/types/interpretation_ministry.py 에 위치.
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


class InterpretationMinistryDocument(Base):
    """
    부처 유권해석 테이블

    data/ingest_source/interpretation_ministry/ (28파일, 다수 부처)
    고용노동부, 국토교통부, 행정안전부 등
    """

    __tablename__ = "interpretation_ministry_documents"

    id = Column(Integer, primary_key=True, autoincrement=True)

    serial_number = Column(
        String(100),
        unique=True,
        index=True,
        nullable=False,
        comment="법령해석 일련번호",
    )
    case_name = Column(
        Text,
        nullable=True,
        comment="안건명",
    )
    interpretation_date = Column(
        Date,
        index=True,
        nullable=True,
        comment="해석일자",
    )

    case_number = Column(
        String(200),
        nullable=True,
        comment="안건번호 (일부 부처만 보유)",
    )

    # 주요 내용
    inquiry = Column(
        Text,
        nullable=True,
        comment="질의요지",
    )
    related_law = Column(
        Text,
        nullable=True,
        comment="관련법령",
    )
    answer = Column(
        Text,
        nullable=True,
        comment="회답",
    )
    reason = Column(
        Text,
        nullable=True,
        comment="이유 (일부 부처만 보유)",
    )

    # 소속 부처 (파일명에서 추출)
    ministry_name = Column(
        String(200),
        index=True,
        nullable=True,
        comment="부처명 (예: 고용노동부)",
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
            f"<InterpretationMinistryDocument(id={self.id}, "
            f"serial={self.serial_number}, "
            f"ministry={self.ministry_name})>"
        )
