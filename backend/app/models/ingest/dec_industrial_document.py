"""
산업재해보상보험재심사위원회 결정례 문서 모델 (순수 테이블 정의)

data/decisions_committee/ 데이터를 PostgreSQL에 저장.
적재 로직(JSON->ORM 변환)은 scripts/ingest/types/dec_industrial.py 에 위치.
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


class DecIndustrialDocument(Base):  # type: ignore[misc]
    """산업재해보상보험재심사위원회 결정례 테이블"""

    __tablename__ = "dec_industrial_documents"

    id = Column(Integer, primary_key=True, autoincrement=True)

    serial_number = Column(
        String(100),
        unique=True,
        index=True,
        nullable=False,
        comment="결정문 일련번호",
    )
    case_number = Column(
        String(200),
        nullable=True,
        comment="사건번호",
    )
    case_label = Column(
        String(200),
        nullable=True,
        comment="사건",
    )
    case_major_category = Column(
        String(200),
        nullable=True,
        comment="사건대분류",
    )
    case_mid_category = Column(
        String(200),
        nullable=True,
        comment="사건중분류",
    )
    case_sub_category = Column(
        String(200),
        nullable=True,
        comment="사건소분류",
    )
    decision_date = Column(
        String(50),
        index=True,
        nullable=True,
        comment="의결일자",
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
    issue = Column(
        Text,
        nullable=True,
        comment="쟁점",
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
    original_authority = Column(
        String(200),
        nullable=True,
        comment="원처분기관",
    )
    document_provision_type = Column(
        String(100),
        nullable=True,
        comment="문서제공구분",
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
            f"<DecIndustrialDocument(id={self.id}, "
            f"serial={self.serial_number}, "
            f"case_number={self.case_number})>"
        )
