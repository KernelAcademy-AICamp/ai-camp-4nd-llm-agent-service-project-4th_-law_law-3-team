"""
행정규칙 문서 모델 (순수 테이블 정의)

data/admin_rule_v3.json 데이터를 PostgreSQL에 저장하기 위한 테이블.
적재 로직(JSON→ORM 변환)은 scripts/ingest/types/admin_rule.py 에 위치.
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


class AdminRuleDocument(Base):
    """
    행정규칙 문서 테이블

    data/admin_rule_v3.json (17,332건, 고유 17,092건)
    """

    __tablename__ = "admin_rule_documents"

    id = Column(Integer, primary_key=True, autoincrement=True)

    admin_rule_id = Column(
        String(50),
        unique=True,
        index=True,
        nullable=False,
        comment="행정규칙 ID",
    )
    serial_number = Column(
        String(50),
        index=True,
        nullable=True,
        comment="행정규칙 일련번호",
    )
    admin_rule_name = Column(
        String(500),
        index=True,
        nullable=False,
        comment="행정규칙명",
    )
    admin_rule_type = Column(
        String(100),
        index=True,
        nullable=True,
        comment="행정규칙 종류 (훈령/예규/고시 등)",
    )
    ministry = Column(
        String(200),
        nullable=True,
        comment="소관부처명",
    )
    ministry_code = Column(
        String(50),
        nullable=True,
        comment="소관부처코드",
    )
    parent_ministry = Column(
        String(200),
        nullable=True,
        comment="상위부처명",
    )
    promulgation_date = Column(
        String(20),
        nullable=True,
        comment="발령일자",
    )
    enforcement_date = Column(
        String(20),
        nullable=True,
        comment="시행일자",
    )

    content = Column(
        Text,
        nullable=True,
        comment="조문내용 전체 텍스트",
    )
    supplementary = Column(
        Text,
        nullable=True,
        comment="부칙내용",
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
            f"<AdminRuleDocument(id={self.id}, "
            f"admin_rule_id={self.admin_rule_id}, "
            f"name={self.admin_rule_name})>"
        )
