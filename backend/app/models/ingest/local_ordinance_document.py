"""
자치법규 문서 모델 (순수 테이블 정의)

data/local_rules_v1.json 데이터를 PostgreSQL에 저장하기 위한 테이블.
적재 로직(JSON→ORM 변환)은 scripts/ingest/types/local_ordinance.py 에 위치.
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


class LocalOrdinanceDocument(Base):
    """
    자치법규 문서 테이블

    data/local_rules_v1.json (160,276건)
    """

    __tablename__ = "local_ordinance_documents"

    id = Column(Integer, primary_key=True, autoincrement=True)

    ordinance_id = Column(
        String(20),
        unique=True,
        index=True,
        nullable=False,
        comment="자치법규ID",
    )
    ordinance_serial = Column(
        String(20),
        nullable=True,
        comment="자치법규일련번호",
    )
    ordinance_name = Column(
        Text,
        nullable=False,
        comment="자치법규명",
    )
    local_government = Column(
        String(100),
        nullable=True,
        comment="지자체기관명",
    )
    overall_summary = Column(
        Text,
        nullable=True,
        comment="전체요약 (Basic)",
    )
    content = Column(
        Text,
        nullable=True,
        comment="조문 전체 텍스트 (concat)",
    )
    supplementary = Column(
        Text,
        nullable=True,
        comment="부칙내용",
    )
    ai_summary = Column(
        Text,
        nullable=True,
        comment="AI 생성 요약 (= overall_summary, 인제스트 호환)",
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
            f"<LocalOrdinanceDocument(id={self.id}, "
            f"ordinance_id={self.ordinance_id}, "
            f"name={self.ordinance_name})>"
        )
