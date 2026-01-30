"""
법령 문서 모델 (LanceDB 전용)

data/law_cleaned.json 데이터를 PostgreSQL에 저장하기 위한 테이블
LanceDB 벡터 검색 후 원본 데이터 조회에 사용
"""

from datetime import datetime, date
from typing import Optional

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Date,
    DateTime,
    Index,
)
from sqlalchemy.dialects.postgresql import JSONB

from app.core.database import Base


class LawDocument(Base):
    """
    법령 문서 테이블 (LanceDB 전용)

    data/law_cleaned.json의 원본 데이터 저장용
    LanceDB에서 벡터 검색 후 source_id로 원본 조회

    사용 예시:
        # LanceDB 검색 결과에서 source_id 추출 후
        result = await session.execute(
            select(LawDocument).where(LawDocument.law_id == source_id)
        )
        law = result.scalar_one_or_none()
    """

    __tablename__ = "law_documents"

    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # 문서 식별
    law_id = Column(
        String(50),
        unique=True,
        index=True,
        nullable=False,
        comment="법령 ID (원본 law_id)",
    )

    # 기본 정보
    law_name = Column(
        String(500),
        index=True,
        nullable=False,
        comment="법령명 (민법, 형법 등)",
    )
    law_type = Column(
        String(50),
        index=True,
        nullable=True,
        comment="법령 유형 (법률/시행령/시행규칙)",
    )
    ministry = Column(
        String(200),
        nullable=True,
        comment="소관부처",
    )

    # 일자 정보
    promulgation_date = Column(
        String(20),
        nullable=True,
        comment="공포일자 (YYYYMMDD 형식)",
    )
    promulgation_no = Column(
        String(50),
        nullable=True,
        comment="공포번호",
    )
    enforcement_date = Column(
        Date,
        nullable=True,
        comment="시행일",
    )

    # 내용
    content = Column(
        Text,
        nullable=True,
        comment="조문 전체 텍스트",
    )
    supplementary = Column(
        Text,
        nullable=True,
        comment="부칙",
    )

    # 원본 데이터
    raw_data = Column(
        JSONB,
        nullable=False,
        comment="원본 JSON 데이터 전체",
    )

    # 메타데이터
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

    # 인덱스
    __table_args__ = (
        Index("idx_law_docs_name", "law_name"),
        Index("idx_law_docs_type", "law_type"),
        Index("idx_law_docs_ministry", "ministry"),
        Index("idx_law_docs_enforcement", "enforcement_date"),
    )

    def __repr__(self) -> str:
        return f"<LawDocument(id={self.id}, law_id={self.law_id}, name={self.law_name})>"

    @classmethod
    def from_json(cls, data: dict) -> "LawDocument":
        """
        JSON 데이터에서 인스턴스 생성

        Args:
            data: law_cleaned.json의 개별 item

        Returns:
            LawDocument 인스턴스
        """
        enforcement_date = None
        enforcement_str = data.get("enforcement_date", "")
        if enforcement_str:
            try:
                enforcement_date = date.fromisoformat(enforcement_str[:10])
            except (ValueError, TypeError):
                pass

        return cls(
            law_id=data.get("law_id", ""),
            law_name=data.get("law_name", ""),
            law_type=data.get("law_type"),
            ministry=data.get("ministry"),
            promulgation_date=data.get("promulgation_date"),
            promulgation_no=data.get("promulgation_no"),
            enforcement_date=enforcement_date,
            content=data.get("content"),
            supplementary=data.get("supplementary"),
            raw_data=data,
        )

    @property
    def embedding_text(self) -> str:
        """
        임베딩용 텍스트 생성

        조문 내용을 반환 (prefix는 임베딩 생성 시 추가)
        """
        return self.content or ""
