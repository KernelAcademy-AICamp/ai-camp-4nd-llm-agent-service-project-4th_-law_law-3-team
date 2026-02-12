"""
법령 문서 모델 (순수 테이블 정의)

data/raw/law.json 데이터를 PostgreSQL에 저장하기 위한 테이블.
LanceDB 벡터 검색 후 원본 데이터 조회에 사용.

적재 로직(JSON→ORM 변환)은 scripts/ingest/types/law.py 에 위치.
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

    # AI 생성 요약
    ai_summary = Column(
        Text,
        nullable=True,
        comment="AI 생성 법령요약",
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

    def __repr__(self) -> str:
        return f"<LawDocument(id={self.id}, law_id={self.law_id}, name={self.law_name})>"
