"""
판례 문서 모델 (순수 테이블 정의)

data/precedents.json 데이터를 PostgreSQL에 저장하기 위한 테이블.
LanceDB 벡터 검색 후 원본 데이터 조회에 사용.

적재 로직(JSON→ORM 변환)은 scripts/ingest/types/precedent.py 에 위치.
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


class PrecedentDocument(Base):
    """
    판례 문서 테이블 (LanceDB 전용)

    data/precedents_cleaned.json의 원본 데이터 저장용
    LanceDB에서 벡터 검색 후 source_id로 원본 조회

    검색 흐름:
        1. LanceDB 벡터 검색 → source_id 추출
        2. PostgreSQL 조회 → ruling, claim, reasoning 등 전체 텍스트 접근

    사용 예시:
        # LanceDB 검색 결과에서 source_id 추출 후
        result = await session.execute(
            select(PrecedentDocument).where(
                PrecedentDocument.serial_number == source_id
            )
        )
        precedent = result.scalar_one_or_none()
        print(precedent.ruling)  # 주문
        print(precedent.claim)   # 청구취지
        print(precedent.full_reason)  # 이유
    """

    __tablename__ = "precedent_documents"

    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # 문서 식별
    serial_number = Column(
        String(100),
        unique=True,
        index=True,
        nullable=False,
        comment="판례정보일련번호",
    )

    # 사건 기본 정보
    case_name = Column(
        Text,
        nullable=True,
        comment="사건명",
    )
    case_number = Column(
        Text,
        index=False,
        nullable=True,
        comment="사건번호 (예: 84나3990, 병합사건은 수백자 가능)",
    )
    decision_date = Column(
        Date,
        index=True,
        nullable=True,
        comment="선고일자",
    )

    # 법원 정보
    court_name = Column(
        String(100),
        index=True,
        nullable=True,
        comment="법원명 (대법원, 서울고법 등)",
    )
    case_type = Column(
        String(50),
        index=True,
        nullable=True,
        comment="사건종류명 (민사/형사/행정)",
    )
    judgment_type = Column(
        String(100),
        nullable=True,
        comment="판결유형 (예: 제11민사부판결)",
    )

    # 주요 내용 (RAG 활용 대상)
    summary = Column(
        Text,
        nullable=True,
        comment="판시사항",
    )
    reasoning = Column(
        Text,
        nullable=True,
        comment="판결요지",
    )
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
    full_reason = Column(
        Text,
        nullable=True,
        comment="이유 (전체)",
    )
    full_text = Column(
        Text,
        nullable=True,
        comment="판례내용 (전문)",
    )

    # AI 생성 요약
    ai_summary = Column(
        Text,
        nullable=True,
        comment="AI 생성 판례요약",
    )

    # 참조 정보
    reference_provisions = Column(
        Text,
        nullable=True,
        comment="참조조문 (예: 민법 제750조, 제756조)",
    )
    reference_cases = Column(
        Text,
        nullable=True,
        comment="참조판례",
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
        return (
            f"<PrecedentDocument(id={self.id}, serial={self.serial_number}, "
            f"case_number={self.case_number})>"
        )
