"""
판례 문서 모델 (LanceDB 전용)

data/precedents_cleaned.json 데이터를 PostgreSQL에 저장하기 위한 테이블
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

from app.common.database import Base


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
        String(100),
        index=True,
        nullable=True,
        comment="사건번호 (예: 84나3990)",
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
        Index("idx_precedent_docs_case_number", "case_number"),
        Index("idx_precedent_docs_court", "court_name"),
        Index("idx_precedent_docs_case_type", "case_type"),
        Index("idx_precedent_docs_date", "decision_date"),
    )

    def __repr__(self) -> str:
        return (
            f"<PrecedentDocument(id={self.id}, serial={self.serial_number}, "
            f"case_number={self.case_number})>"
        )

    @classmethod
    def from_json(cls, data: dict) -> "PrecedentDocument":
        """
        JSON 데이터에서 인스턴스 생성

        Args:
            data: precedents_cleaned.json의 개별 item

        Returns:
            PrecedentDocument 인스턴스
        """
        decision_date = cls._parse_date(data.get("선고일자"))

        return cls(
            serial_number=data.get("판례정보일련번호", ""),
            case_name=data.get("사건명"),
            case_number=data.get("사건번호"),
            decision_date=decision_date,
            court_name=data.get("법원명"),
            case_type=data.get("사건종류명"),
            judgment_type=data.get("판결유형"),
            summary=data.get("판시사항"),
            reasoning=data.get("판결요지"),
            ruling=data.get("주문"),
            claim=data.get("청구취지"),
            full_reason=data.get("이유"),
            full_text=data.get("판례내용"),
            reference_provisions=data.get("참조조문"),
            reference_cases=data.get("참조판례"),
            raw_data=data,
        )

    @staticmethod
    def _parse_date(date_str: Optional[str]) -> Optional[date]:
        """날짜 문자열 파싱 (YYYYMMDD 또는 YYYY-MM-DD)"""
        if not date_str:
            return None

        date_str = str(date_str).strip()

        # 숫자만 있는 경우 (20170731)
        if date_str.isdigit() and len(date_str) == 8:
            try:
                return date(
                    int(date_str[:4]),
                    int(date_str[4:6]),
                    int(date_str[6:8])
                )
            except ValueError:
                return None

        # ISO 형식 (2017-07-31)
        try:
            return date.fromisoformat(date_str[:10])
        except (ValueError, IndexError):
            return None

    @property
    def embedding_text(self) -> str:
        """
        임베딩용 텍스트 생성

        판시사항 + 판결요지를 조합하여 반환 (prefix는 임베딩 생성 시 추가)
        """
        parts = []

        if self.case_name:
            parts.append(f"[{self.case_name}]")

        if self.summary:
            parts.append(self.summary)

        if self.reasoning:
            parts.append(self.reasoning)

        return "\n".join(parts)
