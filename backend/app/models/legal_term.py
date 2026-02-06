"""
법률 용어 ORM 모델

legal_terms 테이블: 법령정의사전 기반 법률 용어 저장
MeCab 토크나이저의 법률 복합명사 보강에 사용
"""

import re
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)

from app.core.database import Base

# 한글 전용 여부 판단 정규식 (한글 + 공백만 허용)
_KOREAN_ONLY_PATTERN = re.compile(r"^[가-힣\s]+$")


class LegalTerm(Base):
    """법률 용어 테이블"""

    __tablename__ = "legal_terms"

    id = Column(Integer, primary_key=True, autoincrement=True)
    term = Column(
        String(200), unique=True, nullable=False,
        comment="법령용어명 한글",
    )
    term_hanja = Column(
        String(200), nullable=True,
        comment="법령용어명 한자",
    )
    definition = Column(
        Text, nullable=True,
        comment="법령용어 정의",
    )
    source = Column(
        Text, nullable=True,
        comment="출처 법령명",
    )
    source_code = Column(
        String(20), nullable=True,
        comment="사전유형 (법령정의사전, 법령한영사전 등)",
    )
    serial_number = Column(
        String(20), nullable=True,
        comment="법령용어 일련번호",
    )
    term_length = Column(
        Integer, nullable=False,
        comment="용어 글자 수 (필터링용)",
    )
    is_korean_only = Column(
        Boolean, nullable=False, default=False,
        comment="한글 전용 여부",
    )
    priority = Column(
        Integer, nullable=False, default=0,
        comment="토크나이저 로드 우선순위 (높을수록 우선)",
    )

    # 타임스탬프
    created_at = Column(
        DateTime, default=datetime.utcnow,
        comment="레코드 생성일시",
    )
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow,
        comment="레코드 수정일시",
    )

    __table_args__ = (
        UniqueConstraint("term", name="uq_legal_terms_term"),
        Index("idx_legal_terms_term", "term"),
        Index("idx_legal_terms_source_code", "source_code"),
        Index("idx_legal_terms_is_korean", "is_korean_only"),
        Index("idx_legal_terms_priority", "priority"),
        Index("idx_legal_terms_length", "term_length"),
        {"comment": "법률 용어 사전 (MeCab 토크나이저 보강용)"},
    )

    def __repr__(self) -> str:
        return (
            f"<LegalTerm(id={self.id}, term='{self.term}', "
            f"length={self.term_length}, korean={self.is_korean_only})>"
        )

    @staticmethod
    def compute_is_korean_only(term: str) -> bool:
        """용어가 한글 전용인지 판별"""
        return bool(_KOREAN_ONLY_PATTERN.match(term))

    @staticmethod
    def compute_priority(
        term: str,
        source_code: str,
        term_length: int,
        is_korean_only: bool,
    ) -> int:
        """
        토크나이저 로드 우선순위 계산

        기준:
        - 법령정의사전 +10
        - 한글 전용 +5
        - 2~10자 길이 +3
        """
        priority = 0
        if source_code == "법령정의사전":
            priority += 10
        if is_korean_only:
            priority += 5
        if 2 <= term_length <= 10:
            priority += 3
        return priority
