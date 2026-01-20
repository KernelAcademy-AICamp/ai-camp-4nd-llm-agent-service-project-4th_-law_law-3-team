"""
법률 참조 데이터 모델

조약, 행정규칙, 법률용어 등 참조용 데이터 저장 테이블
"""

from datetime import datetime, date
from typing import Optional
from enum import Enum

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Date,
    DateTime,
    Index,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB

from app.common.database import Base


class RefType(str, Enum):
    """참조 데이터 유형"""
    TREATY = "treaty"          # 조약
    ADMIN_RULE = "admin_rule"  # 행정규칙
    LAW_TERM = "law_term"      # 법률용어


class LegalReference(Base):
    """
    법률 참조 데이터 테이블

    조약, 행정규칙, 법률용어 등 참조 데이터를 저장
    """

    __tablename__ = "legal_references"

    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # 식별
    ref_type = Column(
        String(20),
        nullable=False,
        index=True,
        comment="참조 유형: treaty, admin_rule, law_term",
    )
    serial_number = Column(
        String(100),
        nullable=False,
        comment="일련번호",
    )

    # 공통 필드
    title = Column(
        Text,
        index=True,
        comment="명칭 (조약명/규칙명/용어명)",
    )
    content = Column(
        Text,
        comment="내용/정의",
    )

    # 기관/출처
    organization = Column(
        Text,
        index=True,
        comment="소관기관/체결국가",
    )
    category = Column(
        Text,
        comment="분류/종류",
    )

    # 날짜
    effective_date = Column(
        Date,
        comment="발효일/시행일",
    )

    # 메타데이터
    raw_data = Column(
        JSONB,
        nullable=False,
        comment="원본 데이터 전체 (JSON)",
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

    # Constraints & Indexes
    __table_args__ = (
        UniqueConstraint(
            "ref_type", "serial_number",
            name="uq_legal_refs_type_serial"
        ),
        Index("idx_legal_refs_org", "organization"),
    )

    def __repr__(self) -> str:
        return (
            f"<LegalReference(id={self.id}, type={self.ref_type}, "
            f"title={self.title[:30] if self.title else None})>"
        )

    @property
    def embedding_text(self) -> str:
        """RAG 임베딩용 텍스트 생성"""
        parts = []

        if self.title:
            parts.append(f"명칭: {self.title}")

        if self.content:
            content = self.content[:3000]
            parts.append(f"내용: {content}")

        return "\n".join(parts)

    @classmethod
    def from_treaty(cls, data: dict) -> "LegalReference":
        """조약 데이터에서 인스턴스 생성

        treaty-full.json 형식:
        {
            "조약일련번호": "1400",
            "조약기본정보": {
                "조약명_한글": "...",
                "발효일자": "19491201",
                "서명일자": "19490527",
                ...
            },
            "추가정보": {
                "체결대상국가한글": "미국",
                "양자조약분야명": "재정",
                ...
            },
            "조약내용": {
                "조약내용": "..."
            }
        }
        """
        basic_info = data.get("조약기본정보", {})
        extra_info = data.get("추가정보", {})
        treaty_content = data.get("조약내용", {})

        return cls(
            ref_type=RefType.TREATY.value,
            serial_number=data.get("조약일련번호", ""),
            title=basic_info.get("조약명_한글") or basic_info.get("조약명_영문", ""),
            content=treaty_content.get("조약내용", ""),
            organization=extra_info.get("체결대상국가한글", ""),
            category=extra_info.get("양자조약분야명", ""),
            effective_date=cls._parse_date(basic_info.get("발효일자")),
            raw_data=data,
        )

    @classmethod
    def from_admin_rule(cls, data: dict) -> "LegalReference":
        """행정규칙 데이터에서 인스턴스 생성

        administrative_rules_full.json 형식:
        {
            "행정규칙ID": "93857",
            "행정규칙기본정보": {
                "행정규칙명": "...",
                "시행일자": "20250924",
                "소관부처명": "...",
                "행정규칙종류": "훈령",
                ...
            },
            "조문내용": ["제1조...", "제2조...", ...]
        }
        """
        basic_info = data.get("행정규칙기본정보", {})
        provisions = data.get("조문내용", [])

        # 조문 내용을 텍스트로 결합
        content = "\n".join(provisions) if isinstance(provisions, list) else str(provisions)

        return cls(
            ref_type=RefType.ADMIN_RULE.value,
            serial_number=data.get("행정규칙ID", ""),
            title=basic_info.get("행정규칙명", ""),
            content=content,
            organization=basic_info.get("소관부처명", ""),
            category=basic_info.get("행정규칙종류", ""),
            effective_date=cls._parse_date(basic_info.get("시행일자")),
            raw_data=data,
        )

    @classmethod
    def from_law_term(cls, data: dict) -> "LegalReference":
        """법률용어 데이터에서 인스턴스 생성

        lawterms_full.json 형식:
        {
            "법령용어 일련번호": "4350393",
            "법령용어명_한글": "080착신과금사업자",
            "법령용어명_한자": "080착신과금사업자",
            "법령용어코드명": "법령정의사전",
            "출처": "...",
            "법령용어정의": "..."
        }
        """
        return cls(
            ref_type=RefType.LAW_TERM.value,
            serial_number=data.get("법령용어 일련번호", ""),
            title=data.get("법령용어명_한글", ""),
            content=data.get("법령용어정의", ""),
            organization=data.get("출처", ""),
            category=data.get("법령용어코드명", ""),
            effective_date=None,  # 법률용어는 발효일 없음
            raw_data=data,
        )

    @staticmethod
    def _parse_date(date_str: Optional[str]) -> Optional[date]:
        """날짜 문자열 파싱 (YYYYMMDD 또는 YYYY-MM-DD)"""
        if not date_str:
            return None

        date_str = str(date_str).strip()

        # 숫자만 있는 경우 (20170731)
        if date_str.isdigit():
            if len(date_str) == 8:
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
