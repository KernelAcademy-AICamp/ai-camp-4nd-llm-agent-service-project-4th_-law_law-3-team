"""
법령 모델

법령 조문 정보 저장 테이블
"""

from datetime import date, datetime
from typing import Any, Optional

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB

from app.common.database import Base


class Law(Base):  # type: ignore[misc]
    """
    법령 테이블

    법률, 대통령령, 부령 등 법령 정보를 저장
    """

    __tablename__ = "laws"

    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # 법령 식별
    law_id = Column(
        String(20),
        nullable=False,
        unique=True,
        index=True,
        comment="법령 ID",
    )
    law_name = Column(
        Text,
        nullable=False,
        index=True,
        comment="법령명 (민법, 형법 등)",
    )

    # 법령 정보
    law_type = Column(
        String(20),
        index=True,
        comment="법률, 대통령령, 부령 등",
    )
    ministry = Column(
        Text,
        index=True,
        comment="소관부처",
    )
    promulgation_date = Column(
        Date,
        comment="공포일",
    )
    promulgation_no = Column(
        String(20),
        comment="공포번호",
    )
    enforcement_date = Column(
        Date,
        index=True,
        comment="시행일",
    )

    # 본문
    content = Column(
        Text,
        comment="전체 조문 내용",
    )
    supplementary = Column(
        Text,
        comment="부칙",
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

    # Indexes
    __table_args__ = (
        Index("idx_laws_ministry", "ministry"),
        Index("idx_laws_enforcement", "enforcement_date"),
    )

    def __repr__(self) -> str:
        return f"<Law(id={self.id}, law_id={self.law_id}, name={self.law_name})>"

    @property
    def embedding_text(self) -> str:
        """RAG 임베딩용 텍스트 생성"""
        parts = []

        if self.law_name:
            parts.append(f"법령명: {self.law_name}")

        if self.content:
            # 조문 내용 (너무 길면 자르기)
            content = self.content[:5000]
            parts.append(f"내용: {content}")

        return "\n".join(parts)

    @classmethod
    def from_law_data(cls, data: dict[str, Any]) -> "Law":
        """법령 데이터에서 인스턴스 생성

        law.json의 items 배열 내 객체 형식:
        {
            "law_id": "010719",
            "law_name": "...",
            "promulgation_date": "20230808",
            "promulgation_no": "19592",
            "enforcement_date": "20230808",
            "ministry": "문화체육관광부",
            "law_type": "법률",
            "content": "...",
            "supplementary": "..."
        }
        """
        return cls(
            law_id=data.get("law_id", ""),
            law_name=data.get("law_name", ""),
            law_type=data.get("law_type"),
            ministry=data.get("ministry"),
            promulgation_date=cls._parse_date(data.get("promulgation_date")),
            promulgation_no=data.get("promulgation_no"),
            enforcement_date=cls._parse_date(data.get("enforcement_date")),
            content=data.get("content"),
            supplementary=data.get("supplementary"),
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
