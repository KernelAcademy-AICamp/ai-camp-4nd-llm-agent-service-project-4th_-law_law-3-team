"""
법률 문서 모델

판례, 헌재결정례, 행정심판례, 법령해석례, 위원회 결정문 통합 테이블
"""

from datetime import date, datetime
from enum import Enum
from typing import Any, Optional

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB

from app.core.database import Base


class DocType(str, Enum):
    """법률 문서 유형"""
    PRECEDENT = "precedent"           # 판례
    CONSTITUTIONAL = "constitutional"  # 헌재결정례
    ADMINISTRATION = "administration"  # 행정심판례
    LEGISLATION = "legislation"        # 법령해석례
    COMMITTEE = "committee"            # 위원회 결정문


# 위원회 코드 매핑 (source -> 위원회명)
COMMITTEE_SOURCES = {
    "ftc": "공정거래위원회",
    "nhrck": "국가인권위원회",
    "acrc": "국민권익위원회",
    "ppc": "개인정보보호위원회",
    "kcc": "방송통신위원회",
    "fsc": "금융위원회",
    "ecc": "중앙선거관리위원회",
    "eiac": "환경분쟁조정위원회",
    "sfc": "해양환경관리공단",
    "iaciac": "산업재해보상보험심사위원회",
    "oclt": "원자력안전위원회",
}


class LegalDocument(Base):  # type: ignore[misc]
    """
    법률 문서 통합 테이블

    4가지 유형의 법률 문서를 하나의 테이블에 저장:
    - precedent: 일반 판례
    - constitutional: 헌법재판소 결정례
    - administration: 행정심판례
    - legislation: 법령해석례
    """

    __tablename__ = "legal_documents"

    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # 문서 식별
    doc_type = Column(
        String(20),
        nullable=False,
        index=True,
        comment="문서 유형: precedent, constitutional, administration, legislation, committee",
    )
    serial_number = Column(
        String(100),
        nullable=False,
        comment="원본 일련번호 (판례정보일련번호 등)",
    )
    source = Column(
        String(50),
        nullable=False,
        default="",
        index=True,
        comment="데이터 출처 (precedents, ftc, nhrck 등)",
    )

    # 사건 기본 정보
    case_name = Column(Text, comment="사건명/안건명")
    case_number = Column(
        Text,
        index=True,
        comment="사건번호/안건번호",
    )
    decision_date = Column(
        Date,
        index=True,
        comment="선고일/의결일/종국일",
    )

    # 기관 정보
    court_name = Column(
        Text,
        index=True,
        comment="법원명/재결청/해석기관",
    )
    court_type = Column(
        Text,
        comment="법원종류/재결례유형",
    )
    case_type = Column(
        Text,
        index=True,
        comment="사건종류 (민사/형사/헌마 등)",
    )

    # 주요 내용 (RAG 임베딩 대상)
    summary = Column(
        Text,
        comment="판시사항/결정요지/질의요지/주문",
    )
    reasoning = Column(
        Text,
        comment="판결요지/이유/회답",
    )
    full_text = Column(
        Text,
        comment="판례내용/전문",
    )

    # 추가 필드
    claim = Column(
        Text,
        comment="청구취지 (행정심판례)",
    )

    # 참조 정보
    reference_articles = Column(
        Text,
        comment="참조조문",
    )
    reference_cases = Column(
        Text,
        comment="참조판례",
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

    # Constraints
    __table_args__ = (
        UniqueConstraint(
            "doc_type", "serial_number", "source",
            name="uq_legal_docs_type_serial_source"
        ),
        Index(
            "idx_legal_docs_search",
            "doc_type", "case_type", "court_name", "decision_date",
        ),
        Index(
            "idx_legal_docs_source",
            "source",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<LegalDocument(id={self.id}, type={self.doc_type}, "
            f"source={self.source}, case_number={self.case_number})>"
        )

    @property
    def embedding_text(self) -> str:
        """
        RAG 임베딩용 텍스트 생성

        summary와 reasoning을 조합하여 검색에 사용할 텍스트 반환
        """
        parts = []

        if self.case_name:
            parts.append(f"사건명: {self.case_name}")

        if self.summary:
            parts.append(f"요지: {self.summary}")

        if self.reasoning:
            # 너무 길면 앞부분만 (토큰 제한)
            reasoning = self.reasoning[:3000]
            parts.append(f"내용: {reasoning}")

        return "\n".join(parts)

    @classmethod
    def from_precedent(cls, data: dict[str, Any], source: str = "precedents") -> "LegalDocument":
        """판례 데이터에서 인스턴스 생성"""
        return cls(
            doc_type=DocType.PRECEDENT.value,
            serial_number=data.get("판례정보일련번호", ""),
            source=source,
            case_name=data.get("사건명"),
            case_number=data.get("사건번호"),
            decision_date=cls._parse_date(data.get("선고일자")),
            court_name=data.get("법원명"),
            court_type=data.get("법원종류코드"),
            case_type=data.get("사건종류명"),
            summary=data.get("판시사항"),
            reasoning=data.get("판결요지"),
            full_text=data.get("판례내용"),
            reference_articles=data.get("참조조문"),
            reference_cases=data.get("참조판례"),
            raw_data=data,
        )

    @classmethod
    def from_constitutional(cls, data: dict[str, Any], source: str = "constitutional") -> "LegalDocument":
        """헌재결정례 데이터에서 인스턴스 생성"""
        # 참조조문 + 심판대상조문 결합
        ref_articles = "\n".join(filter(None, [
            data.get("참조조문"),
            data.get("심판대상조문"),
        ]))

        return cls(
            doc_type=DocType.CONSTITUTIONAL.value,
            serial_number=data.get("헌재결정례일련번호", ""),
            source=source,
            case_name=data.get("사건명"),
            case_number=data.get("사건번호"),
            decision_date=cls._parse_date(data.get("종국일자")),
            case_type=data.get("판시사항"),  # 헌마, 헌바 등
            summary=data.get("판시사항"),
            reasoning=data.get("결정요지"),
            full_text=data.get("전문"),
            reference_articles=ref_articles if ref_articles else None,
            reference_cases=data.get("참조판례"),
            raw_data=data,
        )

    @classmethod
    def from_administration(cls, data: dict[str, Any], source: str = "administration") -> "LegalDocument":
        """행정심판례 데이터에서 인스턴스 생성"""
        return cls(
            doc_type=DocType.ADMINISTRATION.value,
            serial_number=data.get("행정심판례일련번호", ""),
            source=source,
            case_name=data.get("사건명"),
            case_number=data.get("사건번호"),
            decision_date=cls._parse_date(data.get("의결일자")),
            court_name=data.get("재결청"),
            court_type=data.get("재결례유형명"),
            summary=data.get("주문"),
            reasoning=data.get("이유"),
            claim=data.get("청구취지"),
            raw_data=data,
        )

    @classmethod
    def from_legislation(cls, data: dict[str, Any], source: str = "legislation") -> "LegalDocument":
        """법령해석례 데이터에서 인스턴스 생성"""
        # 회답 + 이유 결합
        reasoning = "\n\n".join(filter(None, [
            data.get("회답"),
            data.get("이유"),
        ]))

        return cls(
            doc_type=DocType.LEGISLATION.value,
            serial_number=data.get("법령해석례일련번호", ""),
            source=source,
            case_name=data.get("안건명"),
            case_number=data.get("안건번호"),
            decision_date=cls._parse_date(data.get("등록일시")),
            court_name=data.get("해석기관명"),
            summary=data.get("질의요지"),
            reasoning=reasoning if reasoning else None,
            raw_data=data,
        )

    @classmethod
    def from_committee(cls, data: dict[str, Any], source: str) -> "LegalDocument":
        """위원회 결정문 데이터에서 인스턴스 생성

        Args:
            data: 위원회 결정문 JSON 데이터
            source: 위원회 출처 코드 (ftc, nhrck, ppc 등)
        """
        # 위원회명 결정 (source 코드에서 매핑)
        court_name = COMMITTEE_SOURCES.get(source, source)

        # 결정요지 또는 판단요지
        summary = data.get("결정요지") or data.get("판단요지") or data.get("주문", "")

        # 이유 또는 의결문
        reasoning = data.get("이유") or data.get("의결문", "")

        # 사건번호 (다양한 필드명 대응)
        case_number = data.get("사건번호") or data.get("결정번호", "")

        # 사건명
        case_name = data.get("사건명", "")
        if case_name == "null":
            case_name = ""

        # 의결일자 / 결정일자
        decision_date_str = data.get("의결일자") or data.get("결정일자", "")

        return cls(
            doc_type=DocType.COMMITTEE.value,
            serial_number=data.get("결정문일련번호", ""),
            source=source,
            case_name=case_name,
            case_number=case_number,
            decision_date=cls._parse_date(decision_date_str),
            court_name=court_name,
            court_type=data.get("회의종류") or data.get("문서유형", ""),
            summary=summary,
            reasoning=reasoning,
            full_text=data.get("별지", ""),
            claim=data.get("신청취지", ""),
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
