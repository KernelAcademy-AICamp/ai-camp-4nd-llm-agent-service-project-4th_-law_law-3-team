"""
위원회 결정례 문서 모델 (순수 테이블 정의)

data/ingest_source/decisions_committee/ 디렉토리의 JSON 파일들을 PostgreSQL에 저장.
적재 로직(JSON→ORM 변환)은 scripts/ingest/types/decisions_committee.py 에 위치.

10개 위원회별로 JSON 스키마가 상이하므로 공통/빈출 필드를 union으로 정의.
위원회별 필드명 매핑은 _orm_factory에서 처리 (예: 사건명/안건명/제목 → case_name).
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


class DecisionsCommitteeDocument(Base):
    """
    위원회 결정례 테이블

    data/ingest_source/decisions_committee/ (11파일, 10개 위원회)
    공정거래위원회, 국가인권위원회, 노동위원회, 개인정보보호위원회,
    금융위원회, 증권선물위원회, 국민권익위원회, 고용보험심사위원회,
    산업재해보상위험재심사위원회, 중앙환경분쟁조정위원회

    위원회별 필드명이 다르므로 _orm_factory에서 다중 필드명 매핑:
    - 제목: 사건명 / 안건명 / 제목 → case_name
    - 날짜: 의결일자 / 의결일 / 결정일자 / 등록일 → decision_date
    - 번호: 결정번호 / 의결번호 / 의안번호 → decision_number
    - 요지: 결정요지 / 판단요지 / 판정요지 / 판정사항 → decision_summary
    - 전문: 의결문 / 결정례전문 / 내용 → full_text
    """

    __tablename__ = "decisions_committee_documents"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # --- 공통 필드 (10/10) ---
    serial_number = Column(
        String(100),
        unique=True,
        index=True,
        nullable=False,
        comment="결정문 일련번호",
    )
    ai_summary = Column(
        Text,
        nullable=True,
        comment="AI 생성 요약 (결정문요약)",
    )

    # 소속 위원회 (파일명에서 추출)
    committee_name = Column(
        String(200),
        index=True,
        nullable=True,
        comment="위원회명 (예: 공정거래위원회)",
    )

    # --- 제목/날짜/번호 (다중 필드명 매핑) ---
    case_name = Column(
        Text,
        nullable=True,
        comment="안건명 (사건명/안건명/제목 통합)",
    )
    decision_date = Column(
        String(50),
        index=True,
        nullable=True,
        comment="의결일자 (의결일자/의결일/결정일자/등록일 통합)",
    )
    case_number = Column(
        String(200),
        nullable=True,
        comment="사건번호 (5/10 위원회)",
    )
    decision_number = Column(
        String(200),
        nullable=True,
        comment="결정번호/의결번호/의안번호 통합 (5/10 위원회)",
    )

    # --- 주요 내용 ---
    ruling = Column(
        Text,
        nullable=True,
        comment="주문 (6/10 위원회)",
    )
    reason = Column(
        Text,
        nullable=True,
        comment="이유 (6/10 위원회)",
    )
    decision_summary = Column(
        Text,
        nullable=True,
        comment="결정요지/판단요지/판정요지/판정사항 통합 (5/10 위원회)",
    )
    claim = Column(
        Text,
        nullable=True,
        comment="청구취지 (2/10 위원회)",
    )
    appendix = Column(
        Text,
        nullable=True,
        comment="별지 (3/10 위원회)",
    )

    # --- 조치 관련 (금융/증권선물) ---
    action_reason = Column(
        Text,
        nullable=True,
        comment="조치이유 (금융위/증권선물위)",
    )
    action_content = Column(
        Text,
        nullable=True,
        comment="조치내용 (금융위/증권선물위)",
    )

    # --- 전문 (의결문/결정례전문/내용 통합) ---
    full_text = Column(
        Text,
        nullable=True,
        comment="의결문/결정례전문/내용 통합 (3/10 위원회)",
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
            f"<DecisionsCommitteeDocument(id={self.id}, "
            f"serial={self.serial_number}, "
            f"committee={self.committee_name})>"
        )
