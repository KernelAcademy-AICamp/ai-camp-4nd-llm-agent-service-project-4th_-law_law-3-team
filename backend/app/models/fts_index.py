"""
FTS 인덱스 모델 (하이브리드 검색용)

원본 문서의 tsvector를 저장하여 PostgreSQL FTS 키워드 검색 지원.
원문 텍스트는 law_documents/precedent_documents에 있으므로 여기에는 저장하지 않음.
순수 검색 인덱스 + 결과 표시용 메타데이터만 보유.
"""

from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Index,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import TSVECTOR

from app.core.database import Base


class FtsIndex(Base):
    """
    FTS 인덱스 테이블 (문서 단위)

    데이터 흐름:
        PostgreSQL 원본 (law_documents, precedent_documents)
        → 원문 전체 concat → MeCab 토크나이징 → tsvector 생성
        → fts_index 테이블 저장

    검색 흐름:
        1. 벡터 검색 (LanceDB, 요약문 기반) → source_id 목록
        2. 키워드 검색 (PostgreSQL FTS, 원문 기반) → source_id 목록
        3. RRF 병합 (source_id 단위) → 최종 랭킹
    """

    __tablename__ = "fts_index"

    source_id = Column(
        String(100),
        primary_key=True,
        nullable=False,
        comment="원본 문서 ID (law_id 또는 serial_number)",
    )
    data_type = Column(
        String(20),
        nullable=False,
        comment="문서 유형 (법령/판례)",
    )
    title = Column(
        Text,
        nullable=False,
        server_default="",
        comment="법령명 또는 사건명",
    )
    date = Column(
        String(20),
        nullable=True,
        comment="시행일(법령) 또는 선고일(판례)",
    )
    source_name = Column(
        String(200),
        nullable=True,
        comment="소관부처(법령) 또는 법원명(판례)",
    )
    case_number = Column(
        String(100),
        nullable=True,
        comment="판례 사건번호 (법령은 NULL)",
    )
    content_tsvector = Column(
        TSVECTOR,
        nullable=True,
        comment="MeCab 토크나이징 기반 tsvector (GIN 인덱스 대상)",
    )
    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        comment="레코드 생성일시",
    )

    __table_args__ = (
        Index(
            "idx_fts_index_content_tsvector",
            "content_tsvector",
            postgresql_using="gin",
        ),
        Index("idx_fts_index_data_type", "data_type"),
    )

    def __repr__(self) -> str:
        return (
            f"<FtsIndex(source_id={self.source_id}, "
            f"data_type={self.data_type}, title={self.title!r})>"
        )
