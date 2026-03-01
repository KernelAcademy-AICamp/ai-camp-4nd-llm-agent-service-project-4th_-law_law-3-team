"""
FTS 인덱스 모델 (하이브리드 검색용)

BM25 전문 검색 인덱스: MeCab 전처리된 토큰 텍스트를 search_text에 저장하고
pg_textsearch BM25 인덱스로 검색. 원문은 law_documents/precedent_documents에 보유.
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
        primary_key=True,
        nullable=False,
        comment="문서 유형 (법령/판례/헌재결정례/행정심판례 등)",
    )
    title = Column(
        Text,
        nullable=False,
        server_default="",
        comment="법령명 또는 사건명",
    )
    date = Column(
        String(50),
        nullable=True,
        comment="시행일(법령) 또는 선고일(판례)",
    )
    source_name = Column(
        String(200),
        nullable=True,
        comment="소관부처(법령) 또는 법원명(판례)",
    )
    case_number = Column(
        Text,
        nullable=True,
        comment="판례 사건번호 (법령은 NULL, 병합사건은 수백자 가능)",
    )
    search_text = Column(
        Text,
        nullable=True,
        comment="MeCab 전처리된 공백 구분 토큰 텍스트 (BM25 인덱스 대상)",
    )
    content_tsvector = Column(
        TSVECTOR,
        nullable=True,
        comment="롤백용 tsvector (BM25 안정화 후 제거 예정)",
    )
    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        comment="레코드 생성일시",
    )

    __table_args__ = (
        # BM25 인덱스는 Alembic 마이그레이션에서 raw SQL로 생성
        # (ORM Index 객체는 bm25 access method 미지원)
        Index("idx_fts_index_data_type", "data_type"),
    )

    def __repr__(self) -> str:
        return (
            f"<FtsIndex(source_id={self.source_id}, "
            f"data_type={self.data_type}, title={self.title!r})>"
        )
