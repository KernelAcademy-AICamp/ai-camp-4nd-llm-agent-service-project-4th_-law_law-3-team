"""
LanceDB 스키마 v2 - 단일 테이블, 10컬럼 (단순화)

설계 원칙:
- 공통 9개 + 개별 1개(date) = 총 10개 컬럼
- data_type으로 문서 유형 구분
- 상세 필드(ruling, claim, case_number 등)는 PostgreSQL에 저장
- date는 판례/법령만 채움, 나머지 타입은 NULL

문서 타입:
- data_type = "법령" | "판례" | "헌법재판" | "행정심판" | ...

검색 흐름:
1. LanceDB 벡터 검색 → source_id + data_type 추출
2. PostgreSQL에서 원본 조회 (ruling, claim, reasoning 등)
"""

from typing import Optional

import pyarrow as pa
from pydantic import BaseModel

# =============================================================================
# 상수
# =============================================================================

VECTOR_DIM = 1024  # 임베딩 차원
TABLE_NAME = "legal_chunks"


# =============================================================================
# PyArrow 스키마 (LanceDB 테이블 생성용)
# =============================================================================

LEGAL_CHUNKS_SCHEMA = pa.schema([
    # ========== 공통 필드 (9개) ==========
    pa.field("id", pa.utf8()),              # 청크 고유 ID (예: "010719_0")
    pa.field("source_id", pa.utf8()),       # 원본 문서 ID (예: "010719")
    pa.field("data_type", pa.utf8()),       # "법령" | "판례" | "헌법재판" | ...
    pa.field("title", pa.utf8()),           # 제목 (법령명 / 사건명)
    pa.field("content", pa.utf8()),         # 청크 텍스트 (prefix 포함)
    pa.field("vector", pa.list_(pa.float32(), VECTOR_DIM)),  # 임베딩 벡터 (KURE 1024차원)
    pa.field("source_name", pa.utf8()),     # 출처
    pa.field("chunk_index", pa.int32()),    # 청크 인덱스
    pa.field("total_chunks", pa.int32()),   # 해당 문서의 총 청크 수

    # ========== 개별 필드 (1개) ==========
    pa.field("date", pa.utf8()),            # 날짜 (법령: 시행일, 판례: 선고일)
])


# =============================================================================
# 컬럼 그룹 정의
# =============================================================================

COMMON_COLUMNS = [
    "id", "source_id", "data_type", "title", "content",
    "vector", "source_name", "chunk_index", "total_chunks",
]

ALL_COLUMNS = COMMON_COLUMNS + ["date"]


# =============================================================================
# Pydantic 모델 (데이터 검증 및 타입 힌트)
# =============================================================================

class LegalChunk(BaseModel):
    """법률 청크 모델 (전 타입 통합)"""

    # === 공통 필드 ===
    id: str                                 # 청크 고유 ID
    source_id: str                          # 원본 문서 ID
    data_type: str                          # "법령" | "판례" | ...
    title: str                              # 제목
    content: str                            # 청크 텍스트
    vector: list[float]                     # 임베딩 벡터
    source_name: str = ""                   # 출처
    chunk_index: int = 0                    # 청크 인덱스
    total_chunks: int = 1                   # 총 청크 수

    # === 개별 필드 ===
    date: Optional[str] = None              # 날짜 (판례/법령만)

    def to_dict(self) -> dict[str, object]:
        """LanceDB 삽입용 딕셔너리 변환"""
        return self.model_dump()


# =============================================================================
# 헬퍼 함수
# =============================================================================

def create_chunk(
    data_type: str,
    source_id: str,
    title: str,
    content: str,
    vector: list[float],
    source_name: str = "",
    date: Optional[str] = None,
    chunk_index: int = 0,
    total_chunks: int = 1,
) -> dict[str, object]:
    """
    범용 청크 생성

    Args:
        data_type: 문서 유형 ("법령", "판례", "헌법재판" 등)
        source_id: 원본 문서 ID
        title: 제목
        content: 청크 텍스트
        vector: 임베딩 벡터
        source_name: 출처
        date: 날짜 (판례/법령만, 나머지 None)
        chunk_index: 청크 인덱스 (0부터)
        total_chunks: 해당 문서의 총 청크 수

    Returns:
        LanceDB 삽입용 딕셔너리
    """
    return {
        "id": f"{source_id}_{chunk_index}",
        "source_id": source_id,
        "data_type": data_type,
        "title": title,
        "content": content,
        "vector": vector,
        "source_name": source_name,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "date": date,
    }


def create_law_chunk(
    source_id: str,
    chunk_index: int,
    title: str,
    content: str,
    vector: list[float],
    enforcement_date: str = "",
    department: str = "",
    total_chunks: int = 1,
    **_kwargs: object,
) -> dict[str, object]:
    """법령 청크 생성 (하위 호환 래퍼)"""
    return create_chunk(
        data_type="법령",
        source_id=source_id,
        title=title,
        content=content,
        vector=vector,
        source_name=department,
        date=enforcement_date or None,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
    )


def create_precedent_chunk(
    source_id: str,
    chunk_index: int,
    title: str,
    content: str,
    vector: list[float],
    decision_date: str = "",
    court_name: str = "",
    total_chunks: int = 1,
    **_kwargs: object,
) -> dict[str, object]:
    """판례 청크 생성 (하위 호환 래퍼)"""
    return create_chunk(
        data_type="판례",
        source_id=source_id,
        title=title,
        content=content,
        vector=vector,
        source_name=court_name,
        date=decision_date or None,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
    )
