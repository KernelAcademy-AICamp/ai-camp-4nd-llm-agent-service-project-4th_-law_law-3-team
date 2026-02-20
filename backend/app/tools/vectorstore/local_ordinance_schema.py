"""
자치법규 LanceDB 스키마 - 별도 테이블, 12컬럼

기존 legal_chunks(10컬럼)와 별도의 local_ordinance_chunks 테이블.
전체요약(Basic) + 조문요약(Specific)을 각각 임베딩하여
1문서 = 1(전체요약) + N(조문요약) 벡터를 저장합니다.

ID 체계:
- 전체요약: {자치법규ID}_overall_0
- 조문요약: {자치법규ID}_art_{조문번호}_0
"""

from typing import Optional

import pyarrow as pa
from pydantic import BaseModel

# =============================================================================
# 상수
# =============================================================================

VECTOR_DIM = 1024  # 임베딩 차원 (KURE-v1)
TABLE_NAME = "local_ordinance_chunks"


# =============================================================================
# PyArrow 스키마 (LanceDB 테이블 생성용)
# =============================================================================

LOCAL_ORDINANCE_SCHEMA = pa.schema([
    # ========== 기존 호환 필드 (10개) ==========
    pa.field("id", pa.utf8()),              # 청크 고유 ID
    pa.field("source_id", pa.utf8()),       # 자치법규ID
    pa.field("data_type", pa.utf8()),       # "자치법규" (고정)
    pa.field("title", pa.utf8()),           # 자치법규명
    pa.field("content", pa.utf8()),         # 요약 텍스트
    pa.field("vector", pa.list_(pa.float32(), VECTOR_DIM)),
    pa.field("source_name", pa.utf8()),     # 지자체기관명
    pa.field("chunk_index", pa.int32()),    # 0: 전체요약, 1~N: 조문요약
    pa.field("total_chunks", pa.int32()),   # 1 + 조문 수
    pa.field("date", pa.utf8()),            # None (자치법규에 날짜 필드 없음)

    # ========== 추가 필드 (2개) ==========
    pa.field("summary_type", pa.utf8()),    # "Basic" | "Specific"
    pa.field("article_number", pa.utf8()),  # 조문번호 (Basic은 None)
])

ALL_COLUMNS = [
    "id", "source_id", "data_type", "title", "content",
    "vector", "source_name", "chunk_index", "total_chunks",
    "date", "summary_type", "article_number",
]


# =============================================================================
# Pydantic 모델
# =============================================================================

class LocalOrdinanceChunk(BaseModel):
    """자치법규 청크 모델"""

    id: str
    source_id: str
    data_type: str = "자치법규"
    title: str
    content: str
    vector: list[float]
    source_name: str = ""
    chunk_index: int = 0
    total_chunks: int = 1
    date: Optional[str] = None
    summary_type: str = "Basic"         # "Basic" | "Specific"
    article_number: Optional[str] = None

    def to_dict(self) -> dict[str, object]:
        """LanceDB 삽입용 딕셔너리 변환"""
        return self.model_dump()


# =============================================================================
# 헬퍼 함수
# =============================================================================

def create_local_ordinance_chunk(
    source_id: str,
    title: str,
    content: str,
    vector: list[float],
    source_name: str = "",
    chunk_index: int = 0,
    total_chunks: int = 1,
    summary_type: str = "Basic",
    article_number: Optional[str] = None,
) -> dict[str, object]:
    """범용 자치법규 청크 생성"""
    if summary_type == "Basic":
        chunk_id = f"{source_id}_overall_{chunk_index}"
    else:
        art_no = article_number or "unknown"
        chunk_id = f"{source_id}_art_{art_no}_{chunk_index}"

    return {
        "id": chunk_id,
        "source_id": source_id,
        "data_type": "자치법규",
        "title": title,
        "content": content,
        "vector": vector,
        "source_name": source_name,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "date": None,
        "summary_type": summary_type,
        "article_number": article_number,
    }


def create_overall_summary_chunk(
    source_id: str,
    title: str,
    content: str,
    vector: list[float],
    source_name: str = "",
    total_chunks: int = 1,
) -> dict[str, object]:
    """전체요약(Basic) 청크 생성"""
    return create_local_ordinance_chunk(
        source_id=source_id,
        title=title,
        content=content,
        vector=vector,
        source_name=source_name,
        chunk_index=0,
        total_chunks=total_chunks,
        summary_type="Basic",
        article_number=None,
    )


def create_article_summary_chunk(
    source_id: str,
    title: str,
    content: str,
    vector: list[float],
    source_name: str = "",
    chunk_index: int = 1,
    total_chunks: int = 1,
    article_number: str = "",
) -> dict[str, object]:
    """조문요약(Specific) 청크 생성"""
    return create_local_ordinance_chunk(
        source_id=source_id,
        title=title,
        content=content,
        vector=vector,
        source_name=source_name,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
        summary_type="Specific",
        article_number=article_number,
    )
