"""
스키마 re-export + 검증 유틸리티

backend/app/tools/vectorstore/schema_v2.py의 canonical 스키마를 단일 소스로 사용.
임베딩 스크립트에서 사용 가능하도록 re-export합니다.
"""

import sys
from pathlib import Path
from typing import Any, Optional

# 백엔드 app 모듈을 import할 수 있도록 경로 추가
_backend_root = Path(__file__).parent.parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))

try:
    from app.tools.vectorstore.schema_v2 import (  # noqa: F401
        ALL_COLUMNS,
        COMMON_COLUMNS,
        LEGAL_CHUNKS_SCHEMA,
        TABLE_NAME,
        VECTOR_DIM,
        LegalChunk,
        create_chunk,
        create_law_chunk,
        create_precedent_chunk,
    )

    SCHEMA_SOURCE = "app.tools.vectorstore.schema_v2"

except ImportError:
    # 백엔드 미설치 환경 (RunPod, Colab)에서 독립 스키마 정의
    import pyarrow as pa

    VECTOR_DIM = 1024
    TABLE_NAME = "legal_chunks"

    LEGAL_CHUNKS_SCHEMA = pa.schema([
        # ========== 공통 필드 (9개) ==========
        pa.field("id", pa.utf8()),
        pa.field("source_id", pa.utf8()),
        pa.field("data_type", pa.utf8()),
        pa.field("title", pa.utf8()),
        pa.field("content", pa.utf8()),
        pa.field("vector", pa.list_(pa.float32(), VECTOR_DIM)),
        pa.field("source_name", pa.utf8()),
        pa.field("chunk_index", pa.int32()),
        pa.field("total_chunks", pa.int32()),
        # ========== 개별 필드 (3개) ==========
        pa.field("date", pa.utf8()),
        pa.field("summary_type", pa.utf8()),
        pa.field("article_number", pa.utf8()),
    ])

    COMMON_COLUMNS = [
        "id", "source_id", "data_type", "title", "content",
        "vector", "source_name", "chunk_index", "total_chunks",
    ]
    ALL_COLUMNS = COMMON_COLUMNS + ["date", "summary_type", "article_number"]

    LegalChunk = dict  # type: ignore[assignment,misc]

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
        summary_type: str = "Basic",
        article_number: Optional[str] = None,
        chunk_id: Optional[str] = None,
    ) -> dict[str, Any]:
        return {
            "id": chunk_id or f"{source_id}_{chunk_index}",
            "source_id": source_id,
            "data_type": data_type,
            "title": title,
            "content": content,
            "vector": vector,
            "source_name": source_name,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "date": date,
            "summary_type": summary_type,
            "article_number": article_number,
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
        summary_type: str = "Basic",
        article_number: Optional[str] = None,
        **_kwargs: object,
    ) -> dict[str, Any]:
        if summary_type == "Basic":
            chunk_id = f"{source_id}_overall_{chunk_index}"
        else:
            art_no = article_number or "unknown"
            chunk_id = f"{source_id}_art_{art_no}_{chunk_index}"

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
            summary_type=summary_type,
            article_number=article_number,
            chunk_id=chunk_id,
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
    ) -> dict[str, Any]:
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

    SCHEMA_SOURCE = "embedded_fallback"


def validate_chunk(chunk: dict[str, Any]) -> bool:
    """청크 레코드 필수 필드 검증"""
    required = {"id", "source_id", "data_type", "title", "content", "vector"}
    return required.issubset(chunk.keys())
