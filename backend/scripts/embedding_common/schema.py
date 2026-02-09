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
        LAW_COLUMNS,
        LEGAL_CHUNKS_SCHEMA,
        PRECEDENT_COLUMNS,
        TABLE_NAME,
        VECTOR_DIM,
        LegalChunk,
        create_law_chunk,
        create_precedent_chunk,
        get_law_columns,
        get_precedent_columns,
    )

    SCHEMA_SOURCE = "app.tools.vectorstore.schema_v2"

except ImportError:
    # 백엔드 미설치 환경 (RunPod, Colab)에서 독립 스키마 정의
    import pyarrow as pa

    VECTOR_DIM = 1024
    TABLE_NAME = "legal_chunks"

    LEGAL_CHUNKS_SCHEMA = pa.schema([
        pa.field("id", pa.utf8()),
        pa.field("source_id", pa.utf8()),
        pa.field("data_type", pa.utf8()),
        pa.field("title", pa.utf8()),
        pa.field("content", pa.utf8()),
        pa.field("content_tokenized", pa.utf8()),
        pa.field("vector", pa.list_(pa.float32(), VECTOR_DIM)),
        pa.field("date", pa.utf8()),
        pa.field("source_name", pa.utf8()),
        pa.field("chunk_index", pa.int32()),
        pa.field("total_chunks", pa.int32()),
        # 법령 전용
        pa.field("promulgation_date", pa.utf8()),
        pa.field("promulgation_no", pa.utf8()),
        pa.field("law_type", pa.utf8()),
        pa.field("article_no", pa.utf8()),
        # 판례 전용
        pa.field("case_number", pa.utf8()),
        pa.field("case_type", pa.utf8()),
        pa.field("judgment_type", pa.utf8()),
        pa.field("judgment_status", pa.utf8()),
        pa.field("reference_provisions", pa.utf8()),
        pa.field("reference_cases", pa.utf8()),
    ])

    COMMON_COLUMNS = [
        "id", "source_id", "data_type", "title", "content",
        "content_tokenized", "vector", "date", "source_name",
        "chunk_index", "total_chunks",
    ]
    LAW_COLUMNS = ["promulgation_date", "promulgation_no", "law_type", "article_no"]
    PRECEDENT_COLUMNS = [
        "case_number", "case_type", "judgment_type", "judgment_status",
        "reference_provisions", "reference_cases",
    ]
    ALL_COLUMNS = COMMON_COLUMNS + LAW_COLUMNS + PRECEDENT_COLUMNS

    LegalChunk = dict  # type: ignore[assignment,misc]

    def create_law_chunk(
        source_id: str,
        chunk_index: int,
        title: str,
        content: str,
        vector: list[float],
        enforcement_date: str,
        department: str,
        total_chunks: int = 1,
        promulgation_date: Optional[str] = None,
        promulgation_no: Optional[str] = None,
        law_type: Optional[str] = None,
        article_no: Optional[str] = None,
        content_tokenized: Optional[str] = None,
    ) -> dict[str, Any]:
        return {
            "id": f"law_{source_id}_{chunk_index}",
            "source_id": source_id,
            "data_type": "법령",
            "title": title,
            "content": content,
            "content_tokenized": content_tokenized,
            "vector": vector,
            "date": enforcement_date,
            "source_name": department,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "promulgation_date": promulgation_date,
            "promulgation_no": promulgation_no,
            "law_type": law_type,
            "article_no": article_no,
            "case_number": None,
            "case_type": None,
            "judgment_type": None,
            "judgment_status": None,
            "reference_provisions": None,
            "reference_cases": None,
        }

    def create_precedent_chunk(
        source_id: str,
        chunk_index: int,
        title: str,
        content: str,
        vector: list[float],
        decision_date: str,
        court_name: str,
        total_chunks: int = 1,
        case_number: Optional[str] = None,
        case_type: Optional[str] = None,
        judgment_type: Optional[str] = None,
        judgment_status: Optional[str] = None,
        reference_provisions: Optional[str] = None,
        reference_cases: Optional[str] = None,
        content_tokenized: Optional[str] = None,
    ) -> dict[str, Any]:
        return {
            "id": f"prec_{source_id}_{chunk_index}",
            "source_id": source_id,
            "data_type": "판례",
            "title": title,
            "content": content,
            "content_tokenized": content_tokenized,
            "vector": vector,
            "date": decision_date,
            "source_name": court_name,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "promulgation_date": None,
            "promulgation_no": None,
            "law_type": None,
            "article_no": None,
            "case_number": case_number,
            "case_type": case_type,
            "judgment_type": judgment_type,
            "judgment_status": judgment_status,
            "reference_provisions": reference_provisions,
            "reference_cases": reference_cases,
        }

    def get_law_columns() -> list[str]:
        return LAW_COLUMNS

    def get_precedent_columns() -> list[str]:
        return PRECEDENT_COLUMNS

    SCHEMA_SOURCE = "embedded_fallback"


def validate_chunk(chunk: dict[str, Any]) -> bool:
    """청크 레코드 필수 필드 검증"""
    required = {"id", "source_id", "data_type", "title", "content", "vector"}
    return required.issubset(chunk.keys())
