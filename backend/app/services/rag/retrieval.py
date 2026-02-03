"""
RAG 검색 서비스

VectorStore에서 관련 문서를 검색
"""

import logging
import warnings
from typing import Any, Dict, List, Optional

from app.services.rag.embedding import create_query_embedding
from app.tools.vectorstore import get_vector_store

logger = logging.getLogger(__name__)


def _map_data_type(data_type: str) -> str:
    """LanceDB data_type을 표준 doc_type으로 변환"""
    mapping = {
        "판례": "precedent",
        "법령": "law",
        "헌법재판소": "constitutional",
    }
    return mapping.get(data_type, data_type.lower() if data_type else "")


def _get_chunk_content(store: Any, chunk_id: str, source_id: Optional[str] = None) -> str:
    """청크 ID로 content 조회"""
    try:
        result = store.get_by_id(chunk_id)
        if result:
            content = result.get("content", "")
            if content:
                return content
    except Exception as e:
        logger.debug("LanceDB 청크 조회 실패: %s, %s", chunk_id, e)

    if source_id:
        try:
            from app.core.database import sync_session_factory
            from app.models.legal_document import LegalDocument
            from sqlalchemy import select

            with sync_session_factory() as session:
                result = session.execute(
                    select(LegalDocument.embedding_text).where(
                        LegalDocument.serial_number == source_id
                    )
                )
                row = result.scalar_one_or_none()
                if row:
                    return row
        except Exception as e:
            logger.debug("PostgreSQL fallback 실패: %s, %s", source_id, e)

    return ""


def search_relevant_documents(
    query: str,
    n_results: int = 5,
    doc_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    관련 법률 문서 검색

    Args:
        query: 검색 쿼리
        n_results: 반환할 결과 수
        doc_type: 문서 유형 필터 (precedent, law, constitutional)

    Returns:
        관련 문서 목록
    """
    store = get_vector_store()
    query_embedding = create_query_embedding(query)

    # LanceDB는 data_type 필드 사용 (precedent→판례, law→법령)
    if doc_type:
        data_type_map = {"precedent": "판례", "law": "법령"}
        where = {"data_type": data_type_map.get(doc_type, doc_type)}
    else:
        where = None

    results = store.search(
        query_embedding=query_embedding,
        n_results=n_results,
        where=where,
        include=["metadatas", "distances"],
    )

    if not results or not results.get("ids") or not results["ids"][0]:
        return []

    documents = []
    for i, chunk_id in enumerate(results["ids"][0]):
        raw_metadata = results["metadatas"][0][i] if results.get("metadatas") else {}

        metadata = {
            "case_name": raw_metadata.get("title", ""),
            "case_number": raw_metadata.get("case_number", ""),
            "doc_type": _map_data_type(raw_metadata.get("data_type", "")),
            "court_name": raw_metadata.get("source_name", ""),
            "doc_id": raw_metadata.get("source_id"),
            "date": raw_metadata.get("date", ""),
            "chunk_index": raw_metadata.get("chunk_index", 0),
            "total_chunks": raw_metadata.get("total_chunks", 1),
        }

        source_id = raw_metadata.get("source_id")
        content = _get_chunk_content(store, chunk_id, source_id)

        doc = {
            "id": chunk_id,
            "content": content,
            "metadata": metadata,
            "similarity": 1 - results["distances"][0][i] if results.get("distances") else 0,
        }
        documents.append(doc)

    return documents


# =============================================================================
# 하위 호환성 유지용 (Deprecated)
# =============================================================================


class _RetrievalServiceCompat:
    """
    RetrievalService 하위 호환 래퍼

    .. deprecated:: 1.0.0
        `search_relevant_documents()` 함수를 직접 사용하세요.
    """

    def __init__(self) -> None:
        warnings.warn(
            "RetrievalService는 deprecated입니다. "
            "search_relevant_documents() 함수를 직접 사용하세요.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._store = get_vector_store()

    def search(
        self,
        query: str,
        n_results: int = 5,
        doc_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """관련 문서 검색"""
        return search_relevant_documents(query, n_results, doc_type)

    def embed_query(self, text: str) -> List[float]:
        """텍스트 임베딩"""
        return create_query_embedding(text)


# 하위 호환용 alias
RetrievalService = _RetrievalServiceCompat

_retrieval_service: Optional[_RetrievalServiceCompat] = None


def get_retrieval_service() -> _RetrievalServiceCompat:
    """
    RetrievalService 싱글톤 인스턴스 반환

    .. deprecated:: 1.0.0
        `search_relevant_documents()` 함수를 직접 사용하세요.
    """
    global _retrieval_service
    if _retrieval_service is None:
        _retrieval_service = _RetrievalServiceCompat()
    return _retrieval_service
