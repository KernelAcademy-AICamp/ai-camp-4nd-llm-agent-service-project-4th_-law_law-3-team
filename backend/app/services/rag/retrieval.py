"""
RAG 검색 서비스

벡터 검색(요약문, LanceDB) + 키워드 검색(원문, PostgreSQL FTS) 하이브리드 검색.
USE_HYBRID_SEARCH=true 시 양쪽 결과를 source_id 단위 RRF로 병합.
"""

import asyncio
import logging
import warnings
from typing import Any, Dict, List, Optional

from app.core.config import settings
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


async def _get_chunk_content(store: Any, chunk_id: str, source_id: Optional[str] = None) -> str:
    """청크 ID로 content 조회"""
    try:
        result = await asyncio.to_thread(store.get_by_id, chunk_id)
        if result:
            content = result.get("content", "")
            if content:
                return content
    except Exception as e:
        logger.debug("LanceDB 청크 조회 실패: %s, %s", chunk_id, e)

    if source_id:
        try:
            from sqlalchemy import select

            from app.core.database import async_session_factory
            from app.models.legal_document import LegalDocument

            async with async_session_factory() as session:
                result = await session.execute(
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


async def _search_vector(
    query: str,
    n_results: int,
    doc_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """벡터 검색 (LanceDB, 요약문 기반)."""
    store = get_vector_store()
    query_embedding = create_query_embedding(query)

    if doc_type:
        data_type_map = {"precedent": "판례", "law": "법령"}
        where = {"data_type": data_type_map.get(doc_type, doc_type)}
    else:
        where = None

    results = await asyncio.to_thread(
        store.search,
        query_embedding=query_embedding,
        n_results=n_results,
        where=where,
        include=["metadatas", "distances"],
    )

    if not results or not results.get("ids") or not results["ids"][0]:
        return []

    result_documents = results.get("documents", [[]])[0] if results.get("documents") else []

    documents: List[Dict[str, Any]] = []
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

        content = ""
        if i < len(result_documents) and result_documents[i]:
            content = result_documents[i]
        else:
            source_id = raw_metadata.get("source_id")
            content = await _get_chunk_content(store, chunk_id, source_id)

        doc = {
            "id": chunk_id,
            "content": content,
            "metadata": metadata,
            "similarity": 1 - results["distances"][0][i] if results.get("distances") else 0,
        }
        documents.append(doc)

    return documents


def _unique_source_ids(docs: List[Dict[str, Any]]) -> List[str]:
    """문서 리스트에서 순서를 유지하며 고유 source_id 추출."""
    seen: set[str] = set()
    result: List[str] = []
    for doc in docs:
        sid = doc.get("metadata", {}).get("doc_id", "")
        if sid and sid not in seen:
            seen.add(sid)
            result.append(sid)
    return result


def _best_doc_per_source(docs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """source_id별 최고 similarity 문서만 유지."""
    best: Dict[str, Dict[str, Any]] = {}
    for doc in docs:
        sid = doc.get("metadata", {}).get("doc_id", "")
        if not sid:
            continue
        if sid not in best or doc.get("similarity", 0) > best[sid].get("similarity", 0):
            best[sid] = doc
    return best


async def search_relevant_documents(
    query: str,
    n_results: int = 5,
    doc_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    관련 법률 문서 검색 (하이브리드 지원)

    USE_HYBRID_SEARCH=true 시:
        1. 벡터 검색 (LanceDB, 요약문) → source_id 랭킹
        2. 키워드 검색 (PostgreSQL FTS, 원문) → source_id 랭킹
        3. RRF 병합 (source_id 단위) → 최종 랭킹

    USE_HYBRID_SEARCH=false 시:
        기존 벡터 검색만 수행.

    Args:
        query: 검색 쿼리
        n_results: 반환할 결과 수
        doc_type: 문서 유형 필터 (precedent, law, constitutional)

    Returns:
        관련 문서 목록
    """
    # 벡터 검색 수행 (하이브리드 시 더 많은 후보 확보)
    vector_fetch = n_results * 3 if settings.USE_HYBRID_SEARCH else n_results
    vector_results = await _search_vector(query, vector_fetch, doc_type)

    if not settings.USE_HYBRID_SEARCH:
        return vector_results[:n_results]

    # 키워드 검색
    from app.services.rag.keyword_search import (
        is_fts_available,
        search_by_keyword,
    )

    if not await is_fts_available():
        logger.info("fts_index 테이블 비어있음 → 벡터 검색만 사용")
        return vector_results[:n_results]

    keyword_results = await search_by_keyword(query, n_results=vector_fetch, doc_type=doc_type)

    if not keyword_results:
        return vector_results[:n_results]

    # source_id 단위 RRF 병합
    from app.services.rag.fusion import reciprocal_rank_fusion

    vector_source_ids = _unique_source_ids(vector_results)
    keyword_source_ids = _unique_source_ids(keyword_results)
    fused_source_ids = reciprocal_rank_fusion(vector_source_ids, keyword_source_ids)

    # source_id별 최고 문서 선택 (벡터 결과 우선, 키워드 전용 결과도 포함)
    vector_best = _best_doc_per_source(vector_results)
    keyword_best = _best_doc_per_source(keyword_results)

    merged: List[Dict[str, Any]] = []
    for sid in fused_source_ids:
        if sid in vector_best:
            merged.append(vector_best[sid])
        elif sid in keyword_best:
            merged.append(keyword_best[sid])

        if len(merged) >= n_results:
            break

    return merged


# 하위 호환성 별칭
search_relevant_documents_async = search_relevant_documents


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

    async def search(
        self,
        query: str,
        n_results: int = 5,
        doc_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """관련 문서 검색"""
        return await search_relevant_documents(query, n_results, doc_type)

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
