"""
RAG 검색 서비스

VectorStore에서 관련 문서를 검색
"""

import logging
import warnings
from typing import Any, Dict, List, Optional

from app.services.rag.embedding import create_query_embedding
from app.tools.vectorstore import get_vector_store
from app.tools.vectorstore.lancedb import LanceDBStore

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


def _process_search_result(store: Any, result: Any, is_vector: bool = True) -> List[Dict[str, Any]]:
    """검색 결과를 공통 문서 포맷으로 변환"""
    if not result or not result.ids or not result.ids[0]:
        return []

    documents = []
    ids = result.ids[0]
    # distance가 없는 경우 0으로 채움
    distances = result.distances[0] if result.distances and result.distances[0] else [0] * len(ids)

    for i, chunk_id in enumerate(ids):
        # 인덱스 범위 체크
        if i >= len(distances):
            break

        raw_metadata = result.metadatas[0][i] if result.metadatas and result.metadatas[0] else {}

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

        # 벡터 검색의 경우 distance는 작을수록 좋음 (유사도 = 1 - distance)
        # 키워드 검색의 경우 score는 클수록 좋음 (여기서는 그대로 사용)
        score = distances[i]
        similarity = 1 - score if is_vector else score

        doc = {
            "id": chunk_id,
            "content": content,
            "metadata": metadata,
            "similarity": similarity,
            "score": score  # 원본 점수
        }
        documents.append(doc)

    return documents


def rrf_fusion(
    vector_results: List[Dict[str, Any]],
    keyword_results: List[Dict[str, Any]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """
    RRF (Reciprocal Rank Fusion) 알고리즘으로 검색 결과 병합

    Score = 1 / (k + rank)

    Args:
        vector_results: 벡터 검색 결과 리스트
        keyword_results: 키워드 검색 결과 리스트
        k: RRF 상수 (기본값 60)

    Returns:
        병합 및 정렬된 문서 리스트
    """
    fused_scores = {}
    doc_map = {}

    # 벡터 검색 결과 처리
    for rank, doc in enumerate(vector_results):
        doc_id = doc["id"]
        doc_map[doc_id] = doc
        if doc_id not in fused_scores:
            fused_scores[doc_id] = 0
        fused_scores[doc_id] += 1 / (k + rank + 1)

    # 키워드 검색 결과 처리
    for rank, doc in enumerate(keyword_results):
        doc_id = doc["id"]
        # 이미 벡터 검색에 있으면 그 문서 사용, 없으면 추가
        if doc_id not in doc_map:
            doc_map[doc_id] = doc
        if doc_id not in fused_scores:
            fused_scores[doc_id] = 0
        fused_scores[doc_id] += 1 / (k + rank + 1)

    # 점수 기준 정렬
    sorted_doc_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)

    # 결과 리스트 생성
    fused_results = []
    for doc_id in sorted_doc_ids:
        doc = doc_map[doc_id]
        doc["rrf_score"] = fused_scores[doc_id]
        fused_results.append(doc)

    return fused_results


def search_relevant_documents(
    query: str,
    n_results: int = 5,
    doc_type: Optional[str] = None,
    enable_hybrid: bool = False,
) -> List[Dict[str, Any]]:
    """
    관련 법률 문서 검색

    Args:
        query: 검색 쿼리
        n_results: 반환할 결과 수
        doc_type: 문서 유형 필터 (precedent, law, constitutional)
        enable_hybrid: 하이브리드 검색 활성화 여부

    Returns:
        관련 문서 목록
    """
    store = get_vector_store()

    # 필터 조건 설정
    where = None
    if doc_type:
        data_type_map = {"precedent": "판례", "law": "법령"}
        where = {"data_type": data_type_map.get(doc_type, doc_type)}

    # 1. 벡터 검색 수행
    query_embedding = create_query_embedding(query)
    vector_search_result = store.search(
        query_embedding=query_embedding,
        n_results=n_results,
        where=where,
        include=["metadatas", "distances"],
    )
    vector_docs = _process_search_result(store, vector_search_result, is_vector=True)

    # 하이브리드 검색이 아니거나 LanceDB가 아닌 경우 벡터 결과만 반환
    if not enable_hybrid or not isinstance(store, LanceDBStore):
        return vector_docs

    # 2. 키워드 검색 수행 (LanceDB FTS)
    try:
        keyword_search_result = store.search_keyword(
            query_text=query,
            n_results=n_results,
            where=where,
        )
        keyword_docs = _process_search_result(store, keyword_search_result, is_vector=False)
    except Exception as e:
        logger.warning("키워드 검색 실패: %s", e)
        keyword_docs = []

    if not keyword_docs:
        return vector_docs

    # 3. RRF로 결과 병합
    fused_docs = rrf_fusion(vector_docs, keyword_docs)

    # 상위 n_results 개 반환
    return fused_docs[:n_results]


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
