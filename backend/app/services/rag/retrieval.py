"""
RAG 검색 서비스

벡터 검색(LanceDB) + 키워드 검색(PostgreSQL FTS) 하이브리드 검색.
검색은 source_id 기반으로 수행하며, 원문은 PostgreSQL에서 배치 조회.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from langsmith import traceable

from app.core.config import settings
from app.services.rag.document_fetcher import (
    _lookup_data_types,
    apply_law_article_content,
    fetch_ai_summaries,  # noqa: F401  # re-export (pipeline.py, benchmark scripts)
    fetch_ai_summaries_async,  # noqa: F401  # re-export (pipeline.py)
    fetch_document_contents,
    fetch_document_contents_async,  # noqa: F401  # re-export (pipeline.py)
    populate_content,
    populate_law_metadata,
    populate_precedent_metadata,
    populate_rerank_text,
    populate_rerank_text_from_contents,
)
from app.services.rag.embedding import create_query_embedding
from app.services.rag.table_registry import (
    DOCUMENT_TABLE_REGISTRY,
    TableConfig,  # noqa: F401  # re-export (__init__.py)
    group_dec_source_ids,
    resolve_data_type,
    to_doc_type,
)
from app.tools.vectorstore import get_vector_store

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# re-export 선언 (mypy attr-defined 해소)
# ---------------------------------------------------------------------------
__all__ = [
    # document_fetcher re-exports
    "fetch_ai_summaries",
    "fetch_ai_summaries_async",
    "fetch_document_contents",
    "fetch_document_contents_async",
    # table_registry re-exports
    "DOCUMENT_TABLE_REGISTRY",
    "TableConfig",
]

# ---------------------------------------------------------------------------
# 하위 호환 별칭 (기존 코드가 retrieval에서 직접 import하던 이름들)
# ---------------------------------------------------------------------------

# table_registry re-exports
_resolve_data_type = resolve_data_type
_to_doc_type = to_doc_type
_group_dec_source_ids = group_dec_source_ids

# document_fetcher re-exports (내부 함수명 하위 호환)
_populate_content = populate_content
_apply_law_article_content = apply_law_article_content
_populate_law_metadata = populate_law_metadata
_populate_precedent_metadata = populate_precedent_metadata
_populate_rerank_text = populate_rerank_text
_populate_rerank_text_from_contents = populate_rerank_text_from_contents


# ---------------------------------------------------------------------------
# 유틸리티
# ---------------------------------------------------------------------------


def _unique_source_ids(docs: list[dict[str, Any]]) -> list[str]:
    """문서 리스트에서 순서를 유지하며 고유 source_id 추출."""
    seen: set[str] = set()
    result: list[str] = []
    for doc in docs:
        sid = doc.get("metadata", {}).get("doc_id", "")
        if sid and sid not in seen:
            seen.add(sid)
            result.append(sid)
    return result


def _best_doc_per_source(docs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """source_id별 최고 similarity 문서만 유지."""
    best: dict[str, dict[str, Any]] = {}
    for doc in docs:
        sid = doc.get("metadata", {}).get("doc_id", "")
        if not sid:
            continue
        if sid not in best or doc.get("similarity", 0) > best[sid].get(
            "similarity", 0
        ):
            best[sid] = doc
    return best


def _merge_with_source_tag(
    vector_results: list[dict[str, Any]],
    keyword_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """벡터 + 키워드 결과를 RRF 병합하고 search_source 태깅.

    source_id별 첫 등장 문서를 유지하고 (벡터 우선),
    각 문서에 "vector" | "keyword" | "both" 태그를 부여한다.
    """
    from app.services.rag.fusion import reciprocal_rank_fusion

    fused_sids = reciprocal_rank_fusion(
        _unique_source_ids(vector_results),
        _unique_source_ids(keyword_results),
    )

    vector_sids = {d.get("metadata", {}).get("doc_id", "") for d in vector_results}
    keyword_sids = {d.get("metadata", {}).get("doc_id", "") for d in keyword_results}

    doc_map: dict[str, dict[str, Any]] = {}
    for doc in [*vector_results, *keyword_results]:
        sid = doc.get("metadata", {}).get("doc_id", "")
        if sid and sid not in doc_map:
            doc_map[sid] = doc

    merged: list[dict[str, Any]] = []
    for sid in fused_sids:
        if sid not in doc_map:
            continue
        doc = doc_map[sid]
        if sid in vector_sids and sid in keyword_sids:
            doc["search_source"] = "both"
        elif sid in keyword_sids:
            doc["search_source"] = "keyword"
        else:
            doc["search_source"] = "vector"
        merged.append(doc)

    return merged


# ---------------------------------------------------------------------------
# 벡터 검색 (ID 기반)
# ---------------------------------------------------------------------------


@traceable(name="vector_search")
def _search_vector_ids(
    query: str,
    n_results: int,
    doc_type: Optional[str] = None,
    exclude_doc_types: Optional[list[str]] = None,
    *,
    query_embedding: list[float] | None = None,
) -> list[dict[str, Any]]:
    """벡터 검색 — source_id + metadata 반환 (content 미포함).

    동일 source_id 청크 중 최고 유사도만 유지하고
    유사도 내림차순으로 정렬하여 반환.

    Args:
        query: 검색 쿼리
        n_results: 반환할 결과 수
        doc_type: 포함할 문서 유형 필터
        exclude_doc_types: 제외할 data_type 목록 (한국어, doc_type 미지정 시만 적용)
        query_embedding: 사전 계산된 임베딩 (None이면 내부 계산)
    """
    store = get_vector_store()
    if query_embedding is None:
        query_embedding = create_query_embedding(query)

    where: dict[str, Any] | None = None
    if doc_type:
        where = {"data_type": resolve_data_type(doc_type)}
    elif exclude_doc_types:
        where = {"data_type": {"$not_in": exclude_doc_types}}

    results = store.search(
        query_embedding=query_embedding,
        n_results=n_results,
        where=where,
        include=["metadatas", "distances"],
    )

    if not results or not results.get("ids") or not results["ids"][0]:
        return []

    # (source_id, article_number) 복합키로 deduplicate (최고 유사도 유지)
    # 같은 법령이라도 다른 조문이면 별도 결과로 유지
    best: dict[tuple[str, str], dict[str, Any]] = {}

    for i, _chunk_id in enumerate(results["ids"][0]):
        raw_meta = results["metadatas"][0][i] if results.get("metadatas") else {}
        source_id = raw_meta.get("source_id", "")
        if not source_id:
            continue

        similarity = (
            1 - results["distances"][0][i] if results.get("distances") else 0.0
        )
        data_type_val = raw_meta.get("data_type", "")
        article_number = str(raw_meta.get("article_number") or "")
        summary_type = str(raw_meta.get("summary_type") or "")

        dedup_key = (source_id, article_number)
        if dedup_key in best and similarity <= best[dedup_key].get(
            "similarity", 0
        ):
            continue

        best[dedup_key] = {
            "id": source_id,
            "content": "",
            "metadata": {
                "case_name": raw_meta.get("title", ""),
                "case_number": raw_meta.get("case_number", ""),
                "data_type": data_type_val,
                "doc_type": to_doc_type(data_type_val),
                "court_name": raw_meta.get("source_name", ""),
                "doc_id": source_id,
                "date": raw_meta.get("date", ""),
                "article_number": article_number,
                "summary_type": summary_type,
            },
            "similarity": similarity,
            "score_type": "cosine",
            "search_source": "vector",
        }

    docs = list(best.values())
    docs.sort(key=lambda d: d.get("similarity", 0), reverse=True)

    return docs


# ---------------------------------------------------------------------------
# 하이브리드 검색
# ---------------------------------------------------------------------------


def _extract_id_data_type_map(docs: list[dict[str, Any]]) -> dict[str, str]:
    """검색 결과에서 {source_id: data_type(한국어)} 매핑 추출."""
    result: dict[str, str] = {}
    missing_ids: list[str] = []

    for doc in docs:
        meta = doc.get("metadata", {})
        sid = meta.get("doc_id", "")
        if not sid:
            continue

        data_type = meta.get("data_type", "")
        if data_type and data_type in DOCUMENT_TABLE_REGISTRY:
            result[sid] = data_type
        else:
            # doc_type → data_type 변환 시도
            resolved = resolve_data_type(meta.get("doc_type", ""))
            if resolved in DOCUMENT_TABLE_REGISTRY:
                result[sid] = resolved
            else:
                missing_ids.append(sid)

    # fts_index fallback
    if missing_ids:
        result.update(_lookup_data_types(missing_ids))

    return result


@traceable(name="hybrid_search")
def search_without_content(
    query: str,
    n_results: int = 5,
    doc_type: Optional[str] = None,
    exclude_doc_types: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """
    하이브리드 검색 (content 미포함).

    벡터 + FTS + RRF 병합까지 수행하되, 원문 조회는 하지 않음.
    리랭킹 파이프라인에서 요약문 기반 리랭킹 후 top-k만 원문 조회할 때 사용.

    Args:
        query: 검색 쿼리
        n_results: 반환할 결과 수
        doc_type: 문서 유형 필터
        exclude_doc_types: 제외할 data_type 목록 (한국어)

    Returns:
        관련 문서 목록 (content는 빈 문자열)
    """
    vector_results = _search_vector_ids(
        query, n_results, doc_type, exclude_doc_types
    )

    if not settings.USE_HYBRID_SEARCH:
        return vector_results[:n_results]

    from app.services.rag.keyword_search import (
        is_fts_available_sync,
        search_by_keyword,
    )

    if not is_fts_available_sync():
        logger.info("fts_index 비어있음 → 벡터 검색만 사용")
        return vector_results[:n_results]

    keyword_results = search_by_keyword(
        query,
        n_results=n_results,
        doc_type=doc_type,
        exclude_doc_types=exclude_doc_types,
    )

    if not keyword_results:
        return vector_results[:n_results]

    # keyword_results에 data_type 보강 (doc_type → data_type 변환)
    for doc in keyword_results:
        meta = doc.get("metadata", {})
        if "data_type" not in meta:
            meta["data_type"] = resolve_data_type(meta.get("doc_type", ""))

    # RRF 병합 + search_source 태깅
    return _merge_with_source_tag(vector_results, keyword_results)


def search_relevant_documents(
    query: str,
    n_results: int = 5,
    doc_type: Optional[str] = None,
    exclude_doc_types: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """
    관련 법률 문서 하이브리드 검색 (원문 포함).

    search_without_content() + 원문 배치 조회.

    Args:
        query: 검색 쿼리
        n_results: 반환할 결과 수
        doc_type: 문서 유형 필터
        exclude_doc_types: 제외할 data_type 목록 (한국어)

    Returns:
        관련 문서 목록 (content는 PostgreSQL 원문)
    """
    docs = search_without_content(query, n_results, doc_type, exclude_doc_types)
    contents = fetch_document_contents(_extract_id_data_type_map(docs))
    populate_content(docs, contents)
    apply_law_article_content(docs)
    return docs


async def search_relevant_documents_async(
    query: str,
    n_results: int = 5,
    doc_type: Optional[str] = None,
    exclude_doc_types: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """관련 법률 문서 비동기 검색.

    sync 함수를 별도 스레드에서 실행하여 이벤트 루프 블로킹 방지.
    """
    return await asyncio.to_thread(
        search_relevant_documents, query, n_results, doc_type, exclude_doc_types
    )


# ---------------------------------------------------------------------------
# 병렬 async 래퍼
# ---------------------------------------------------------------------------


async def search_without_content_async(
    query: str,
    n_results: int = 5,
    doc_type: Optional[str] = None,
    exclude_doc_types: Optional[list[str]] = None,
    *,
    query_embedding: list[float] | None = None,
) -> list[dict[str, Any]]:
    """벡터 + FTS 병렬 하이브리드 검색 (async, content 미포함).

    벡터 검색과 FTS 키워드 검색을 ``asyncio.gather``로 동시에 실행한 뒤
    RRF로 병합한다. 동기 ``search_without_content``의 병렬 버전.

    Args:
        query: 검색 쿼리
        n_results: 반환할 결과 수
        doc_type: 문서 유형 필터
        exclude_doc_types: 제외할 data_type 목록
        query_embedding: 사전 계산된 임베딩 (focus 모드 공유용)
    """
    from app.services.rag.keyword_search import (
        is_fts_available_sync,
        search_by_keyword,
    )

    # 하이브리드 검색 비활성화 또는 FTS 불가 시 벡터 검색만
    if not settings.USE_HYBRID_SEARCH or not is_fts_available_sync():
        vector_results = await asyncio.to_thread(
            _search_vector_ids, query, n_results, doc_type, exclude_doc_types,
            query_embedding=query_embedding,
        )
        return vector_results[:n_results]

    # 벡터 + FTS 동시 실행
    vector_results, keyword_results = await asyncio.gather(
        asyncio.to_thread(
            _search_vector_ids, query, n_results, doc_type, exclude_doc_types,
            query_embedding=query_embedding,
        ),
        asyncio.to_thread(
            search_by_keyword,
            query,
            n_results=n_results,
            doc_type=doc_type,
            exclude_doc_types=exclude_doc_types,
        ),
    )

    if not keyword_results:
        return vector_results[:n_results]

    # keyword_results에 data_type 보강
    for doc in keyword_results:
        meta = doc.get("metadata", {})
        if "data_type" not in meta:
            meta["data_type"] = resolve_data_type(meta.get("doc_type", ""))

    # RRF 병합 + search_source 태깅
    return _merge_with_source_tag(vector_results, keyword_results)
