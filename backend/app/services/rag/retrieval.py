"""
RAG 검색 서비스

벡터 검색(LanceDB) + 키워드 검색(PostgreSQL FTS) 하이브리드 검색.
검색은 source_id 기반으로 수행하며, 원문은 PostgreSQL에서 배치 조회.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, NamedTuple, Optional

from langsmith import traceable
from sqlalchemy import select, text

from app.core.config import settings
from app.core.database import sync_session_factory
from app.models.fts_index import FtsIndex
from app.services.rag.embedding import create_query_embedding
from app.tools.vectorstore import get_vector_store

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 테이블 레지스트리
# ---------------------------------------------------------------------------


_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def _validate_identifier(name: str) -> str:
    """SQL 식별자(테이블명/컬럼명)가 안전한 형식인지 검증."""
    if not _IDENTIFIER_PATTERN.match(name):
        raise ValueError(f"안전하지 않은 SQL 식별자: {name!r}")
    return name


class TableConfig(NamedTuple):
    """PostgreSQL 원문 테이블 조회 설정."""

    table_name: str
    id_column: str
    content_columns: tuple[str, ...]


# fts_index.data_type / LanceDB data_type (한국어) → PostgreSQL 테이블 매핑
DOCUMENT_TABLE_REGISTRY: dict[str, list[TableConfig]] = {
    "판례": [
        TableConfig("precedent_documents", "serial_number", ("ruling", "reasoning")),
    ],
    "법령": [
        TableConfig("law_documents", "law_id", ("content",)),
    ],
    "행정규칙": [
        TableConfig("admin_rule_documents", "serial_number", ("content",)),
    ],
    "부처유권해석": [
        TableConfig(
            "interpretation_ministry_documents",
            "serial_number",
            ("answer", "reason"),
        ),
    ],
    "헌재결정례": [
        TableConfig(
            "constitutional_documents", "serial_number", ("ruling", "reasoning")
        ),
    ],
    "행정심판례": [
        TableConfig("administration_documents", "serial_number", ("ruling", "reason")),
    ],
    "법령해석례": [
        TableConfig("legislation_documents", "serial_number", ("answer", "reason")),
    ],
    "조약": [
        TableConfig("treaty_documents", "serial_number", ("content",)),
    ],
    "특별행정심판": [
        TableConfig(
            "special_admin_appeal_documents", "serial_number", ("ruling", "reason")
        ),
    ],
    "위원회결정례": [
        TableConfig(
            "dec_labor_documents",
            "serial_number",
            ("judgment_summary", "judgment_result"),
        ),
        TableConfig(
            "dec_human_rights_documents", "serial_number", ("ruling", "reason")
        ),
        TableConfig("dec_privacy_documents", "serial_number", ("reason",)),
        TableConfig(
            "dec_employment_documents", "serial_number", ("ruling", "reason")
        ),
        TableConfig(
            "dec_financial_documents",
            "serial_number",
            ("action_reason", "action_content"),
        ),
        TableConfig(
            "dec_industrial_documents", "serial_number", ("ruling", "reason")
        ),
        TableConfig(
            "dec_environment_documents", "serial_number", ("ruling", "case_overview")
        ),
        TableConfig(
            "dec_securities_documents",
            "serial_number",
            ("action_reason", "action_content"),
        ),
        TableConfig(
            "dec_civil_rights_documents", "serial_number", ("ruling", "reason")
        ),
        TableConfig(
            "dec_fair_trade_documents", "serial_number", ("ruling", "reason")
        ),
    ],
}

# doc_type (API, 영어) ↔ data_type (DB, 한국어) 변환
_DOC_TYPE_TO_DATA_TYPE: dict[str, str] = {"precedent": "판례", "law": "법령"}
_DATA_TYPE_TO_DOC_TYPE: dict[str, str] = {"판례": "precedent", "법령": "law"}


# ---------------------------------------------------------------------------
# 유틸리티
# ---------------------------------------------------------------------------


def _resolve_data_type(doc_type_or_data_type: str) -> str:
    """doc_type(영어) 또는 data_type(한국어) → 항상 한국어 data_type."""
    return _DOC_TYPE_TO_DATA_TYPE.get(doc_type_or_data_type, doc_type_or_data_type)


def _to_doc_type(data_type: str) -> str:
    """data_type(한국어) → doc_type. 매핑 없으면 원본 반환."""
    return _DATA_TYPE_TO_DOC_TYPE.get(data_type, data_type)


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


# ---------------------------------------------------------------------------
# 벡터 검색 (ID 기반)
# ---------------------------------------------------------------------------


@traceable(name="vector_search")
def _search_vector_ids(
    query: str,
    n_results: int,
    doc_type: Optional[str] = None,
    exclude_doc_types: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """벡터 검색 — source_id + metadata 반환 (content 미포함).

    동일 source_id 청크 중 최고 유사도만 유지하고
    유사도 내림차순으로 정렬하여 반환.

    Args:
        query: 검색 쿼리
        n_results: 반환할 결과 수
        doc_type: 포함할 문서 유형 필터
        exclude_doc_types: 제외할 data_type 목록 (한국어, doc_type 미지정 시만 적용)
    """
    store = get_vector_store()
    query_embedding = create_query_embedding(query)

    where: dict[str, Any] | None = None
    if doc_type:
        where = {"data_type": _resolve_data_type(doc_type)}
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

    # source_id 단위 deduplicate (최고 유사도 유지)
    best: dict[str, dict[str, Any]] = {}

    for i, _chunk_id in enumerate(results["ids"][0]):
        raw_meta = results["metadatas"][0][i] if results.get("metadatas") else {}
        source_id = raw_meta.get("source_id", "")
        if not source_id:
            continue

        similarity = (
            1 - results["distances"][0][i] if results.get("distances") else 0.0
        )
        data_type_val = raw_meta.get("data_type", "")

        if source_id in best and similarity <= best[source_id].get("similarity", 0):
            continue

        best[source_id] = {
            "id": source_id,
            "content": "",
            "metadata": {
                "case_name": raw_meta.get("title", ""),
                "case_number": raw_meta.get("case_number", ""),
                "data_type": data_type_val,
                "doc_type": _to_doc_type(data_type_val),
                "court_name": raw_meta.get("source_name", ""),
                "doc_id": source_id,
                "date": raw_meta.get("date", ""),
            },
            "similarity": similarity,
            "score_type": "cosine",
        }

    # 유사도 내림차순 정렬 (RRF 랭킹용)
    docs = list(best.values())
    docs.sort(key=lambda d: d.get("similarity", 0), reverse=True)
    return docs


# ---------------------------------------------------------------------------
# 원문 배치 조회
# ---------------------------------------------------------------------------


@traceable(name="fetch_contents")
def fetch_document_contents(
    id_to_data_type: dict[str, str],
) -> dict[str, str]:
    """source_id별 원문 텍스트를 PostgreSQL에서 배치 조회.

    테이블별 원문 컬럼을 결합하여 반환. ai_summary는 사용하지 않음.
    위원회결정례(1:10 테이블)는 순차 조회로 처리.

    Args:
        id_to_data_type: {source_id: data_type(한국어)} 매핑

    Returns:
        {source_id: "컬럼1 내용\\n\\n컬럼2 내용"} 매핑
    """
    if not id_to_data_type:
        return {}

    # data_type별 source_id 그룹화
    type_groups: dict[str, list[str]] = {}
    for source_id, data_type in id_to_data_type.items():
        type_groups.setdefault(data_type, []).append(source_id)

    result: dict[str, str] = {}

    with sync_session_factory() as session:
        for data_type, source_ids in type_groups.items():
            table_configs = DOCUMENT_TABLE_REGISTRY.get(data_type)
            if not table_configs:
                logger.warning(
                    "미등록 data_type: %s (%d건)", data_type, len(source_ids)
                )
                continue

            remaining_ids = set(source_ids)

            for tc in table_configs:
                if not remaining_ids:
                    break

                # 화이트리스트 검증: 테이블명/컬럼명이 안전한 식별자인지 확인
                safe_table = _validate_identifier(tc.table_name)
                safe_id_col = _validate_identifier(tc.id_column)
                safe_content_cols = [
                    _validate_identifier(c) for c in tc.content_columns
                ]

                cols = ", ".join([safe_id_col, *safe_content_cols])
                sql = text(
                    f"SELECT {cols} FROM {safe_table} "
                    f"WHERE {safe_id_col} = ANY(:ids)"
                )
                rows = session.execute(
                    sql, {"ids": list(remaining_ids)}
                ).fetchall()

                for row in rows:
                    sid = str(row[0])
                    parts = [
                        str(row[col_idx + 1])
                        for col_idx in range(len(tc.content_columns))
                        if row[col_idx + 1]
                    ]
                    result[sid] = "\n\n".join(parts)
                    remaining_ids.discard(sid)

            if remaining_ids:
                logger.debug(
                    "%s: %d건 원문 미발견 (예: %s)",
                    data_type,
                    len(remaining_ids),
                    list(remaining_ids)[:3],
                )

    return result


def _lookup_data_types(source_ids: list[str]) -> dict[str, str]:
    """fts_index에서 source_id → data_type 매핑 조회 (fallback용)."""
    if not source_ids:
        return {}

    with sync_session_factory() as session:
        rows = session.execute(
            select(FtsIndex.source_id, FtsIndex.data_type).where(
                FtsIndex.source_id.in_(source_ids)
            )
        ).all()
        return {row.source_id: row.data_type for row in rows}


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
            resolved = _resolve_data_type(meta.get("doc_type", ""))
            if resolved in DOCUMENT_TABLE_REGISTRY:
                result[sid] = resolved
            else:
                missing_ids.append(sid)

    # fts_index fallback
    if missing_ids:
        result.update(_lookup_data_types(missing_ids))

    return result


def _populate_content(
    docs: list[dict[str, Any]],
    contents: dict[str, str],
) -> None:
    """검색 결과에 원문 content 주입 (in-place)."""
    for doc in docs:
        sid = doc.get("metadata", {}).get("doc_id", "")
        if sid and sid in contents:
            doc["content"] = contents[sid]


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
            meta["data_type"] = _resolve_data_type(meta.get("doc_type", ""))

    # RRF 병합 — 리랭킹을 위해 전체 반환 (잘라내지 않음)
    from app.services.rag.fusion import reciprocal_rank_fusion

    vector_source_ids = _unique_source_ids(vector_results)
    keyword_source_ids = _unique_source_ids(keyword_results)
    fused_source_ids = reciprocal_rank_fusion(vector_source_ids, keyword_source_ids)

    # source_id별 최고 문서 선택 (벡터 결과 우선)
    vector_best = _best_doc_per_source(vector_results)
    keyword_best = _best_doc_per_source(keyword_results)

    merged: list[dict[str, Any]] = []
    for sid in fused_source_ids:
        if sid in vector_best:
            merged.append(vector_best[sid])
        elif sid in keyword_best:
            merged.append(keyword_best[sid])

    return merged


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
    _populate_content(docs, contents)
    return docs


@traceable(name="fetch_summaries")
def fetch_ai_summaries(
    id_to_data_type: dict[str, str],
) -> dict[str, str]:
    """source_id별 ai_summary를 PostgreSQL에서 배치 조회.

    리랭킹용 요약문으로 사용. 모든 인제스트 테이블에 ai_summary 컬럼이 존재.

    Args:
        id_to_data_type: {source_id: data_type(한국어)} 매핑

    Returns:
        {source_id: ai_summary 텍스트} 매핑
    """
    if not id_to_data_type:
        return {}

    # data_type별 source_id 그룹화
    type_groups: dict[str, list[str]] = {}
    for source_id, data_type in id_to_data_type.items():
        type_groups.setdefault(data_type, []).append(source_id)

    result: dict[str, str] = {}

    with sync_session_factory() as session:
        for data_type, source_ids in type_groups.items():
            table_configs = DOCUMENT_TABLE_REGISTRY.get(data_type)
            if not table_configs:
                continue

            remaining_ids = set(source_ids)

            for tc in table_configs:
                if not remaining_ids:
                    break

                safe_table = _validate_identifier(tc.table_name)
                safe_id_col = _validate_identifier(tc.id_column)

                sql = text(
                    f"SELECT {safe_id_col}, ai_summary "
                    f"FROM {safe_table} "
                    f"WHERE {safe_id_col} = ANY(:ids) "
                    f"AND ai_summary IS NOT NULL"
                )
                rows = session.execute(
                    sql, {"ids": list(remaining_ids)}
                ).fetchall()

                for row in rows:
                    sid = str(row[0])
                    summary = str(row[1]) if row[1] else ""
                    if summary:
                        result[sid] = summary
                        remaining_ids.discard(sid)

    return result


def fetch_lancedb_summaries(source_ids: list[str]) -> dict[str, str]:
    """(deprecated) LanceDB에서 source_id별 청크 텍스트 조회.

    리랭킹에는 fetch_ai_summaries()를 사용하세요.
    """
    if not source_ids:
        return {}

    try:
        import lancedb

        db = lancedb.connect(settings.LANCEDB_URI)
        table = db.open_table(settings.LANCEDB_TABLE_NAME)

        safe_pattern = re.compile(r"^[\w\\-]+$")
        safe_ids = [sid for sid in source_ids if safe_pattern.match(sid)]
        if not safe_ids:
            return {}

        ids_str = ", ".join(
            "'{}'".format(sid.replace("'", "''")) for sid in safe_ids
        )
        df = table.search().where(
            f"source_id IN ({ids_str})", prefilter=True
        ).select(["source_id", "content"]).limit(len(safe_ids) * 2).to_pandas()

        result: dict[str, str] = {}
        for _, row in df.iterrows():
            sid = row["source_id"]
            if sid not in result:
                result[sid] = row["content"]
        return result
    except Exception as e:
        logger.warning("LanceDB 요약문 조회 실패: %s", e)
        return {}


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
) -> list[dict[str, Any]]:
    """벡터 + FTS 병렬 하이브리드 검색 (async, content 미포함).

    벡터 검색과 FTS 키워드 검색을 ``asyncio.gather``로 동시에 실행한 뒤
    RRF로 병합한다. 동기 ``search_without_content``의 병렬 버전.
    """
    from app.services.rag.keyword_search import (
        is_fts_available_sync,
        search_by_keyword,
    )

    # 하이브리드 검색 비활성화 또는 FTS 불가 시 벡터 검색만
    if not settings.USE_HYBRID_SEARCH or not is_fts_available_sync():
        vector_results = await asyncio.to_thread(
            _search_vector_ids, query, n_results, doc_type, exclude_doc_types,
        )
        return vector_results[:n_results]

    # 벡터 + FTS 동시 실행
    vector_results, keyword_results = await asyncio.gather(
        asyncio.to_thread(
            _search_vector_ids, query, n_results, doc_type, exclude_doc_types,
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
            meta["data_type"] = _resolve_data_type(meta.get("doc_type", ""))

    from app.services.rag.fusion import reciprocal_rank_fusion

    vector_source_ids = _unique_source_ids(vector_results)
    keyword_source_ids = _unique_source_ids(keyword_results)
    fused_source_ids = reciprocal_rank_fusion(vector_source_ids, keyword_source_ids)

    vector_best = _best_doc_per_source(vector_results)
    keyword_best = _best_doc_per_source(keyword_results)

    merged: list[dict[str, Any]] = []
    for sid in fused_source_ids:
        if sid in vector_best:
            merged.append(vector_best[sid])
        elif sid in keyword_best:
            merged.append(keyword_best[sid])

    return merged


def _fetch_contents_for_type(
    data_type: str,
    source_ids: list[str],
) -> dict[str, str]:
    """단일 data_type에 대한 원문 조회 (스레드 풀 병렬화용)."""
    table_configs = DOCUMENT_TABLE_REGISTRY.get(data_type)
    if not table_configs:
        logger.warning("미등록 data_type: %s (%d건)", data_type, len(source_ids))
        return {}

    result: dict[str, str] = {}
    remaining_ids = set(source_ids)

    with sync_session_factory() as session:
        for tc in table_configs:
            if not remaining_ids:
                break

            safe_table = _validate_identifier(tc.table_name)
            safe_id_col = _validate_identifier(tc.id_column)
            safe_content_cols = [_validate_identifier(c) for c in tc.content_columns]

            cols = ", ".join([safe_id_col, *safe_content_cols])
            sql = text(
                f"SELECT {cols} FROM {safe_table} "
                f"WHERE {safe_id_col} = ANY(:ids)"
            )
            rows = session.execute(sql, {"ids": list(remaining_ids)}).fetchall()

            for row in rows:
                sid = str(row[0])
                parts = [
                    str(row[col_idx + 1])
                    for col_idx in range(len(tc.content_columns))
                    if row[col_idx + 1]
                ]
                result[sid] = "\n\n".join(parts)
                remaining_ids.discard(sid)

    return result


async def fetch_document_contents_async(
    id_to_data_type: dict[str, str],
) -> dict[str, str]:
    """source_id별 원문을 data_type 그룹별로 병렬 조회 (async).

    동기 ``fetch_document_contents``의 병렬 버전.
    """
    if not id_to_data_type:
        return {}

    type_groups: dict[str, list[str]] = {}
    for source_id, data_type in id_to_data_type.items():
        type_groups.setdefault(data_type, []).append(source_id)

    tasks = [
        asyncio.to_thread(_fetch_contents_for_type, dt, sids)
        for dt, sids in type_groups.items()
    ]
    group_results = await asyncio.gather(*tasks)

    merged: dict[str, str] = {}
    for partial in group_results:
        merged.update(partial)
    return merged


def _fetch_summaries_for_type(
    data_type: str,
    source_ids: list[str],
) -> dict[str, str]:
    """단일 data_type에 대한 ai_summary 조회 (스레드 풀 병렬화용)."""
    table_configs = DOCUMENT_TABLE_REGISTRY.get(data_type)
    if not table_configs:
        return {}

    result: dict[str, str] = {}
    remaining_ids = set(source_ids)

    with sync_session_factory() as session:
        for tc in table_configs:
            if not remaining_ids:
                break

            safe_table = _validate_identifier(tc.table_name)
            safe_id_col = _validate_identifier(tc.id_column)

            sql = text(
                f"SELECT {safe_id_col}, ai_summary "
                f"FROM {safe_table} "
                f"WHERE {safe_id_col} = ANY(:ids) "
                f"AND ai_summary IS NOT NULL"
            )
            rows = session.execute(sql, {"ids": list(remaining_ids)}).fetchall()

            for row in rows:
                sid = str(row[0])
                summary = str(row[1]) if row[1] else ""
                if summary:
                    result[sid] = summary
                    remaining_ids.discard(sid)

    return result


async def fetch_ai_summaries_async(
    id_to_data_type: dict[str, str],
) -> dict[str, str]:
    """source_id별 ai_summary를 data_type 그룹별로 병렬 조회 (async).

    동기 ``fetch_ai_summaries``의 병렬 버전.
    """
    if not id_to_data_type:
        return {}

    type_groups: dict[str, list[str]] = {}
    for source_id, data_type in id_to_data_type.items():
        type_groups.setdefault(data_type, []).append(source_id)

    tasks = [
        asyncio.to_thread(_fetch_summaries_for_type, dt, sids)
        for dt, sids in type_groups.items()
    ]
    group_results = await asyncio.gather(*tasks)

    merged: dict[str, str] = {}
    for partial in group_results:
        merged.update(partial)
    return merged
