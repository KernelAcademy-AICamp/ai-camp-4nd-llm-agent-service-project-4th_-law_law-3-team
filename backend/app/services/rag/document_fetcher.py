"""
RAG 문서 원문/요약문 조회 서비스

PostgreSQL에서 source_id 기반으로 원문(content_columns)과
ai_summary를 배치 조회하는 함수 모음.
법령 조문 단위 교체, 판례/법령 메타데이터 보강 포함.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langsmith import traceable
from sqlalchemy import select, text

from app.core.database import sync_session_factory
from app.models.fts_index import FtsIndex
from app.models.law_article import LawArticle
from app.services.rag.table_registry import (
    _DEC_TABLE_BY_PREFIX,
    DOCUMENT_TABLE_REGISTRY,
    TableConfig,
    _validate_identifier,
    group_dec_source_ids,
)

logger = logging.getLogger(__name__)

__all__ = [
    "fetch_document_contents",
    "fetch_document_contents_async",
    "fetch_ai_summaries",
    "fetch_ai_summaries_async",
    "populate_content",
    "apply_law_article_content",
    "populate_law_metadata",
    "populate_precedent_metadata",
    "populate_rerank_text",
    "populate_rerank_text_from_contents",
]


@traceable(name="fetch_contents")
def fetch_document_contents(
    id_to_data_type: dict[str, str],
) -> dict[str, dict[str, str]]:
    """source_id별 원문을 컬럼별 구조화 dict로 PostgreSQL에서 배치 조회.

    DOCUMENT_TABLE_REGISTRY의 content_columns를 개별 키로 반환.
    위원회결정례는 접두사 기반 직접 라우팅으로 11테이블 순차 스캔을 회피.

    Args:
        id_to_data_type: {source_id: data_type(한국어)} 매핑

    Returns:
        {source_id: {column_name: value, ...}} 매핑
    """
    if not id_to_data_type:
        return {}

    # data_type별 source_id 그룹화
    type_groups: dict[str, list[str]] = {}
    for source_id, data_type in id_to_data_type.items():
        type_groups.setdefault(data_type, []).append(source_id)

    result: dict[str, dict[str, str]] = {}
    for data_type, source_ids in type_groups.items():
        partial = _fetch_contents_for_type(data_type, source_ids)
        result.update(partial)

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


def populate_content(
    docs: list[dict[str, Any]],
    contents: dict[str, dict[str, str]],
) -> None:
    """검색 결과에 구조화 원문 주입 (in-place).

    content_fields에 컬럼별 dict, content에 조인 문자열(하위 호환)을 세팅.
    법령은 제외 — apply_law_article_content()에서 조문 단위로 별도 처리.
    """
    for doc in docs:
        meta = doc.get("metadata", {})
        if meta.get("data_type") == "법령":
            continue
        sid = meta.get("doc_id", "")
        if sid and sid in contents:
            fields = contents[sid]
            doc["content_fields"] = fields
            doc["content"] = "\n\n".join(fields.values())


def apply_law_article_content(docs: list[dict[str, Any]]) -> None:
    """법령 문서의 content를 조문 단위로 교체 (in-place).

    벡터 검색에서 article_number가 전파된 법령 문서:
        → law_articles 테이블에서 해당 조문만 조회하여 content 교체
    키워드 검색 only (article_number 없음):
        → content 비움 (법령명만 참조, 전체 텍스트 주입 방지)
    """
    # article_number가 있는 법령 문서 수집
    article_queries: list[tuple[int, str, str]] = []  # (doc_idx, source_id, article_number)
    for idx, doc in enumerate(docs):
        meta = doc.get("metadata", {})
        if meta.get("data_type") != "법령":
            continue

        article_number = meta.get("article_number", "")
        source_id = meta.get("doc_id", "")

        if article_number and source_id:
            article_queries.append((idx, source_id, article_number))
        else:
            # 키워드 only: content 비움 (법령명만 참조)
            doc["content"] = ""
            doc["content_fields"] = {}

    if not article_queries:
        return

    # law_articles 배치 조회
    from sqlalchemy import and_, or_

    conditions = [
        and_(
            LawArticle.law_id == source_id,
            LawArticle.article_number == article_num,
        )
        for _, source_id, article_num in article_queries
    ]

    row_map: dict[tuple[str, str], LawArticle] = {}
    with sync_session_factory() as session:
        rows = session.execute(
            select(LawArticle).where(or_(*conditions))
        ).scalars().all()
        for row in rows:
            row_map[(str(row.law_id), str(row.article_number))] = row

    # content 교체
    for doc_idx, source_id, article_num in article_queries:
        article = row_map.get((source_id, article_num))
        if article:
            docs[doc_idx]["content"] = article.article_content
            docs[doc_idx]["content_fields"] = {"content": article.article_content}
        else:
            # law_articles에 없으면 기존 content 유지
            logger.debug(
                "law_articles 미발견: law_id=%s, article_number=%s",
                source_id, article_num,
            )


def populate_law_metadata(docs: list[dict[str, Any]]) -> None:
    """법령 문서의 law_type·ministry·article_title 보강 (in-place).

    벡터 검색 결과의 법령 문서에 law_documents/law_articles 테이블에서
    메타데이터를 배치 조회하여 주입.
    """
    law_docs = [
        (idx, doc)
        for idx, doc in enumerate(docs)
        if doc.get("metadata", {}).get("data_type") == "법령"
    ]
    if not law_docs:
        return

    # 1) law_documents 배치 조회 (law_type, ministry)
    law_ids = list({
        doc.get("metadata", {}).get("doc_id", "")
        for _, doc in law_docs
        if doc.get("metadata", {}).get("doc_id")
    })
    law_meta_map: dict[str, dict[str, str]] = {}
    if law_ids:
        with sync_session_factory() as session:
            rows = session.execute(
                text(
                    "SELECT law_id, law_type, ministry "
                    "FROM law_documents WHERE law_id = ANY(:ids)"
                ),
                {"ids": law_ids},
            ).fetchall()
            for row in rows:
                law_meta_map[str(row[0])] = {
                    "law_type": row[1] or "",
                    "ministry": row[2] or "",
                }

    # 2) law_articles 배치 조회 (article_title)
    article_queries: list[tuple[str, str]] = [
        (
            doc.get("metadata", {}).get("doc_id", ""),
            doc.get("metadata", {}).get("article_number", ""),
        )
        for _, doc in law_docs
        if doc.get("metadata", {}).get("article_number")
        and doc.get("metadata", {}).get("doc_id")
    ]
    article_title_map: dict[tuple[str, str], str] = {}
    if article_queries:
        from sqlalchemy import and_, or_

        conditions = [
            and_(
                LawArticle.law_id == lid,
                LawArticle.article_number == anum,
            )
            for lid, anum in article_queries
        ]
        with sync_session_factory() as session:
            rows = session.execute(
                select(
                    LawArticle.law_id,
                    LawArticle.article_number,
                    LawArticle.article_title,
                ).where(or_(*conditions))
            ).fetchall()
            for row in rows:
                article_title_map[(str(row[0]), str(row[1]))] = row[2] or ""

    # 3) in-place 주입
    for _, doc in law_docs:
        meta = doc.get("metadata", {})
        lid = meta.get("doc_id", "")
        if lid in law_meta_map:
            extra = law_meta_map[lid]
            if not meta.get("law_type"):
                meta["law_type"] = extra["law_type"]
            if not meta.get("ministry"):
                meta["ministry"] = extra["ministry"]
        anum = meta.get("article_number", "")
        if anum and (lid, anum) in article_title_map:
            meta["article_title"] = article_title_map[(lid, anum)]


def populate_precedent_metadata(docs: list[dict[str, Any]]) -> None:
    """판례 문서의 case_number·decision_date·court_name 보강 (in-place).

    벡터 검색 결과는 LanceDB에 case_number가 없어 빈 문자열이므로,
    원문 조회 후 precedent_documents 테이블에서 메타 컬럼을 배치 조회하여 주입.
    """
    precedent_sids = [
        doc.get("metadata", {}).get("doc_id", "")
        for doc in docs
        if doc.get("metadata", {}).get("data_type") == "판례"
        and not doc.get("metadata", {}).get("case_number")
    ]
    if not precedent_sids:
        return

    meta_map: dict[str, dict[str, str]] = {}
    with sync_session_factory() as session:
        rows = session.execute(
            text(
                "SELECT serial_number, case_number, decision_date, court_name "
                "FROM precedent_documents "
                "WHERE serial_number = ANY(:ids)"
            ),
            {"ids": precedent_sids},
        ).fetchall()
        for row in rows:
            meta_map[str(row[0])] = {
                "case_number": row[1] or "",
                "decision_date": str(row[2]) if row[2] else "",
                "court_name": row[3] or "",
            }

    for doc in docs:
        meta = doc.get("metadata", {})
        sid = meta.get("doc_id", "")
        if sid in meta_map:
            extra = meta_map[sid]
            if not meta.get("case_number"):
                meta["case_number"] = extra["case_number"]
            if not meta.get("date") and extra["decision_date"]:
                meta["date"] = extra["decision_date"]
            if not meta.get("court_name") and extra["court_name"]:
                meta["court_name"] = extra["court_name"]


def populate_rerank_text(
    docs: list[dict[str, Any]],
    summaries: dict[str, str],
) -> None:
    """검색 결과에 ai_summary를 rerank_text로 주입 (리랭킹용, in-place)."""
    for doc in docs:
        sid = doc.get("metadata", {}).get("doc_id", "")
        if sid and sid in summaries:
            doc["rerank_text"] = summaries[sid]


def populate_rerank_text_from_contents(
    docs: list[dict[str, Any]],
    contents: dict[str, dict[str, str]],
) -> None:
    """원문 컬럼을 rerank_text로 주입 (리랭킹용, in-place).

    DOCUMENT_TABLE_REGISTRY에 정의된 content_columns(ruling, reasoning 등)를
    결합하여 rerank_text에 세팅한다. ai_summary 대비 cross-encoder 입력 품질 향상.
    """
    for doc in docs:
        sid = doc.get("metadata", {}).get("doc_id", "")
        if sid and sid in contents:
            doc["rerank_text"] = "\n\n".join(contents[sid].values())


def _fetch_contents_for_type(
    data_type: str,
    source_ids: list[str],
) -> dict[str, dict[str, str]]:
    """단일 data_type에 대한 원문 조회 (스레드 풀 병렬화용)."""
    table_configs = DOCUMENT_TABLE_REGISTRY.get(data_type)
    if not table_configs:
        logger.warning("미등록 data_type: %s (%d건)", data_type, len(source_ids))
        return {}

    result: dict[str, dict[str, str]] = {}

    # 위원회결정례: 접두사 기반 직접 라우팅 (11테이블 순차 스캔 회피)
    if data_type == "위원회결정례" and _DEC_TABLE_BY_PREFIX:
        routed, unrouted = group_dec_source_ids(source_ids)
        remaining_ids = set(unrouted)

        with sync_session_factory() as session:
            # 접두사 있는 ID: 해당 테이블 1개만 조회
            for tc, sid_serial_map in routed.items():
                _query_content_rows(session, tc, sid_serial_map, result)

            # 접두사 없는 ID (벡터 검색 결과): 기존 순차 스캔
            for tc in table_configs:
                if not remaining_ids:
                    break
                found = _query_content_rows(
                    session, tc, {sid: sid for sid in remaining_ids}, result,
                )
                remaining_ids -= found

        if remaining_ids:
            logger.debug(
                "%s: %d건 원문 미발견 (예: %s)",
                data_type, len(remaining_ids), list(remaining_ids)[:3],
            )

        return result

    # 기타 data_type: 기존 로직 (테이블 1개)
    remaining_ids = set(source_ids)

    with sync_session_factory() as session:
        for tc in table_configs:
            if not remaining_ids:
                break
            found = _query_content_rows(
                session, tc, {sid: sid for sid in remaining_ids}, result,
            )
            remaining_ids -= found

    if remaining_ids:
        logger.debug(
            "%s: %d건 원문 미발견 (예: %s)",
            data_type, len(remaining_ids), list(remaining_ids)[:3],
        )

    return result


def _query_content_rows(
    session: Any,
    tc: TableConfig,
    sid_serial_map: dict[str, str],
    result: dict[str, dict[str, str]],
) -> set[str]:
    """단일 테이블에서 원문 컬럼 조회.

    Args:
        sid_serial_map: {original_source_id: query_serial_number}
        result: 결과를 누적할 dict (in-place 수정)

    Returns:
        조회 성공한 original_source_id 집합
    """
    if not sid_serial_map:
        return set()

    safe_table = _validate_identifier(tc.table_name)
    safe_id_col = _validate_identifier(tc.id_column)
    safe_content_cols = [_validate_identifier(c) for c in tc.content_columns]

    cols = ", ".join([safe_id_col, *safe_content_cols])
    query_ids = list(sid_serial_map.values())
    sql = text(
        f"SELECT {cols} FROM {safe_table} "
        f"WHERE {safe_id_col} = ANY(:ids)"
    )
    rows = session.execute(sql, {"ids": query_ids}).fetchall()

    # serial → original_sid 역매핑
    serial_to_sid = {v: k for k, v in sid_serial_map.items()}
    found: set[str] = set()

    for row in rows:
        serial = str(row[0])
        orig_sid = serial_to_sid.get(serial, serial)
        fields: dict[str, str] = {}
        for col_idx, col_name in enumerate(tc.content_columns):
            val = row[col_idx + 1]
            if val:
                fields[col_name] = str(val)
        if fields:
            result[orig_sid] = fields
            found.add(orig_sid)

    return found


async def fetch_document_contents_async(
    id_to_data_type: dict[str, str],
) -> dict[str, dict[str, str]]:
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

    merged: dict[str, dict[str, str]] = {}
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

    # 위원회결정례: 접두사 기반 직접 라우팅 (11테이블 순차 스캔 회피)
    if data_type == "위원회결정례" and _DEC_TABLE_BY_PREFIX:
        routed, unrouted = group_dec_source_ids(source_ids)
        remaining_ids = set(unrouted)

        with sync_session_factory() as session:
            for tc, sid_serial_map in routed.items():
                _query_summary_rows(session, tc, sid_serial_map, result)

            for tc in table_configs:
                if not remaining_ids:
                    break
                found = _query_summary_rows(
                    session, tc, {sid: sid for sid in remaining_ids}, result,
                )
                remaining_ids -= found

        if remaining_ids:
            logger.debug(
                "%s: %d건 요약문 미발견 (예: %s)",
                data_type, len(remaining_ids), list(remaining_ids)[:3],
            )

        return result

    # 기타 data_type: 기존 로직
    remaining_ids = set(source_ids)

    with sync_session_factory() as session:
        for tc in table_configs:
            if not remaining_ids:
                break
            found = _query_summary_rows(
                session, tc, {sid: sid for sid in remaining_ids}, result,
            )
            remaining_ids -= found

    if remaining_ids:
        logger.debug(
            "%s: %d건 요약문 미발견 (예: %s)",
            data_type, len(remaining_ids), list(remaining_ids)[:3],
        )

    return result


def _query_summary_rows(
    session: Any,
    tc: TableConfig,
    sid_serial_map: dict[str, str],
    result: dict[str, str],
) -> set[str]:
    """단일 테이블에서 ai_summary 조회.

    Args:
        sid_serial_map: {original_source_id: query_serial_number}
        result: 결과를 누적할 dict (in-place 수정)

    Returns:
        조회 성공한 original_source_id 집합
    """
    if not sid_serial_map:
        return set()

    safe_table = _validate_identifier(tc.table_name)
    safe_id_col = _validate_identifier(tc.id_column)

    query_ids = list(sid_serial_map.values())
    sql = text(
        f"SELECT {safe_id_col}, ai_summary "
        f"FROM {safe_table} "
        f"WHERE {safe_id_col} = ANY(:ids) "
        f"AND ai_summary IS NOT NULL"
    )
    rows = session.execute(sql, {"ids": query_ids}).fetchall()

    serial_to_sid = {v: k for k, v in sid_serial_map.items()}
    found: set[str] = set()

    for row in rows:
        serial = str(row[0])
        orig_sid = serial_to_sid.get(serial, serial)
        summary = str(row[1]) if row[1] else ""
        if summary:
            result[orig_sid] = summary
            found.add(orig_sid)

    return found


@traceable(name="fetch_summaries")
def fetch_ai_summaries(
    id_to_data_type: dict[str, str],
) -> dict[str, str]:
    """source_id별 ai_summary를 PostgreSQL에서 배치 조회.

    리랭킹용 요약문으로 사용. DOCUMENT_TABLE_REGISTRY 기반으로
    각 data_type에 해당하는 원문 테이블에서 ai_summary 컬럼을 조회.
    위원회결정례는 접두사 기반 직접 라우팅으로 11테이블 순차 스캔을 회피.

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
    for data_type, source_ids in type_groups.items():
        partial = _fetch_summaries_for_type(data_type, source_ids)
        result.update(partial)

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
