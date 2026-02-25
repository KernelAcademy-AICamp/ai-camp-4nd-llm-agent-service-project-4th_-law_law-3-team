"""
PostgreSQL tsvector 기반 키워드 검색

fts_index 테이블의 원문 tsvector를 검색하여
벡터 검색(요약문)과 상호보완하는 키워드 매칭 수행.

검색 전략: 개념 단위 AND → OR fallback
1. 사용자 입력을 공백 분리하여 개념 단위로 그룹핑
2. 각 개념을 MeCab으로 분해 → 개념 내 OR, 개념 간 AND
3. 결과가 부족하면 전체 OR로 fallback

retrieval.py의 search_relevant_documents()와 동일한 반환 형식을 유지하여
RRF 병합이 가능하도록 한다.
content는 fts_index에 미저장이므로 빈 문자열로 반환하며,
retrieval.py에서 원본 테이블 조회로 보충한다.
"""

import logging
import re
from typing import Any, Optional

from langsmith import traceable
from sqlalchemy import func, select

from app.core.database import async_session_factory, sync_session_factory
from app.models.fts_index import FtsIndex

logger = logging.getLogger(__name__)

# tsvector 토큰에 허용되지 않는 문자
_INVALID_TOKEN_RE = re.compile(r"['\\\x00]")

# 개념 AND 결과가 이 수보다 적으면 OR fallback
_CONCEPT_AND_MIN_RESULTS = 5


@traceable(name="mecab_tokenize")
def _tokenize(text: str) -> list[str]:
    """텍스트를 MeCab 토큰으로 분해 (명사만, 2자 이상)."""
    try:
        from app.tools.vectorstore.lancedb import _get_thread_tokenizer
        from app.tools.vectorstore.mecab_tokenizer import FTS_POS_TAGS

        tokenizer = _get_thread_tokenizer()
        tokens = tokenizer.morphs(text, pos_filter=FTS_POS_TAGS)
    except Exception:
        logger.warning("MeCab 사용 불가, 공백 분리 fallback")
        tokens = text.strip().split()

    return [t for t in tokens if len(t) >= 2]


def _clean_token(token: str) -> str:
    """tsquery에 안전한 토큰으로 정리."""
    token = token.strip()
    return _INVALID_TOKEN_RE.sub("", token)


@traceable(name="build_concept_and_tsquery")
def _build_concept_and_tsquery(query: str) -> str:
    """개념 단위 AND tsquery 생성.

    사용자 입력을 공백으로 분리하여 각 개념을 MeCab으로 분해.
    개념 간 AND, 개념 내 OR로 연결.

    예: "교통사고 손해배상 판례"
        → (교통사고 | 교통 | 사고) & (손해 | 배상) & 판례
    """
    concepts = [w for w in query.strip().split() if len(w) >= 2]
    if not concepts:
        return ""

    groups: list[list[str]] = []
    for concept in concepts:
        tokens = [_clean_token(t) for t in _tokenize(concept)]
        tokens = [t for t in tokens if t]
        if tokens:
            groups.append(tokens)

    if not groups:
        return ""

    parts: list[str] = []
    for tokens in groups:
        if len(tokens) == 1:
            parts.append(tokens[0])
        else:
            parts.append("(" + " | ".join(tokens) + ")")

    return " & ".join(parts)


@traceable(name="build_or_tsquery")
def _build_or_tsquery(query: str) -> str:
    """전체 OR tsquery 생성 (fallback용)."""
    tokens = [_clean_token(t) for t in _tokenize(query)]
    tokens = [t for t in tokens if t]
    if not tokens:
        return ""
    return " | ".join(tokens)


def _get_query_tokens(query: str) -> list[str]:
    """쿼리를 MeCab 토큰으로 분해 (하위 호환용).

    1자 토큰은 FTS에서 고빈도 매칭을 유발하므로 제거한다.
    """
    return _tokenize(query)


def _map_doc_type_to_data_type(doc_type: str) -> str:
    """API doc_type을 DB data_type으로 변환."""
    mapping = {"precedent": "판례", "law": "법령"}
    return mapping.get(doc_type, doc_type)


def _map_data_type_to_doc_type(data_type: str) -> str:
    """DB data_type을 API doc_type으로 변환."""
    mapping = {"판례": "precedent", "법령": "law"}
    return mapping.get(data_type, data_type.lower() if data_type else "")


@traceable(name="execute_fts_query")
def _execute_fts_query(
    session: Any,
    tsquery_str: str,
    n_results: int,
    doc_type: Optional[str],
    exclude_doc_types: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """tsquery 문자열로 FTS 검색 실행."""
    tsquery_expr = func.to_tsquery("simple", tsquery_str)
    rank_expr = func.ts_rank(FtsIndex.content_tsvector, tsquery_expr)

    stmt = select(
        FtsIndex.source_id,
        FtsIndex.data_type,
        FtsIndex.title,
        FtsIndex.date,
        FtsIndex.source_name,
        FtsIndex.case_number,
        rank_expr.label("rank"),
    ).where(FtsIndex.content_tsvector.op("@@")(tsquery_expr))

    if doc_type:
        data_type = _map_doc_type_to_data_type(doc_type)
        stmt = stmt.where(FtsIndex.data_type == data_type)
    elif exclude_doc_types:
        stmt = stmt.where(FtsIndex.data_type.not_in(exclude_doc_types))

    stmt = stmt.order_by(rank_expr.desc()).limit(n_results)

    rows = session.execute(stmt).all()

    if not rows:
        return []

    max_rank = max(row.rank for row in rows)
    if max_rank == 0:
        max_rank = 1.0

    documents: list[dict[str, Any]] = []
    for row in rows:
        metadata = {
            "case_name": row.title or "",
            "case_number": row.case_number or "",
            "doc_type": _map_data_type_to_doc_type(row.data_type),
            "court_name": row.source_name or "",
            "doc_id": row.source_id,
            "date": row.date or "",
        }

        documents.append({
            "id": row.source_id,
            "content": "",
            "metadata": metadata,
            "similarity": row.rank / max_rank,
            "score_type": "fts_rank",
        })

    return documents


@traceable(name="keyword_search")
def search_by_keyword(
    query: str,
    n_results: int = 50,
    doc_type: Optional[str] = None,
    exclude_doc_types: Optional[list[str]] = None,
    *,
    precomputed_concept_tsq: Optional[str] = None,
    precomputed_or_tsq: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    PostgreSQL tsvector 기반 키워드 검색.

    개념 단위 AND를 먼저 시도하고, 결과가 부족하면 전체 OR로 fallback.

    Args:
        query: 검색 쿼리
        n_results: 반환할 최대 결과 수
        doc_type: 문서 유형 필터 ("precedent", "law")
        exclude_doc_types: 제외할 data_type 목록 (한국어)
        precomputed_concept_tsq: 사전 계산된 개념 AND tsquery (focus 모드 공유용)
        precomputed_or_tsq: 사전 계산된 OR tsquery (focus 모드 공유용)

    Returns:
        [{"id": source_id, "content": "", "metadata": dict, "similarity": float}, ...]
    """
    concept_tsq = precomputed_concept_tsq or _build_concept_and_tsquery(query)
    if not concept_tsq:
        return []

    try:
        with sync_session_factory() as session:
            # Step 1: 개념 AND 검색 (빠름, GIN 인덱스 활용)
            results = _execute_fts_query(
                session, concept_tsq, n_results, doc_type, exclude_doc_types
            )

            if len(results) >= _CONCEPT_AND_MIN_RESULTS:
                return results

            # Step 2: OR fallback (느리지만 recall 보장)
            logger.info(
                "개념 AND 결과 부족 (%d건 < %d), OR fallback",
                len(results),
                _CONCEPT_AND_MIN_RESULTS,
            )
            or_tsq = precomputed_or_tsq or _build_or_tsquery(query)
            if not or_tsq:
                return results

            return _execute_fts_query(
                session, or_tsq, n_results, doc_type, exclude_doc_types
            )

    except Exception as e:
        logger.warning("키워드 검색 실패 (fts_index 미생성 또는 비어있음): %s", e)
        return []


def is_fts_available_sync() -> bool:
    """fts_index 테이블에 데이터가 있는지 확인 (동기 버전)."""
    try:
        with sync_session_factory() as session:
            result = session.execute(
                select(func.count()).select_from(FtsIndex)
            )
            count = result.scalar_one()
            return count > 0
    except Exception:
        return False


async def is_fts_available() -> bool:
    """fts_index 테이블에 데이터가 있는지 확인 (비동기 버전)."""
    try:
        async with async_session_factory() as session:
            result = await session.execute(
                select(func.count()).select_from(FtsIndex)
            )
            count = result.scalar_one()
            return count > 0
    except Exception:
        return False
