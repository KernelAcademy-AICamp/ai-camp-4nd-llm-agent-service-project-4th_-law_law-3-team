"""
PostgreSQL tsvector 기반 키워드 검색

fts_index 테이블의 원문 tsvector를 검색하여
벡터 검색(요약문)과 상호보완하는 키워드 매칭 수행.

retrieval.py의 search_relevant_documents()와 동일한 반환 형식을 유지하여
RRF 병합이 가능하도록 한다.
content는 fts_index에 미저장이므로 빈 문자열로 반환하며,
retrieval.py에서 원본 테이블 조회로 보충한다.
"""

import logging
from typing import Any, Optional

from sqlalchemy import func, select

from app.core.database import sync_session_factory
from app.models.fts_index import FtsIndex
from app.services.rag.tsvector_builder import tokens_to_tsquery

logger = logging.getLogger(__name__)


def _get_query_tokens(query: str) -> list[str]:
    """쿼리를 MeCab 토큰으로 분해 (thread-local 캐싱, legal_dict/userdic 포함)."""
    try:
        from app.tools.vectorstore.lancedb import _get_thread_tokenizer

        tokenizer = _get_thread_tokenizer()
        return tokenizer.morphs(query)
    except Exception:
        logger.warning("MeCab 사용 불가, 공백 분리 fallback")
        return query.strip().split()


def _map_doc_type_to_data_type(doc_type: str) -> str:
    """API doc_type을 DB data_type으로 변환."""
    mapping = {"precedent": "판례", "law": "법령"}
    return mapping.get(doc_type, doc_type)


def _map_data_type_to_doc_type(data_type: str) -> str:
    """DB data_type을 API doc_type으로 변환."""
    mapping = {"판례": "precedent", "법령": "law"}
    return mapping.get(data_type, data_type.lower() if data_type else "")


def search_by_keyword(
    query: str,
    n_results: int = 50,
    doc_type: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    PostgreSQL tsvector 기반 키워드 검색.

    fts_index 테이블에서 원문 tsvector를 검색하여 메타데이터를 반환.
    content는 fts_index에 미저장이므로 빈 문자열로 반환.
    retrieval.py에서 원본 테이블 조회로 content를 보충한다.

    Args:
        query: 검색 쿼리
        n_results: 반환할 최대 결과 수
        doc_type: 문서 유형 필터 ("precedent", "law")

    Returns:
        [{"id": source_id, "content": "", "metadata": dict, "similarity": float}, ...]
    """
    tokens = _get_query_tokens(query)
    if not tokens:
        return []

    tsquery_str = tokens_to_tsquery(tokens, operator="|")
    if not tsquery_str:
        return []

    try:
        with sync_session_factory() as session:
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

            stmt = stmt.order_by(rank_expr.desc()).limit(n_results)

            rows = session.execute(stmt).all()

            if not rows:
                return []

            max_rank = max(row.rank for row in rows) if rows else 1.0
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
                })

            return documents

    except Exception as e:
        logger.warning("키워드 검색 실패 (fts_index 미생성 또는 비어있음): %s", e)
        return []


def is_fts_available() -> bool:
    """fts_index 테이블에 데이터가 있는지 확인."""
    try:
        with sync_session_factory() as session:
            count = session.execute(
                select(func.count()).select_from(FtsIndex)
            ).scalar_one()
            return count > 0
    except Exception:
        return False
