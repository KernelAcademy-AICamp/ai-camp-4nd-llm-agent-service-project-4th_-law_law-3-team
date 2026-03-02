"""
PostgreSQL pg_textsearch BM25 키워드 검색

fts_index 테이블의 search_text 컬럼에 대해 BM25 스코어링으로 검색.
pg_textsearch의 <@> 연산자는 **음수** BM25 점수를 반환 (PG는 ASC 인덱스 스캔만 지원).
더 작은(더 음수인) 값이 더 높은 관련성을 의미.
ORDER BY <@> ASC LIMIT n 패턴으로 BMW(Block-Max WAND) 최적화 트리거.

retrieval.py의 search_relevant_documents()와 동일한 반환 형식을 유지하여
RRF 병합이 가능하도록 한다.
content는 fts_index에 미저장이므로 빈 문자열로 반환하며,
retrieval.py에서 원본 테이블 조회로 보충한다.
"""

import logging
from typing import Any, Optional

from langsmith import traceable
from sqlalchemy import func, select, text

from app.core.database import async_session_factory, sync_session_factory
from app.models.fts_index import FtsIndex

logger = logging.getLogger(__name__)

# BM25 인덱스 이름 (to_bm25query에 명시 전달 — IDF 정확성 보장)
_BM25_INDEX_NAME = "idx_fts_bm25"

# FTS 가용성 캐시 (서버 수명 동안 유효 — 인제스트 후 재시작 필요)
_fts_available_cache: bool | None = None


@traceable(name="mecab_tokenize")
def _tokenize(query: str) -> list[str]:
    """텍스트를 MeCab 토큰으로 분해 (명사만, 2자 이상)."""
    from app.tools.vectorstore.lancedb import _get_thread_tokenizer

    tokenizer = _get_thread_tokenizer()
    return tokenizer.morphs(query)


def _map_doc_type_to_data_type(doc_type: str) -> str:
    """API doc_type을 DB data_type으로 변환."""
    mapping = {"precedent": "판례", "law": "법령"}
    return mapping.get(doc_type, doc_type)


def _map_data_type_to_doc_type(data_type: str) -> str:
    """DB data_type을 API doc_type으로 변환."""
    mapping = {"판례": "precedent", "법령": "law"}
    return mapping.get(data_type, data_type.lower() if data_type else "")


@traceable(name="execute_bm25_query")
def _execute_bm25_query(
    session: Any,
    query: str,
    n_results: int,
    doc_type: Optional[str],
    exclude_doc_types: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """BM25 스코어링으로 FTS 검색 실행.

    <@> 연산자는 음수 BM25 점수 반환 (더 음수 = 더 관련성 높음).
    BMW(Block-Max WAND) 최적화 트리거:
    ORDER BY <@> ASC LIMIT n 패턴 필수.
    """
    tokens = _tokenize(query)
    if not tokens:
        return []

    search_query = " ".join(tokens)

    # to_bm25query(query_text, index_name): 인덱스명 명시 전달 (IDF 정확성)
    bm25_query = func.to_bm25query(search_query, _BM25_INDEX_NAME)
    score_expr = FtsIndex.search_text.op("<@>")(bm25_query)

    stmt = select(
        FtsIndex.source_id,
        FtsIndex.data_type,
        FtsIndex.title,
        FtsIndex.date,
        FtsIndex.source_name,
        FtsIndex.case_number,
        score_expr.label("rank"),
    )

    if doc_type:
        data_type = _map_doc_type_to_data_type(doc_type)
        stmt = stmt.where(FtsIndex.data_type == data_type)
    elif exclude_doc_types:
        stmt = stmt.where(FtsIndex.data_type.not_in(exclude_doc_types))

    # BMW 트리거: ORDER BY <@> ASC LIMIT n (음수 점수 → ASC가 관련성 높은 순)
    stmt = stmt.order_by(score_expr.asc()).limit(n_results)

    rows = session.execute(stmt).all()

    if not rows:
        return []

    # <@> 음수 점수를 양수로 변환 (abs) — RRF 및 로깅 가독성
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
            "similarity": abs(float(row.rank)),
            "score_type": "bm25",
        })

    return documents


@traceable(name="keyword_search")
def search_by_keyword(
    query: str,
    n_results: int = 50,
    doc_type: Optional[str] = None,
    exclude_doc_types: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """
    PostgreSQL BM25 키워드 검색.

    MeCab 토크나이징 후 to_bm25query()로 BM25 스코어링.
    BMW 최적화로 368K+ 문서에서 고속 검색.

    Args:
        query: 검색 쿼리
        n_results: 반환할 최대 결과 수
        doc_type: 문서 유형 필터 ("precedent", "law")
        exclude_doc_types: 제외할 data_type 목록 (한국어)

    Returns:
        [{"id": source_id, "content": "", "metadata": dict, "similarity": float}, ...]
    """
    if not query.strip():
        return []

    try:
        with sync_session_factory() as session:
            return _execute_bm25_query(
                session, query, n_results, doc_type, exclude_doc_types
            )
    except Exception as e:
        logger.warning("BM25 키워드 검색 실패: %s", e)
        return []


def is_fts_available_sync() -> bool:
    """BM25 FTS 사용 가능 여부 확인 (동기, 캐시).

    USE_BM25_SEARCH=False → 즉시 False (FTS 비활성화).
    USE_BM25_SEARCH=True → pg_indexes에서 idx_fts_bm25 존재 확인.
    서버 수명 동안 결과를 캐시합니다.
    """
    from app.core.config import settings

    if not settings.USE_BM25_SEARCH:
        return False

    global _fts_available_cache  # noqa: PLW0603
    if _fts_available_cache is not None:
        return _fts_available_cache
    try:
        with sync_session_factory() as session:
            result = session.execute(
                text(
                    "SELECT 1 FROM pg_indexes "
                    "WHERE indexname = :idx_name LIMIT 1"
                ),
                {"idx_name": _BM25_INDEX_NAME},
            )
            _fts_available_cache = result.scalar_one_or_none() is not None
            return _fts_available_cache
    except Exception:
        return False


async def is_fts_available() -> bool:
    """BM25 FTS 사용 가능 여부 확인 (비동기, 캐시).

    USE_BM25_SEARCH=False → 즉시 False (FTS 비활성화).
    USE_BM25_SEARCH=True → pg_indexes에서 idx_fts_bm25 존재 확인.
    """
    from app.core.config import settings

    if not settings.USE_BM25_SEARCH:
        return False

    global _fts_available_cache  # noqa: PLW0603
    if _fts_available_cache is not None:
        return _fts_available_cache
    try:
        async with async_session_factory() as session:
            result = await session.execute(
                text(
                    "SELECT 1 FROM pg_indexes "
                    "WHERE indexname = :idx_name LIMIT 1"
                ),
                {"idx_name": _BM25_INDEX_NAME},
            )
            _fts_available_cache = result.scalar_one_or_none() is not None
            return _fts_available_cache
    except Exception:
        return False
