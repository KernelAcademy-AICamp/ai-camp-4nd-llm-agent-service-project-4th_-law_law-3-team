"""
LanceDB 마이크로서비스 FastAPI 앱

벡터/FTS/하이브리드 검색 API를 제공하는 독립 서비스.
backend에서 HTTP로 통신하여 LanceDB 데이터에 접근.
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException

from schemas import (
    CountResponse,
    DocumentsResponse,
    FtsIndexRequest,
    FtsIndexResponse,
    FtsSearchRequest,
    GetByIdsRequest,
    GetBySourceIdRequest,
    HealthResponse,
    HybridSearchRequest,
    SearchResponse,
    VectorIndexRequest,
    VectorIndexResponse,
    VectorSearchRequest,
)
from store import LanceDBServiceStore
from tokenizer import LegalTermDictionary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# 글로벌 스토어 인스턴스
_store: LanceDBServiceStore | None = None


def _init_store() -> LanceDBServiceStore:
    """스토어 초기화 (법률 용어 사전 + userdic 로드 포함)"""
    legal_dict = None
    userdic_path = None

    # 법률 용어 사전 로드 (JSON 기반)
    use_legal_dict = os.environ.get("USE_LEGAL_TERM_DICT", "false").lower() == "true"
    if use_legal_dict:
        dict_json_path = os.environ.get(
            "LEGAL_TERM_DICT_JSON",
            "/app/legal_term_dict.json",
        )
        if Path(dict_json_path).exists():
            legal_dict = LegalTermDictionary()
            count = legal_dict.load_from_json(dict_json_path)
            logger.info("법률 용어 사전 로드: %d개 (%s)", count, dict_json_path)

            # 분해맵 로드
            decomp_path = os.environ.get(
                "DECOMPOSITION_MAP_JSON",
                "/app/mecab_userdic/decomposition_map.json",
            )
            if Path(decomp_path).exists():
                decomp_count = legal_dict.load_decomposition_map(decomp_path)
                logger.info("분해맵 로드: %d개", decomp_count)
        else:
            logger.warning("법률 용어 사전 JSON 없음: %s", dict_json_path)

    # MeCab userdic 경로
    use_userdic = os.environ.get("USE_MECAB_USERDIC", "false").lower() == "true"
    if use_userdic:
        _userdic = os.environ.get(
            "MECAB_USERDIC_PATH",
            "/app/mecab_userdic/legal_terms.dic",
        )
        if Path(_userdic).exists():
            userdic_path = _userdic
            logger.info("MeCab userdic: %s", userdic_path)
        else:
            logger.warning("MeCab userdic 없음: %s", _userdic)

    return LanceDBServiceStore(
        legal_dict=legal_dict,
        userdic_path=userdic_path,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """앱 생명주기"""
    global _store  # noqa: PLW0603
    _store = _init_store()
    logger.info(
        "LanceDB 서비스 시작: table=%s, count=%d",
        _store.table_name,
        _store.count(),
    )
    yield
    logger.info("LanceDB 서비스 종료")


app = FastAPI(
    title="LanceDB Search Service",
    version="1.0.0",
    lifespan=lifespan,
)


def _get_store() -> LanceDBServiceStore:
    if _store is None:
        raise HTTPException(status_code=503, detail="Store not initialized")
    return _store


# =========================================================================
# 헬스체크
# =========================================================================


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    store = _get_store()
    return HealthResponse(
        status="ok",
        table_name=store.table_name,
        count=store.count(),
    )


# =========================================================================
# 검색 엔드포인트
# =========================================================================


@app.post("/search", response_model=SearchResponse)
async def vector_search(req: VectorSearchRequest) -> SearchResponse:
    store = _get_store()
    result = store.search(
        query_embedding=req.query_embedding,
        n_results=req.n_results,
        where=req.where,
    )
    return SearchResponse(**result)


@app.post("/search/fts", response_model=SearchResponse)
async def fts_search(req: FtsSearchRequest) -> SearchResponse:
    store = _get_store()
    result = store.search_fts(
        query=req.query,
        n_results=req.n_results,
        where=req.where,
    )
    return SearchResponse(**result)


@app.post("/search/hybrid", response_model=SearchResponse)
async def hybrid_search(req: HybridSearchRequest) -> SearchResponse:
    store = _get_store()
    result = store.hybrid_search(
        query_embedding=req.query_embedding,
        query_text=req.query_text,
        n_results=req.n_results,
        where=req.where,
        rrf_k=req.rrf_k,
    )
    return SearchResponse(**result)


# =========================================================================
# 문서 조회
# =========================================================================


@app.post("/documents/by-ids", response_model=DocumentsResponse)
async def get_documents_by_ids(req: GetByIdsRequest) -> DocumentsResponse:
    store = _get_store()
    result = store.get_by_ids(req.ids)
    return DocumentsResponse(**result)


@app.post("/documents/by-source-id", response_model=DocumentsResponse)
async def get_documents_by_source_id(req: GetBySourceIdRequest) -> DocumentsResponse:
    store = _get_store()
    result = store.get_by_source_id(req.source_id)
    return DocumentsResponse(**result)


# =========================================================================
# 카운트
# =========================================================================


@app.get("/count", response_model=CountResponse)
async def get_count() -> CountResponse:
    store = _get_store()
    return CountResponse(count=store.count())


@app.get("/count/{data_type}", response_model=CountResponse)
async def get_count_by_type(data_type: str) -> CountResponse:
    store = _get_store()
    return CountResponse(count=store.count_by_type(data_type))


# =========================================================================
# 인덱스 관리
# =========================================================================


@app.post("/index/vector", response_model=VectorIndexResponse)
async def create_vector_index(req: VectorIndexRequest) -> VectorIndexResponse:
    store = _get_store()
    created = store.create_vector_index(req.index_type)
    return VectorIndexResponse(created=created)


@app.post("/index/fts", response_model=FtsIndexResponse)
async def create_fts_index(req: FtsIndexRequest) -> FtsIndexResponse:
    store = _get_store()
    store.create_fts_index(req.field)
    return FtsIndexResponse(ok=True)
