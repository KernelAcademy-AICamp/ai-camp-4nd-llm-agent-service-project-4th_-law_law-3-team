"""
LanceDB 마이크로서비스 Pydantic 스키마

API 요청/응답 모델 정의.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# =========================================================================
# 공통 응답
# =========================================================================


class HealthResponse(BaseModel):
    status: str = "ok"
    table_name: str = ""
    count: int = 0


class CountResponse(BaseModel):
    count: int = 0


# =========================================================================
# 검색 요청/응답
# =========================================================================


class VectorSearchRequest(BaseModel):
    """벡터 검색 요청"""
    query_embedding: List[float]
    n_results: int = Field(default=10, ge=1, le=200)
    where: Optional[Dict[str, Any]] = None


class FtsSearchRequest(BaseModel):
    """FTS 검색 요청"""
    query: str
    n_results: int = Field(default=10, ge=1, le=200)
    where: Optional[Dict[str, Any]] = None


class HybridSearchRequest(BaseModel):
    """하이브리드 검색 요청"""
    query_embedding: List[float]
    query_text: str
    n_results: int = Field(default=10, ge=1, le=200)
    where: Optional[Dict[str, Any]] = None
    rrf_k: int = Field(default=60, ge=1)


class SearchResponse(BaseModel):
    """검색 결과 응답 (SearchResult 호환)"""
    ids: List[List[str]]
    distances: Optional[List[List[float]]] = None
    metadatas: Optional[List[List[Dict[str, Any]]]] = None
    documents: Optional[List[List[str]]] = None


# =========================================================================
# 문서 조회 요청/응답
# =========================================================================


class GetByIdsRequest(BaseModel):
    """ID 기반 문서 조회 요청"""
    ids: List[str]


class GetBySourceIdRequest(BaseModel):
    """source_id 기반 문서 조회 요청"""
    source_id: str


class DocumentsResponse(BaseModel):
    """문서 조회 응답"""
    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []


# =========================================================================
# 인덱스 요청/응답
# =========================================================================


class VectorIndexRequest(BaseModel):
    """벡터 인덱스 생성 요청"""
    index_type: str = "IVF_FLAT"


class VectorIndexResponse(BaseModel):
    """벡터 인덱스 생성 응답"""
    created: bool = False


class FtsIndexRequest(BaseModel):
    """FTS 인덱스 생성 요청"""
    field: str = "content_tokenized"


class FtsIndexResponse(BaseModel):
    """FTS 인덱스 생성 응답"""
    ok: bool = True
