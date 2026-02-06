"""
RAG 서비스 모듈

검색, 리랭킹, 쿼리 리라이팅, 파이프라인 함수 제공
"""

# 임베딩 함수
from app.services.rag.embedding import (
    check_embedding_model_availability,
    create_query_embedding,
    create_query_embedding_async,
    get_local_model,
    is_embedding_model_cached,
)

# 파이프라인
from app.services.rag.pipeline import (
    PipelineConfig,
    PipelineResult,
    search_with_pipeline,
    search_with_pipeline_async,
    search_with_rerank,
    search_with_rewrite,
)

# 쿼리 리라이팅
from app.services.rag.query_rewrite import (
    extract_legal_keywords,
    rewrite_query,
)

# 리랭킹 함수
from app.services.rag.rerank import (
    RerankerService,
    get_reranker_service,
    is_reranker_available,
    rerank_documents,
)

# 검색 함수
# 하위 호환용 (deprecated)
from app.services.rag.retrieval import (
    RetrievalService,
    get_retrieval_service,
    search_relevant_documents,
    search_relevant_documents_async,
)

__all__ = [
    # 임베딩
    "create_query_embedding",
    "create_query_embedding_async",
    "check_embedding_model_availability",
    "get_local_model",
    "is_embedding_model_cached",
    # 검색
    "search_relevant_documents",
    "search_relevant_documents_async",
    # 리랭킹
    "rerank_documents",
    "is_reranker_available",
    # 쿼리 리라이팅
    "rewrite_query",
    "extract_legal_keywords",
    # 파이프라인
    "PipelineConfig",
    "PipelineResult",
    "search_with_pipeline",
    "search_with_pipeline_async",
    "search_with_rerank",
    "search_with_rewrite",
    # 하위 호환용 (deprecated)
    "RetrievalService",
    "get_retrieval_service",
    "RerankerService",
    "get_reranker_service",
]
