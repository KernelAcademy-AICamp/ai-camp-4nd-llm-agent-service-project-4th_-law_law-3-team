"""
RAG 서비스 모듈

검색, 리랭킹, 쿼리 리라이팅, 파이프라인 함수 제공
"""

# Context 압축
from app.services.rag.compression import ContextCompressor, get_context_compressor

# 임베딩 함수
from app.services.rag.embedding import (
    check_embedding_model_availability,
    create_query_embedding,
    create_query_embedding_async,
    get_local_model,
    is_embedding_model_cached,
)

# 포맷팅 유틸리티
from app.services.rag.format_utils import (
    format_generic_context,
    format_law_context,
    format_law_sources,
    format_precedent_context,
    format_precedent_sources,
    format_supplementary_context,
    format_supplementary_sources,
)

# 하이브리드 검색
from app.services.rag.fusion import reciprocal_rank_fusion
from app.services.rag.keyword_search import search_by_keyword

# ONNX 세션 관리
from app.services.rag.onnx_session import (
    get_platform_info,
    is_embedding_onnx_loaded,
    is_reranker_onnx_loaded,
)

# 파이프라인
from app.services.rag.pipeline import (
    PRESETS,
    PipelineConfig,
    PipelineMetrics,
    PipelineResult,
    RAGPipeline,
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
    is_reranker_available,
    rerank_documents,
    rerank_documents_async,
)

# 검색 함수
from app.services.rag.retrieval import (
    DOCUMENT_TABLE_REGISTRY,
    TableConfig,
    fetch_document_contents,
    search_relevant_documents,
    search_relevant_documents_async,
)

__all__ = [
    # Context 압축
    "ContextCompressor",
    "get_context_compressor",
    # 임베딩
    "create_query_embedding",
    "create_query_embedding_async",
    "check_embedding_model_availability",
    "get_local_model",
    "is_embedding_model_cached",
    # 검색
    "search_relevant_documents",
    "search_relevant_documents_async",
    "fetch_document_contents",
    "TableConfig",
    "DOCUMENT_TABLE_REGISTRY",
    # 하이브리드 검색
    "search_by_keyword",
    "reciprocal_rank_fusion",
    # ONNX
    "is_embedding_onnx_loaded",
    "is_reranker_onnx_loaded",
    "get_platform_info",
    # 리랭킹
    "rerank_documents",
    "rerank_documents_async",
    "is_reranker_available",
    # 쿼리 리라이팅
    "rewrite_query",
    "extract_legal_keywords",
    # 포맷팅 유틸리티
    "format_precedent_context",
    "format_law_context",
    "format_supplementary_context",
    "format_generic_context",
    "format_precedent_sources",
    "format_law_sources",
    "format_supplementary_sources",
    # 파이프라인
    "RAGPipeline",
    "PRESETS",
    "PipelineConfig",
    "PipelineMetrics",
    "PipelineResult",
    "search_with_pipeline",
    "search_with_pipeline_async",
    "search_with_rerank",
    "search_with_rewrite",
]
