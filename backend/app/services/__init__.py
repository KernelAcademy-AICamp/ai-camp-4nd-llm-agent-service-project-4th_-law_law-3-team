"""
Services 패키지

비즈니스 로직을 담당하는 서비스 모듈
"""

# RAG 서비스
from app.services.rag import (
    PipelineConfig,
    PipelineResult,
    RerankerService,
    # 하위 호환용 (deprecated)
    RetrievalService,
    create_query_embedding,
    rerank_documents,
    rewrite_query,
    search_relevant_documents,
    search_with_pipeline,
    search_with_rerank,
)

# Cases 서비스
from app.services.service_function import LawService, PrecedentService

__all__ = [
    # RAG
    "create_query_embedding",
    "search_relevant_documents",
    "rerank_documents",
    "rewrite_query",
    "search_with_pipeline",
    "search_with_rerank",
    "PipelineConfig",
    "PipelineResult",
    # Cases
    "PrecedentService",
    "LawService",
    # 하위 호환용 (deprecated)
    "RetrievalService",
    "RerankerService",
]
