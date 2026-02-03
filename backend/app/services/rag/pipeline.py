"""
RAG 파이프라인

검색, 리랭킹, 쿼리 리라이팅을 통합한 파이프라인 함수
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.services.rag.query_rewrite import rewrite_query
from app.services.rag.rerank import rerank_documents
from app.services.rag.retrieval import search_relevant_documents

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """
    파이프라인 설정

    Attributes:
        n_results: 검색 결과 수
        doc_type: 문서 유형 필터 (precedent, law, constitutional)
        enable_rewrite: 쿼리 리라이팅 활성화
        num_rewrite_queries: 리라이팅 시 생성할 쿼리 수
        enable_rerank: 리랭킹 활성화
        rerank_top_k: 리랭킹 후 반환할 결과 수
        use_llm_rewrite: LLM 기반 리라이팅 사용
    """

    n_results: int = 10
    doc_type: Optional[str] = None
    enable_rewrite: bool = False
    num_rewrite_queries: int = 3
    enable_rerank: bool = False
    rerank_top_k: int = 5
    use_llm_rewrite: bool = True


@dataclass
class PipelineResult:
    """
    파이프라인 결과

    Attributes:
        documents: 검색된 문서 목록
        original_query: 원본 쿼리
        rewritten_queries: 리라이팅된 쿼리 목록 (활성화 시)
        reranked: 리랭킹 적용 여부
        total_retrieved: 리랭킹 전 검색 결과 수
    """

    documents: List[Dict[str, Any]] = field(default_factory=list)
    original_query: str = ""
    rewritten_queries: List[str] = field(default_factory=list)
    reranked: bool = False
    total_retrieved: int = 0


def search_with_pipeline(
    query: str,
    config: Optional[PipelineConfig] = None,
) -> PipelineResult:
    """
    통합 RAG 파이프라인으로 검색 수행

    Args:
        query: 검색 쿼리
        config: 파이프라인 설정 (기본값: 검색만 수행)

    Returns:
        파이프라인 결과
    """
    config = config or PipelineConfig()

    result = PipelineResult(original_query=query)

    # 1. 쿼리 리라이팅 (선택)
    queries = [query]
    if config.enable_rewrite:
        queries = rewrite_query(
            query=query,
            num_queries=config.num_rewrite_queries,
            use_llm=config.use_llm_rewrite,
        )
        result.rewritten_queries = queries

    # 2. 검색 수행 (모든 쿼리에 대해)
    all_documents: List[Dict[str, Any]] = []
    seen_ids: set = set()

    for q in queries:
        docs = search_relevant_documents(
            query=q,
            n_results=config.n_results,
            doc_type=config.doc_type,
        )

        # 중복 제거
        for doc in docs:
            doc_id = doc.get("id")
            if doc_id and doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_documents.append(doc)

    result.total_retrieved = len(all_documents)

    # 3. 리랭킹 (선택)
    if config.enable_rerank and all_documents:
        result.documents = rerank_documents(
            query=query,
            documents=all_documents,
            top_k=config.rerank_top_k,
        )
        result.reranked = True
    else:
        # 리랭킹 미사용 시 similarity 기준 정렬
        all_documents.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        result.documents = all_documents[: config.n_results]

    return result


def search_with_rerank(
    query: str,
    n_results: int = 10,
    top_k: int = 5,
    doc_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    검색 + 리랭킹 간편 함수

    Args:
        query: 검색 쿼리
        n_results: 검색 결과 수
        top_k: 리랭킹 후 반환할 결과 수
        doc_type: 문서 유형 필터

    Returns:
        리랭킹된 문서 목록
    """
    config = PipelineConfig(
        n_results=n_results,
        doc_type=doc_type,
        enable_rerank=True,
        rerank_top_k=top_k,
    )
    result = search_with_pipeline(query, config)
    return result.documents


def search_with_rewrite(
    query: str,
    n_results: int = 10,
    doc_type: Optional[str] = None,
    use_llm: bool = True,
) -> List[Dict[str, Any]]:
    """
    쿼리 리라이팅 + 검색 간편 함수

    Args:
        query: 검색 쿼리
        n_results: 검색 결과 수
        doc_type: 문서 유형 필터
        use_llm: LLM 기반 리라이팅 사용

    Returns:
        검색된 문서 목록
    """
    config = PipelineConfig(
        n_results=n_results,
        doc_type=doc_type,
        enable_rewrite=True,
        use_llm_rewrite=use_llm,
    )
    result = search_with_pipeline(query, config)
    return result.documents
