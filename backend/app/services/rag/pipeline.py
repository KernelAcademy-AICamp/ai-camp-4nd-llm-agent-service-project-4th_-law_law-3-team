"""
RAG 파이프라인

검색, 원문 조회, 리랭킹, 쿼리 리라이팅을 통합한 파이프라인.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from app.core.config import settings
from app.services.rag.query_rewrite import rewrite_query
from app.services.rag.rerank import rerank_documents
from app.services.rag.retrieval import (
    _extract_id_data_type_map,
    _populate_content,
    fetch_ai_summaries,
    fetch_ai_summaries_async,
    fetch_document_contents,
    fetch_document_contents_async,
    search_relevant_documents,
    search_without_content,
    search_without_content_async,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 설정 / 결과 데이터 클래스
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """파이프라인 설정.

    Attributes:
        n_results: 검색할 후보 수
        doc_type: 문서 유형 필터 ("precedent", "law" 또는 한국어 data_type)
        exclude_doc_types: 제외할 data_type 목록 (한국어, 예: ["판례"]).
            doc_type과 상호 배타적 — 동시 지정 시 doc_type 우선.
        enable_rewrite: 쿼리 리라이팅 활성화
        enable_rerank: 리랭킹 활성화
        rerank_top_k: 리랭킹 후 반환할 결과 수
        use_llm_rewrite: LLM 기반 리라이팅 사용
    """

    n_results: int = 10
    doc_type: Optional[str] = None
    exclude_doc_types: Optional[list[str]] = None
    enable_rewrite: bool = True
    enable_rerank: bool = False
    rerank_top_k: int = 5
    use_llm_rewrite: bool = True


@dataclass
class PipelineMetrics:
    """파이프라인 실행 메트릭.

    Attributes:
        search_time_ms: 검색 + 원문 조회 소요 시간 (ms)
        rerank_time_ms: 리랭킹 소요 시간 (ms)
        total_time_ms: 전체 소요 시간 (ms)
        total_searched: 검색된 총 후보 수
        total_reranked: 리랭킹 후 반환 수
    """

    search_time_ms: float = 0.0
    rerank_time_ms: float = 0.0
    total_time_ms: float = 0.0
    total_searched: int = 0
    total_reranked: int = 0


@dataclass
class PipelineResult:
    """파이프라인 결과.

    Attributes:
        documents: 검색된 문서 목록
        original_query: 원본 쿼리
        rewritten_queries: 리라이팅된 쿼리 목록 (활성화 시)
        reranked: 리랭킹 적용 여부
        total_retrieved: 리랭킹 전 검색 결과 수
        hybrid_search_used: 하이브리드 검색 사용 여부
        metrics: 실행 메트릭
    """

    documents: list[dict[str, Any]] = field(default_factory=list)
    original_query: str = ""
    rewritten_queries: list[str] = field(default_factory=list)
    reranked: bool = False
    total_retrieved: int = 0
    hybrid_search_used: bool = field(
        default_factory=lambda: settings.USE_HYBRID_SEARCH
    )
    metrics: PipelineMetrics = field(default_factory=PipelineMetrics)


# ---------------------------------------------------------------------------
# 프리셋
# ---------------------------------------------------------------------------

PRESETS: dict[str, PipelineConfig] = {
    "legal_search_precedent": PipelineConfig(
        n_results=15,
        doc_type="precedent",
        enable_rerank=True,
        rerank_top_k=5,
    ),
    "legal_search_law": PipelineConfig(
        n_results=15,
        doc_type="law",
        enable_rerank=True,
        rerank_top_k=5,
    ),
    "legal_search_all": PipelineConfig(
        n_results=20,
        enable_rerank=True,
        rerank_top_k=7,
    ),
    "law_study": PipelineConfig(
        n_results=10,
        enable_rerank=True,
        rerank_top_k=5,
        enable_rewrite=True,
    ),
    "small_claims": PipelineConfig(
        n_results=10,
        doc_type="precedent",
        enable_rerank=True,
        rerank_top_k=3,
    ),
    "quick_search": PipelineConfig(
        n_results=5,
        enable_rerank=False,
    ),
}


# ---------------------------------------------------------------------------
# RAGPipeline
# ---------------------------------------------------------------------------


class RAGPipeline:
    """RAG 파이프라인.

    1. 쿼리 리라이팅 (선택)
    2. 하이브리드 검색 (content 미포함)
    3. 요약문 기반 Cross-encoder 리랭킹 (선택)
    4. top-k만 원문 배치 조회 (PostgreSQL)
    5. 결과 포맷팅 + 메트릭
    """

    @traceable(name="rag_pipeline")
    def execute(
        self,
        query: str,
        config: Optional[PipelineConfig] = None,
    ) -> PipelineResult:
        """파이프라인 실행 (동기)."""
        config = config or PipelineConfig()
        pipeline_start = time.monotonic()

        result = PipelineResult(original_query=query)
        metrics = result.metrics

        # Step 1: 쿼리 리라이팅 (선택)
        queries = [query]
        if config.enable_rewrite:
            queries = rewrite_query(
                query=query,
                use_llm=config.use_llm_rewrite,
            )
            result.rewritten_queries = queries

        # LangSmith extra에 쿼리 + 파이프라인 설정 기록
        run_tree = get_current_run_tree()
        if run_tree is not None:
            run_tree.extra = {
                **(run_tree.extra or {}),
                "metadata": {
                    **(run_tree.extra or {}).get("metadata", {}),
                    "original_query": query,
                    "rewritten_queries": queries,
                    "[config] doc_type": config.doc_type,
                    "[config] exclude_doc_types": config.exclude_doc_types,
                    "[config] n_results": config.n_results,
                    "[config] enable_rewrite": config.enable_rewrite,
                    "[config] enable_rerank": config.enable_rerank,
                    "[config] rerank_top_k": config.rerank_top_k,
                    "[config] use_llm_rewrite": config.use_llm_rewrite,
                },
            }

        # Step 2: 검색
        search_start = time.monotonic()

        all_documents: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        search_fn = (
            search_without_content if config.enable_rerank
            else search_relevant_documents
        )

        # exclude_doc_types는 doc_type이 없을 때만 적용
        exclude = (
            config.exclude_doc_types
            if not config.doc_type and config.exclude_doc_types
            else None
        )

        for q in queries:
            docs = search_fn(
                query=q,
                n_results=config.n_results,
                doc_type=config.doc_type,
                exclude_doc_types=exclude,
            )
            for doc in docs:
                doc_id = doc.get("metadata", {}).get("doc_id", "")
                if doc_id and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_documents.append(doc)

        metrics.search_time_ms = (time.monotonic() - search_start) * 1000
        metrics.total_searched = len(all_documents)
        result.total_retrieved = len(all_documents)

        # Step 3: 리랭킹 (선택)
        if config.enable_rerank and all_documents:
            # ai_summary 조회 (PostgreSQL) → 리랭킹용 content 주입
            id_type_map = _extract_id_data_type_map(all_documents)
            summaries = fetch_ai_summaries(id_type_map)
            _populate_content(all_documents, summaries)

            rerank_start = time.monotonic()
            reranked = rerank_documents(
                query=query,
                documents=all_documents,
                top_k=config.rerank_top_k,
            )
            metrics.rerank_time_ms = (time.monotonic() - rerank_start) * 1000
            metrics.total_reranked = len(reranked)
            result.reranked = True

            # top-k만 원본 조회 (PostgreSQL)
            contents = fetch_document_contents(
                _extract_id_data_type_map(reranked)
            )
            _populate_content(reranked, contents)
            result.documents = reranked
        else:
            # 리랭킹 미사용 시 similarity 기준 정렬
            all_documents.sort(
                key=lambda x: x.get("similarity", 0), reverse=True
            )
            result.documents = all_documents[: config.n_results]

        metrics.total_time_ms = (time.monotonic() - pipeline_start) * 1000

        logger.info(
            "RAG 파이프라인 완료: %d건 검색 → %d건 반환 (%.0fms)",
            result.total_retrieved,
            len(result.documents),
            metrics.total_time_ms,
        )

        return result

    async def execute_async(
        self,
        query: str,
        config: Optional[PipelineConfig] = None,
    ) -> PipelineResult:
        """파이프라인 실행 (비동기, 내부 병렬화).

        동기 execute()와 동일한 로직이지만 검색/조회를 병렬로 수행한다.
        - 다중 리라이팅 쿼리를 asyncio.gather로 병렬 검색
        - 각 검색 내부에서 벡터 + FTS를 병렬 실행
        - 요약문/원문 조회를 data_type별 병렬 실행
        """
        config = config or PipelineConfig()
        pipeline_start = time.monotonic()

        result = PipelineResult(original_query=query)
        metrics = result.metrics

        # Step 1: 쿼리 리라이팅 (CPU-bound → to_thread)
        queries = [query]
        if config.enable_rewrite:
            queries = await asyncio.to_thread(
                rewrite_query,
                query=query,
                use_llm=config.use_llm_rewrite,
            )
            result.rewritten_queries = queries

        # LangSmith extra 기록
        run_tree = get_current_run_tree()
        if run_tree is not None:
            run_tree.extra = {
                **(run_tree.extra or {}),
                "metadata": {
                    **(run_tree.extra or {}).get("metadata", {}),
                    "original_query": query,
                    "rewritten_queries": queries,
                    "[config] doc_type": config.doc_type,
                    "[config] exclude_doc_types": config.exclude_doc_types,
                    "[config] n_results": config.n_results,
                    "[config] enable_rewrite": config.enable_rewrite,
                    "[config] enable_rerank": config.enable_rerank,
                    "[config] rerank_top_k": config.rerank_top_k,
                    "[config] use_llm_rewrite": config.use_llm_rewrite,
                },
            }

        # Step 2: 다중 쿼리 병렬 검색 (각 쿼리 내 벡터+FTS도 병렬)
        search_start = time.monotonic()

        exclude = (
            config.exclude_doc_types
            if not config.doc_type and config.exclude_doc_types
            else None
        )

        if config.enable_rerank:
            search_tasks = [
                search_without_content_async(
                    query=q,
                    n_results=config.n_results,
                    doc_type=config.doc_type,
                    exclude_doc_types=exclude,
                )
                for q in queries
            ]
        else:
            search_tasks = [
                asyncio.to_thread(
                    search_relevant_documents,
                    query=q,
                    n_results=config.n_results,
                    doc_type=config.doc_type,
                    exclude_doc_types=exclude,
                )
                for q in queries
            ]

        query_results = await asyncio.gather(*search_tasks)

        # 결과 병합 (deduplicate)
        all_documents: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for docs in query_results:
            for doc in docs:
                doc_id = doc.get("metadata", {}).get("doc_id", "")
                if doc_id and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_documents.append(doc)

        metrics.search_time_ms = (time.monotonic() - search_start) * 1000
        metrics.total_searched = len(all_documents)
        result.total_retrieved = len(all_documents)

        # Step 3: 리랭킹
        if config.enable_rerank and all_documents:
            # 요약문 병렬 조회
            id_type_map = _extract_id_data_type_map(all_documents)
            summaries = await fetch_ai_summaries_async(id_type_map)
            _populate_content(all_documents, summaries)

            # 리랭킹 (CPU-bound → to_thread)
            rerank_start = time.monotonic()
            reranked = await asyncio.to_thread(
                rerank_documents,
                query=query,
                documents=all_documents,
                top_k=config.rerank_top_k,
            )
            metrics.rerank_time_ms = (time.monotonic() - rerank_start) * 1000
            metrics.total_reranked = len(reranked)
            result.reranked = True

            # 원문 병렬 조회
            contents = await fetch_document_contents_async(
                _extract_id_data_type_map(reranked)
            )
            _populate_content(reranked, contents)
            result.documents = reranked
        else:
            all_documents.sort(
                key=lambda x: x.get("similarity", 0), reverse=True
            )
            result.documents = all_documents[: config.n_results]

        metrics.total_time_ms = (time.monotonic() - pipeline_start) * 1000

        logger.info(
            "RAG 파이프라인(async) 완료: %d건 검색 → %d건 반환 (%.0fms)",
            result.total_retrieved,
            len(result.documents),
            metrics.total_time_ms,
        )

        return result


# ---------------------------------------------------------------------------
# 편의 함수 (하위 호환)
# ---------------------------------------------------------------------------

_default_pipeline = RAGPipeline()


def search_with_pipeline(
    query: str,
    config: Optional[PipelineConfig] = None,
) -> PipelineResult:
    """통합 RAG 파이프라인으로 검색 수행."""
    return _default_pipeline.execute(query, config)


def search_with_rerank(
    query: str,
    n_results: int = 10,
    top_k: int = 5,
    doc_type: Optional[str] = None,
) -> list[dict[str, Any]]:
    """검색 + 리랭킹 간편 함수."""
    config = PipelineConfig(
        n_results=n_results,
        doc_type=doc_type,
        enable_rerank=True,
        rerank_top_k=top_k,
    )
    result = _default_pipeline.execute(query, config)
    return result.documents


def search_with_rewrite(
    query: str,
    n_results: int = 10,
    doc_type: Optional[str] = None,
    use_llm: bool = True,
) -> list[dict[str, Any]]:
    """쿼리 리라이팅 + 검색 간편 함수."""
    config = PipelineConfig(
        n_results=n_results,
        doc_type=doc_type,
        enable_rewrite=True,
        use_llm_rewrite=use_llm,
    )
    result = _default_pipeline.execute(query, config)
    return result.documents


async def search_with_pipeline_async(
    query: str,
    config: Optional[PipelineConfig] = None,
) -> PipelineResult:
    """통합 RAG 파이프라인 비동기 검색."""
    return await _default_pipeline.execute_async(query, config)
