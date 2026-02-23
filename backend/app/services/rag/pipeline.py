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

from app.core.config import settings
from app.services.rag.query_rewrite import rewrite_query
from app.services.rag.rerank import rerank_documents
from app.services.rag.retrieval import (
    _extract_id_data_type_map,
    _populate_content,
    fetch_ai_summaries,
    fetch_document_contents,
    search_relevant_documents,
    search_without_content,
    search_without_content_traced,
)
from app.services.rag.trace_store import RagStepResult, summarize_doc

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
        enable_rewrite: 쿼리 리라이팅 활성화
        num_rewrite_queries: 리라이팅 시 생성할 쿼리 수
        enable_rerank: 리랭킹 활성화
        rerank_top_k: 리랭킹 후 반환할 결과 수
        use_llm_rewrite: LLM 기반 리라이팅 사용
    """

    n_results: int = 10
    doc_type: Optional[str] = None
    enable_rewrite: bool = True
    num_rewrite_queries: int = 3
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
                num_queries=config.num_rewrite_queries,
                use_llm=config.use_llm_rewrite,
            )
            result.rewritten_queries = queries

        # Step 2: 검색
        search_start = time.monotonic()

        all_documents: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        search_fn = (
            search_without_content if config.enable_rerank
            else search_relevant_documents
        )

        for q in queries:
            docs = search_fn(
                query=q,
                n_results=config.n_results,
                doc_type=config.doc_type,
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

    def execute_traced(
        self,
        query: str,
        config: Optional[PipelineConfig] = None,
    ) -> tuple[PipelineResult, list[RagStepResult]]:
        """파이프라인 실행 + 단계별 트레이스 수집.

        execute()와 동일한 결과를 반환하면서,
        각 단계(리라이팅, 벡터, 키워드, RRF, 리랭킹, 최종)의
        중간 결과를 RagStepResult 리스트로 함께 반환.
        """
        config = config or PipelineConfig()
        pipeline_start = time.monotonic()

        result = PipelineResult(original_query=query)
        metrics = result.metrics
        steps: list[RagStepResult] = []

        # Step 1: 쿼리 리라이팅 (선택)
        queries = [query]
        if config.enable_rewrite:
            rewrite_start = time.monotonic()
            queries = rewrite_query(
                query=query,
                num_queries=config.num_rewrite_queries,
                use_llm=config.use_llm_rewrite,
            )
            result.rewritten_queries = queries
            steps.append(RagStepResult(
                step_name="query_rewrite",
                documents=[],
                count=len(queries),
                time_ms=(time.monotonic() - rewrite_start) * 1000,
                metadata={"original": query, "rewritten": queries},
            ))

        # Step 2: 검색 (traced)
        search_start = time.monotonic()
        all_documents: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        # traced 검색은 리랭킹 활성화 시에만 (search_without_content 대체)
        all_vector: list[dict[str, Any]] = []
        all_keyword: list[dict[str, Any]] = []
        all_fused_ids: list[str] = []
        vector_time_total = 0.0
        keyword_time_total = 0.0

        for q in queries:
            if config.enable_rerank:
                docs, intermediates = search_without_content_traced(
                    query=q,
                    n_results=config.n_results,
                    doc_type=config.doc_type,
                )
                all_vector.extend(intermediates.get("vector_results", []))
                all_keyword.extend(intermediates.get("keyword_results", []))
                all_fused_ids.extend(intermediates.get("fused_source_ids", []))
                vector_time_total += intermediates.get("vector_time_ms", 0)
                keyword_time_total += intermediates.get("keyword_time_ms", 0)
            else:
                docs = search_relevant_documents(
                    query=q,
                    n_results=config.n_results,
                    doc_type=config.doc_type,
                )

            for doc in docs:
                doc_id = doc.get("metadata", {}).get("doc_id", "")
                if doc_id and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_documents.append(doc)

        metrics.search_time_ms = (time.monotonic() - search_start) * 1000
        metrics.total_searched = len(all_documents)
        result.total_retrieved = len(all_documents)

        # ai_summary 조회 (PostgreSQL) → 트레이스 표시 + 리랭킹 공용
        # all_documents(RRF 융합)뿐 아니라 벡터/키워드 전체 ID를 대상으로 조회
        summaries: dict[str, str] = {}
        if config.enable_rerank and all_documents:
            all_trace_docs = all_vector + all_keyword + all_documents
            id_type_map = _extract_id_data_type_map(all_trace_docs)
            summaries = fetch_ai_summaries(id_type_map)

            # 벡터/키워드 결과에도 ai_summary를 content로 주입 (트레이스용)
            for doc in all_vector:
                doc_id = doc.get("metadata", {}).get("doc_id", "")
                if doc_id and doc_id in summaries:
                    doc["content"] = summaries[doc_id]
            for doc in all_keyword:
                doc_id = doc.get("metadata", {}).get("doc_id", "")
                if doc_id and doc_id in summaries:
                    doc["content"] = summaries[doc_id]

            # 리랭킹용 content 주입
            _populate_content(all_documents, summaries)

        # 벡터 검색 스텝
        steps.append(RagStepResult(
            step_name="vector_search",
            documents=[summarize_doc(d, include_content=True) for d in all_vector],
            count=len(all_vector),
            time_ms=vector_time_total,
        ))

        # 키워드 검색 스텝
        if all_keyword:
            steps.append(RagStepResult(
                step_name="keyword_search",
                documents=[summarize_doc(d, include_content=True) for d in all_keyword],
                count=len(all_keyword),
                time_ms=keyword_time_total,
            ))

        # RRF 융합 스텝 (건수만 표시)
        if all_fused_ids:
            steps.append(RagStepResult(
                step_name="rrf_fusion",
                documents=[],
                count=len(all_fused_ids),
                time_ms=0,
            ))

        # Step 3: 리랭킹 (선택)
        if config.enable_rerank and all_documents:

            rerank_start = time.monotonic()
            reranked = rerank_documents(
                query=query,
                documents=all_documents,
                top_k=config.rerank_top_k,
            )
            rerank_ms = (time.monotonic() - rerank_start) * 1000
            metrics.rerank_time_ms = rerank_ms
            metrics.total_reranked = len(reranked)
            result.reranked = True

            steps.append(RagStepResult(
                step_name="rerank",
                documents=[summarize_doc(d, include_content=True) for d in reranked],
                count=len(reranked),
                time_ms=rerank_ms,
            ))

            contents = fetch_document_contents(
                _extract_id_data_type_map(reranked)
            )
            _populate_content(reranked, contents)
            result.documents = reranked
        else:
            all_documents.sort(
                key=lambda x: x.get("similarity", 0), reverse=True
            )
            result.documents = all_documents[: config.n_results]

        # 최종 컨텍스트 스텝
        steps.append(RagStepResult(
            step_name="final_context",
            documents=[summarize_doc(d) for d in result.documents],
            count=len(result.documents),
            time_ms=0,
        ))

        metrics.total_time_ms = (time.monotonic() - pipeline_start) * 1000

        logger.info(
            "RAG 파이프라인(traced) 완료: %d건 검색 → %d건 반환 (%.0fms)",
            result.total_retrieved,
            len(result.documents),
            metrics.total_time_ms,
        )

        return result, steps

    async def execute_async(
        self,
        query: str,
        config: Optional[PipelineConfig] = None,
    ) -> PipelineResult:
        """파이프라인 실행 (비동기)."""
        return await asyncio.to_thread(self.execute, query, config)

    async def execute_traced_async(
        self,
        query: str,
        config: Optional[PipelineConfig] = None,
    ) -> tuple[PipelineResult, list[RagStepResult]]:
        """파이프라인 실행 + 트레이스 (비동기)."""
        return await asyncio.to_thread(self.execute_traced, query, config)


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


async def search_with_pipeline_traced_async(
    query: str,
    config: Optional[PipelineConfig] = None,
) -> tuple[PipelineResult, list[RagStepResult]]:
    """통합 RAG 파이프라인 비동기 검색 + 트레이스."""
    return await _default_pipeline.execute_traced_async(query, config)
