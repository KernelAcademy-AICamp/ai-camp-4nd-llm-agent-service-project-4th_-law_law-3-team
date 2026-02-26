"""
모의 법정 RAG 검색 서비스

EvidenceSearcher Protocol + LanceDB 구현체 + RAG 파이프라인 통합
Design 문서 Section 6.5 기반
"""

import logging
from typing import Any, Protocol

from app.services.rag.pipeline import PipelineConfig, search_with_pipeline_async

logger = logging.getLogger(__name__)

# ── 역할별 쿼리 템플릿 ──

ROLE_QUERY_TEMPLATES: dict[str, str] = {
    "prosecutor": "{query} 유죄 판결 양형",
    "attorney": "{query} 무죄 감형 정상참작",
    "judge": "{query} 판결 기준 양형",
    "defendant": "{query} 피고인 권리 진술",
}


class EvidenceSearcher(Protocol):
    """모의재판 증거 검색 인터페이스"""

    async def search_cases(
        self, query: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        """판례 검색"""
        ...

    async def search_articles(
        self, query: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        """법령 검색"""
        ...


class LanceDBEvidenceSearcher:
    """기존 LanceDB RAG를 활용한 증거 검색 구현체"""

    async def search_cases(
        self, query: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        """판례 검색 (REST API 호환용)

        Args:
            query: 검색 쿼리
            limit: 최대 결과 수

        Returns:
            판례 검색 결과 리스트
        """
        try:
            from app.services.rag import search_relevant_documents_async

            results = await search_relevant_documents_async(
                query=query, n_results=limit
            )
            return [
                {
                    "id": doc.get("id", ""),
                    "title": doc.get("metadata", {}).get("case_name", ""),
                    "summary": doc.get("content", "")[:300],
                    "relevance_score": round(doc.get("similarity", 0.0), 3),
                    "source": "lancedb",
                }
                for doc in results
                if doc.get("metadata", {}).get("doc_type") == "precedent"
                or not doc.get("metadata", {}).get("doc_type")
            ]
        except Exception as e:
            logger.warning("판례 검색 실패: %s", e)
            return []

    async def search_articles(
        self, query: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        """법령 검색 (REST API 호환용)

        Args:
            query: 검색 쿼리
            limit: 최대 결과 수

        Returns:
            법령 검색 결과 리스트
        """
        try:
            from app.services.rag import search_relevant_documents_async

            results = await search_relevant_documents_async(
                query=query, n_results=limit
            )
            return [
                {
                    "id": doc.get("id", ""),
                    "title": doc.get("metadata", {}).get("law_name", ""),
                    "content": doc.get("content", "")[:300],
                    "relevance_score": round(doc.get("similarity", 0.0), 3),
                    "source": "lancedb",
                }
                for doc in results
                if doc.get("metadata", {}).get("doc_type") == "law"
                or not doc.get("metadata", {}).get("doc_type")
            ]
        except Exception as e:
            logger.warning("법령 검색 실패: %s", e)
            return []


# 싱글톤 인스턴스
_evidence_searcher: EvidenceSearcher | None = None


def get_evidence_searcher() -> EvidenceSearcher:
    """EvidenceSearcher 싱글톤 반환"""
    global _evidence_searcher
    if _evidence_searcher is None:
        _evidence_searcher = LanceDBEvidenceSearcher()
    return _evidence_searcher


# ── RAG 통합 함수 (파이프라인 기반) ──


async def search_for_role(
    role: str,
    query: str,
    case_type: str = "criminal",
    limit: int = 3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """역할별 편향 쿼리로 판례/법령을 검색합니다.

    Args:
        role: 에이전트 역할 (prosecutor, attorney, judge, defendant)
        query: 사건 개요
        case_type: 사건 유형 (criminal, civil)
        limit: 리랭킹 후 반환 결과 수

    Returns:
        (판례 결과 리스트, 법령 결과 리스트) 튜플
    """
    template = ROLE_QUERY_TEMPLATES.get(role, "{query}")
    biased_query = template.format(query=query)

    case_config = PipelineConfig(
        n_results=10,
        doc_type="precedent",
        enable_rerank=True,
        rerank_top_k=limit,
        enable_rewrite=False,
    )
    law_config = PipelineConfig(
        n_results=10,
        doc_type="law",
        enable_rerank=True,
        rerank_top_k=limit,
        enable_rewrite=False,
    )

    try:
        case_result = await search_with_pipeline_async(biased_query, case_config)
        law_result = await search_with_pipeline_async(biased_query, law_config)
        return case_result.documents, law_result.documents
    except Exception as e:
        logger.warning("역할별 RAG 검색 실패 (role=%s): %s", role, e)
        return [], []


async def search_for_verdict(
    query: str,
    case_type: str = "criminal",
    limit: int = 5,
) -> list[dict[str, Any]]:
    """양형 참조용 유사 판례를 검색합니다.

    Args:
        query: 사건 개요
        case_type: 사건 유형
        limit: 결과 수

    Returns:
        양형 참조 판례 리스트
    """
    verdict_query = f"{query} 양형 기준 판결 선고"
    config = PipelineConfig(
        n_results=10,
        doc_type="precedent",
        enable_rerank=True,
        rerank_top_k=limit,
        enable_rewrite=False,
    )
    try:
        result = await search_with_pipeline_async(verdict_query, config)
        return result.documents
    except Exception as e:
        logger.warning("양형 판례 검색 실패: %s", e)
        return []


async def search_rebuttal(
    opponent_statement: str,
    case_summary: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """상대방 발언에서 핵심 주장을 추출하여 반박 증거를 검색합니다.

    Args:
        opponent_statement: 상대측 최근 발언
        case_summary: 사건 개요

    Returns:
        (반박 판례, 반박 법령) 튜플
    """
    rebuttal_query = f"{opponent_statement[:200]} 반박 {case_summary[:100]}"
    config = PipelineConfig(
        n_results=5,
        enable_rerank=True,
        rerank_top_k=2,
        enable_rewrite=False,
    )
    try:
        result = await search_with_pipeline_async(rebuttal_query, config)
        cases = [
            d for d in result.documents
            if d.get("metadata", {}).get("doc_type") == "precedent"
            or d.get("metadata", {}).get("data_type", "").startswith("판례")
        ]
        articles = [
            d for d in result.documents
            if d.get("metadata", {}).get("doc_type") == "law"
            or d.get("metadata", {}).get("data_type", "").startswith("법령")
        ]
        return cases, articles
    except Exception as e:
        logger.warning("반박 증거 검색 실패: %s", e)
        return [], []


def build_rag_context(
    role: str,
    cases: list[dict[str, Any]],
    articles: list[dict[str, Any]],
) -> str:
    """검색 결과를 LLM 프롬프트용 컨텍스트 문자열로 포맷합니다.

    LegalSearchAgent._build_context 패턴 참고.

    Args:
        role: 에이전트 역할
        cases: 판례 검색 결과
        articles: 법령 검색 결과

    Returns:
        포맷된 컨텍스트 문자열
    """
    parts: list[str] = []

    if cases:
        parts.append("## 관련 판례")
        for i, doc in enumerate(cases, 1):
            metadata = doc.get("metadata", {})
            case_name = metadata.get("case_name", "")
            case_number = metadata.get("case_number", "")
            content = doc.get("content", "")[:500]
            parts.append(
                f"[{i}] {case_name} ({case_number})\n요지: {content}"
            )

    if articles:
        parts.append("## 관련 법령")
        for i, doc in enumerate(articles, 1):
            metadata = doc.get("metadata", {})
            law_name = (
                metadata.get("case_name", "")
                or metadata.get("title", "")
                or metadata.get("law_name", "")
            )
            content = doc.get("content", "")[:500]
            parts.append(f"[{i}] {law_name}\n내용: {content}")

    return "\n\n".join(parts) if parts else ""


def build_user_hints(
    cases: list[dict[str, Any]],
    articles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """검색 결과를 사용자 힌트용 요약 리스트로 변환합니다.

    Args:
        cases: 판례 검색 결과
        articles: 법령 검색 결과

    Returns:
        UserHint 형태의 dict 리스트
    """
    hints: list[dict[str, Any]] = []

    for doc in cases:
        metadata = doc.get("metadata", {})
        hints.append({
            "type": "case",
            "title": metadata.get("case_name", ""),
            "summary": doc.get("content", "")[:200],
            "relevance_score": round(doc.get("similarity", 0.0), 3),
            "suggestion": (
                f"판례 {metadata.get('case_number', '')}을 "
                "인용하여 주장을 뒷받침하세요."
            ),
        })

    for doc in articles:
        metadata = doc.get("metadata", {})
        law_name = (
            metadata.get("case_name", "")
            or metadata.get("title", "")
            or metadata.get("law_name", "")
        )
        hints.append({
            "type": "law",
            "title": law_name,
            "summary": doc.get("content", "")[:200],
            "relevance_score": round(doc.get("similarity", 0.0), 3),
            "suggestion": f"{law_name} 조문을 인용하여 법적 근거를 제시하세요.",
        })

    return hints
