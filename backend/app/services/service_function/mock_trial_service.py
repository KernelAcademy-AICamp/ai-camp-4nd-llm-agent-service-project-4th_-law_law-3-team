"""
모의 법정 RAG 검색 서비스

EvidenceSearcher Protocol + LanceDB 구현체
Design 문서 Section 6.5 기반
"""

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


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
        """판례 검색

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
        """법령 검색

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
