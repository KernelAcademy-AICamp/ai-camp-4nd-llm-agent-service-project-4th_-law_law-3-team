"""
RemoteLanceDBStore - LanceDB 마이크로서비스 HTTP 클라이언트 어댑터

LanceDB 마이크로서비스(services/lancedb/)와 HTTP로 통신하여
기존 VectorStoreBase 인터페이스를 유지.

Usage:
    store = RemoteLanceDBStore()
    results = store.search(query_embedding=[0.1, 0.2, ...], n_results=5)
"""

import logging
from typing import Any, Dict, List, Optional

import httpx

from app.core.config import settings
from app.tools.vectorstore.base import SearchResult, VectorStoreBase

logger = logging.getLogger(__name__)


class RemoteLanceDBStore(VectorStoreBase):
    """
    LanceDB 마이크로서비스 HTTP 클라이언트

    VectorStoreBase를 구현하여 기존 코드에서 투명하게 교체 가능.
    데이터 추가/삭제는 임베딩 스크립트가 직접 LanceDB 파일에 접근하므로
    NotImplementedError를 발생시킨다.
    """

    def __init__(self) -> None:
        self._base_url = settings.LANCEDB_SERVICE_URL.rstrip("/")
        self._timeout = settings.LANCEDB_SERVICE_TIMEOUT
        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=httpx.Timeout(self._timeout),
        )

    def _post(self, path: str, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """POST 요청 헬퍼"""
        try:
            response = self._client.post(path, json=json_data)
            response.raise_for_status()
            result: Dict[str, Any] = response.json()
            return result
        except httpx.HTTPStatusError as e:
            logger.error("LanceDB 서비스 HTTP 오류: %s %s", e.response.status_code, path)
            raise
        except httpx.ConnectError:
            logger.error("LanceDB 서비스 연결 실패: %s", self._base_url)
            raise

    def _get(self, path: str) -> Dict[str, Any]:
        """GET 요청 헬퍼"""
        try:
            response = self._client.get(path)
            response.raise_for_status()
            result: Dict[str, Any] = response.json()
            return result
        except httpx.HTTPStatusError as e:
            logger.error("LanceDB 서비스 HTTP 오류: %s %s", e.response.status_code, path)
            raise
        except httpx.ConnectError:
            logger.error("LanceDB 서비스 연결 실패: %s", self._base_url)
            raise

    # =========================================================================
    # VectorStoreBase 구현
    # =========================================================================

    def search(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
    ) -> SearchResult:
        """벡터 유사 검색"""
        data = self._post("/search", {
            "query_embedding": query_embedding,
            "n_results": n_results,
            "where": where,
        })
        return SearchResult(
            ids=data.get("ids", [[]]),
            distances=data.get("distances"),
            metadatas=data.get("metadatas"),
            documents=data.get("documents"),
        )

    def get_by_ids(self, ids: List[str]) -> Dict[str, Any]:
        """ID로 문서 조회"""
        return self._post("/documents/by-ids", {"ids": ids})

    def get_by_source_id(self, source_id: str) -> Dict[str, Any]:
        """source_id로 모든 청크 조회"""
        return self._post("/documents/by-source-id", {"source_id": source_id})

    def count(self) -> int:
        """전체 카운트"""
        data = self._get("/count")
        count: int = data.get("count", 0)
        return count

    def count_by_type(self, data_type: str) -> int:
        """타입별 카운트"""
        data = self._get(f"/count/{data_type}")
        count: int = data.get("count", 0)
        return count

    # =========================================================================
    # FTS / 하이브리드 검색
    # =========================================================================

    def search_fts(
        self,
        query: str,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> SearchResult:
        """FTS 검색"""
        data = self._post("/search/fts", {
            "query": query,
            "n_results": n_results,
            "where": where,
        })
        return SearchResult(
            ids=data.get("ids", [[]]),
            distances=data.get("distances"),
            metadatas=data.get("metadatas"),
            documents=data.get("documents"),
        )

    def hybrid_search(
        self,
        query_embedding: List[float],
        query_text: str,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        rrf_k: int = 60,
    ) -> SearchResult:
        """하이브리드 검색 (벡터 + FTS, RRF)"""
        data = self._post("/search/hybrid", {
            "query_embedding": query_embedding,
            "query_text": query_text,
            "n_results": n_results,
            "where": where,
            "rrf_k": rrf_k,
        })
        return SearchResult(
            ids=data.get("ids", [[]]),
            distances=data.get("distances"),
            metadatas=data.get("metadatas"),
            documents=data.get("documents"),
        )

    # =========================================================================
    # 인덱스 관리
    # =========================================================================

    def create_vector_index(self, index_type: str = "IVF_FLAT") -> bool:
        """벡터 인덱스 생성 (마이크로서비스에 위임)"""
        data = self._post("/index/vector", {"index_type": index_type})
        created: bool = data.get("created", False)
        return created

    def create_fts_index(self, field: str = "content_tokenized") -> None:
        """FTS 인덱스 생성 (마이크로서비스에 위임)"""
        self._post("/index/fts", {"field": field})

    # =========================================================================
    # 미지원 (임베딩 스크립트가 직접 LanceDB 파일 접근)
    # =========================================================================

    def add_documents(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[str]] = None,
    ) -> None:
        raise NotImplementedError(
            "Remote 모드에서는 문서 추가를 지원하지 않습니다. "
            "임베딩 스크립트로 직접 LanceDB 데이터에 접근하세요."
        )

    def delete_by_ids(self, ids: List[str]) -> None:
        raise NotImplementedError(
            "Remote 모드에서는 문서 삭제를 지원하지 않습니다."
        )

    def reset(self) -> None:
        raise NotImplementedError(
            "Remote 모드에서는 테이블 초기화를 지원하지 않습니다."
        )
