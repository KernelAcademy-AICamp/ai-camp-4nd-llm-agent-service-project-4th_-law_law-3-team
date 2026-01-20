"""
벡터 저장소 추상 인터페이스

ChromaDB, Qdrant 등 다양한 벡터 DB를 지원하기 위한 공통 인터페이스
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class SearchResult:
    """
    검색 결과 데이터 클래스

    dict처럼 접근 가능 (하위 호환성)
    """
    ids: List[List[str]]
    distances: Optional[List[List[float]]] = None
    metadatas: Optional[List[List[Dict[str, Any]]]] = None
    documents: Optional[List[List[str]]] = None

    def get(self, key: str, default: Any = None) -> Any:
        """dict.get() 호환 메서드"""
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        """dict[key] 호환 메서드"""
        return getattr(self, key)


class VectorStoreBase(ABC):
    """
    벡터 저장소 추상 기본 클래스

    모든 벡터 DB 구현체는 이 클래스를 상속받아야 합니다.

    Usage:
        store = get_vector_store()  # 환경에 따라 ChromaDB 또는 Qdrant 반환

        # 문서 추가
        store.add_documents(
            ids=["doc_1", "doc_2"],
            embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
            metadatas=[{"type": "precedent"}, {"type": "precedent"}],
        )

        # 유사 문서 검색
        results = store.search(query_embedding=[0.1, 0.2, ...], n_results=5)
    """

    @abstractmethod
    def add_documents(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[str]] = None,
    ) -> None:
        """
        문서 추가 (임베딩 포함)

        Args:
            ids: 문서 ID 목록
            embeddings: 임베딩 벡터 목록
            metadatas: 메타데이터 목록 (필터링용)
            documents: 문서 텍스트 목록 (선택 - 용량 최적화 시 None)
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
    ) -> SearchResult:
        """
        유사 문서 검색

        Args:
            query_embedding: 검색 쿼리 임베딩 벡터
            n_results: 반환할 결과 수
            where: 필터 조건 (예: {"doc_type": "precedent"})
            include: 결과에 포함할 필드 (metadatas, documents, distances)

        Returns:
            SearchResult 객체
        """
        pass

    @abstractmethod
    def get_by_ids(self, ids: List[str]) -> Dict[str, Any]:
        """ID로 문서 조회"""
        pass

    @abstractmethod
    def delete_by_ids(self, ids: List[str]) -> None:
        """ID로 문서 삭제"""
        pass

    @abstractmethod
    def count(self) -> int:
        """컬렉션 문서 수"""
        pass

    @abstractmethod
    def reset(self) -> None:
        """컬렉션 초기화 (주의: 모든 데이터 삭제)"""
        pass

    def get_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """단일 ID로 문서 조회"""
        result = self.get_by_ids([doc_id])
        if result and result.get("ids") and len(result["ids"]) > 0:
            return {
                "id": result["ids"][0],
                "content": result["documents"][0] if result.get("documents") else "",
                "metadata": result["metadatas"][0] if result.get("metadatas") else {},
            }
        return None
