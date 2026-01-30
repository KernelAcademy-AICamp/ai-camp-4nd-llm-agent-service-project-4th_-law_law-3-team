"""
ChromaDB 벡터 저장소 구현체

법률 문서 임베딩 저장 및 유사도 검색 (ChromaDB 사용)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.tools.vectorstore.base import SearchResult, VectorStoreBase
from app.core.config import settings


def get_chroma_client() -> chromadb.ClientAPI:  # type: ignore[name-defined]
    """
    ChromaDB 클라이언트 생성

    - 개발: 로컬 persistent storage
    - 프로덕션: ChromaDB 서버 연결 가능
    """
    persist_dir = Path(settings.CHROMA_PERSIST_DIR)
    persist_dir.mkdir(parents=True, exist_ok=True)

    return chromadb.PersistentClient(
        path=str(persist_dir),
        settings=ChromaSettings(
            anonymized_telemetry=False,
            allow_reset=True,
        ),
    )


def get_collection(
    client: Optional["chromadb.ClientAPI"] = None,  # type: ignore[name-defined]
    collection_name: Optional[str] = None,
) -> "chromadb.Collection":
    """법률 문서 컬렉션 가져오기 (없으면 생성)"""
    if client is None:
        client = get_chroma_client()

    name = collection_name or settings.CHROMA_COLLECTION_NAME

    return client.get_or_create_collection(  # type: ignore[no-any-return]
        name=name,
        metadata={
            "description": "법률 문서 임베딩 (판례, 헌재결정, 행정심판, 법령해석)",
            "hnsw:space": "cosine",  # 코사인 유사도 사용
        },
    )


class ChromaVectorStore(VectorStoreBase):
    """ChromaDB 기반 벡터 저장소 구현체"""

    def __init__(self, collection_name: Optional[str] = None):
        self.client = get_chroma_client()
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
        self.collection = get_collection(self.client, self.collection_name)

    def add_documents(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[str]] = None,
    ) -> None:
        """문서 추가 (임베딩 포함)"""
        kwargs: Dict[str, Any] = {
            "ids": ids,
            "embeddings": embeddings,
        }

        if metadatas is not None:
            kwargs["metadatas"] = metadatas

        if documents is not None:
            kwargs["documents"] = documents

        self.collection.add(**kwargs)

    def search(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
    ) -> SearchResult:
        """유사 문서 검색"""
        if include is None:
            include = ["metadatas", "distances"]

        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": include,
        }

        if where:
            kwargs["where"] = where

        result = self.collection.query(**kwargs)

        return SearchResult(
            ids=result.get("ids", []),
            distances=result.get("distances"),
            metadatas=result.get("metadatas"),  # type: ignore[arg-type]
            documents=result.get("documents"),
        )

    def search_by_text(
        self,
        query_text: str,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
    ) -> SearchResult:
        """텍스트로 직접 검색 (ChromaDB 내장 임베딩 사용)"""
        if include is None:
            include = ["metadatas", "distances"]

        kwargs: Dict[str, Any] = {
            "query_texts": [query_text],
            "n_results": n_results,
            "include": include,
        }

        if where:
            kwargs["where"] = where

        result = self.collection.query(**kwargs)

        return SearchResult(
            ids=result.get("ids", []),
            distances=result.get("distances"),
            metadatas=result.get("metadatas"),  # type: ignore[arg-type]
            documents=result.get("documents"),
        )

    def get_by_ids(self, ids: List[str]) -> Dict[str, Any]:
        """ID로 문서 조회"""
        result = self.collection.get(ids=ids)
        return dict(result)

    def delete_by_ids(self, ids: List[str]) -> None:
        """ID로 문서 삭제"""
        self.collection.delete(ids=ids)

    def count(self) -> int:
        """컬렉션 문서 수"""
        return self.collection.count()

    def reset(self) -> None:
        """컬렉션 초기화 (주의: 모든 데이터 삭제)"""
        self.client.delete_collection(self.collection_name)
        self.collection = get_collection(self.client, self.collection_name)
