"""
ChromaDB 벡터 저장소 설정

법률 문서 임베딩 저장 및 유사도 검색
"""

from pathlib import Path
from typing import List, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import settings


def get_chroma_client() -> chromadb.ClientAPI:
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
    client: Optional[chromadb.ClientAPI] = None,
    collection_name: Optional[str] = None,
) -> chromadb.Collection:
    """
    법률 문서 컬렉션 가져오기 (없으면 생성)

    Args:
        client: ChromaDB 클라이언트 (없으면 새로 생성)
        collection_name: 컬렉션 이름 (기본값: settings에서)

    Returns:
        chromadb.Collection
    """
    if client is None:
        client = get_chroma_client()

    name = collection_name or settings.CHROMA_COLLECTION_NAME

    return client.get_or_create_collection(
        name=name,
        metadata={
            "description": "법률 문서 임베딩 (판례, 헌재결정, 행정심판, 법령해석)",
            "hnsw:space": "cosine",  # 코사인 유사도 사용
        },
    )


class VectorStore:
    """
    법률 문서 벡터 저장소 클래스

    Usage:
        store = VectorStore()

        # 문서 추가
        store.add_documents(
            ids=["doc_1", "doc_2"],
            documents=["판례 내용...", "판례 내용..."],
            metadatas=[{"type": "precedent"}, {"type": "precedent"}],
            embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]]
        )

        # 유사 문서 검색
        results = store.search("손해배상 의료사고", n_results=5)
    """

    def __init__(self, collection_name: Optional[str] = None):
        self.client = get_chroma_client()
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
        self.collection = get_collection(self.client, self.collection_name)

    def add_documents(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: Optional[List[dict]] = None,
        embeddings: Optional[List[List[float]]] = None,
    ) -> None:
        """
        문서 추가 (임베딩 포함)

        Args:
            ids: 문서 ID 목록
            documents: 문서 텍스트 목록
            metadatas: 메타데이터 목록 (필터링용)
            embeddings: 임베딩 벡터 목록 (없으면 ChromaDB 기본 임베딩 사용)
        """
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def search(
        self,
        query_text: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
        n_results: int = 10,
        where: Optional[dict] = None,
        include: List[str] = ["documents", "metadatas", "distances"],
    ) -> dict:
        """
        유사 문서 검색

        Args:
            query_text: 검색 쿼리 텍스트 (임베딩 자동 생성)
            query_embedding: 검색 쿼리 임베딩 (직접 제공 시)
            n_results: 반환할 결과 수
            where: 필터 조건 (예: {"doc_type": "precedent"})
            include: 결과에 포함할 필드

        Returns:
            검색 결과 딕셔너리
        """
        kwargs = {
            "n_results": n_results,
            "include": include,
        }

        if where:
            kwargs["where"] = where

        if query_embedding:
            kwargs["query_embeddings"] = [query_embedding]
        elif query_text:
            kwargs["query_texts"] = [query_text]
        else:
            raise ValueError("query_text 또는 query_embedding 필요")

        return self.collection.query(**kwargs)

    def get_by_ids(self, ids: List[str]) -> dict:
        """ID로 문서 조회"""
        return self.collection.get(ids=ids)

    def get_by_id(self, doc_id: str) -> Optional[dict]:
        """단일 ID로 문서 조회"""
        result = self.collection.get(ids=[doc_id])
        if result and result.get("ids") and len(result["ids"]) > 0:
            return {
                "id": result["ids"][0],
                "content": result["documents"][0] if result.get("documents") else "",
                "metadata": result["metadatas"][0] if result.get("metadatas") else {},
            }
        return None

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
