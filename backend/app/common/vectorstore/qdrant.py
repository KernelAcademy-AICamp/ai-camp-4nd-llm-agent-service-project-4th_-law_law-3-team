"""
Qdrant 벡터 저장소 구현체

법률 문서 임베딩 저장 및 유사도 검색 (Qdrant 사용)

사용하려면 qdrant-client 패키지 설치 필요:
    uv add qdrant-client
"""

from typing import List, Optional, Dict, Any

from app.core.config import settings
from app.common.vectorstore.base import VectorStoreBase, SearchResult


class QdrantVectorStore(VectorStoreBase):
    """
    Qdrant 기반 벡터 저장소 구현체

    Qdrant Cloud 또는 로컬 Qdrant 서버에 연결하여 사용합니다.

    환경 변수:
        QDRANT_URL: Qdrant 서버 URL (예: http://localhost:6333)
        QDRANT_API_KEY: Qdrant Cloud API 키 (선택)
        QDRANT_COLLECTION_NAME: 컬렉션 이름

    Usage:
        store = QdrantVectorStore()

        # 문서 추가
        store.add_documents(
            ids=["doc_1", "doc_2"],
            embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
            metadatas=[{"type": "precedent"}, {"type": "precedent"}],
        )

        # 유사 문서 검색
        results = store.search(query_embedding=[0.1, 0.2, ...], n_results=5)
    """

    def __init__(self, collection_name: Optional[str] = None):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError:
            raise ImportError(
                "qdrant-client 패키지가 필요합니다. "
                "설치: uv add qdrant-client"
            )

        self.collection_name = collection_name or getattr(
            settings, "QDRANT_COLLECTION_NAME", "legal_documents"
        )

        # Qdrant 클라이언트 초기화
        qdrant_url = getattr(settings, "QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = getattr(settings, "QDRANT_API_KEY", None)

        if qdrant_api_key:
            # Qdrant Cloud
            self.client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key,
            )
        else:
            # 로컬 Qdrant
            self.client = QdrantClient(url=qdrant_url)

        # 컬렉션 생성 (없으면)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """컬렉션이 없으면 생성"""
        from qdrant_client.models import Distance, VectorParams

        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if not exists:
            # 임베딩 차원 (로컬 모델: 384, OpenAI: 1536)
            vector_size = getattr(settings, "EMBEDDING_DIMENSION", 384)

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def add_documents(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[str]] = None,
    ) -> None:
        """문서 추가 (임베딩 포함)"""
        from qdrant_client.models import PointStruct

        points = []
        for i, (doc_id, embedding) in enumerate(zip(ids, embeddings)):
            payload: Dict[str, Any] = {}

            if metadatas and i < len(metadatas):
                payload.update(metadatas[i])

            if documents and i < len(documents):
                payload["document"] = documents[i]

            points.append(
                PointStruct(
                    id=doc_id,
                    vector=embedding,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def search(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
    ) -> SearchResult:
        """유사 문서 검색"""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        # 필터 조건 변환
        query_filter = None
        if where:
            conditions = []
            for key, value in where.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value),
                    )
                )
            query_filter = Filter(must=conditions)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=n_results,
            query_filter=query_filter,
            with_payload=True,
        )

        # SearchResult 형식으로 변환
        ids = [[str(r.id) for r in results]]
        distances = [[1 - r.score for r in results]]  # Qdrant는 similarity 반환, distance로 변환
        metadatas = [[r.payload or {} for r in results]]
        documents = [[r.payload.get("document", "") for r in results]] if include and "documents" in include else None

        return SearchResult(
            ids=ids,
            distances=distances,
            metadatas=metadatas,
            documents=documents,
        )

    def get_by_ids(self, ids: List[str]) -> Dict[str, Any]:
        """ID로 문서 조회"""
        results = self.client.retrieve(
            collection_name=self.collection_name,
            ids=ids,
            with_payload=True,
        )

        return {
            "ids": [str(r.id) for r in results],
            "metadatas": [r.payload or {} for r in results],
            "documents": [r.payload.get("document", "") for r in results if r.payload],
        }

    def delete_by_ids(self, ids: List[str]) -> None:
        """ID로 문서 삭제"""
        from qdrant_client.models import PointIdsList

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=ids),
        )

    def count(self) -> int:
        """컬렉션 문서 수"""
        collection_info = self.client.get_collection(self.collection_name)
        return collection_info.points_count

    def reset(self) -> None:
        """컬렉션 초기화 (주의: 모든 데이터 삭제)"""
        self.client.delete_collection(self.collection_name)
        self._ensure_collection()
