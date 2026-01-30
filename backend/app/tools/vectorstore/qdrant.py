"""
Qdrant 벡터 저장소 구현체

법률 문서 임베딩 저장 및 유사도 검색 (Qdrant 사용)

사용하려면 qdrant-client 패키지 설치 필요:
    uv add qdrant-client
"""

from typing import Any, Dict, List, Optional

from app.tools.vectorstore.base import SearchResult, VectorStoreBase
from app.core.config import settings


class QdrantVectorStore(VectorStoreBase):
    """Qdrant 기반 벡터 저장소 구현체"""

    def __init__(self, collection_name: Optional[str] = None):
        try:
            from qdrant_client import (  # type: ignore[import-not-found]
                QdrantClient,  # noqa: F401
            )
            from qdrant_client.models import (  # type: ignore[import-not-found]
                Distance,  # noqa: F401
                VectorParams,  # noqa: F401
            )
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
            self.client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key,
            )
        else:
            self.client = QdrantClient(url=qdrant_url)

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """컬렉션이 없으면 생성"""
        from qdrant_client.models import Distance, VectorParams

        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if not exists:
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
        from qdrant_client.models import FieldCondition, Filter, MatchValue

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

        ids = [[str(r.id) for r in results]]
        distances = [[1 - r.score for r in results]]
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
        return int(collection_info.points_count)

    def reset(self) -> None:
        """컬렉션 초기화 (주의: 모든 데이터 삭제)"""
        self.client.delete_collection(self.collection_name)
        self._ensure_collection()
