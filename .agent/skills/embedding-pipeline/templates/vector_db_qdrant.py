"""
Qdrant Vector Database Manager
"""

from typing import List, Dict, Any, Optional
import uuid

import numpy as np

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import Distance, VectorParams, PointStruct
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False


class QdrantManager:
    """
    Qdrant 벡터 데이터베이스 매니저
    
    Args:
        host: Qdrant 서버 호스트
        port: Qdrant 서버 포트
        collection_name: 컬렉션 이름
        embedding_dim: 임베딩 차원
        use_grpc: gRPC 사용 여부
        api_key: API 키 (클라우드용)
    
    Example:
        # 로컬 서버
        db = QdrantManager("localhost", 6333, "documents", 1536)
        
        # Qdrant Cloud
        db = QdrantManager(
            host="xxx.cloud.qdrant.io",
            port=6333,
            collection_name="documents",
            api_key="your-api-key"
        )
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "documents",
        embedding_dim: int = 1536,
        use_grpc: bool = False,
        api_key: Optional[str] = None,
        path: Optional[str] = None  # 로컬 파일 모드
    ):
        if not HAS_QDRANT:
            raise ImportError("qdrant-client가 필요합니다: pip install qdrant-client")
        
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        
        # 클라이언트 생성
        if path:
            # 로컬 파일 모드 (서버 없이)
            self.client = QdrantClient(path=path)
        elif api_key:
            # 클라우드 모드
            self.client = QdrantClient(
                host=host,
                port=port,
                api_key=api_key,
                prefer_grpc=use_grpc
            )
        else:
            # 로컬 서버 모드
            self.client = QdrantClient(
                host=host,
                port=port,
                prefer_grpc=use_grpc
            )
        
        # 컬렉션 생성/확인
        self._ensure_collection()
    
    def _ensure_collection(self):
        """컬렉션 존재 확인 및 생성"""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )
            print(f"Created collection: {self.collection_name}")
        else:
            print(f"Using existing collection: {self.collection_name}")
    
    def add_documents(
        self,
        ids: List[str],
        embeddings: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None,
        texts: Optional[List[str]] = None,
        batch_size: int = 100
    ):
        """
        문서 추가
        
        Args:
            ids: 문서 ID 리스트
            embeddings: 임베딩 배열
            metadata: 메타데이터 리스트
            texts: 원본 텍스트 리스트
            batch_size: 배치 크기
        """
        points = []
        
        for i, (doc_id, embedding) in enumerate(zip(ids, embeddings)):
            # ID를 UUID로 변환 (Qdrant는 UUID 또는 정수 ID 사용)
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(doc_id)))
            
            payload = {"original_id": str(doc_id)}
            
            if texts and i < len(texts):
                payload["text"] = texts[i]
            
            if metadata and i < len(metadata):
                payload.update(metadata[i])
            
            vector = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
            
            points.append(PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            ))
        
        # 배치로 업로드
        for start in range(0, len(points), batch_size):
            batch = points[start:start + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
        
        print(f"Added {len(points)} documents")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None,
        include_vectors: bool = False
    ) -> List[Dict[str, Any]]:
        """
        유사도 검색
        
        Args:
            query_embedding: 쿼리 임베딩
            top_k: 반환할 결과 수
            filter_conditions: 필터 조건
                예: {"must": [{"key": "법령종류", "match": {"value": "법률"}}]}
            include_vectors: 벡터 포함 여부
        
        Returns:
            검색 결과 리스트
        """
        query_vec = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
        
        # 필터 빌드
        query_filter = None
        if filter_conditions:
            query_filter = models.Filter(**filter_conditions)
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vec,
            limit=top_k,
            query_filter=query_filter,
            with_vectors=include_vectors
        )
        
        cleaned_results = []
        for r in results:
            result = {
                "id": r.payload.get("original_id", str(r.id)),
                "score": r.score,
                "text": r.payload.get("text", ""),
                "metadata": {k: v for k, v in r.payload.items() if k not in ["original_id", "text"]}
            }
            if include_vectors:
                result["vector"] = r.vector
            cleaned_results.append(result)
        
        return cleaned_results
    
    def delete_by_ids(self, ids: List[str]):
        """ID로 문서 삭제"""
        point_ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, str(doc_id))) for doc_id in ids]
        
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=point_ids)
        )
        print(f"Deleted {len(ids)} documents")
    
    def delete_by_filter(self, filter_conditions: Dict[str, Any]):
        """필터로 문서 삭제"""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(**filter_conditions)
            )
        )
    
    def count(self) -> int:
        """문서 수 반환"""
        info = self.client.get_collection(self.collection_name)
        return info.points_count
    
    def get_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """ID로 문서 조회"""
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(doc_id)))
        
        results = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[point_id],
            with_vectors=True
        )
        
        if results:
            r = results[0]
            return {
                "id": r.payload.get("original_id", str(r.id)),
                "vector": r.vector,
                **r.payload
            }
        return None
    
    def scroll_all(
        self,
        batch_size: int = 100,
        with_vectors: bool = False
    ):
        """모든 문서 순회 (Generator)"""
        offset = None
        
        while True:
            results, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=batch_size,
                offset=offset,
                with_vectors=with_vectors
            )
            
            if not results:
                break
            
            for r in results:
                yield {
                    "id": r.payload.get("original_id", str(r.id)),
                    "vector": r.vector if with_vectors else None,
                    **r.payload
                }
            
            offset = next_offset
            if offset is None:
                break
    
    def create_index(self, field_name: str, field_type: str = "keyword"):
        """
        페이로드 필드에 인덱스 생성
        
        Args:
            field_name: 필드 이름
            field_type: 필드 타입 ("keyword", "integer", "float", "geo", "text")
        """
        schema_type = {
            "keyword": models.PayloadSchemaType.KEYWORD,
            "integer": models.PayloadSchemaType.INTEGER,
            "float": models.PayloadSchemaType.FLOAT,
            "text": models.PayloadSchemaType.TEXT,
        }
        
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name=field_name,
            field_schema=schema_type.get(field_type, models.PayloadSchemaType.KEYWORD)
        )
        print(f"Created index on field: {field_name}")
    
    def get_stats(self) -> Dict[str, Any]:
        """컬렉션 통계"""
        info = self.client.get_collection(self.collection_name)
        
        return {
            "name": self.collection_name,
            "count": info.points_count,
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "embedding_dim": self.embedding_dim,
            "status": info.status.value
        }
    
    def optimize(self):
        """컬렉션 최적화"""
        self.client.update_collection(
            collection_name=self.collection_name,
            optimizer_config=models.OptimizersConfigDiff(
                indexing_threshold=10000
            )
        )
        print("Collection optimization triggered")
    
    def delete_collection(self):
        """컬렉션 삭제"""
        self.client.delete_collection(self.collection_name)
        print(f"Deleted collection: {self.collection_name}")


def create_filter(
    must: Optional[List[Dict]] = None,
    should: Optional[List[Dict]] = None,
    must_not: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """
    Qdrant 필터 생성 헬퍼
    
    Example:
        filter = create_filter(
            must=[
                {"key": "법령종류", "match": {"value": "법률"}},
                {"key": "시행일자", "range": {"gte": "2020-01-01"}}
            ]
        )
        results = db.search(query_emb, filter_conditions=filter)
    """
    conditions = {}
    
    if must:
        conditions["must"] = must
    if should:
        conditions["should"] = should
    if must_not:
        conditions["must_not"] = must_not
    
    return conditions


if __name__ == "__main__":
    print("Qdrant Manager 모듈 로드 완료")
    print(f"Qdrant available: {HAS_QDRANT}")
    
    if HAS_QDRANT:
        # 로컬 파일 모드 테스트
        db = QdrantManager(path="./test_qdrant", collection_name="test", embedding_dim=128)
        
        # 테스트 데이터
        ids = ["doc1", "doc2", "doc3"]
        embeddings = np.random.randn(3, 128).astype(np.float32)
        metadata = [
            {"title": "문서1", "type": "법률"},
            {"title": "문서2", "type": "시행령"},
            {"title": "문서3", "type": "법률"},
        ]
        
        db.add_documents(ids, embeddings, metadata)
        
        query = np.random.randn(128).astype(np.float32)
        results = db.search(query, top_k=2)
        
        print(f"Search results: {len(results)}")
        for r in results:
            print(f"  - {r['id']}: {r['score']:.4f}")
        
        print(f"Stats: {db.get_stats()}")
