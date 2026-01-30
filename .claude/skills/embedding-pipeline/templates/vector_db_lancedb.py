"""
LanceDB Vector Database Manager
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

try:
    import lancedb
    HAS_LANCEDB = True
except ImportError:
    HAS_LANCEDB = False

try:
    import pyarrow as pa
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False


class LanceDBManager:
    """
    LanceDB 벡터 데이터베이스 매니저
    
    Args:
        db_path: 데이터베이스 경로
        table_name: 테이블 이름
        embedding_dim: 임베딩 차원
    
    Example:
        db = LanceDBManager("./legal_vectors", "documents", 1536)
        db.add_documents(ids, embeddings, metadata)
        results = db.search(query_embedding, top_k=10)
    """
    
    def __init__(
        self,
        db_path: str = "./lancedb",
        table_name: str = "documents",
        embedding_dim: int = 1536
    ):
        if not HAS_LANCEDB:
            raise ImportError("lancedb가 필요합니다: pip install lancedb")
        
        self.db_path = Path(db_path)
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        
        self.db = lancedb.connect(str(self.db_path))
        self.table = None
        
        if table_name in self.db.table_names():
            self.table = self.db.open_table(table_name)
            print(f"Loaded existing table: {table_name}")
    
    def add_documents(
        self,
        ids: List[str],
        embeddings: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None,
        texts: Optional[List[str]] = None,
        batch_size: int = 1000
    ):
        """문서 추가"""
        data = []
        for i, (doc_id, embedding) in enumerate(zip(ids, embeddings)):
            record = {
                "id": str(doc_id),
                "vector": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                "text": texts[i] if texts else "",
            }
            
            if metadata and i < len(metadata):
                for key, value in metadata[i].items():
                    if key not in ['id', 'vector', 'text']:
                        record[key] = str(value) if not isinstance(value, (int, float)) else value
            
            data.append(record)
        
        if self.table is None:
            self.table = self.db.create_table(self.table_name, data)
        else:
            for start in range(0, len(data), batch_size):
                batch = data[start:start + batch_size]
                self.table.add(batch)
        
        print(f"Added {len(data)} documents")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        include_vectors: bool = False
    ) -> List[Dict[str, Any]]:
        """유사도 검색"""
        if self.table is None:
            return []
        
        query_vec = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
        
        search_query = self.table.search(query_vec)
        
        if filter:
            where_clauses = []
            for key, value in filter.items():
                if isinstance(value, str):
                    where_clauses.append(f"{key} = '{value}'")
                else:
                    where_clauses.append(f"{key} = {value}")
            if where_clauses:
                search_query = search_query.where(" AND ".join(where_clauses))
        
        results = search_query.limit(top_k).to_list()
        
        cleaned_results = []
        for r in results:
            result = {
                "id": r.get("id"),
                "score": 1 - r.get("_distance", 0),
                "text": r.get("text", ""),
                "metadata": {k: v for k, v in r.items() if k not in ["id", "vector", "text", "_distance"]}
            }
            if include_vectors:
                result["vector"] = r.get("vector")
            cleaned_results.append(result)
        
        return cleaned_results
    
    def delete_by_ids(self, ids: List[str]):
        """ID로 문서 삭제"""
        if self.table is None:
            return
        ids_str = ", ".join([f"'{id}'" for id in ids])
        self.table.delete(f"id IN ({ids_str})")
    
    def count(self) -> int:
        """문서 수 반환"""
        return self.table.count_rows() if self.table else 0
    
    def get_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """ID로 문서 조회"""
        if self.table is None:
            return None
        results = self.table.search().where(f"id = '{doc_id}'").limit(1).to_list()
        return results[0] if results else None
    
    def list_all_ids(self) -> List[str]:
        """모든 문서 ID 조회"""
        if self.table is None:
            return []
        df = self.table.to_pandas()
        return df['id'].tolist()
    
    def create_index(self, num_partitions: int = 256, num_sub_vectors: int = 96):
        """IVF-PQ 인덱스 생성"""
        if self.table:
            self.table.create_index(
                metric="cosine",
                num_partitions=num_partitions,
                num_sub_vectors=num_sub_vectors
            )
            print(f"Created IVF-PQ index")
    
    def compact(self):
        """테이블 압축"""
        if self.table:
            self.table.compact_files()
    
    def get_stats(self) -> Dict[str, Any]:
        """테이블 통계"""
        return {
            "exists": self.table is not None,
            "name": self.table_name,
            "count": self.count(),
            "embedding_dim": self.embedding_dim,
            "path": str(self.db_path)
        }


if __name__ == "__main__":
    print("LanceDB Manager 모듈 로드 완료")
    print(f"LanceDB available: {HAS_LANCEDB}")
