"""
Cache Manager - 임베딩 캐싱 및 증분 업데이트
"""

import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Set
import pickle

import numpy as np


class EmbeddingCache:
    """
    임베딩 캐시 매니저
    
    Args:
        cache_dir: 캐시 디렉토리
        model_name: 모델 이름 (캐시 키에 포함)
    
    Example:
        cache = EmbeddingCache("./cache", "nlpai-lab/KURE-v1")
        
        embedding = cache.get(text)
        if embedding is None:
            embedding = generator.encode(text)
            cache.set(text, embedding)
    """
    
    def __init__(
        self,
        cache_dir: str = "./embedding_cache",
        model_name: str = "default"
    ):
        self.cache_dir = Path(cache_dir)
        self.model_name = model_name
        self.model_hash = hashlib.md5(model_name.encode()).hexdigest()[:8]
        
        self.cache_path = self.cache_dir / f"cache_{self.model_hash}"
        self.cache_path.mkdir(parents=True, exist_ok=True)
        
        self.index_path = self.cache_path / "index.json"
        self.index = self._load_index()
    
    def _load_index(self) -> Dict[str, str]:
        """인덱스 로드"""
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_index(self):
        """인덱스 저장"""
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f)
    
    def _hash_text(self, text: str) -> str:
        """텍스트 해시 생성"""
        return hashlib.sha256(text.encode()).hexdigest()
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """캐시에서 임베딩 조회"""
        text_hash = self._hash_text(text)
        
        if text_hash not in self.index:
            return None
        
        file_path = self.cache_path / self.index[text_hash]
        
        if not file_path.exists():
            del self.index[text_hash]
            self._save_index()
            return None
        
        return np.load(file_path)
    
    def set(self, text: str, embedding: np.ndarray):
        """캐시에 임베딩 저장"""
        text_hash = self._hash_text(text)
        filename = f"{text_hash[:16]}.npy"
        
        file_path = self.cache_path / filename
        np.save(file_path, embedding)
        
        self.index[text_hash] = filename
        self._save_index()
    
    def get_batch(self, texts: List[str]) -> Dict[str, Optional[np.ndarray]]:
        """배치 캐시 조회"""
        return {text: self.get(text) for text in texts}
    
    def set_batch(self, texts: List[str], embeddings: np.ndarray):
        """배치 캐시 저장"""
        for text, embedding in zip(texts, embeddings):
            self.set(text, embedding)
    
    def get_or_compute_batch(
        self,
        texts: List[str],
        compute_fn: Callable[[List[str]], np.ndarray]
    ) -> np.ndarray:
        """캐시 확인 후 없는 것만 계산"""
        results = [None] * len(texts)
        texts_to_compute = []
        indices_to_compute = []
        
        for i, text in enumerate(texts):
            cached = self.get(text)
            if cached is not None:
                results[i] = cached
            else:
                texts_to_compute.append(text)
                indices_to_compute.append(i)
        
        if texts_to_compute:
            computed = compute_fn(texts_to_compute)
            
            for idx, text, embedding in zip(indices_to_compute, texts_to_compute, computed):
                results[idx] = embedding
                self.set(text, embedding)
        
        return np.array(results)
    
    def contains(self, text: str) -> bool:
        """캐시 존재 여부 확인"""
        return self._hash_text(text) in self.index
    
    def clear(self):
        """캐시 전체 삭제"""
        import shutil
        shutil.rmtree(self.cache_path)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.index = {}
        self._save_index()
    
    def stats(self) -> Dict[str, Any]:
        """캐시 통계"""
        total_size = sum(
            (self.cache_path / f).stat().st_size
            for f in self.cache_path.iterdir()
            if f.suffix == '.npy'
        )
        
        return {
            "count": len(self.index),
            "size_mb": total_size / 1e6,
            "cache_dir": str(self.cache_path)
        }


class IncrementalUpdater:
    """
    증분 업데이트 매니저
    
    신규/변경된 문서만 처리
    
    Args:
        db: 벡터 DB 매니저 (LanceDBManager 또는 QdrantManager)
        cache: EmbeddingCache 인스턴스
        hash_store_path: 문서 해시 저장 경로
    
    Example:
        updater = IncrementalUpdater(db, cache, "hashes.json")
        
        changed_docs = updater.get_changed_documents(new_documents)
        if changed_docs:
            updater.process_updates(changed_docs, generator)
    """
    
    def __init__(
        self,
        db,
        cache: Optional[EmbeddingCache] = None,
        hash_store_path: str = "document_hashes.json"
    ):
        self.db = db
        self.cache = cache
        self.hash_store_path = Path(hash_store_path)
        self.document_hashes = self._load_hashes()
    
    def _load_hashes(self) -> Dict[str, str]:
        """해시 저장소 로드"""
        if self.hash_store_path.exists():
            with open(self.hash_store_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_hashes(self):
        """해시 저장소 저장"""
        with open(self.hash_store_path, 'w') as f:
            json.dump(self.document_hashes, f)
    
    def _hash_document(self, doc: Dict[str, Any], text_field: str = 'text') -> str:
        """문서 해시 생성"""
        text = doc.get(text_field, '')
        return hashlib.sha256(text.encode()).hexdigest()
    
    def get_changed_documents(
        self,
        documents: List[Dict[str, Any]],
        id_field: str = 'id',
        text_field: str = 'text'
    ) -> List[Dict[str, Any]]:
        """변경된 문서 찾기"""
        changed = []
        
        for doc in documents:
            doc_id = str(doc[id_field])
            doc_hash = self._hash_document(doc, text_field)
            
            if doc_id not in self.document_hashes:
                changed.append(doc)
            elif self.document_hashes[doc_id] != doc_hash:
                changed.append(doc)
        
        return changed
    
    def get_deleted_ids(
        self,
        current_documents: List[Dict[str, Any]],
        id_field: str = 'id'
    ) -> List[str]:
        """삭제된 문서 ID 찾기"""
        current_ids = {str(doc[id_field]) for doc in current_documents}
        stored_ids = set(self.document_hashes.keys())
        
        return list(stored_ids - current_ids)
    
    def delete_old_embeddings(self, documents: List[Dict[str, Any]], id_field: str = 'id'):
        """기존 임베딩 삭제"""
        ids_to_delete = [str(doc[id_field]) for doc in documents]
        
        if ids_to_delete:
            self.db.delete_by_ids(ids_to_delete)
            
            for doc_id in ids_to_delete:
                if doc_id in self.document_hashes:
                    del self.document_hashes[doc_id]
            
            self._save_hashes()
    
    def process_updates(
        self,
        documents: List[Dict[str, Any]],
        generator,
        id_field: str = 'id',
        text_field: str = 'text'
    ):
        """업데이트 처리"""
        if not documents:
            return
        
        ids = [str(doc[id_field]) for doc in documents]
        texts = [doc[text_field] for doc in documents]
        metadata = [
            {k: v for k, v in doc.items() if k not in [id_field, text_field]}
            for doc in documents
        ]
        
        if self.cache:
            embeddings = self.cache.get_or_compute_batch(
                texts,
                lambda t: generator.encode_batch(t, show_progress=True)
            )
        else:
            embeddings = generator.encode_batch(texts, show_progress=True)
        
        self.db.add_documents(
            ids=ids,
            embeddings=embeddings,
            metadata=metadata,
            texts=texts
        )
        
        for doc in documents:
            doc_id = str(doc[id_field])
            doc_hash = self._hash_document(doc, text_field)
            self.document_hashes[doc_id] = doc_hash
        
        self._save_hashes()
        print(f"Processed {len(documents)} documents")
    
    def full_sync(
        self,
        documents: List[Dict[str, Any]],
        generator,
        id_field: str = 'id',
        text_field: str = 'text'
    ):
        """전체 동기화"""
        changed = self.get_changed_documents(documents, id_field, text_field)
        deleted_ids = self.get_deleted_ids(documents, id_field)
        
        print(f"Changed: {len(changed)}, Deleted: {len(deleted_ids)}")
        
        if deleted_ids:
            self.db.delete_by_ids(deleted_ids)
            for doc_id in deleted_ids:
                if doc_id in self.document_hashes:
                    del self.document_hashes[doc_id]
        
        if changed:
            self.delete_old_embeddings(changed, id_field)
            self.process_updates(changed, generator, id_field, text_field)
        
        self._save_hashes()


class DuplicateDetector:
    """
    중복 문서 탐지기
    
    Example:
        detector = DuplicateDetector()
        
        exact_dups = detector.find_exact_duplicates(documents, 'content')
        similar_dups = detector.find_similar_documents(embeddings, threshold=0.95)
    """
    
    def __init__(self):
        pass
    
    def _hash_text(self, text: str) -> str:
        """텍스트 해시"""
        normalized = ' '.join(text.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def find_exact_duplicates(
        self,
        documents: List[Dict[str, Any]],
        text_field: str = 'text'
    ) -> Dict[str, List[int]]:
        """정확히 동일한 문서 찾기"""
        hash_to_indices: Dict[str, List[int]] = {}
        
        for i, doc in enumerate(documents):
            text = doc.get(text_field, '')
            text_hash = self._hash_text(text)
            
            if text_hash not in hash_to_indices:
                hash_to_indices[text_hash] = []
            hash_to_indices[text_hash].append(i)
        
        return {
            h: indices for h, indices in hash_to_indices.items()
            if len(indices) > 1
        }
    
    def find_similar_documents(
        self,
        embeddings: np.ndarray,
        threshold: float = 0.95
    ) -> List[tuple]:
        """
        유사한 문서 쌍 찾기
        
        Args:
            embeddings: 임베딩 배열
            threshold: 유사도 임계값
        
        Returns:
            유사한 문서 쌍 리스트 [(i, j, similarity), ...]
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        n = len(embeddings)
        similar_pairs = []
        
        batch_size = 1000
        
        for i in range(0, n, batch_size):
            end_i = min(i + batch_size, n)
            batch_i = embeddings[i:end_i]
            
            for j in range(i, n, batch_size):
                end_j = min(j + batch_size, n)
                batch_j = embeddings[j:end_j]
                
                similarities = cosine_similarity(batch_i, batch_j)
                
                for bi, row in enumerate(similarities):
                    for bj, sim in enumerate(row):
                        actual_i = i + bi
                        actual_j = j + bj
                        
                        if actual_i < actual_j and sim >= threshold:
                            similar_pairs.append((actual_i, actual_j, float(sim)))
        
        return similar_pairs
    
    def remove_duplicates(
        self,
        documents: List[Dict[str, Any]],
        text_field: str = 'text',
        keep: str = 'first'
    ) -> List[Dict[str, Any]]:
        """
        중복 문서 제거
        
        Args:
            documents: 문서 리스트
            text_field: 텍스트 필드
            keep: 'first' 또는 'last'
        
        Returns:
            중복 제거된 문서 리스트
        """
        seen_hashes: Set[str] = set()
        unique_documents = []
        
        doc_list = documents if keep == 'first' else reversed(documents)
        
        for doc in doc_list:
            text = doc.get(text_field, '')
            text_hash = self._hash_text(text)
            
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                unique_documents.append(doc)
        
        if keep == 'last':
            unique_documents.reverse()
        
        return unique_documents
    
    def deduplicate_with_embeddings(
        self,
        documents: List[Dict[str, Any]],
        embeddings: np.ndarray,
        threshold: float = 0.95
    ) -> tuple:
        """
        임베딩 기반 중복 제거
        
        Returns:
            (unique_documents, unique_embeddings, removed_indices)
        """
        n = len(documents)
        keep_mask = [True] * n
        
        similar_pairs = self.find_similar_documents(embeddings, threshold)
        
        for i, j, sim in similar_pairs:
            if keep_mask[j]:
                keep_mask[j] = False
        
        unique_docs = [doc for doc, keep in zip(documents, keep_mask) if keep]
        unique_embs = embeddings[keep_mask]
        removed = [i for i, keep in enumerate(keep_mask) if not keep]
        
        return unique_docs, unique_embs, removed


if __name__ == "__main__":
    print("Cache Manager 모듈 로드 완료")
    
    cache = EmbeddingCache("./test_cache", "test-model")
    
    test_embedding = np.random.randn(768).astype(np.float32)
    cache.set("테스트 텍스트", test_embedding)
    
    retrieved = cache.get("테스트 텍스트")
    print(f"Cache hit: {retrieved is not None}")
    print(f"Cache stats: {cache.stats()}")
