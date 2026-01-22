"""
벡터 저장소 모듈

환경 변수 VECTOR_DB에 따라 적절한 벡터 DB 구현체를 선택합니다.
- chroma (기본값): ChromaDB 사용
- qdrant: Qdrant 사용

Usage:
    from app.common.vectorstore import get_vector_store, VectorStore

    # 팩토리 함수 사용 (권장)
    store = get_vector_store()

    # 또는 VectorStore 별칭 사용 (하위 호환)
    store = VectorStore()
"""

from typing import Optional

from app.core.config import settings
from app.common.vectorstore.base import VectorStoreBase, SearchResult


def get_vector_store(collection_name: Optional[str] = None) -> VectorStoreBase:
    """
    환경 설정에 따라 적절한 벡터 저장소 인스턴스 반환

    Args:
        collection_name: 컬렉션 이름 (기본값: settings에서)

    Returns:
        VectorStoreBase 구현체 (ChromaVectorStore 또는 QdrantVectorStore)

    환경 변수:
        VECTOR_DB: 사용할 벡터 DB (chroma, qdrant)
    """
    vector_db = getattr(settings, "VECTOR_DB", "chroma").lower()

    if vector_db == "qdrant":
        from app.common.vectorstore.qdrant import QdrantVectorStore
        return QdrantVectorStore(collection_name=collection_name)
    else:
        # 기본값: ChromaDB
        from app.common.vectorstore.chroma import ChromaVectorStore
        return ChromaVectorStore(collection_name=collection_name)


# 하위 호환성을 위한 별칭
# 기존 코드: from app.common.vectorstore import VectorStore
# 새 코드: from app.common.vectorstore import get_vector_store
class VectorStore(VectorStoreBase):
    """
    하위 호환성을 위한 VectorStore 클래스

    기존 코드와의 호환성을 위해 제공됩니다.
    새 코드에서는 get_vector_store() 팩토리 함수 사용을 권장합니다.
    """

    def __new__(cls, collection_name: Optional[str] = None):
        # 실제로는 환경에 맞는 구현체를 반환
        return get_vector_store(collection_name)


# 직접 import할 수 있도록 export
__all__ = [
    "VectorStoreBase",
    "SearchResult",
    "VectorStore",
    "get_vector_store",
]
