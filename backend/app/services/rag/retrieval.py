"""
RAG 검색 서비스

VectorStore에서 관련 문서를 검색하고 임베딩을 생성하는 서비스
chat_service.py에서 추출
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from openai import OpenAI

from app.core.config import settings
from app.core.errors import EmbeddingModelNotFoundError
from app.tools.vectorstore import get_vector_store

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# 임베딩 모델 관련 상수
MODEL_CACHE_DIR = Path(__file__).parent.parent.parent.parent / "data" / "models"
_embedding_model_available: Optional[bool] = None
_embedding_model_warning_shown = False


def _get_model_cache_path(model_name: str) -> Path:
    """모델 캐시 경로 반환 (HuggingFace 캐시 구조)"""
    sanitized = model_name.replace("/", "--")
    return MODEL_CACHE_DIR / f"models--{sanitized}"


def is_embedding_model_cached(model_name: Optional[str] = None) -> bool:
    """임베딩 모델이 로컬에 캐시되어 있는지 확인"""
    model_name = model_name or settings.LOCAL_EMBEDDING_MODEL
    cache_path = _get_model_cache_path(model_name)

    if not cache_path.exists():
        return False

    # blobs 디렉토리에 .incomplete 파일이 있으면 다운로드 미완료
    blobs_dir = cache_path / "blobs"
    if blobs_dir.exists():
        for file in blobs_dir.iterdir():
            if file.name.endswith(".incomplete"):
                return False

    # snapshots 디렉토리에 실제 모델 파일이 있어야 함
    snapshots_dir = cache_path / "snapshots"
    if not snapshots_dir.exists():
        return False

    snapshots = list(snapshots_dir.iterdir())
    return len(snapshots) > 0


def check_embedding_model_availability() -> bool:
    """임베딩 모델 사용 가능 여부 확인 (서버 시작 시 호출)"""
    global _embedding_model_available, _embedding_model_warning_shown

    if _embedding_model_available is not None:
        return _embedding_model_available

    if not settings.USE_LOCAL_EMBEDDING:
        _embedding_model_available = True
        return True

    _embedding_model_available = is_embedding_model_cached()

    if not _embedding_model_available and not _embedding_model_warning_shown:
        _embedding_model_warning_shown = True
        warning_msg = (
            "\n" + "=" * 60 + "\n"
            "[WARNING] 임베딩 모델이 캐시되지 않았습니다.\n"
            f"모델명: {settings.LOCAL_EMBEDDING_MODEL}\n"
            "검색 API 사용 전 먼저 모델을 다운로드해주세요:\n"
            "  uv run python scripts/download_models.py\n"
            "=" * 60
        )
        print(warning_msg)
        logger.warning("임베딩 모델 미캐시: %s", settings.LOCAL_EMBEDDING_MODEL)

    return _embedding_model_available


@lru_cache(maxsize=1)
def get_local_model() -> "SentenceTransformer":
    """sentence-transformers 모델 로드 (캐싱)"""
    global _embedding_model_available

    if _embedding_model_available is None:
        _embedding_model_available = is_embedding_model_cached()

    if not _embedding_model_available:
        raise EmbeddingModelNotFoundError(settings.LOCAL_EMBEDDING_MODEL)

    from sentence_transformers import SentenceTransformer

    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    return SentenceTransformer(
        settings.LOCAL_EMBEDDING_MODEL,
        cache_folder=str(MODEL_CACHE_DIR),
        trust_remote_code=True,
        local_files_only=True,
    )


def create_query_embedding(query: str) -> List[float]:
    """쿼리 텍스트를 임베딩 벡터로 변환"""
    if settings.USE_LOCAL_EMBEDDING:
        model = get_local_model()
        embedding = model.encode(
            query,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embedding.tolist()
    else:
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        response = client.embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input=query,
        )
        return response.data[0].embedding


def _map_data_type(data_type: str) -> str:
    """LanceDB data_type을 표준 doc_type으로 변환"""
    mapping = {
        "판례": "precedent",
        "법령": "law",
        "헌법재판소": "constitutional",
    }
    return mapping.get(data_type, data_type.lower() if data_type else "")


def search_relevant_documents(
    query: str,
    n_results: int = 5,
    doc_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    관련 법률 문서 검색

    Args:
        query: 검색 쿼리
        n_results: 반환할 결과 수
        doc_type: 문서 유형 필터 (precedent, constitutional, etc.)

    Returns:
        관련 문서 목록
    """
    store = get_vector_store()
    query_embedding = create_query_embedding(query)

    # LanceDB는 data_type 필드 사용 (precedent→판례, law→법령)
    if doc_type:
        data_type_map = {"precedent": "판례", "law": "법령"}
        where = {"data_type": data_type_map.get(doc_type, doc_type)}
    else:
        where = None

    results = store.search(
        query_embedding=query_embedding,
        n_results=n_results,
        where=where,
        include=["metadatas", "distances"],
    )

    if not results or not results.get("ids") or not results["ids"][0]:
        return []

    documents = []
    for i, chunk_id in enumerate(results["ids"][0]):
        raw_metadata = results["metadatas"][0][i] if results.get("metadatas") else {}

        metadata = {
            "case_name": raw_metadata.get("title", ""),
            "case_number": raw_metadata.get("case_number", ""),
            "doc_type": _map_data_type(raw_metadata.get("data_type", "")),
            "court_name": raw_metadata.get("source_name", ""),
            "doc_id": raw_metadata.get("source_id"),
            "date": raw_metadata.get("date", ""),
            "chunk_index": raw_metadata.get("chunk_index", 0),
            "total_chunks": raw_metadata.get("total_chunks", 1),
        }

        source_id = raw_metadata.get("source_id")
        content = _get_chunk_content(store, chunk_id, source_id)

        doc = {
            "id": chunk_id,
            "content": content,
            "metadata": metadata,
            "similarity": 1 - results["distances"][0][i] if results.get("distances") else 0,
        }
        documents.append(doc)

    return documents


def _get_chunk_content(store: Any, chunk_id: str, source_id: Optional[str] = None) -> str:
    """청크 ID로 content 조회"""
    try:
        result = store.get_by_id(chunk_id)
        if result:
            content = result.get("content", "")
            if content:
                return content
    except Exception as e:
        logger.debug("LanceDB 청크 조회 실패: %s, %s", chunk_id, e)

    if source_id:
        try:
            from app.core.database import sync_session_factory
            from sqlalchemy import select
            from app.models.legal_document import LegalDocument

            with sync_session_factory() as session:
                result = session.execute(
                    select(LegalDocument.embedding_text)
                    .where(LegalDocument.serial_number == source_id)
                )
                row = result.scalar_one_or_none()
                if row:
                    return row
        except Exception as e:
            logger.debug("PostgreSQL fallback 실패: %s, %s", source_id, e)

    return ""


class RetrievalService:
    """검색 서비스 클래스"""

    def __init__(self) -> None:
        self._store = get_vector_store()

    def search(
        self,
        query: str,
        n_results: int = 5,
        doc_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """관련 문서 검색"""
        return search_relevant_documents(query, n_results, doc_type)

    def embed_query(self, text: str) -> List[float]:
        """텍스트 임베딩"""
        return create_query_embedding(text)


_retrieval_service: Optional[RetrievalService] = None


def get_retrieval_service() -> RetrievalService:
    """RetrievalService 싱글톤 인스턴스 반환"""
    global _retrieval_service
    if _retrieval_service is None:
        _retrieval_service = RetrievalService()
    return _retrieval_service
