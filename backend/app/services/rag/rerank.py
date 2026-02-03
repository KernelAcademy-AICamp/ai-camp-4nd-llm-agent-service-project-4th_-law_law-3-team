"""
리랭킹 서비스

검색 결과를 리랭킹하여 정확도 향상
"""

import logging
import warnings
from functools import lru_cache
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 기본 리랭커 모델명
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# 모듈 레벨 모델 캐싱
_reranker_model: Any = None
_reranker_model_name: Optional[str] = None


@lru_cache(maxsize=1)
def _load_reranker_model(model_name: str = DEFAULT_RERANKER_MODEL) -> Any:
    """리랭커 모델 로드 (캐싱)"""
    try:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(model_name)
        logger.info("리랭커 모델 로드 완료: %s", model_name)
        return model
    except ImportError:
        logger.warning(
            "sentence-transformers 패키지가 필요합니다. " "리랭킹이 비활성화됩니다."
        )
        return None
    except Exception as e:
        logger.warning("리랭커 모델 로드 실패: %s", e)
        return None


def is_reranker_available(model_name: str = DEFAULT_RERANKER_MODEL) -> bool:
    """
    리랭커 사용 가능 여부 확인

    Args:
        model_name: 리랭커 모델명

    Returns:
        리랭커 사용 가능 여부
    """
    return _load_reranker_model(model_name) is not None


def rerank_documents(
    query: str,
    documents: List[Dict[str, Any]],
    top_k: int = 5,
    model_name: str = DEFAULT_RERANKER_MODEL,
) -> List[Dict[str, Any]]:
    """
    문서 리랭킹

    Args:
        query: 검색 쿼리
        documents: 검색된 문서 목록
        top_k: 반환할 최상위 결과 수
        model_name: 리랭커 모델명

    Returns:
        리랭킹된 문서 목록 (rerank_score 필드 추가)
    """
    if not documents:
        return []

    model = _load_reranker_model(model_name)
    if model is None:
        # 모델 로드 실패 시 원본 반환
        return documents[:top_k]

    try:
        # 쿼리-문서 쌍 생성
        pairs = [
            (query, doc.get("content", "")[:512])  # 최대 512자 제한
            for doc in documents
        ]

        # Cross-encoder 점수 계산
        scores = model.predict(pairs)

        # 점수와 함께 문서 정렬
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # 상위 k개 반환 (리랭크 점수 추가)
        reranked = []
        for doc, score in scored_docs[:top_k]:
            doc_copy = doc.copy()
            doc_copy["rerank_score"] = float(score)
            reranked.append(doc_copy)

        return reranked

    except Exception as e:
        logger.warning("리랭킹 실패: %s", e)
        return documents[:top_k]


# =============================================================================
# 하위 호환성 유지용 (Deprecated)
# =============================================================================


class _RerankerServiceCompat:
    """
    RerankerService 하위 호환 래퍼

    .. deprecated:: 1.0.0
        `rerank_documents()` 함수를 직접 사용하세요.
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        warnings.warn(
            "RerankerService는 deprecated입니다. "
            "rerank_documents() 함수를 직접 사용하세요.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._model_name = model_name or DEFAULT_RERANKER_MODEL

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """문서 리랭킹"""
        return rerank_documents(query, documents, top_k, self._model_name)

    def is_available(self) -> bool:
        """리랭커 사용 가능 여부"""
        return is_reranker_available(self._model_name)


# 하위 호환용 alias
RerankerService = _RerankerServiceCompat

_reranker_service: Optional[_RerankerServiceCompat] = None


def get_reranker_service() -> _RerankerServiceCompat:
    """
    RerankerService 싱글톤 인스턴스 반환

    .. deprecated:: 1.0.0
        `rerank_documents()` 함수를 직접 사용하세요.
    """
    global _reranker_service
    if _reranker_service is None:
        _reranker_service = _RerankerServiceCompat()
    return _reranker_service
