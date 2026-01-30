"""
리랭킹 서비스

검색 결과를 리랭킹하여 정확도 향상
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RerankerService:
    """
    리랭킹 서비스

    검색된 문서를 재순위화하여 관련성 높은 결과 반환
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        """
        Args:
            model_name: 리랭킹 모델명 (기본값: cross-encoder)
        """
        self._model_name = model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self._model: Any = None

    def _load_model(self) -> Any:
        """리랭커 모델 로드 (lazy loading)"""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self._model_name)
                logger.info("리랭커 모델 로드 완료: %s", self._model_name)
            except ImportError:
                logger.warning(
                    "sentence-transformers 패키지가 필요합니다. "
                    "리랭킹이 비활성화됩니다."
                )
            except Exception as e:
                logger.warning("리랭커 모델 로드 실패: %s", e)
        return self._model

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        문서 리랭킹

        Args:
            query: 검색 쿼리
            documents: 검색된 문서 목록
            top_k: 반환할 최상위 결과 수

        Returns:
            리랭킹된 문서 목록
        """
        if not documents:
            return []

        model = self._load_model()
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

    def is_available(self) -> bool:
        """리랭커 사용 가능 여부"""
        return self._load_model() is not None


_reranker_service: Optional[RerankerService] = None


def get_reranker_service() -> RerankerService:
    """RerankerService 싱글톤 인스턴스 반환"""
    global _reranker_service
    if _reranker_service is None:
        _reranker_service = RerankerService()
    return _reranker_service
