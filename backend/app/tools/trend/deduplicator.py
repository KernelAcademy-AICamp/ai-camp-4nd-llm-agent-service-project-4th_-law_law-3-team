"""의미 기반 중복 제거 (Semantic Deduplication)

2단계 병합 전략:
  Stage 1 (자동 병합): cosine similarity >= 0.90 → 무조건 병합
  Stage 2 (검증 병합): cosine similarity >= 0.82 → 소스가 다르면 병합

최소 15건 미만이면 URL 기반 단순 중복 제거만 수행 (임베딩 비용 절약).
KURE-v1 임베딩 모델은 기존 RAG 서비스의 get_local_model()을 lazy load로 재사용.
"""

import logging
from typing import TYPE_CHECKING, cast

import numpy as np

from app.tools.trend.models import RawTrendItem

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# 병합 임계치
AUTO_MERGE_THRESHOLD: float = 0.90
VERIFY_MERGE_THRESHOLD: float = 0.82
MIN_ITEMS_FOR_DEDUP: int = 15


def _get_embeddings(texts: list[str]) -> "NDArray[np.float32]":
    """텍스트 목록을 임베딩 벡터로 변환 (KURE-v1 lazy load)

    Raises:
        EmbeddingModelNotFoundError: 모델 미캐시 시
    """
    from app.services.rag.embedding import get_local_model

    model = get_local_model()
    embeddings: NDArray[np.float32] = model.encode(
        texts,
        show_progress_bar=False,
        normalize_embeddings=True,
        batch_size=32,
    )
    return embeddings


def _cosine_similarity_matrix(embeddings: "NDArray[np.float32]") -> "NDArray[np.float32]":
    """정규화된 임베딩의 코사인 유사도 행렬 계산"""
    # 이미 normalize_embeddings=True이므로 dot product = cosine similarity
    return cast("NDArray[np.float32]", np.dot(embeddings, embeddings.T))


class SemanticDeduplicator:
    """2단계 의미 기반 중복 제거기

    - 15건 미만: URL 기반 단순 중복 제거만 수행
    - 15건 이상: Stage 1 (자동 병합 0.90) + Stage 2 (검증 병합 0.82)
    """

    def deduplicate(self, items: list[RawTrendItem]) -> list[RawTrendItem]:
        """의미 기반 중복 제거

        Args:
            items: 중복 가능성이 있는 RawTrendItem 리스트

        Returns:
            중복 제거된 RawTrendItem 리스트 (원본 순서 유지)
        """
        if len(items) < 2:
            return items

        # URL 기반 1차 중복 제거
        url_deduped = self._url_deduplicate(items)

        if len(url_deduped) < MIN_ITEMS_FOR_DEDUP:
            logger.debug(
                "SemanticDedup 건너뜀: %d건 < %d (최소 기준)",
                len(url_deduped), MIN_ITEMS_FOR_DEDUP,
            )
            return url_deduped

        # 임베딩 기반 2단계 중복 제거
        try:
            return self._semantic_deduplicate(url_deduped)
        except Exception:
            logger.warning("SemanticDedup 실패, URL 기반 결과 반환", exc_info=True)
            return url_deduped

    def _url_deduplicate(self, items: list[RawTrendItem]) -> list[RawTrendItem]:
        """URL 기반 단순 중복 제거"""
        seen: set[str] = set()
        unique: list[RawTrendItem] = []
        for item in items:
            key = item.url.rstrip("/").lower()
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return unique

    def _semantic_deduplicate(self, items: list[RawTrendItem]) -> list[RawTrendItem]:
        """임베딩 기반 의미 중복 제거"""
        # 제목+snippet 결합 텍스트로 임베딩
        texts = [
            f"{item.title} {item.snippet[:100]}" for item in items
        ]

        embeddings = _get_embeddings(texts)
        sim_matrix = _cosine_similarity_matrix(embeddings)

        n = len(items)
        removed: set[int] = set()

        # Stage 1: 자동 병합 (similarity >= AUTO_MERGE_THRESHOLD)
        for i in range(n):
            if i in removed:
                continue
            for j in range(i + 1, n):
                if j in removed:
                    continue
                if sim_matrix[i, j] >= AUTO_MERGE_THRESHOLD:
                    # j를 제거하고 i에 소스 병합 정보 기록
                    removed.add(j)
                    if items[j].source not in items[i].merged_sources:
                        items[i].merged_sources.append(items[j].source)

        # Stage 2: 검증 병합 (similarity >= VERIFY_MERGE_THRESHOLD, 다른 소스)
        for i in range(n):
            if i in removed:
                continue
            for j in range(i + 1, n):
                if j in removed:
                    continue
                if sim_matrix[i, j] >= VERIFY_MERGE_THRESHOLD:
                    # 소스가 다르면 병합
                    if items[i].source != items[j].source:
                        removed.add(j)
                        if items[j].source not in items[i].merged_sources:
                            items[i].merged_sources.append(items[j].source)

        result = [item for idx, item in enumerate(items) if idx not in removed]

        merged_count = len(items) - len(result)
        if merged_count > 0:
            logger.info(
                "SemanticDedup: %d → %d건 (병합 %d건)",
                len(items), len(result), merged_count,
            )

        return result
