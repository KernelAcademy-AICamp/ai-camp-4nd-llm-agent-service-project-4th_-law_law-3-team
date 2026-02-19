"""
리랭킹 서비스

Cross-encoder 기반 문서 리랭킹.
원문 기반 적응형 truncation + 배치 처리.
"""

from __future__ import annotations

import asyncio
import logging
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

# 기본 리랭커 모델명 (한국어 특화, BGE v2-m3 기반)
DEFAULT_RERANKER_MODEL = "dragonkue/bge-reranker-v2-m3-ko"

# bge-reranker-v2-m3-ko 최대 8192 토큰 ≈ 한글 4000자
_MAX_RERANK_CHARS = 4_000
_HEAD_CHARS = 3_000
_TAIL_CHARS = 1_000

# Cross-encoder 배치 크기
_RERANK_BATCH_SIZE = 32

# 최소 리랭킹 점수 (sigmoid 출력, 이하 필터링)
_MIN_RERANK_SCORE = 0.01


@lru_cache(maxsize=1)
def _load_reranker_model(model_name: str = DEFAULT_RERANKER_MODEL) -> Any:
    """리랭커 모델 로드 (캐싱)."""
    try:
        import torch
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(
            model_name,
            activation_fn=torch.nn.Sigmoid(),
        )
        logger.info("리랭커 모델 로드 완료: %s", model_name)
        return model
    except ImportError:
        logger.warning("sentence-transformers 미설치 → 리랭킹 비활성화")
        return None
    except Exception as e:
        logger.warning("리랭커 모델 로드 실패: %s", e)
        return None


def is_reranker_available(model_name: str = DEFAULT_RERANKER_MODEL) -> bool:
    """리랭커 사용 가능 여부 확인."""
    return _load_reranker_model(model_name) is not None


def _adaptive_truncate(content: str) -> str:
    """적응형 텍스트 truncation.

    ≤ _MAX_RERANK_CHARS 이면 전문 사용.
    초과 시 head(_HEAD_CHARS) + tail(_TAIL_CHARS) 결합.
    """
    if len(content) <= _MAX_RERANK_CHARS:
        return content
    return content[:_HEAD_CHARS] + "\n...\n" + content[-_TAIL_CHARS:]


def rerank_documents(
    query: str,
    documents: list[dict[str, Any]],
    top_k: int = 5,
    model_name: str = DEFAULT_RERANKER_MODEL,
    min_score: float = _MIN_RERANK_SCORE,
    batch_size: int = _RERANK_BATCH_SIZE,
) -> list[dict[str, Any]]:
    """
    문서 리랭킹.

    Cross-encoder로 쿼리-문서 관련성 점수를 계산하고 재정렬.
    원문 기반 적응형 truncation 적용 (head 3000 + tail 1000자).

    Args:
        query: 검색 쿼리
        documents: 검색된 문서 목록 (content 필드 사용)
        top_k: 반환할 최상위 결과 수
        model_name: 리랭커 모델명
        min_score: 최소 점수 (이하 필터링, sigmoid 출력 0~1)
        batch_size: Cross-encoder 배치 크기

    Returns:
        리랭킹된 문서 목록 (rerank_score 필드 추가)
    """
    if not documents:
        return []

    model = _load_reranker_model(model_name)
    if model is None:
        return documents[:top_k]

    try:
        # 쿼리-문서 쌍 생성 (적응형 truncation)
        pairs: list[tuple[str, str]] = [
            (query, _adaptive_truncate(doc.get("content", "")))
            for doc in documents
        ]

        # 배치 처리
        all_scores: list[float] = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            scores = model.predict(batch)
            all_scores.extend(float(s) for s in scores)

        # 점수 내림차순 정렬
        scored_docs = sorted(
            zip(documents, all_scores),
            key=lambda x: x[1],
            reverse=True,
        )

        # 최소 점수 필터링 + top_k
        reranked: list[dict[str, Any]] = []
        for doc, score in scored_docs:
            if score < min_score:
                continue
            doc_copy = doc.copy()
            doc_copy["rerank_score"] = score
            reranked.append(doc_copy)
            if len(reranked) >= top_k:
                break

        # min_score 필터로 전부 제거된 경우 원본 반환
        if not reranked:
            return documents[:top_k]

        return reranked

    except Exception as e:
        logger.warning("리랭킹 실패: %s", e)
        return documents[:top_k]


async def rerank_documents_async(
    query: str,
    documents: list[dict[str, Any]],
    top_k: int = 5,
    model_name: str = DEFAULT_RERANKER_MODEL,
    min_score: float = _MIN_RERANK_SCORE,
    batch_size: int = _RERANK_BATCH_SIZE,
) -> list[dict[str, Any]]:
    """문서 리랭킹 (비동기 래퍼)."""
    return await asyncio.to_thread(
        rerank_documents, query, documents, top_k, model_name, min_score, batch_size
    )
