"""
ONNX 품질 게이트

서버 시작 시 PyTorch vs ONNX 임베딩/리랭킹 품질을 비교하여
품질 기준(cosine >= 0.995) 미달 시 자동으로 PyTorch로 폴백한다.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)

# 품질 기준
_EMBEDDING_COSINE_THRESHOLD = 0.995
_RERANKER_CORRELATION_THRESHOLD = 0.990

# 법률 도메인 대표 쿼리 (품질 검증용)
_GATE_QUERIES = [
    "교통사고 손해배상 판례",
    "임대차 보증금 반환 청구",
    "근로기준법 해고 부당해고",
    "이혼 재산분할 위자료",
    "민법 제750조 불법행위",
    "형사소송법 증거능력 배제",
    "행정소송 처분 취소 요건",
    "상속 포기 절차와 기한",
]

_GATE_RERANK_QUERY = "교통사고 손해배상 판례"
_GATE_RERANK_DOCS = [
    "피고는 원고에게 금 50,000,000원 및 이에 대한 지연손해금을 지급하라.",
    "자동차손해배상 보장법 제3조에 의하면 운행자는 배상 책임을 진다.",
    "임대차보증금 반환 청구 사건에서 임대인은 보증금 전액을 반환할 의무가 있다.",
    "근로기준법 제23조 제1항은 정당한 이유 없이 해고를 하지 못한다.",
    "형법 제307조 제1항의 명예훼손죄가 성립하려면 사실을 적시하여야 한다.",
]


@dataclass
class QualityGateResult:
    """품질 게이트 결과."""

    passed: bool
    metric_name: str  # cosine | pearson
    metric_value: float
    threshold: float
    elapsed_ms: float
    detail: str = ""


def _cosine_similarity_vectors(a: list[float], b: list[float]) -> float:
    """두 벡터의 cosine similarity."""
    arr_a = np.array(a)
    arr_b = np.array(b)
    dot = np.dot(arr_a, arr_b)
    norm_a = np.linalg.norm(arr_a)
    norm_b = np.linalg.norm(arr_b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(dot / (norm_a * norm_b))


def _pearson_correlation(a: list[float], b: list[float]) -> float:
    """Pearson 상관계수."""
    arr_a = np.array(a)
    arr_b = np.array(b)
    if len(arr_a) < 2:
        return 0.0
    return float(np.corrcoef(arr_a, arr_b)[0, 1])


def check_embedding_quality() -> QualityGateResult:
    """임베딩 ONNX vs PyTorch 품질을 비교한다.

    8개 법률 쿼리에 대해 양쪽 임베딩을 생성하고
    평균 cosine similarity를 계산한다.

    Returns:
        QualityGateResult (passed=True이면 ONNX 사용 가능)
    """
    from app.services.rag.embedding import get_local_model
    from app.services.rag.onnx_session import encode_embedding_onnx

    t0 = time.perf_counter()

    cosines: list[float] = []
    for query in _GATE_QUERIES:
        # PyTorch 임베딩
        model = get_local_model()
        pt_emb = model.encode(
            query,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).tolist()

        # ONNX 임베딩
        onnx_emb = encode_embedding_onnx(query)

        cosine = _cosine_similarity_vectors(pt_emb, onnx_emb)
        cosines.append(cosine)

    avg_cosine = float(np.mean(cosines))
    min_cosine = float(np.min(cosines))
    elapsed_ms = (time.perf_counter() - t0) * 1000

    passed = avg_cosine >= _EMBEDDING_COSINE_THRESHOLD

    detail = (
        f"avg_cosine={avg_cosine:.6f}, min_cosine={min_cosine:.6f}, "
        f"queries={len(_GATE_QUERIES)}"
    )

    if passed:
        logger.info(
            "임베딩 품질 게이트 통과: cosine=%.6f (>= %.4f) [%.0fms]",
            avg_cosine,
            _EMBEDDING_COSINE_THRESHOLD,
            elapsed_ms,
        )
    else:
        logger.warning(
            "임베딩 품질 게이트 실패: cosine=%.6f (< %.4f) [%.0fms]",
            avg_cosine,
            _EMBEDDING_COSINE_THRESHOLD,
            elapsed_ms,
        )

    return QualityGateResult(
        passed=passed,
        metric_name="cosine",
        metric_value=avg_cosine,
        threshold=_EMBEDDING_COSINE_THRESHOLD,
        elapsed_ms=elapsed_ms,
        detail=detail,
    )


def check_reranker_quality() -> QualityGateResult:
    """리랭커 ONNX vs PyTorch 품질을 비교한다.

    동일 쿼리-문서 쌍에 대해 양쪽 점수를 계산하고
    Pearson 상관계수를 비교한다.

    Returns:
        QualityGateResult (passed=True이면 ONNX 사용 가능)
    """
    from app.services.rag.onnx_session import predict_reranker_onnx
    from app.services.rag.rerank import _load_reranker_model

    t0 = time.perf_counter()

    # PyTorch 리랭킹
    model = _load_reranker_model()
    if model is None:
        return QualityGateResult(
            passed=False,
            metric_name="pearson",
            metric_value=0.0,
            threshold=_RERANKER_CORRELATION_THRESHOLD,
            elapsed_ms=0.0,
            detail="PyTorch 리랭커 모델 로드 실패",
        )

    pairs = [(_GATE_RERANK_QUERY, doc) for doc in _GATE_RERANK_DOCS]
    pt_scores = [float(s) for s in model.predict(pairs)]

    # ONNX 리랭킹
    onnx_scores = predict_reranker_onnx(_GATE_RERANK_QUERY, _GATE_RERANK_DOCS)

    correlation = _pearson_correlation(pt_scores, onnx_scores)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    passed = correlation >= _RERANKER_CORRELATION_THRESHOLD

    detail = (
        f"pearson={correlation:.6f}, "
        f"pt_scores={[f'{s:.4f}' for s in pt_scores]}, "
        f"onnx_scores={[f'{s:.4f}' for s in onnx_scores]}"
    )

    if passed:
        logger.info(
            "리랭커 품질 게이트 통과: pearson=%.6f (>= %.4f) [%.0fms]",
            correlation,
            _RERANKER_CORRELATION_THRESHOLD,
            elapsed_ms,
        )
    else:
        logger.warning(
            "리랭커 품질 게이트 실패: pearson=%.6f (< %.4f) [%.0fms]",
            correlation,
            _RERANKER_CORRELATION_THRESHOLD,
            elapsed_ms,
        )

    return QualityGateResult(
        passed=passed,
        metric_name="pearson",
        metric_value=correlation,
        threshold=_RERANKER_CORRELATION_THRESHOLD,
        elapsed_ms=elapsed_ms,
        detail=detail,
    )


def run_quality_gate() -> dict[str, QualityGateResult]:
    """임베딩 + 리랭커 품질 게이트를 실행한다.

    품질 미달 시 settings.ONNX_QUALITY_GATE_FALLBACK이 True이면
    해당 ONNX 세션을 비활성화하고 PyTorch로 폴백.

    Returns:
        {"embedding": result, "reranker": result}
    """
    from app.services.rag.onnx_session import (
        is_embedding_onnx_loaded,
        is_reranker_onnx_loaded,
    )

    results: dict[str, QualityGateResult] = {}

    if not settings.ONNX_QUALITY_GATE_ENABLED:
        logger.info("ONNX 품질 게이트 비활성화 (ONNX_QUALITY_GATE_ENABLED=false)")
        return results

    # 임베딩 품질 검증
    if is_embedding_onnx_loaded():
        emb_result = check_embedding_quality()
        results["embedding"] = emb_result

        if not emb_result.passed and settings.ONNX_QUALITY_GATE_FALLBACK:
            logger.warning(
                "임베딩 ONNX 품질 미달 → PyTorch 폴백 (USE_ONNX_EMBEDDING=false)"
            )
            settings.USE_ONNX_EMBEDDING = False

    # 리랭커 품질 검증
    if is_reranker_onnx_loaded():
        rr_result = check_reranker_quality()
        results["reranker"] = rr_result

        if not rr_result.passed and settings.ONNX_QUALITY_GATE_FALLBACK:
            logger.warning(
                "리랭커 ONNX 품질 미달 → PyTorch 폴백 (USE_ONNX_RERANKER=false)"
            )
            settings.USE_ONNX_RERANKER = False

    return results
