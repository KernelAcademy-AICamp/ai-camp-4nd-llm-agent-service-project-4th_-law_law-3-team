"""
Retrieval 평가 메트릭

검색 품질을 평가하기 위한 지표들:
- Recall@K: 상위 K개 결과에서 정답 문서가 포함된 비율
- MRR (Mean Reciprocal Rank): 첫 번째 정답 문서의 역순위 평균
- NDCG@K: 정규화된 할인 누적 이득
- Hit Rate: 정답이 1개 이상 포함된 쿼리 비율
"""

import math
from typing import Optional


def calculate_recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int = 10,
) -> float:
    """
    Recall@K 계산

    상위 K개 검색 결과에서 정답 문서가 얼마나 포함되었는지 계산

    Args:
        retrieved_ids: 검색된 문서 ID 목록 (순위대로)
        relevant_ids: 정답 문서 ID 목록
        k: 상위 K개

    Returns:
        Recall@K 값 (0.0 ~ 1.0)

    Example:
        >>> calculate_recall_at_k(["A", "B", "C", "D", "E"], ["B", "D", "F"], k=5)
        0.6667  # 3개 중 2개 포함
    """
    if not relevant_ids:
        return 0.0

    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)

    intersection = retrieved_set & relevant_set
    return len(intersection) / len(relevant_set)


def calculate_mrr(
    retrieved_ids: list[str],
    relevant_ids: list[str],
) -> float:
    """
    MRR (Mean Reciprocal Rank) 계산

    첫 번째 정답 문서의 역순위 반환 (단일 쿼리용)
    전체 데이터셋 MRR은 개별 RR의 평균으로 계산

    Args:
        retrieved_ids: 검색된 문서 ID 목록 (순위대로)
        relevant_ids: 정답 문서 ID 목록

    Returns:
        Reciprocal Rank 값 (0.0 ~ 1.0)

    Example:
        >>> calculate_mrr(["A", "B", "C", "D", "E"], ["C", "D"])
        0.3333  # 첫 정답 C가 3번째 위치 = 1/3
    """
    if not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)

    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank

    return 0.0


def calculate_ndcg_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    relevance_scores: Optional[dict[str, float]] = None,
    k: int = 10,
) -> float:
    """
    NDCG@K (Normalized Discounted Cumulative Gain) 계산

    순위에 따른 할인을 적용한 정규화된 이득 계산

    Args:
        retrieved_ids: 검색된 문서 ID 목록 (순위대로)
        relevant_ids: 정답 문서 ID 목록
        relevance_scores: 문서별 관련성 점수 (없으면 binary로 처리)
        k: 상위 K개

    Returns:
        NDCG@K 값 (0.0 ~ 1.0)

    Example:
        >>> calculate_ndcg_at_k(["A", "B", "C"], ["B", "C"], k=3)
        0.8154  # B가 2번째, C가 3번째
    """
    if not relevant_ids:
        return 0.0

    # 관련성 점수가 없으면 binary로 처리
    if relevance_scores is None:
        relevance_scores = {doc_id: 1.0 for doc_id in relevant_ids}

    # DCG 계산
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        rel = relevance_scores.get(doc_id, 0.0)
        dcg += rel / math.log2(rank + 1)

    # IDCG 계산 (이상적인 순서)
    sorted_scores = sorted(
        [relevance_scores.get(doc_id, 0.0) for doc_id in relevant_ids],
        reverse=True,
    )
    idcg = 0.0
    for rank, rel in enumerate(sorted_scores[:k], start=1):
        idcg += rel / math.log2(rank + 1)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def calculate_hit_rate(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int = 10,
) -> float:
    """
    Hit Rate 계산 (단일 쿼리용)

    상위 K개에 정답이 1개 이상 있으면 1.0, 없으면 0.0
    전체 Hit Rate는 개별 값의 평균으로 계산

    Args:
        retrieved_ids: 검색된 문서 ID 목록 (순위대로)
        relevant_ids: 정답 문서 ID 목록
        k: 상위 K개

    Returns:
        Hit 여부 (0.0 또는 1.0)
    """
    if not relevant_ids:
        return 0.0

    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)

    return 1.0 if retrieved_set & relevant_set else 0.0


def calculate_all_retrieval_metrics(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    relevance_scores: Optional[dict[str, float]] = None,
) -> dict[str, float]:
    """
    모든 Retrieval 메트릭 계산

    Args:
        retrieved_ids: 검색된 문서 ID 목록 (순위대로)
        relevant_ids: 정답 문서 ID 목록
        relevance_scores: 문서별 관련성 점수 (optional)

    Returns:
        모든 메트릭이 담긴 딕셔너리
    """
    return {
        "recall_at_5": calculate_recall_at_k(retrieved_ids, relevant_ids, k=5),
        "recall_at_10": calculate_recall_at_k(retrieved_ids, relevant_ids, k=10),
        "mrr": calculate_mrr(retrieved_ids, relevant_ids),
        "hit_at_10": calculate_hit_rate(retrieved_ids, relevant_ids, k=10),
        "ndcg_at_10": calculate_ndcg_at_k(
            retrieved_ids, relevant_ids, relevance_scores, k=10
        ),
    }


def aggregate_metrics(
    all_results: list[dict[str, float]],
) -> dict[str, float]:
    """
    여러 쿼리의 메트릭을 집계

    Args:
        all_results: 개별 쿼리 메트릭 목록

    Returns:
        평균 메트릭
    """
    if not all_results:
        return {
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "mrr": 0.0,
            "hit_rate": 0.0,
            "ndcg_at_10": 0.0,
        }

    n = len(all_results)
    return {
        "recall_at_5": sum(r.get("recall_at_5", 0) for r in all_results) / n,
        "recall_at_10": sum(r.get("recall_at_10", 0) for r in all_results) / n,
        "mrr": sum(r.get("mrr", 0) for r in all_results) / n,
        "hit_rate": sum(r.get("hit_at_10", 0) for r in all_results) / n,
        "ndcg_at_10": sum(r.get("ndcg_at_10", 0) for r in all_results) / n,
    }
