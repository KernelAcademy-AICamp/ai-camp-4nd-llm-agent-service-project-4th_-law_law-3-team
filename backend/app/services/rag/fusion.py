"""
Reciprocal Rank Fusion (RRF) 유틸리티

여러 검색 소스의 랭킹 결과를 단일 랭킹으로 결합.
벡터 검색과 키워드 검색 결과를 source_id 단위로 병합할 때 사용.
"""

from collections import defaultdict


def reciprocal_rank_fusion(
    *ranked_id_lists: list[str],
    k: int = 60,
) -> list[str]:
    """
    여러 랭킹 리스트를 RRF로 결합하여 단일 랭킹 반환.

    RRF 스코어 = Σ 1/(k + rank_i) for each list i

    Args:
        *ranked_id_lists: 순위대로 정렬된 ID 리스트들 (가변 인자)
        k: RRF 파라미터 (기본 60, 높을수록 상위 순위의 영향력 감소)

    Returns:
        RRF 스코어 내림차순으로 정렬된 ID 리스트.

    Example:
        >>> vector_ids = ["doc_A", "doc_B", "doc_C"]
        >>> keyword_ids = ["doc_B", "doc_D", "doc_A"]
        >>> fused = reciprocal_rank_fusion(vector_ids, keyword_ids)
        >>> # doc_B가 양쪽 모두 상위 → 최상위
    """
    scores: dict[str, float] = defaultdict(float)

    for ranked_list in ranked_id_lists:
        for rank, doc_id in enumerate(ranked_list):
            scores[doc_id] += 1.0 / (k + rank + 1)

    # 스코어 내림차순 정렬
    return sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)
