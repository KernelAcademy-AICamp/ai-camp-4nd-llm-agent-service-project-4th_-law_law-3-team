"""Weighted Reciprocal Rank Fusion (RRF)

소스별 신뢰도 가중치를 적용한 RRF 병합.
소스 권위도 × 1/(k + rank) 로 최종 점수를 산출하여
다양한 소스의 결과를 단일 순위로 통합한다.
"""

import logging
from collections import defaultdict

from app.modules.content_marketing.schema import NewsArticle, TrendSource

logger = logging.getLogger(__name__)

# RRF 상수 (rank 60은 RRF 표준값)
_RRF_K: int = 60

# 소스별 신뢰도 가중치 (설계 문서 §2.5)
SOURCE_AUTHORITY_WEIGHTS: dict[str, float] = {
    TrendSource.NAVER.value: 1.2,
    TrendSource.GOOGLE_TRENDS.value: 0.9,
    TrendSource.NEWSAPI.value: 0.8,
    TrendSource.YOUTUBE.value: 0.7,
    TrendSource.NEWSDATA.value: 0.6,
    TrendSource.TAVILY.value: 0.5,
}

_DEFAULT_WEIGHT: float = 0.5


def _get_source_weight(source_name: str) -> float:
    """소스명으로 가중치 반환 (미등록 소스는 기본값)"""
    return SOURCE_AUTHORITY_WEIGHTS.get(source_name, _DEFAULT_WEIGHT)


def weighted_rrf_merge(
    source_results: dict[str, list[NewsArticle]],
) -> list[NewsArticle]:
    """소스별 랭킹 리스트를 WeightedRRF로 병합

    Args:
        source_results: {소스명: [NewsArticle, ...]} — 각 소스의 순위별 기사 리스트

    Returns:
        RRF 점수 내림차순으로 정렬된 NewsArticle 리스트 (중복 URL 제거)
    """
    if not source_results:
        return []

    # URL → (최고 RRF 점수, 최상위 기사 객체)
    url_scores: dict[str, float] = defaultdict(float)
    url_articles: dict[str, NewsArticle] = {}

    for source_name, articles in source_results.items():
        weight = _get_source_weight(source_name)

        for rank_idx, article in enumerate(articles):
            rrf_score = weight * (1.0 / (_RRF_K + rank_idx + 1))
            url_scores[article.url] += rrf_score

            # 같은 URL이면 더 높은 RRF 기여 소스의 기사를 채택
            if article.url not in url_articles:
                url_articles[article.url] = article

    # RRF 점수 내림차순 정렬
    sorted_urls = sorted(url_scores.keys(), key=lambda u: url_scores[u], reverse=True)

    merged = [url_articles[url] for url in sorted_urls]

    logger.debug(
        "WeightedRRF 병합: %d 소스 → %d 기사 (중복 제거 후)",
        len(source_results),
        len(merged),
    )

    return merged
