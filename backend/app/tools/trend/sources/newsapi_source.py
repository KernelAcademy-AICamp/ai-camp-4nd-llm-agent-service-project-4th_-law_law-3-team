"""NewsAPI.org Everything API 데이터 소스"""

import logging
from datetime import datetime, timedelta, timezone

import httpx

from app.core.config import settings
from app.modules.content_marketing.schema import TrendSource
from app.tools.trend.models import RawTrendItem
from app.tools.trend.sources import BaseTrendSource, SourceConfig

logger = logging.getLogger(__name__)

NEWSAPI_EVERYTHING_URL = "https://newsapi.org/v2/everything"

# time_range → from 날짜 오프셋 (시간 단위)
_TIME_RANGE_HOURS: dict[str, int] = {
    "48h": 48,
    "7d": 168,
    "30d": 720,
}


def _parse_newsapi_date(date_str: str | None) -> datetime | None:
    """NewsAPI.org 날짜 파싱 (ISO 8601)"""
    if not date_str:
        return None
    for fmt in (
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%S%z",
    ):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


class NewsAPISource(BaseTrendSource):
    """NewsAPI.org Everything API 데이터 소스"""

    @property
    def name(self) -> TrendSource:
        return TrendSource.NEWSAPI

    @property
    def is_available(self) -> bool:
        return bool(settings.NEWSAPI_API_KEY)

    async def fetch(
        self,
        query: str | None,
        config: SourceConfig,
    ) -> list[RawTrendItem]:
        search_query = config.search_query or query or "최신 뉴스"
        hours = _TIME_RANGE_HOURS.get(config.time_range, 48)
        from_date = (
            datetime.now(tz=timezone.utc) - timedelta(hours=hours)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

        params: dict[str, str | int] = {
            "apiKey": settings.NEWSAPI_API_KEY,
            "q": search_query,
            "language": "ko",
            "from": from_date,
            "sortBy": "publishedAt",
            "pageSize": min(config.max_results, 100),
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(NEWSAPI_EVERYTHING_URL, params=params)
            response.raise_for_status()
            data = response.json()

        if data.get("status") != "ok":
            logger.warning("NewsAPI 응답 오류: %s", data.get("message", ""))
            return []

        items: list[RawTrendItem] = []
        now = datetime.now(tz=timezone.utc)

        for article in data.get("articles", []):
            title = article.get("title") or ""
            url = article.get("url") or ""
            description = article.get("description") or ""

            if not title or not url or title == "[Removed]":
                continue

            items.append(
                RawTrendItem(
                    title=title,
                    url=url,
                    snippet=description[:300],
                    source=TrendSource.NEWSAPI,
                    published_at=_parse_newsapi_date(article.get("publishedAt")) or now,
                    raw_data=article,
                )
            )

        logger.info("NewsAPI.org 수집 완료: %d건", len(items))
        return items
