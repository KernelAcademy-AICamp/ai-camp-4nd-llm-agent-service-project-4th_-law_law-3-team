"""NewsData.io Latest News API 데이터 소스"""

import logging
from datetime import datetime, timezone

import httpx

from app.core.config import settings
from app.modules.content_marketing.schema import TrendSource
from app.tools.trend.models import RawTrendItem
from app.tools.trend.sources import BaseTrendSource, SourceConfig

logger = logging.getLogger(__name__)

NEWSDATA_API_URL = "https://newsdata.io/api/1/latest"

# time_range → timeframe 매핑 (시간 단위, latest 엔드포인트 최대 48시간)
_TIMEFRAME_MAP: dict[str, int] = {
    "48h": 48,
    "7d": 48,
    "30d": 48,
}


def _parse_newsdata_date(date_str: str | None) -> datetime | None:
    """NewsData.io 날짜 파싱 (ISO 8601 형식)"""
    if not date_str:
        return None
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%fZ",
    ):
        try:
            return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


class NewsDataSource(BaseTrendSource):
    """NewsData.io Latest News API 데이터 소스"""

    @property
    def name(self) -> TrendSource:
        return TrendSource.NEWSDATA

    @property
    def is_available(self) -> bool:
        return bool(settings.NEWSDATA_API_KEY)

    async def fetch(
        self,
        query: str | None,
        config: SourceConfig,
    ) -> list[RawTrendItem]:
        search_query = config.search_query or query or "최신 뉴스"
        timeframe = _TIMEFRAME_MAP.get(config.time_range, 48)

        params: dict[str, str | int] = {
            "apikey": settings.NEWSDATA_API_KEY,
            "q": search_query,
            "language": "ko",
            "country": "kr",
            "timeframe": timeframe,
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(NEWSDATA_API_URL, params=params)
            response.raise_for_status()
            data = response.json()

        items: list[RawTrendItem] = []
        now = datetime.now(tz=timezone.utc)

        for article in data.get("results", []):
            title = article.get("title", "")
            link = article.get("link", "")
            description = article.get("description") or ""

            if not title or not link:
                continue

            items.append(
                RawTrendItem(
                    title=title,
                    url=link,
                    snippet=description[:300],
                    source=TrendSource.NEWSDATA,
                    published_at=_parse_newsdata_date(article.get("pubDate")) or now,
                    raw_data=article,
                )
            )

        logger.info("NewsData.io 수집 완료: %d건", len(items))
        return items
