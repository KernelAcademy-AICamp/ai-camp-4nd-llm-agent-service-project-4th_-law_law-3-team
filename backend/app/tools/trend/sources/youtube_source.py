"""YouTube Data API v3 데이터 소스"""

import logging
from datetime import datetime

import httpx

from app.core.config import settings
from app.modules.content_marketing.schema import TrendSource
from app.tools.trend.models import RawTrendItem
from app.tools.trend.sources import BaseTrendSource, SourceConfig

logger = logging.getLogger(__name__)

YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"

# time_range → publishedAfter 매핑용 (시간 단위)
_TIME_RANGE_HOURS: dict[str, int] = {
    "48h": 48,
    "7d": 168,
    "30d": 720,
}


def _parse_youtube_date(date_str: str | None) -> datetime | None:
    """YouTube ISO 8601 날짜 파싱"""
    if not date_str:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


class YouTubeSource(BaseTrendSource):
    """YouTube Data API v3 데이터 소스"""

    @property
    def name(self) -> TrendSource:
        return TrendSource.YOUTUBE

    @property
    def is_available(self) -> bool:
        return bool(settings.YOUTUBE_API_KEY)

    async def fetch(
        self,
        query: str | None,
        config: SourceConfig,
    ) -> list[RawTrendItem]:
        search_query = config.search_query or query or "법률 뉴스"

        # publishedAfter 계산
        hours = _TIME_RANGE_HOURS.get(config.time_range, 24)
        from datetime import timedelta, timezone

        published_after = (
            datetime.now(tz=timezone.utc) - timedelta(hours=hours)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

        params: dict[str, str | int] = {
            "part": "snippet",
            "q": search_query,
            "type": "video",
            "order": "date",
            "relevanceLanguage": "ko",
            "regionCode": "KR",
            "maxResults": min(config.max_results, 25),
            "publishedAfter": published_after,
            "key": settings.YOUTUBE_API_KEY,
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(YOUTUBE_SEARCH_URL, params=params)
            response.raise_for_status()
            data = response.json()

        items: list[RawTrendItem] = []
        for item in data.get("items", []):
            snippet = item.get("snippet", {})
            video_id = item.get("id", {}).get("videoId", "")
            if not video_id:
                continue

            items.append(
                RawTrendItem(
                    title=snippet.get("title", ""),
                    url=f"https://www.youtube.com/watch?v={video_id}",
                    snippet=snippet.get("description", "")[:300],
                    source=TrendSource.YOUTUBE,
                    published_at=_parse_youtube_date(snippet.get("publishedAt")),
                    raw_data=item,
                )
            )

        logger.info("YouTube 수집 완료: %d건", len(items))
        return items
