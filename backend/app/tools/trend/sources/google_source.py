"""Google Custom Search JSON API 데이터 소스"""

import logging
from datetime import datetime, timezone

import httpx

from app.core.config import settings
from app.modules.content_marketing.schema import TrendSource
from app.tools.trend.models import RawTrendItem
from app.tools.trend.sources import BaseTrendSource, SourceConfig

logger = logging.getLogger(__name__)

GOOGLE_CSE_URL = "https://www.googleapis.com/customsearch/v1"

# time_range → dateRestrict 매핑
_DATE_RESTRICT_MAP: dict[str, str] = {
    "48h": "d2",
    "7d": "d7",
    "30d": "d30",
}


class GoogleSource(BaseTrendSource):
    """Google Custom Search JSON API 데이터 소스"""

    @property
    def name(self) -> TrendSource:
        return TrendSource.GOOGLE_TRENDS

    @property
    def is_available(self) -> bool:
        return bool(settings.GOOGLE_CSE_API_KEY and settings.GOOGLE_CSE_ID)

    async def fetch(
        self,
        query: str | None,
        config: SourceConfig,
    ) -> list[RawTrendItem]:
        search_query = config.search_query or query or "최신 뉴스"
        date_restrict = _DATE_RESTRICT_MAP.get(config.time_range, "d2")

        params: dict[str, str | int] = {
            "key": settings.GOOGLE_CSE_API_KEY,
            "cx": settings.GOOGLE_CSE_ID,
            "q": search_query,
            "lr": "lang_ko",
            "dateRestrict": date_restrict,
            "num": min(config.max_results, 10),
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(GOOGLE_CSE_URL, params=params)
            response.raise_for_status()
            data = response.json()

        items: list[RawTrendItem] = []
        now = datetime.now(tz=timezone.utc)

        for item in data.get("items", []):
            title = item.get("title", "")
            link = item.get("link", "")
            snippet = item.get("snippet", "")

            if not title or not link:
                continue

            items.append(
                RawTrendItem(
                    title=title,
                    url=link,
                    snippet=snippet[:300],
                    source=TrendSource.GOOGLE_TRENDS,
                    published_at=now,
                    raw_data=item,
                )
            )

        logger.info("Google CSE 수집 완료: %d건", len(items))
        return items
