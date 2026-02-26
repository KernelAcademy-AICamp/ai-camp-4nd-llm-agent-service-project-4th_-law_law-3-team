"""Naver Search API 데이터 소스"""

import logging
import re
from datetime import datetime

import httpx

from app.core.config import settings
from app.modules.content_marketing.schema import TrendSource
from app.tools.trend.models import RawTrendItem
from app.tools.trend.sources import BaseTrendSource, SourceConfig

logger = logging.getLogger(__name__)

NAVER_SEARCH_URL = "https://openapi.naver.com/v1/search/news.json"

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    """HTML 태그 제거"""
    return _HTML_TAG_RE.sub("", text).strip()


def _parse_naver_date(date_str: str | None) -> datetime | None:
    """Naver pubDate 파싱 (RFC 2822 형식)"""
    if not date_str:
        return None
    try:
        from email.utils import parsedate_to_datetime

        return parsedate_to_datetime(date_str)
    except (TypeError, ValueError):
        return None


class NaverSource(BaseTrendSource):
    """Naver Search API 데이터 소스"""

    @property
    def name(self) -> TrendSource:
        return TrendSource.NAVER

    @property
    def is_available(self) -> bool:
        return bool(settings.NAVER_CLIENT_ID and settings.NAVER_CLIENT_SECRET)

    async def fetch(
        self,
        query: str | None,
        config: SourceConfig,
    ) -> list[RawTrendItem]:
        search_query = config.search_query or query or "법률 뉴스"

        headers = {
            "X-Naver-Client-Id": settings.NAVER_CLIENT_ID,
            "X-Naver-Client-Secret": settings.NAVER_CLIENT_SECRET,
        }
        params: dict[str, str | int] = {
            "query": search_query,
            "display": config.max_results,
            "sort": "date",
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                NAVER_SEARCH_URL,
                headers=headers,
                params=params,
            )
            response.raise_for_status()
            data = response.json()

        items: list[RawTrendItem] = []
        for item in data.get("items", []):
            items.append(
                RawTrendItem(
                    title=_strip_html(item.get("title", "")),
                    url=item.get("originallink") or item.get("link", ""),
                    snippet=_strip_html(item.get("description", ""))[:300],
                    source=TrendSource.NAVER,
                    published_at=_parse_naver_date(item.get("pubDate")),
                    raw_data=item,
                )
            )

        logger.info("Naver 수집 완료: %d건", len(items))
        return items
