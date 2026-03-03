"""Tavily Search API 데이터 소스"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from app.core.config import settings
from app.modules.content_marketing.schema import TrendSource
from app.tools.trend.models import RawTrendItem
from app.tools.trend.sources import BaseTrendSource, SourceConfig

if TYPE_CHECKING:
    from tavily import AsyncTavilyClient  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


def _parse_date(date_str: str | None) -> datetime | None:
    """Tavily 날짜 문자열 파싱"""
    if not date_str:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


class TavilySource(BaseTrendSource):
    """Tavily Search API 데이터 소스"""

    @property
    def name(self) -> TrendSource:
        return TrendSource.TAVILY

    @property
    def is_available(self) -> bool:
        return bool(settings.TAVILY_API_KEY)

    def __init__(self) -> None:
        from tavily import AsyncTavilyClient as _AsyncTavilyClient

        self._client_cls = _AsyncTavilyClient
        self._client: Any = None

    def _get_client(self) -> AsyncTavilyClient:
        if self._client is None:
            self._client = self._client_cls(api_key=settings.TAVILY_API_KEY)
        return self._client

    async def fetch(
        self,
        query: str | None,
        config: SourceConfig,
    ) -> list[RawTrendItem]:
        client = self._get_client()
        search_query = config.search_query or query or "최신 법률 뉴스"

        search_kwargs: dict[str, object] = {
            "query": search_query,
            "search_depth": "advanced",
            "max_results": config.max_results,
            "topic": "general",
        }

        # include_domains 지원 (커뮤니티 특화 수집)
        if config.include_domains:
            search_kwargs["include_domains"] = config.include_domains

        response = await client.search(**search_kwargs)

        items: list[RawTrendItem] = []
        for result in response.get("results", []):
            items.append(
                RawTrendItem(
                    title=result.get("title", ""),
                    url=result.get("url", ""),
                    snippet=result.get("content", "")[:300],
                    source=TrendSource.TAVILY,
                    published_at=_parse_date(result.get("published_date")),
                    raw_data=result,
                )
            )

        logger.info("Tavily 수집 완료: %d건", len(items))
        return items
