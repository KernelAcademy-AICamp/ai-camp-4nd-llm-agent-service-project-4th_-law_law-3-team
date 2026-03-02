"""네이버 뉴스 검색 API 기반 수집"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

import httpx

from app.core.config import settings
from app.tools.news_pipeline.config import NewsPipelineConfig
from app.tools.news_pipeline.models import NewsSourceType, RawArticle
from app.tools.news_pipeline.sources import BaseNewsSource

logger = logging.getLogger(__name__)

NAVER_SEARCH_URL = "https://openapi.naver.com/v1/search/news.json"
_HTML_TAG_RE = re.compile(r"<[^>]+>")


class NaverNewsSource(BaseNewsSource):
    """네이버 뉴스 검색 API 데이터 소스

    키워드 목록 순회하며 검색, 날짜 필터링, 원문 URL 추출.
    """

    def __init__(self, config: NewsPipelineConfig) -> None:
        self._config = config

    @property
    def source_type(self) -> NewsSourceType:
        return NewsSourceType.NAVER

    @property
    def is_available(self) -> bool:
        return bool(
            self._config.naver_enabled
            and settings.NAVER_CLIENT_ID
            and settings.NAVER_CLIENT_SECRET
        )

    async def fetch(self, target_date: date) -> list[RawArticle]:
        """네이버 뉴스 키워드 검색으로 수집"""
        if not self.is_available:
            return []

        all_articles: list[RawArticle] = []
        seen_urls: set[str] = set()

        headers = {
            "X-Naver-Client-Id": settings.NAVER_CLIENT_ID,
            "X-Naver-Client-Secret": settings.NAVER_CLIENT_SECRET,
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            for keyword in self._config.naver_keywords:
                try:
                    articles = await self._search_keyword(
                        client, headers, keyword, target_date, seen_urls,
                    )
                    all_articles.extend(articles)
                except Exception as exc:
                    logger.warning("네이버 키워드 [%s] 검색 실패: %s", keyword, exc)
                    continue

                if len(all_articles) >= self._config.max_articles_per_run:
                    break

        logger.info("네이버 뉴스 수집 완료: %d건 (대상: %s)", len(all_articles), target_date)
        return all_articles

    async def _search_keyword(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        keyword: str,
        target_date: date,
        seen_urls: set[str],
    ) -> list[RawArticle]:
        """단일 키워드 검색"""
        params: dict[str, str | int] = {
            "query": keyword,
            "display": 100,
            "sort": "date",
        }

        resp = await client.get(NAVER_SEARCH_URL, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()

        articles: list[RawArticle] = []
        for item in data.get("items", []):
            url = item.get("originallink") or item.get("link", "")
            if not url or url in seen_urls:
                continue

            published_at = self._parse_date(item.get("pubDate"))
            if published_at and published_at.date() != target_date:
                continue

            seen_urls.add(url)
            articles.append(RawArticle(
                url=url,
                title=self._strip_html(item.get("title", "")),
                raw_html=item.get("description", ""),
                source=NewsSourceType.NAVER,
                publisher=self._extract_publisher(url),
                published_at=published_at,
                section=None,
                tags=[keyword],
                raw_metadata={"naver_link": item.get("link", "")},
            ))

        return articles

    @staticmethod
    def _strip_html(text: str) -> str:
        return _HTML_TAG_RE.sub("", text).strip()

    @staticmethod
    def _parse_date(date_str: str | None) -> datetime | None:
        if not date_str:
            return None
        try:
            return parsedate_to_datetime(date_str)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_publisher(url: str) -> str:
        """URL에서 매체명 추출 (도메인 기반)"""
        domain = urlparse(url).netloc
        publisher_map: dict[str, str] = {
            "www.lawtimes.co.kr": "법률신문",
            "www.lec.co.kr": "법률저널",
            "www.legaltimes.co.kr": "리걸타임즈",
            "news.law.go.kr": "법제처",
        }
        return publisher_map.get(domain, domain)
