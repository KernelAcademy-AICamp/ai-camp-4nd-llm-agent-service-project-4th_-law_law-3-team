"""4단계 중복 제거 (v0.3.0: Stage 4 SimHash Fuzzy Dedup 추가)"""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.news_article import NewsArticle
from app.tools.news_pipeline.models import RawArticle

logger = logging.getLogger(__name__)


class Deduplicator:
    """4단계 중복 제거

    Stage 1: URL 완전 일치 (DB 조회)
    Stage 2: 제목 + 발행일 조합 (DB 조회)
    Stage 3: 본문 해시 비교 (정제 후, content_hash)
    Stage 4: SimHash Fuzzy Dedup (v0.3.0, 유사도 95%+ 기사 탐지)
    """

    def __init__(self, simhash_threshold: int = 3) -> None:
        self._simhash_threshold = simhash_threshold

    async def deduplicate(
        self,
        articles: list[RawArticle],
        db: AsyncSession,
    ) -> list[RawArticle]:
        """중복 제거된 기사 목록 반환"""
        if not articles:
            return []

        # Stage 1: URL 기반 (DB에 이미 있는 URL 제외)
        urls = [a.url for a in articles]
        existing_urls = await self._get_existing_urls(db, urls)
        after_url = [a for a in articles if a.url not in existing_urls]
        logger.info("중복 제거 Stage 1 (URL): %d → %d", len(articles), len(after_url))

        # Stage 2: 제목+발행일 기반 (같은 배치 내 중복)
        after_title = self._deduplicate_by_title_date(after_url)
        logger.info("중복 제거 Stage 2 (제목+일자): %d → %d", len(after_url), len(after_title))

        # Stage 4: SimHash Fuzzy Dedup (v0.3.0)
        after_fuzzy = self._deduplicate_by_simhash(after_title)
        logger.info("중복 제거 Stage 4 (SimHash): %d → %d", len(after_title), len(after_fuzzy))

        return after_fuzzy

    async def check_content_hash(
        self, content_hash: str, db: AsyncSession,
    ) -> bool:
        """Stage 3: 본문 해시 중복 여부 확인 (정제 후 호출)"""
        result = await db.execute(
            select(NewsArticle.id).where(
                NewsArticle.content_hash == content_hash,
            ).limit(1),
        )
        return result.scalar_one_or_none() is not None

    def _deduplicate_by_simhash(
        self, articles: list[RawArticle],
    ) -> list[RawArticle]:
        """v0.3.0 Stage 4: SimHash 기반 유사 기사 제거 (배치 내)"""
        from app.tools.news_pipeline.fuzzy_dedup import (
            compute_simhash,
            hamming_distance,
        )

        unique: list[RawArticle] = []
        seen_hashes: list[int] = []

        for article in articles:
            text = f"{article.title} {article.raw_html[:2000]}"
            article_hash = compute_simhash(text)

            is_duplicate = False
            for existing_hash in seen_hashes:
                if hamming_distance(article_hash, existing_hash) <= self._simhash_threshold:
                    is_duplicate = True
                    logger.debug("SimHash 유사 기사 탐지: %s", article.url)
                    break

            if not is_duplicate:
                unique.append(article)
                seen_hashes.append(article_hash)

        return unique

    async def _get_existing_urls(
        self, db: AsyncSession, urls: list[str],
    ) -> set[str]:
        """DB에 이미 존재하는 URL 집합 반환"""
        if not urls:
            return set()
        result = await db.execute(
            select(NewsArticle.url).where(NewsArticle.url.in_(urls)),
        )
        return {row[0] for row in result.all()}

    @staticmethod
    def _deduplicate_by_title_date(articles: list[RawArticle]) -> list[RawArticle]:
        """제목+발행일 기반 배치 내 중복 제거"""
        seen: set[str] = set()
        unique: list[RawArticle] = []
        for article in articles:
            pub_date = article.published_at.date().isoformat() if article.published_at else "unknown"
            key = f"{article.title.strip()}|{pub_date}"
            if key not in seen:
                seen.add(key)
                unique.append(article)
        return unique
