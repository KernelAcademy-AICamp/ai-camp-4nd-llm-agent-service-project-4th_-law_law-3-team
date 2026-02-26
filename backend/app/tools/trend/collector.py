"""멀티소스 트렌드 수집기 (2단계 파이프라인)

Stage 1: Tavily → 커뮤니티 인기글 수집
Stage 2: LLM 키워드 추출
Stage 3: Naver/YouTube/Perplexity → 키워드 기반 병렬 뉴스 검색

v2.1: Keyword Flow 전용 메서드 추가
  - collect_community_keywords(): 커뮤니티 수집 + 4차원 스코어링 키워드 반환
  - search_news_for_keyword(): 특정 키워드로 뉴스 병렬 검색
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from app.core.config import settings
from app.modules.content_marketing.schema import (
    LawyerPersona,
    NewsArticle,
    TrendRequest,
    TrendResponse,
    TrendSource,
)
from app.tools.trend.exceptions import TrendSourceError
from app.tools.trend.keyword_blacklist import sanitize_keyword
from app.tools.trend.keyword_extractor import CommunityKeywordExtractor
from app.tools.trend.models import (
    CommunityTopic,
    RawTrendItem,
    ScoredKeyword,
    TrendCacheEntry,
)
from app.tools.trend.sources import BaseTrendSource, SourceConfig
from app.tools.trend.sources.google_source import GoogleSource
from app.tools.trend.sources.naver_source import NaverSource
from app.tools.trend.sources.newsapi_source import NewsAPISource
from app.tools.trend.sources.newsdata_source import NewsDataSource
from app.tools.trend.sources.perplexity_source import PerplexitySource
from app.tools.trend.sources.tavily_source import TavilySource
from app.tools.trend.sources.youtube_source import YouTubeSource

logger = logging.getLogger(__name__)


class TrendCollector:
    """멀티소스 트렌드 수집기 (2단계 파이프라인)"""

    def __init__(self) -> None:
        self._community_source = TavilySource()
        self._news_sources: list[BaseTrendSource] = [
            NaverSource(),
            YouTubeSource(),
            PerplexitySource(),
            GoogleSource(),
            NewsDataSource(),
            NewsAPISource(),
        ]
        self._all_sources: list[BaseTrendSource] = [
            self._community_source,
            *self._news_sources,
        ]
        self._keyword_extractor = CommunityKeywordExtractor()
        self._cache: dict[str, TrendCacheEntry] = {}

    def _get_available_sources(self) -> list[BaseTrendSource]:
        """API 키가 설정된 소스만 반환"""
        return [s for s in self._all_sources if s.is_available]

    def get_available_source_names(self) -> list[TrendSource]:
        """사용 가능한 소스 이름 목록"""
        return [s.name for s in self._get_available_sources()]

    def get_available_source_names_str(self) -> list[str]:
        """사용 가능한 소스 이름 문자열 목록"""
        return [s.name.value for s in self._get_available_sources()]

    def _build_search_query(
        self,
        user_query: str | None,
        persona: LawyerPersona | None,
    ) -> str:
        """중앙 쿼리 빌더: 사용자 쿼리 + 페르소나 기반 검색 쿼리 생성"""
        base = user_query if user_query else "최신 뉴스"

        if persona is None:
            return base

        # 관심 토픽 상위 2개
        focus_terms = persona.focus_topics[:2] if persona.focus_topics else []
        if not focus_terms:
            return base

        return f"{base} {' '.join(focus_terms)}"

    def _make_cache_key(self, request: TrendRequest) -> str:
        """캐시 키 생성 (persona_id 포함)"""
        return (
            f"{request.time_range.value}:{request.category.value}"
            f":{request.query or ''}:{request.persona_id or ''}"
        )

    def get_cached(self, request: TrendRequest) -> TrendResponse | None:
        """캐시된 응답 반환 (만료 시 None)"""
        key = self._make_cache_key(request)
        entry = self._cache.get(key)
        if entry is None:
            return None
        elapsed = (datetime.now(tz=timezone.utc) - entry.created_at).total_seconds()
        if elapsed > entry.ttl_seconds:
            del self._cache[key]
            return None
        response = entry.response.model_copy()
        response.cache_hit = True
        return response

    def set_cache(self, request: TrendRequest, response: TrendResponse) -> None:
        """응답 캐시 저장"""
        key = self._make_cache_key(request)
        self._cache[key] = TrendCacheEntry(
            key=key,
            response=response,
            created_at=datetime.now(tz=timezone.utc),
            ttl_seconds=settings.CONTENT_MARKETING_CACHE_TTL,
        )

    async def _collect_community(
        self,
        request: TrendRequest,
        persona: LawyerPersona | None,
    ) -> CommunityTopic:
        """Stage 1+2: 커뮤니티 인기글 수집 + 키워드 추출"""
        if not self._community_source.is_available:
            logger.info("Tavily API 키 미설정, 커뮤니티 수집 건너뜀")
            return CommunityTopic(raw_items=[], extracted_keywords=[], context_summary="")

        search_query = self._build_search_query(request.query, persona)
        config = SourceConfig(
            time_range=request.time_range.value,
            max_results=request.limit * 2,
            search_query=search_query,
        )

        community_items = await self._community_source.safe_fetch(search_query, config)
        logger.info("Stage 1 커뮤니티 수집: %d건", len(community_items))

        if not community_items:
            logger.info("커뮤니티 수집 0건, 빈 CommunityTopic 반환")
            return CommunityTopic(raw_items=[], extracted_keywords=[], context_summary="")

        community_topic = await self._keyword_extractor.extract(community_items)
        logger.info("Stage 2 키워드 추출 완료: %s", community_topic.extracted_keywords)
        return community_topic

    async def _collect_news(
        self,
        request: TrendRequest,
        persona: LawyerPersona | None,
        community_topic: CommunityTopic,
    ) -> list[RawTrendItem]:
        """Stage 3: 추출된 키워드로 뉴스/영상 병렬 검색"""
        available_news = [s for s in self._news_sources if s.is_available]
        if not available_news:
            logger.info("뉴스 소스 전체 미설정, Stage 3 건너뜀")
            return []

        # 키워드가 있으면 키워드 기반 쿼리, 없으면 원래 쿼리 사용
        if community_topic.extracted_keywords:
            keyword_query = " ".join(community_topic.extracted_keywords)
        else:
            keyword_query = self._build_search_query(request.query, persona)

        config = SourceConfig(
            time_range=request.time_range.value,
            max_results=request.limit * 2,
            search_query=keyword_query,
            community_context=community_topic.context_summary,
        )

        results = await asyncio.gather(
            *[source.safe_fetch(keyword_query, config) for source in available_news]
        )

        news_items: list[RawTrendItem] = []
        for items in results:
            news_items.extend(items)

        logger.info("Stage 3 뉴스 수집: %d건 (소스 %d개)", len(news_items), len(available_news))
        return news_items

    async def collect(
        self,
        request: TrendRequest,
        persona: LawyerPersona | None = None,
    ) -> list[RawTrendItem]:
        """2단계 파이프라인으로 트렌드 수집 + 중복 제거

        Stage 1: Tavily → 커뮤니티 인기글 수집
        Stage 2: LLM → 핵심 키워드 추출
        Stage 3: Naver/YouTube/Perplexity → 키워드 기반 병렬 검색
        """
        if not self._get_available_sources():
            raise TrendSourceError("사용 가능한 트렌드 소스가 없습니다. API 키를 설정하세요.")

        # Stage 1+2: 커뮤니티 수집 + 키워드 추출
        community_topic = await self._collect_community(request, persona)

        # Stage 3: 키워드 기반 뉴스 병렬 검색
        news_items = await self._collect_news(request, persona, community_topic)

        # 합산 + 중복 제거
        all_items = community_topic.raw_items + news_items
        deduplicated = self._deduplicate(all_items)
        logger.info(
            "트렌드 수집 완료: 총 %d건 (중복 제거 후 %d건)",
            len(all_items),
            len(deduplicated),
        )
        return deduplicated

    def _deduplicate(self, items: list[RawTrendItem]) -> list[RawTrendItem]:
        """URL 유효성 검증 + 중복 제거"""
        from app.tools.trend.url_filter import is_article_url

        seen_urls: set[str] = set()
        unique: list[RawTrendItem] = []
        filtered_count = 0
        for item in items:
            if not is_article_url(item.url):
                filtered_count += 1
                continue
            normalized_url = item.url.rstrip("/").lower()
            if normalized_url not in seen_urls:
                seen_urls.add(normalized_url)
                unique.append(item)
        if filtered_count:
            logger.info("URL 필터링: %d건 제거 (홈페이지/무효 URL)", filtered_count)
        return unique

    # ── Keyword Flow 전용 메서드 (v2.1 NEW) ──

    async def collect_community_keywords(
        self,
        community_domains: list[str] | None = None,
        max_keywords: int = 10,
    ) -> list[ScoredKeyword]:
        """커뮤니티 수집 + 4차원 스코어링 키워드 반환

        Args:
            community_domains: 수집 대상 도메인 (None이면 config 기본값)
            max_keywords: 최대 키워드 수

        Returns:
            ScoredKeyword 리스트 (total_score 내림차순)
        """
        domains = community_domains or settings.KEYWORD_COMMUNITY_DOMAINS

        if not self._community_source.is_available:
            raise TrendSourceError("Tavily API 키가 설정되지 않았습니다.")

        # Tavily 커뮤니티 수집 (include_domains 사용)
        config = SourceConfig(
            time_range="48h",
            max_results=max_keywords * 3,
            search_query="사건 사고 논란 이슈",
            include_domains=domains,
        )

        # Tavily + Naver 뉴스 병렬 수집 (하이브리드)
        naver_source = next(
            (s for s in self._news_sources if s.name == TrendSource.NAVER and s.is_available),
            None,
        )

        tasks = [self._community_source.safe_fetch("사건 사고 논란 이슈", config)]
        if naver_source:
            naver_config = SourceConfig(
                time_range="48h",
                max_results=max_keywords * 2,
                search_query="사건 사고 논란 법률",
            )
            tasks.append(naver_source.safe_fetch("사건 사고 논란 법률", naver_config))

        results = await asyncio.gather(*tasks)
        community_items: list[RawTrendItem] = []
        for items in results:
            community_items.extend(items)

        logger.info("커뮤니티 키워드 수집: %d건 (소스 %d개)", len(community_items), len(tasks))

        if not community_items:
            return []

        # LLM 키워드 + 스코어링 추출
        scored_keywords = await self._keyword_extractor.extract_with_scores(
            community_items, max_keywords=max_keywords,
        )

        return scored_keywords

    async def search_news_for_keyword(
        self,
        keyword: str,
        max_results: int = 10,
    ) -> tuple[list[NewsArticle], list[str]]:
        """특정 키워드로 뉴스 소스 병렬 검색

        Args:
            keyword: 검색 키워드 (Sanitize 완료)
            max_results: 최대 결과 수

        Returns:
            (NewsArticle 리스트, 사용된 소스명 리스트) 튜플
        """
        sanitized = sanitize_keyword(keyword)
        if not sanitized:
            logger.warning("키워드 Sanitize 실패: %s", keyword[:30])
            return [], []

        available_news = [s for s in self._news_sources if s.is_available]
        if not available_news:
            raise TrendSourceError("사용 가능한 뉴스 소스가 없습니다.")

        config = SourceConfig(
            time_range="7d",
            max_results=max_results,
            search_query=sanitized,
        )

        # 소스별 차등 타임아웃 (§8.6)
        source_timeouts: dict[str, int] = {
            "naver": 5,
            "google_trends": 8,
            "newsdata": 10,
            "newsapi": 8,
            "youtube": 10,
            "perplexity": 10,
            "tavily": 15,
        }

        async def _fetch_with_timeout(source: BaseTrendSource) -> list[RawTrendItem]:
            timeout = source_timeouts.get(source.name.value, 15)
            try:
                return await asyncio.wait_for(
                    source.safe_fetch(sanitized, config),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                logger.warning("소스 %s 타임아웃 (%ds)", source.name.value, timeout)
                return []

        results = await asyncio.gather(
            *[_fetch_with_timeout(s) for s in available_news],
        )

        # 결과 합산 + 중복 제거
        all_items: list[RawTrendItem] = []
        for items in results:
            all_items.extend(items)

        deduplicated = self._deduplicate(all_items)

        # RawTrendItem → NewsArticle 변환
        articles: list[NewsArticle] = []
        for item in deduplicated[:max_results]:
            articles.append(NewsArticle(
                title=item.title,
                url=item.url,
                source=item.source.value,
                published_at=item.published_at,
                snippet=item.snippet,
            ))

        # 실제 결과를 반환한 소스 목록
        sources_used: list[str] = []
        for source, result in zip(available_news, results):
            if result:
                sources_used.append(source.name.value)

        logger.info(
            "키워드 뉴스 검색 '%s': %d건 (소스 %d개)",
            sanitized, len(articles), len(available_news),
        )
        return articles, sources_used
