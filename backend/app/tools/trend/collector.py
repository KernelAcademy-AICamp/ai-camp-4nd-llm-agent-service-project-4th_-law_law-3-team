"""멀티소스 트렌드 수집기 (2단계 파이프라인)

Stage 1: Tavily → 커뮤니티 인기글 수집
Stage 2: LLM 키워드 추출
Stage 3: Naver/YouTube/Perplexity → 키워드 기반 병렬 뉴스 검색

v2.1: Keyword Flow 전용 메서드 추가
  - collect_community_keywords(): 커뮤니티 수집 + 4차원 스코어링 키워드 반환
  - search_news_for_keyword(): 특정 키워드로 뉴스 병렬 검색

v3: Phase 3 통합
  - CircuitBreaker: 소스별 장애 감지 + 점증 backoff
  - URL Canonicalization: 정규화된 URL로 중복 제거 정확도 향상
  - Engagement passthrough: RawTrendItem → NewsArticle 변환 시 engagement 필드 전달
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from app.core.config import settings
from app.modules.content_marketing.schema import (
    LawyerPersona,
    NewsArticle,
    SourceFailInfo,
    TrendRequest,
    TrendResponse,
    TrendSource,
)
from app.tools.trend.circuit_breaker import CircuitBreaker
from app.tools.trend.exceptions import TrendSourceError
from app.tools.trend.keyword_blacklist import sanitize_keyword
from app.tools.trend.keyword_extractor import CommunityKeywordExtractor
from app.tools.trend.models import (
    CommunityTopic,
    RawTrendItem,
    ScoredKeyword,
    TrendCacheEntry,
)
from app.tools.trend.rrf import SOURCE_AUTHORITY_WEIGHTS
from app.tools.trend.sources import BaseTrendSource, SourceConfig, SourceFetchResult
from app.tools.trend.sources.google_source import GoogleSource
from app.tools.trend.sources.naver_source import NaverSource
from app.tools.trend.sources.newsapi_source import NewsAPISource
from app.tools.trend.sources.newsdata_source import NewsDataSource
from app.tools.trend.sources.perplexity_source import PerplexitySource
from app.tools.trend.sources.tavily_source import TavilySource
from app.tools.trend.sources.youtube_source import YouTubeSource
from app.tools.trend.url_canonicalizer import canonicalize_url

logger = logging.getLogger(__name__)

# ── 카테고리별 동적 쿼리 맵 (Phase 1: 하드코딩 쿼리 → 카테고리 관통) ──

_CATEGORY_KEYWORD_QUERIES: dict[str, list[str]] = {
    "all": ["사건 사고 논란 이슈", "법률 사건 사고 논란", "법률 이슈 판결"],
    "criminal": ["형사 사건 판결 논란", "형사 범죄 수사 이슈", "형법 판결 사건"],
    "civil": ["민사 소송 판결 논란", "민사 분쟁 손해배상 이슈", "민법 판결 사건"],
    "labor": ["노동 근로 분쟁 이슈", "부당해고 임금체불 논란", "노동법 판결 사건"],
    "family": ["가족 이혼 상속 분쟁", "가사 양육권 논란 이슈", "가정법원 판결 사건"],
    "administrative": ["행정 소송 규제 논란", "행정법 처분 취소 이슈", "공법 규제 판결 사건"],
    "corporate": ["기업 법무 분쟁 이슈", "상법 회사 경영권 논란", "기업법 판결 사건"],
    "ip": ["지식재산 특허 저작권 분쟁", "상표권 침해 논란 이슈", "지재권 판결 사건"],
}

_CATEGORY_NEWS_QUERIES: dict[str, str] = {
    "all": "법률",
    "criminal": "형사 범죄",
    "civil": "민사 소송",
    "labor": "노동 근로",
    "family": "가사 이혼 상속",
    "administrative": "행정 소송 규제",
    "corporate": "기업 상법",
    "ip": "지식재산 특허 저작권",
}


def _build_keyword_query(category: str) -> list[str]:
    """카테고리 기반 키워드 수집용 동적 쿼리 생성"""
    return _CATEGORY_KEYWORD_QUERIES.get(category, _CATEGORY_KEYWORD_QUERIES["all"])


def _build_news_query(keyword: str, category: str) -> str:
    """카테고리 기반 뉴스 검색용 동적 쿼리 생성"""
    category_context = _CATEGORY_NEWS_QUERIES.get(category, _CATEGORY_NEWS_QUERIES["all"])
    return f"{keyword} {category_context}"


MAX_CONCURRENT_SOURCES: int = 4


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
        self._circuit_breaker = CircuitBreaker(failure_threshold=3, half_open_max_requests=2)
        self._source_semaphore = asyncio.Semaphore(MAX_CONCURRENT_SOURCES)

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
        """URL 유효성 검증 + 정규화 중복 제거

        Phase 3: canonicalize_url()로 UTM 파라미터 제거, 모바일 도메인 통일 등
        정규화된 URL 기준 중복 제거로 정확도 향상.
        """
        from app.tools.trend.url_filter import is_article_url

        seen_urls: set[str] = set()
        unique: list[RawTrendItem] = []
        filtered_count = 0
        for item in items:
            if not is_article_url(item.url):
                filtered_count += 1
                continue
            canonical = canonicalize_url(item.url)
            if canonical not in seen_urls:
                seen_urls.add(canonical)
                unique.append(item)
        if filtered_count:
            logger.info("URL 필터링: %d건 제거 (홈페이지/무효 URL)", filtered_count)
        return unique

    # ── Keyword Flow 전용 메서드 (v2.1 NEW) ──

    async def collect_community_keywords(
        self,
        community_domains: list[str] | None = None,
        max_keywords: int = 10,
        category: str = "all",
    ) -> list[ScoredKeyword]:
        """커뮤니티 수집 + 4차원 스코어링 키워드 반환

        Args:
            community_domains: 수집 대상 도메인 (None이면 config 기본값)
            max_keywords: 최대 키워드 수
            category: 법률 카테고리 (동적 쿼리 생성에 사용)

        Returns:
            ScoredKeyword 리스트 (total_score 내림차순)
        """
        domains = community_domains or settings.KEYWORD_COMMUNITY_DOMAINS
        queries = _build_keyword_query(category)

        if not self._community_source.is_available:
            raise TrendSourceError("Tavily API 키가 설정되지 않았습니다.")

        # Tavily 커뮤니티 수집 (include_domains 사용)
        primary_query = queries[0]
        config = SourceConfig(
            time_range="48h",
            max_results=max_keywords * 3,
            search_query=primary_query,
            include_domains=domains,
            category=category,
        )

        # Tavily + Naver 뉴스 병렬 수집 (하이브리드)
        naver_source = next(
            (s for s in self._news_sources if s.name == TrendSource.NAVER and s.is_available),
            None,
        )

        tasks = [self._community_source.safe_fetch(primary_query, config)]
        if naver_source:
            secondary_query = queries[1] if len(queries) > 1 else primary_query
            naver_config = SourceConfig(
                time_range="48h",
                max_results=max_keywords * 2,
                search_query=secondary_query,
                category=category,
            )
            tasks.append(naver_source.safe_fetch(secondary_query, naver_config))

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

    async def collect_community_keywords_with_status(
        self,
        community_domains: list[str] | None = None,
        max_keywords: int = 10,
        time_range: str = "48h",
        category: str = "all",
    ) -> tuple[list[ScoredKeyword], list[str], list[SourceFailInfo]]:
        """커뮤니티 수집 + 4차원 스코어링 키워드 반환 (소스 상태 포함)

        Args:
            community_domains: 수집 대상 도메인 (None이면 config 기본값)
            max_keywords: 최대 키워드 수
            time_range: 수집 기간
            category: 법률 카테고리 (동적 쿼리 생성에 사용)

        Returns:
            (ScoredKeyword 리스트, 성공 소스명 리스트, 실패 소스 정보 리스트) 튜플
        """
        domains = community_domains or settings.KEYWORD_COMMUNITY_DOMAINS
        queries = _build_keyword_query(category)

        if not self._community_source.is_available:
            raise TrendSourceError("Tavily API 키가 설정되지 않았습니다.")

        primary_query = queries[0]
        config = SourceConfig(
            time_range=time_range,
            max_results=max_keywords * 3,
            search_query=primary_query,
            include_domains=domains,
            category=category,
        )

        # Tavily + Naver 병렬 수집 (safe_fetch_with_status 사용)
        naver_source = next(
            (s for s in self._news_sources if s.name == TrendSource.NAVER and s.is_available),
            None,
        )

        secondary_query = queries[1] if len(queries) > 1 else primary_query
        tasks = [self._community_source.safe_fetch_with_status(primary_query, config)]
        if naver_source:
            naver_config = SourceConfig(
                time_range=time_range,
                max_results=max_keywords * 2,
                search_query=secondary_query,
                category=category,
            )
            tasks.append(naver_source.safe_fetch_with_status(secondary_query, naver_config))

        fetch_results: list[SourceFetchResult] = await asyncio.gather(*tasks)

        # 소스 상태 분류 + 아이템 합산
        community_items: list[RawTrendItem] = []
        sources_used: list[str] = []
        sources_failed: list[SourceFailInfo] = []

        for result in fetch_results:
            community_items.extend(result.items)
            if result.is_success:
                sources_used.append(result.source_name)
            else:
                sources_failed.append(SourceFailInfo(
                    source_name=result.source_name,
                    error_type=result.error_type or "unknown",
                    error_message=result.error_message,
                ))

        logger.info(
            "커뮤니티 키워드 수집: %d건 (성공 %d개, 실패 %d개)",
            len(community_items), len(sources_used), len(sources_failed),
        )

        if not community_items:
            return [], sources_used, sources_failed

        scored_keywords = await self._keyword_extractor.extract_with_scores(
            community_items, max_keywords=max_keywords,
        )

        return scored_keywords, sources_used, sources_failed

    async def search_news_for_keyword(
        self,
        keyword: str,
        max_results: int = 10,
        category: str = "all",
    ) -> tuple[list[NewsArticle], list[str], list[SourceFailInfo]]:
        """특정 키워드로 뉴스 소스 병렬 검색

        Args:
            keyword: 검색 키워드 (Sanitize 완료)
            max_results: 최대 결과 수
            category: 법률 카테고리 (쿼리 보강에 사용)

        Returns:
            (NewsArticle 리스트, 사용된 소스명 리스트, 실패 소스 정보 리스트) 튜플
        """
        sanitized = sanitize_keyword(keyword)
        if not sanitized:
            logger.warning("키워드 Sanitize 실패: %s", keyword[:30])
            return [], [], []

        available_news = [s for s in self._news_sources if s.is_available]
        if not available_news:
            raise TrendSourceError("사용 가능한 뉴스 소스가 없습니다.")

        # 카테고리 기반 쿼리 보강
        search_query = _build_news_query(sanitized, category)

        config = SourceConfig(
            time_range="7d",
            max_results=max_results,
            search_query=search_query,
            category=category,
        )

        # 소스별 차등 타임아웃 (§8.6)
        source_timeouts: dict[str, int] = {
            "naver": 5,
            "google_trends": 8,
            "newsdata": 10,
            "newsapi": 8,
            "youtube": 10,
            "perplexity": 15,
            "tavily": 15,
        }

        async def _fetch_with_status(source: BaseTrendSource) -> SourceFetchResult:
            """safe_fetch_with_status() + 소스별 타임아웃 + CircuitBreaker"""
            source_key = source.name.value

            # CircuitBreaker: OPEN 상태면 즉시 스킵
            if self._circuit_breaker.is_open(source_key):
                status = self._circuit_breaker.get_status(source_key)
                logger.info("CircuitBreaker %s: 소스 '%s' 스킵", status, source_key)
                return SourceFetchResult(
                    items=[],
                    source_name=source_key,
                    is_success=False,
                    error_type="circuit_open",
                    error_message=f"서킷 브레이커 {status} 상태",
                )

            timeout = source_timeouts.get(source_key, 15)
            try:
                result = await asyncio.wait_for(
                    source.safe_fetch_with_status(sanitized, config),
                    timeout=timeout,
                )
                if result.is_success:
                    self._circuit_breaker.record_success(source_key)
                else:
                    self._circuit_breaker.record_failure(source_key)
                return result
            except asyncio.TimeoutError:
                self._circuit_breaker.record_failure(source_key)
                logger.warning("소스 %s 타임아웃 (%ds)", source_key, timeout)
                return SourceFetchResult(
                    items=[],
                    source_name=source_key,
                    is_success=False,
                    error_type="timeout",
                    error_message=f"응답 시간 초과 ({timeout}s)",
                )

        async def _fetch_with_semaphore(source: BaseTrendSource) -> SourceFetchResult:
            async with self._source_semaphore:
                return await _fetch_with_status(source)

        fetch_results: list[SourceFetchResult] = await asyncio.gather(
            *[_fetch_with_semaphore(s) for s in available_news],
        )

        # 결과 합산 + 중복 제거
        all_items: list[RawTrendItem] = []
        for result in fetch_results:
            all_items.extend(result.items)

        deduplicated = self._deduplicate(all_items)

        # RawTrendItem → NewsArticle 변환 (v3: engagement 필드 패스스루)
        articles: list[NewsArticle] = []
        for item in deduplicated[:max_results]:
            source_key = item.source.value
            articles.append(NewsArticle(
                title=item.title,
                url=item.url,
                source=source_key,
                published_at=item.published_at,
                snippet=item.snippet,
                view_count=item.view_count,
                comment_count=item.comment_count,
                is_early_signal=item.is_early_signal,
                source_weight=SOURCE_AUTHORITY_WEIGHTS.get(source_key, 0.5),
            ))

        # SourceFetchResult로부터 성공/실패 소스 분류
        sources_used: list[str] = []
        sources_failed: list[SourceFailInfo] = []
        for result in fetch_results:
            if result.is_success and result.items:
                sources_used.append(result.source_name)
            elif not result.is_success:
                sources_failed.append(SourceFailInfo(
                    source_name=result.source_name,
                    error_type=result.error_type or "unknown",
                    error_message=result.error_message,
                ))
            else:
                # 성공했지만 결과 0건 (빈 결과)
                sources_used.append(result.source_name)

        logger.info(
            "키워드 뉴스 검색 '%s': %d건 (성공 %d개, 실패 %d개)",
            sanitized, len(articles), len(sources_used), len(sources_failed),
        )
        return articles, sources_used, sources_failed
