"""콘텐츠 마케팅 서비스 함수

모듈 라우터에서 호출되는 비즈니스 로직 진입점.
v2.0: 페르소나 CRUD + 분석 서비스 + Legal Gate 스코어링
v2.1: Keyword Flow (키워드 수집 + 뉴스 검색) 추가
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.modules.content_marketing.schema import (
    AnalysisInsights,
    KeywordCollectRequest,
    KeywordCollectResponse,
    KeywordItem,
    KeywordNewsRequest,
    KeywordNewsResponse,
    KeywordScoreSchema,
    KeywordStreamEvent,
    LawyerPersona,
    MetadataRequest,
    NewsArticle,
    PersonaAnalysisRequest,
    PersonaFeedbackRequest,
    PersonaOnboardingRequest,
    PersonaTone,
    PersonaUpdateRequest,
    RelatedLawBrief,
    ScriptMetadata,
    ScriptRequest,
    SourceFailInfo,
    TrendDetailResponse,
    TrendRequest,
    TrendResponse,
)
from app.services.service_function import persona_db_service
from app.tools.persona.onboarding import OnboardingProcessor
from app.tools.trend.collector import TrendCollector
from app.tools.trend.models import ScoredKeyword
from app.tools.trend.rate_limiter import InMemoryRateLimiter, RateLimitExceededError
from app.tools.trend.scorer import LegalGateScorer, TrendScorer
from app.tools.trend.summarizer import IssueSummarizer

logger = logging.getLogger(__name__)


class KeywordNotFoundError(Exception):
    """키워드 캐시 미스 또는 키워드 ID 불일치"""


# 싱글턴 인스턴스 (서버 수명 동안 유지)
_collector = TrendCollector()
_scorer = TrendScorer()
_scorer_v2 = LegalGateScorer()
_summarizer = IssueSummarizer()
_onboarding = OnboardingProcessor()

# 트렌드 상세 조회용 캐시 (trend_id → TrendDetailResponse, LRU 상한 200)
_TREND_DETAIL_CACHE_MAX = 200
_trend_detail_cache: dict[str, TrendDetailResponse] = {}

# Keyword Flow 캐시 (user_id → ScoredKeyword 리스트 + 소스 정보)
# 형식: (keywords, cached_at, sources_used, sources_failed)
_keyword_cache: dict[
    str,
    tuple[list[ScoredKeyword], datetime, list[str], list[SourceFailInfo]],
] = {}

# News 캐시 (user_id:keyword_id:news → KeywordNewsResponse + cached_at)
_news_cache: dict[str, tuple[KeywordNewsResponse, datetime]] = {}

# Rate Limiter 인스턴스 (버킷 분리)
_collect_limiter = InMemoryRateLimiter(
    max_requests=settings.KEYWORD_COLLECT_RATE_LIMIT,
    window_seconds=3600,
)
_news_limiter = InMemoryRateLimiter(
    max_requests=settings.KEYWORD_NEWS_RATE_LIMIT,
    window_seconds=3600,
)


# ── Persona 서비스 (v2.0 NEW) ──


async def get_current_persona(
    db: AsyncSession,
    user_id: str,
) -> LawyerPersona | None:
    """현재 페르소나 조회"""
    return await persona_db_service.get_persona(db, user_id)


async def analyze_persona(
    db: AsyncSession,
    user_id: str,
    request: PersonaAnalysisRequest,
) -> tuple[LawyerPersona, AnalysisInsights]:
    """Track 1: 대화 이력 기반 자동 분석

    Returns:
        (LawyerPersona, AnalysisInsights) 튜플
    """
    from app.tools.persona.analyzer import (
        InsufficientHistoryError,
        LowConfidenceError,
        PersonaAnalyzer,
    )

    analyzer = PersonaAnalyzer()

    # 대화 이력 조회 (인증 시스템 구현 전까지 빈 리스트 → InsufficientHistoryError)
    messages = await _fetch_chat_history(
        db, user_id, request.max_history, request.days_back
    )

    try:
        result = await analyzer.analyze(
            user_id=user_id,
            messages=messages,
            max_history=request.max_history,
            days_back=request.days_back,
        )
    except InsufficientHistoryError:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=422,
            detail="대화 이력이 부족합니다. 온보딩을 진행해주세요.",
        )
    except LowConfidenceError:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=409,
            detail="분석 신뢰도가 낮습니다. 온보딩을 진행해주세요.",
        )

    persona = await persona_db_service.create_persona(db, result.persona)
    insights = result.insights  # PersonaAnalyzer가 insights도 함께 반환

    return persona, insights


async def _fetch_chat_history(
    db: AsyncSession,
    user_id: str,
    max_history: int,
    days_back: int,
) -> list[dict[str, str]]:
    """사용자 대화 이력 조회 (chat_sessions 테이블)

    NOTE: 인증 시스템 구현 전까지 빈 리스트 반환 → InsufficientHistoryError 발생
    """
    # TODO: chat_sessions 테이블 또는 LangSmith에서 대화 이력 조회
    return []


async def get_chat_history_count(
    db: AsyncSession,
    user_id: str,
) -> int:
    """사용자 대화 이력 건수 조회"""
    # TODO: 실제 구현 시 chat_sessions 테이블에서 COUNT
    return 0


async def create_persona_from_onboarding(
    db: AsyncSession,
    user_id: str,
    request: PersonaOnboardingRequest,
) -> LawyerPersona:
    """Track 2: 온보딩 결과로 페르소나 생성"""
    persona = _onboarding.process(user_id, request)
    return await persona_db_service.create_persona(db, persona)


async def update_current_persona(
    db: AsyncSession,
    user_id: str,
    request: PersonaUpdateRequest,
) -> LawyerPersona | None:
    """페르소나 부분 수정"""
    return await persona_db_service.update_persona(db, user_id, request)


async def save_persona_feedback(
    db: AsyncSession,
    request: PersonaFeedbackRequest,
) -> None:
    """피드백 저장"""
    await persona_db_service.save_feedback(db, request)


# ── Trend 서비스 ──


async def collect_trends(
    request: TrendRequest,
    db: AsyncSession | None = None,
) -> TrendResponse:
    """트렌드 수집 + 스코어링 + 요약

    v2.0: persona_id가 있으면 LegalGateScorer + 5차원 스코어링 사용

    1. 캐시 확인
    2. 멀티소스 병렬 수집
    3. 그룹화 + 스코어링 (v1 or v2)
    4. LLM 요약 + RAG 법령/판례 매칭
    5. 캐시 저장 + 반환
    """
    cached = _collector.get_cached(request)
    if cached is not None:
        logger.info("트렌드 캐시 히트: %s", _collector._make_cache_key(request))
        return cached

    # persona 조회 (수집 쿼리 개인화 + 스코어링 모두에서 사용)
    persona: LawyerPersona | None = None
    if request.persona_id and db is not None:
        persona = await persona_db_service.get_persona_by_id(db, request.persona_id)

    raw_items = await _collector.collect(request, persona=persona)

    if persona is not None:
        scored_v2 = await _scorer_v2.score_v2(raw_items, persona=persona)
        issues = await _summarizer.summarize_v2(scored_v2, limit=request.limit)
        logger.info("v2.0 Legal Gate 스코어링 완료: %d건", len(issues))
    else:
        # persona 없으면 v2 스코어러로 기본 스코어링 (fitness=0.5)
        scored_v2 = await _scorer_v2.score_v2(raw_items, persona=None)
        issues = await _summarizer.summarize_v2(scored_v2, limit=request.limit)
        logger.info("v2.0 스코어링 완료 (페르소나 없음): %d건", len(issues))

    response = TrendResponse(
        trends=issues,
        total_count=len(issues),
        collected_at=datetime.now(tz=timezone.utc),
        sources_used=_collector.get_available_source_names(),
        cache_hit=False,
    )

    _collector.set_cache(request, response)

    # 상세 조회용 캐시 업데이트 (LRU eviction: 오래된 항목부터 제거)
    evict_count = max(0, len(_trend_detail_cache) + len(issues) - _TREND_DETAIL_CACHE_MAX)
    for _ in range(min(evict_count, len(_trend_detail_cache))):
        oldest_key = next(iter(_trend_detail_cache))
        del _trend_detail_cache[oldest_key]

    for issue in issues:
        _trend_detail_cache[issue.id] = TrendDetailResponse(
            issue=issue,
            source_articles=issue.source_articles,
            related_laws_detail=[
                {
                    "law_id": law.law_id,
                    "law_name": law.law_name,
                    "relevance_score": law.relevance_score,
                }
                for law in issue.related_laws
            ],
            related_cases_detail=[
                {
                    "case_id": case.case_id,
                    "case_number": case.case_number,
                    "case_name": case.case_name,
                    "relevance_score": case.relevance_score,
                }
                for case in issue.related_cases
            ],
        )

    logger.info("트렌드 수집 완료: %d건", len(issues))
    return response


async def get_trend_detail(trend_id: str) -> TrendDetailResponse | None:
    """캐시된 트렌드 이슈 상세 조회"""
    return _trend_detail_cache.get(trend_id)


# ── Script 서비스 ──


async def generate_script_stream(
    request: ScriptRequest,
    db: AsyncSession | None = None,
) -> AsyncGenerator[str, None]:
    """대본 SSE 스트리밍 생성 (ScriptGenerator 연동)

    v2.0: persona_id가 있으면 DB에서 페르소나 조회 → PersonaTone 전달
    v2.1: SSE heartbeat 코멘트 지원 (연결 유지)
    v2.2: 백그라운드 heartbeat로 프록시 유휴 타이머 방지
    """
    from app.tools.script.generator import ScriptGenerator

    # persona_id → PersonaTone 해석
    persona_tone: PersonaTone | None = None
    if request.persona_id and db is not None:
        persona = await persona_db_service.get_persona_by_id(db, request.persona_id)
        if persona is not None:
            persona_tone = persona.preferred_tone

    generator = ScriptGenerator()

    # 이벤트 큐 기반: 생성기와 heartbeat을 비동기로 조율
    # LLM 스트리밍 중 오랜 공백이 생기면 프록시가 유휴 타이머에 걸리므로
    # 15초마다 SSE 코멘트를 보내 연결을 유지한다
    event_queue: asyncio.Queue[str | None] = asyncio.Queue()
    heartbeat_interval = 15  # 초

    async def _producer() -> None:
        """생성기 이벤트를 큐에 넣기"""
        try:
            async for event in generator.generate_stream(request, persona_tone=persona_tone):
                await event_queue.put(f"event: {event.event}\ndata: {event.model_dump_json()}\n\n")
        except Exception:
            logger.exception("대본 생성 스트림 오류")
            await event_queue.put(
                'event: error\ndata: {"event":"error","error":"서버 내부 오류가 발생했습니다."}\n\n'
            )
        finally:
            await event_queue.put(None)  # 종료 신호

    producer_task = asyncio.create_task(_producer())

    try:
        while True:
            try:
                item = await asyncio.wait_for(
                    event_queue.get(), timeout=heartbeat_interval,
                )
            except TimeoutError:
                # 큐가 비어 있음 → heartbeat 전송 (SSE 코멘트)
                yield ": heartbeat\n\n"
                continue

            if item is None:
                break
            yield item
    finally:
        producer_task.cancel()
        try:
            await producer_task
        except asyncio.CancelledError:
            pass

    # 스트림 종료 시 최종 SSE 코멘트 (클라이언트가 연결 정리하도록)
    yield ": stream-end\n\n"


async def generate_metadata(request: MetadataRequest) -> ScriptMetadata:
    """대본 메타데이터 LLM 생성"""
    from app.tools.script.generator import generate_metadata as _gen_metadata

    return await _gen_metadata(request)


# Rate Limiter: 캐시 클리어 (5 req/min)
_cache_clear_limiter = InMemoryRateLimiter(
    max_requests=5,
    window_seconds=60,
)


def clear_keyword_cache(user_id: str) -> dict[str, int]:
    """키워드/뉴스 캐시 전체 클리어

    Returns:
        삭제된 캐시 항목 수 딕셔너리
    """
    _cache_clear_limiter.check_and_record(user_id, bucket="cache_clear")

    keyword_count = len(_keyword_cache)
    trend_detail_count = len(_trend_detail_cache)
    collector_count = len(_collector._cache)
    news_count = len(_news_cache)

    _keyword_cache.clear()
    _trend_detail_cache.clear()
    _collector._cache.clear()
    _news_cache.clear()

    logger.info(
        "캐시 클리어: user=%s, keyword=%d, trend_detail=%d, collector=%d, news=%d",
        user_id, keyword_count, trend_detail_count, collector_count, news_count,
    )
    return {
        "keyword_cache": keyword_count,
        "trend_detail_cache": trend_detail_count,
        "collector_cache": collector_count,
    }


# ── Keyword Flow 서비스 (v2.1 NEW) ──


async def collect_keywords(
    request: KeywordCollectRequest,
    user_id: str,
) -> KeywordCollectResponse:
    """키워드 수집 + 4차원 스코어링

    1. Rate limit 검사
    2. 캐시 확인
    3. 커뮤니티 수집 + LLM 스코어링
    4. 캐시 저장
    """
    # Rate limit 검사
    _collect_limiter.check_and_record(user_id, bucket="collect")

    # 캐시 확인 (시간 범위 + 카테고리별 캐시 분리)
    category_value = request.category.value if request.category else "all"
    cache_key = f"{user_id}:{request.time_range.value}:{category_value}:keywords"
    cached = _keyword_cache.get(cache_key)
    if cached is not None:
        cached_keywords, cached_at, cached_sources_used, cached_sources_failed = cached
        elapsed = (datetime.now(tz=timezone.utc) - cached_at).total_seconds()
        if elapsed < settings.KEYWORD_COLLECT_CACHE_TTL:
            logger.info("키워드 캐시 히트: user=%s, time_range=%s, category=%s", user_id, request.time_range.value, category_value)
            return KeywordCollectResponse(
                keywords=[_scored_to_item(kw) for kw in cached_keywords],
                total_count=len(cached_keywords),
                collected_at=cached_at,
                sources_used=cached_sources_used,
                sources_failed=cached_sources_failed,
                cache_hit=True,
                prompt_version="1.0",
            )

    # 커뮤니티 수집 + 스코어링 (소스 정보 포함, 카테고리 전달)
    scored_keywords, sources_used, sources_failed = (
        await _collector.collect_community_keywords_with_status(
            community_domains=request.community_domains,
            max_keywords=request.max_keywords,
            time_range=request.time_range.value,
            category=category_value,
        )
    )

    # 캐시 저장 (소스 정보 포함)
    now = datetime.now(tz=timezone.utc)
    _keyword_cache[cache_key] = (scored_keywords, now, sources_used, sources_failed)

    logger.info("키워드 수집 완료: user=%s, count=%d", user_id, len(scored_keywords))
    return KeywordCollectResponse(
        keywords=[_scored_to_item(kw) for kw in scored_keywords],
        total_count=len(scored_keywords),
        collected_at=now,
        sources_used=sources_used,
        sources_failed=sources_failed,
        cache_hit=False,
        prompt_version="1.0",
    )


_SSE_STREAM_TIMEOUT_SECONDS = 120  # SSE 최대 연결 시간 (DoS 방지)


async def collect_keywords_stream(
    request: KeywordCollectRequest,
    user_id: str,
    force_refresh: bool = False,
    db: AsyncSession | None = None,
) -> AsyncGenerator[str, None]:
    """키워드 수집 SSE 스트리밍 (§7.4)

    단계별로 진행 상황을 SSE 이벤트로 전송한다.
    tavily_start → tavily_done → llm_start → llm_done → scoring → done
    캐시 히트 시: cache_hit → done
    """
    def _sse(event: KeywordStreamEvent) -> str:
        return f"data: {event.model_dump_json()}\n\n"

    try:
        # Rate limit 검사
        _collect_limiter.check_and_record(user_id, bucket="collect")

        # 페르소나 조회 (persona_id가 있으면 DB에서 로드)
        persona: LawyerPersona | None = None
        if request.persona_id and db is not None:
            persona = await persona_db_service.get_persona_by_id(db, request.persona_id)

        # 캐시 확인 (시간 범위 + 카테고리 + 페르소나별 캐시 분리, force_refresh 시 건너뜀)
        category_value = request.category.value if request.category else "all"
        persona_key = request.persona_id or "none"
        cache_key = f"{user_id}:{request.time_range.value}:{category_value}:{persona_key}:keywords"
        cached = _keyword_cache.get(cache_key)
        if cached is not None and not force_refresh:
            cached_keywords, cached_at, cached_sources_used, cached_sources_failed = cached
            elapsed = (datetime.now(tz=timezone.utc) - cached_at).total_seconds()
            if elapsed < settings.KEYWORD_COLLECT_CACHE_TTL:
                response = KeywordCollectResponse(
                    keywords=[_scored_to_item(kw) for kw in cached_keywords],
                    total_count=len(cached_keywords),
                    collected_at=cached_at,
                    sources_used=cached_sources_used,
                    sources_failed=cached_sources_failed,
                    cache_hit=True,
                    prompt_version="1.0",
                )
                yield _sse(KeywordStreamEvent(
                    step="cache_hit", progress=50, message="캐시에서 불러오는 중...",
                ))
                yield _sse(KeywordStreamEvent(
                    step="done", progress=100, message="캐시 결과",
                    count=len(cached_keywords), data=response,
                ))
                return

        # Step 1: 커뮤니티 수집 시작
        yield _sse(KeywordStreamEvent(
            step="tavily_start", progress=10, message="커뮤니티 인기글 수집 중...",
        ))

        from app.tools.trend.collector import _build_keyword_query
        from app.tools.trend.sources import SourceConfig, SourceFetchResult

        domains = request.community_domains or settings.KEYWORD_COMMUNITY_DOMAINS
        time_range_value = request.time_range.value
        queries = _build_keyword_query(category_value)
        primary_query = queries[0]
        secondary_query = queries[1] if len(queries) > 1 else primary_query
        tertiary_query = queries[2] if len(queries) > 2 else primary_query

        # 페르소나 기반 쿼리 보강 (관심 토픽 상위 2개 추가)
        if persona is not None:
            primary_query = _collector._build_search_query(primary_query, persona)
            secondary_query = _collector._build_search_query(secondary_query, persona)
            tertiary_query = _collector._build_search_query(tertiary_query, persona)

        tavily_config = SourceConfig(
            time_range=time_range_value,
            max_results=(request.max_keywords or 10) * 3,
            search_query=primary_query,
            include_domains=domains,
            category=category_value,
        )
        fetch_tasks = [_collector._community_source.safe_fetch_with_status(
            primary_query, tavily_config,
        )]

        from app.modules.content_marketing.schema import TrendSource as TrendSourceEnum
        naver_source = next(
            (s for s in _collector._news_sources
             if s.name == TrendSourceEnum.NAVER and s.is_available),
            None,
        )
        if naver_source:
            naver_config = SourceConfig(
                time_range=time_range_value,
                max_results=(request.max_keywords or 10) * 2,
                search_query=secondary_query,
                category=category_value,
            )
            fetch_tasks.append(naver_source.safe_fetch_with_status(
                secondary_query, naver_config,
            ))

        # Secondary 소스 추가 (가용한 소스만)
        secondary_sources = [
            s for s in _collector._news_sources
            if s.is_available and s.name != TrendSourceEnum.NAVER
        ]
        if secondary_sources:
            secondary_config = SourceConfig(
                time_range=time_range_value,
                max_results=request.max_keywords or 10,
                search_query=tertiary_query,
                category=category_value,
            )
            for source in secondary_sources:
                fetch_tasks.append(source.safe_fetch_with_status(
                    tertiary_query, secondary_config,
                ))

        # 타임아웃 적용 (DoS 방지)
        fetch_results: list[SourceFetchResult] = await asyncio.wait_for(
            asyncio.gather(*fetch_tasks),
            timeout=_SSE_STREAM_TIMEOUT_SECONDS,
        )

        # 결과 집계
        community_items = [
            item for result in fetch_results for item in result.items
        ]
        actual_sources_used = [
            r.source_name for r in fetch_results if r.is_success and r.items
        ]
        actual_sources_failed = [
            SourceFailInfo(
                source_name=r.source_name,
                error_type=r.error_type or "unknown",
                error_message=r.error_message,
            )
            for r in fetch_results if not r.is_success
        ]

        yield _sse(KeywordStreamEvent(
            step="tavily_done", progress=40, message=f"{len(community_items)}개 게시글 수집 완료",
            count=len(community_items),
        ))

        if not community_items:
            response = KeywordCollectResponse(
                keywords=[], total_count=0,
                collected_at=datetime.now(tz=timezone.utc),
                sources_used=actual_sources_used,
                sources_failed=actual_sources_failed,
                cache_hit=False, prompt_version="1.0",
            )
            yield _sse(KeywordStreamEvent(
                step="done", progress=100, message="수집된 게시글이 없습니다.",
                count=0, data=response,
            ))
            return

        # Step 2: LLM 키워드 추출
        yield _sse(KeywordStreamEvent(
            step="llm_start", progress=50, message="키워드 분석 중...",
        ))

        scored_keywords = await asyncio.wait_for(
            _collector._keyword_extractor.extract_with_scores(
                community_items, max_keywords=request.max_keywords or 10,
            ),
            timeout=60,
        )

        yield _sse(KeywordStreamEvent(
            step="llm_done", progress=80,
            message=f"{len(scored_keywords)}개 키워드 추출 완료",
            count=len(scored_keywords),
        ))

        # Step 3: 점수 산출 + 캐시 저장
        yield _sse(KeywordStreamEvent(
            step="scoring", progress=90, message="키워드 점수 산출 중...",
        ))

        now = datetime.now(tz=timezone.utc)
        _keyword_cache[cache_key] = (
            scored_keywords, now, actual_sources_used, actual_sources_failed,
        )

        response = KeywordCollectResponse(
            keywords=[_scored_to_item(kw) for kw in scored_keywords],
            total_count=len(scored_keywords),
            collected_at=now,
            sources_used=actual_sources_used,
            sources_failed=actual_sources_failed,
            cache_hit=False,
            prompt_version="1.0",
        )

        yield _sse(KeywordStreamEvent(
            step="done", progress=100, message="완료",
            count=len(scored_keywords), data=response,
        ))

    except RateLimitExceededError as exc:
        logger.warning("키워드 수집 Rate Limit 초과: user=%s", user_id)
        yield _sse(KeywordStreamEvent(
            step="error", progress=0,
            message=f"요청 한도 초과. {exc.retry_after_seconds}초 후 재시도해주세요.",
            error="rate_limit",
        ))
    except TimeoutError:
        logger.warning("키워드 수집 스트리밍 타임아웃: user=%s", user_id)
        yield _sse(KeywordStreamEvent(
            step="error", progress=0, message="요청 시간이 초과되었습니다.",
            error="timeout",
        ))
    except Exception:
        logger.exception("키워드 수집 스트리밍 오류")
        yield _sse(KeywordStreamEvent(
            step="error", progress=0, message="키워드 수집 실패",
            error="internal_error",
        ))


async def search_keyword_news(
    keyword_id: str,
    request: KeywordNewsRequest,
    user_id: str,
) -> KeywordNewsResponse:
    """캐시에서 키워드 조회 + 뉴스 검색 + RAG enrichment + 점수 계산

    1. Rate limit 검사
    2. 캐시에서 keyword_id로 키워드 조회
    3. 뉴스 소스 병렬 검색 + 경량 법률 enrichment 병렬 실행
    4. RAG 결과를 개별 기사에 분배
    5. 점수 계산 + 정렬
    """
    from app.tools.trend.article_scorer import score_articles_v2

    # Rate limit 검사
    _news_limiter.check_and_record(user_id, bucket="news")

    # 뉴스 캐시 확인
    news_cache_key = f"{user_id}:{keyword_id}:news"
    cached_news = _news_cache.get(news_cache_key)
    if cached_news is not None:
        cached_response, cached_at = cached_news
        elapsed = (datetime.now(tz=timezone.utc) - cached_at).total_seconds()
        if elapsed < settings.KEYWORD_NEWS_CACHE_TTL:
            logger.info("뉴스 캐시 히트: user=%s, keyword=%s", user_id, keyword_id)
            return cached_response

    # 캐시에서 키워드 조회 (모든 캐시 키를 순회하여 keyword_id 탐색)
    # 캐시 키 형식: "{user_id}:{time_range}:{category}:keywords" 또는
    #              "{user_id}:{time_range}:{category}:{persona_key}:keywords"
    target_keyword = None
    found_category = "all"
    expired_keys: list[str] = []
    for cache_key, cached in _keyword_cache.items():
        if not cache_key.startswith(f"{user_id}:"):
            continue
        cached_keywords, cached_at = cached[0], cached[1]
        elapsed = (datetime.now(tz=timezone.utc) - cached_at).total_seconds()
        if elapsed >= settings.KEYWORD_COLLECT_CACHE_TTL:
            expired_keys.append(cache_key)
            continue
        found = next((kw for kw in cached_keywords if kw.id == keyword_id), None)
        if found is not None:
            target_keyword = found
            # 캐시 키에서 카테고리 추출 (3번째 세그먼트)
            segments = cache_key.split(":")
            found_category = segments[2] if len(segments) >= 3 else "all"
            break
    for key in expired_keys:
        del _keyword_cache[key]

    if target_keyword is None:
        raise KeywordNotFoundError(
            "키워드를 찾을 수 없습니다. 키워드를 다시 수집해주세요.",
        )

    # 뉴스 검색 + 경량 법률 enrichment 병렬 실행 (개별 예외 격리)
    async def _safe_law_enrichment(keyword: str) -> list[RelatedLawBrief]:
        try:
            return await _search_related_laws(keyword)
        except Exception:
            logger.warning("법률 enrichment 실패, 빈 리스트로 대체", exc_info=True)
            return []

    news_task = _collector.search_news_for_keyword(
        keyword=target_keyword.keyword,
        max_results=request.max_results,
        category=found_category,
    )
    law_task = _safe_law_enrichment(target_keyword.keyword)

    (articles, sources_used, sources_failed), related_laws = await asyncio.gather(
        news_task, law_task,
    )

    # RAG 결과를 개별 기사에 분배 + 점수 계산
    enriched_articles = _enrich_articles(
        articles=articles,
        related_laws=related_laws,
        keyword=target_keyword.keyword,
    )
    scored_articles = score_articles_v2(
        keyword=target_keyword.keyword,
        articles=enriched_articles,
        keyword_convergence_score=0.0,
        category=found_category,
    )

    response = KeywordNewsResponse(
        keyword_id=keyword_id,
        keyword=target_keyword.keyword,
        articles=scored_articles,
        related_laws=related_laws,
        total_count=len(scored_articles),
        sources_used=sources_used,
        sources_failed=sources_failed,
        searched_at=datetime.now(tz=timezone.utc),
    )

    # 뉴스 캐시 저장
    _news_cache[news_cache_key] = (response, datetime.now(tz=timezone.utc))

    return response


def _enrich_articles(
    articles: list[NewsArticle],
    related_laws: list[RelatedLawBrief],
    keyword: str,
) -> list[NewsArticle]:
    """RAG enrichment 결과를 개별 기사에 분배

    기사별 title+snippet과 법령명 간의 키워드 겹침 비율로 차등 할당.
    겹침이 없는 법령도 기본으로 포함하되, 기사와 관련도가 높은 법령을 우선 배치.
    """
    if not related_laws:
        return articles

    law_names = [law.law_name for law in related_laws]
    # 첫 번째 법률의 issue_label을 기본 legal_issue_label로 사용
    default_issue_label = next(
        (law.issue_label for law in related_laws if law.issue_label),
        None,
    )

    enriched: list[NewsArticle] = []
    for article in articles:
        # 기사별 법령 관련도 차등화: title+snippet 내 법령명 키워드 겹침
        article_text = (article.title + " " + article.snippet).lower()
        matched_laws: list[str] = []
        unmatched_laws: list[str] = []

        for name in law_names:
            # 법령명에서 핵심 토큰 추출 (조사/접미사 제거)
            name_lower = name.lower()
            if name_lower in article_text or name_lower.replace(" ", "") in article_text.replace(" ", ""):
                matched_laws.append(name)
            else:
                # 법령명 토큰 분리 후 부분 매칭
                tokens = [t for t in name_lower.split() if len(t) >= 2]
                hits = sum(1 for t in tokens if t in article_text) if tokens else 0
                if tokens and hits / len(tokens) >= 0.5:
                    matched_laws.append(name)
                else:
                    unmatched_laws.append(name)

        # 매칭된 법령 우선 + 나머지도 포함 (키워드 기반 검색이므로)
        article_laws = matched_laws + unmatched_laws

        # legal_issue_label: 매칭된 법령이 있으면 해당 법률의 issue_label 사용
        issue_label = default_issue_label
        if matched_laws:
            matched_brief = next(
                (law for law in related_laws if law.law_name == matched_laws[0]),
                None,
            )
            if matched_brief and matched_brief.issue_label:
                issue_label = matched_brief.issue_label

        enriched.append(article.model_copy(update={
            "related_laws": article_laws,
            "legal_issue_label": issue_label,
        }))

    logger.info(
        "기사 enrichment 완료: keyword='%s', articles=%d, laws=%d",
        keyword[:20], len(enriched), len(law_names),
    )
    return enriched


async def _search_related_laws(
    keyword: str,
    max_results: int = 2,
) -> list[RelatedLawBrief]:
    """경량 법률 enrichment: LanceDB 벡터 검색으로 관련 법령 조회

    LLM 호출 없이 벡터 유사도 검색만 수행하여 응답 지연을 최소화한다.
    """
    try:
        from app.services.rag.retrieval import _search_vector_ids

        results: list[dict[str, Any]] = await asyncio.to_thread(
            _search_vector_ids,
            query=keyword,
            n_results=max_results * 2,
            doc_type="law",
        )

        seen_names: set[str] = set()
        laws: list[RelatedLawBrief] = []

        for doc in results:
            metadata = doc.get("metadata", {})
            law_name = metadata.get("case_name", "").strip()
            if not law_name or law_name in seen_names:
                continue
            seen_names.add(law_name)
            laws.append(RelatedLawBrief(
                law_name=law_name,
                issue_label=metadata.get("data_type", ""),
            ))
            if len(laws) >= max_results:
                break

        logger.info("경량 법률 enrichment '%s': %d건", keyword[:20], len(laws))
        return laws
    except Exception:
        logger.warning("경량 법률 enrichment 실패, 빈 리스트 반환", exc_info=True)
        return []


def _scored_to_item(kw: ScoredKeyword) -> KeywordItem:
    """내부 ScoredKeyword → API 응답용 KeywordItem 변환"""
    return KeywordItem(
        id=kw.id,
        keyword=kw.keyword,
        context=kw.context,
        scores=KeywordScoreSchema(
            virality=kw.scores.virality,
            social_impact=kw.scores.social_impact,
            legal_relevance=kw.scores.legal_relevance,
            content_fitness=kw.scores.content_fitness,
        ),
        total_score=kw.total_score,
        rank=kw.rank,
        score_reason=kw.score_reason,
        confidence=kw.confidence,
    )
