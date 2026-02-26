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
    KeywordCollectRequest,
    KeywordCollectResponse,
    KeywordItem,
    KeywordNewsRequest,
    KeywordNewsResponse,
    KeywordScoreSchema,
    KeywordStreamEvent,
    LawyerPersona,
    MetadataRequest,
    PersonaAnalysisRequest,
    PersonaFeedbackRequest,
    PersonaOnboardingRequest,
    PersonaTone,
    PersonaUpdateRequest,
    RelatedLawBrief,
    ScriptMetadata,
    ScriptRequest,
    TrendDetailResponse,
    TrendRequest,
    TrendResponse,
)
from app.services.service_function import persona_db_service
from app.tools.persona.onboarding import OnboardingProcessor
from app.tools.trend.collector import TrendCollector
from app.tools.trend.models import ScoredKeyword
from app.tools.trend.rate_limiter import InMemoryRateLimiter
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

# Keyword Flow 캐시 (user_id → ScoredKeyword 리스트)
_keyword_cache: dict[str, tuple[list[ScoredKeyword], datetime]] = {}

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
) -> LawyerPersona:
    """Track 1: 대화 이력 기반 자동 분석"""
    from app.tools.persona.analyzer import (
        InsufficientHistoryError,
        LowConfidenceError,
        PersonaAnalyzer,
    )

    analyzer = PersonaAnalyzer()

    # TODO: 실제 대화 이력 조회 로직 구현
    # 현재는 빈 리스트 → InsufficientHistoryError 발생 → Track 2로 폴백
    messages: list[dict[str, str]] = []

    try:
        persona = await analyzer.analyze(
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

    return await persona_db_service.create_persona(db, persona)


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
    """
    from app.tools.script.generator import ScriptGenerator

    # persona_id → PersonaTone 해석
    persona_tone: PersonaTone | None = None
    if request.persona_id and db is not None:
        persona = await persona_db_service.get_persona_by_id(db, request.persona_id)
        if persona is not None:
            persona_tone = persona.preferred_tone

    generator = ScriptGenerator()

    async for event in generator.generate_stream(request, persona_tone=persona_tone):
        yield f"event: {event.event}\ndata: {event.model_dump_json()}\n\n"


async def generate_metadata(request: MetadataRequest) -> ScriptMetadata:
    """대본 메타데이터 LLM 생성"""
    from app.tools.script.generator import generate_metadata as _gen_metadata

    return await _gen_metadata(request)


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

    # 캐시 확인
    cache_key = f"{user_id}:keywords"
    cached = _keyword_cache.get(cache_key)
    if cached is not None:
        cached_keywords, cached_at = cached
        elapsed = (datetime.now(tz=timezone.utc) - cached_at).total_seconds()
        if elapsed < settings.KEYWORD_COLLECT_CACHE_TTL:
            logger.info("키워드 캐시 히트: user=%s", user_id)
            return KeywordCollectResponse(
                keywords=[_scored_to_item(kw) for kw in cached_keywords],
                total_count=len(cached_keywords),
                collected_at=cached_at,
                sources_used=["tavily", "naver"],
                cache_hit=True,
                prompt_version="1.0",
            )

    # 커뮤니티 수집 + 스코어링
    scored_keywords = await _collector.collect_community_keywords(
        community_domains=request.community_domains,
        max_keywords=request.max_keywords,
    )

    # 캐시 저장
    now = datetime.now(tz=timezone.utc)
    _keyword_cache[cache_key] = (scored_keywords, now)

    logger.info("키워드 수집 완료: user=%s, count=%d", user_id, len(scored_keywords))
    return KeywordCollectResponse(
        keywords=[_scored_to_item(kw) for kw in scored_keywords],
        total_count=len(scored_keywords),
        collected_at=now,
        sources_used=_collector.get_available_source_names_str(),
        cache_hit=False,
        prompt_version="1.0",
    )


_SSE_STREAM_TIMEOUT_SECONDS = 120  # SSE 최대 연결 시간 (DoS 방지)


async def collect_keywords_stream(
    request: KeywordCollectRequest,
    user_id: str,
) -> AsyncGenerator[str, None]:
    """키워드 수집 SSE 스트리밍 (§7.4)

    단계별로 진행 상황을 SSE 이벤트로 전송한다.
    tavily_start → tavily_done → llm_start → llm_done → scoring → done
    """
    def _sse(event: KeywordStreamEvent) -> str:
        return f"data: {event.model_dump_json()}\n\n"

    try:
        # Rate limit 검사
        _collect_limiter.check_and_record(user_id, bucket="collect")

        # 캐시 확인
        cache_key = f"{user_id}:keywords"
        cached = _keyword_cache.get(cache_key)
        if cached is not None:
            cached_keywords, cached_at = cached
            elapsed = (datetime.now(tz=timezone.utc) - cached_at).total_seconds()
            if elapsed < settings.KEYWORD_COLLECT_CACHE_TTL:
                response = KeywordCollectResponse(
                    keywords=[_scored_to_item(kw) for kw in cached_keywords],
                    total_count=len(cached_keywords),
                    collected_at=cached_at,
                    sources_used=["tavily", "naver"],
                    cache_hit=True,
                    prompt_version="1.0",
                )
                yield _sse(KeywordStreamEvent(
                    step="done", progress=100, message="캐시 결과",
                    count=len(cached_keywords), data=response,
                ))
                return

        # Step 1: 커뮤니티 수집 시작
        yield _sse(KeywordStreamEvent(
            step="tavily_start", progress=10, message="커뮤니티 인기글 수집 중...",
        ))

        from app.tools.trend.sources import SourceConfig

        domains = request.community_domains or settings.KEYWORD_COMMUNITY_DOMAINS
        tavily_config = SourceConfig(
            time_range="48h",
            max_results=(request.max_keywords or 10) * 3,
            search_query="사건 사고 논란 이슈",
            include_domains=domains,
        )
        fetch_tasks = [_collector._community_source.safe_fetch(
            "사건 사고 논란 이슈", tavily_config,
        )]

        from app.modules.content_marketing.schema import TrendSource as TrendSourceEnum
        naver_source = next(
            (s for s in _collector._news_sources
             if s.name == TrendSourceEnum.NAVER and s.is_available),
            None,
        )
        if naver_source:
            naver_config = SourceConfig(
                time_range="48h",
                max_results=(request.max_keywords or 10) * 2,
                search_query="사건 사고 논란 법률",
            )
            fetch_tasks.append(naver_source.safe_fetch("사건 사고 논란 법률", naver_config))

        # 타임아웃 적용 (DoS 방지)
        results = await asyncio.wait_for(
            asyncio.gather(*fetch_tasks),
            timeout=_SSE_STREAM_TIMEOUT_SECONDS,
        )
        community_items = [item for batch in results for item in batch]

        yield _sse(KeywordStreamEvent(
            step="tavily_done", progress=40, message=f"{len(community_items)}개 게시글 수집 완료",
            count=len(community_items),
        ))

        if not community_items:
            response = KeywordCollectResponse(
                keywords=[], total_count=0,
                collected_at=datetime.now(tz=timezone.utc),
                sources_used=_collector.get_available_source_names_str(),
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
        _keyword_cache[cache_key] = (scored_keywords, now)

        response = KeywordCollectResponse(
            keywords=[_scored_to_item(kw) for kw in scored_keywords],
            total_count=len(scored_keywords),
            collected_at=now,
            sources_used=_collector.get_available_source_names_str(),
            cache_hit=False,
            prompt_version="1.0",
        )

        yield _sse(KeywordStreamEvent(
            step="done", progress=100, message="완료",
            count=len(scored_keywords), data=response,
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
    """캐시에서 키워드 조회 + 뉴스 검색

    1. Rate limit 검사
    2. 캐시에서 keyword_id로 키워드 조회
    3. 뉴스 소스 병렬 검색
    """
    # Rate limit 검사
    _news_limiter.check_and_record(user_id, bucket="news")

    # 캐시에서 키워드 조회 (TTL 검증 포함)
    cache_key = f"{user_id}:keywords"
    cached = _keyword_cache.get(cache_key)
    if cached is None:
        raise KeywordNotFoundError("키워드 캐시가 만료되었습니다. 키워드를 다시 수집해주세요.")

    cached_keywords, cached_at = cached
    elapsed = (datetime.now(tz=timezone.utc) - cached_at).total_seconds()
    if elapsed >= settings.KEYWORD_COLLECT_CACHE_TTL:
        del _keyword_cache[cache_key]
        raise KeywordNotFoundError("키워드 캐시가 만료되었습니다. 키워드를 다시 수집해주세요.")
    target_keyword = next(
        (kw for kw in cached_keywords if kw.id == keyword_id), None,
    )
    if target_keyword is None:
        raise KeywordNotFoundError(f"키워드를 찾을 수 없습니다: {keyword_id}")

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
    )
    law_task = _safe_law_enrichment(target_keyword.keyword)

    (articles, sources_used), related_laws = await asyncio.gather(
        news_task, law_task,
    )

    return KeywordNewsResponse(
        keyword_id=keyword_id,
        keyword=target_keyword.keyword,
        articles=articles,
        related_laws=related_laws,
        total_count=len(articles),
        sources_used=sources_used,
        searched_at=datetime.now(tz=timezone.utc),
    )


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
