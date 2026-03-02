"""
콘텐츠 마케팅 모듈 라우터

Design 문서 Section 4 기반 엔드포인트 구현
v2.0: Persona API 5개 추가 (§4.1)
v2.1: Keyword Flow API 2개 추가 (키워드 수집 + 뉴스 검색)
v3.0: Webtoon Storyboard API 4개 추가
"""

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.modules.content_marketing.schema import (
    ChatHistoryCountResponse,
    KeywordCollectRequest,
    KeywordCollectResponse,
    KeywordNewsRequest,
    KeywordNewsResponse,
    LawyerPersona,
    MetadataRequest,
    PersonaAnalysisRequest,
    PersonaAnalysisResponse,
    PersonaFeedbackRequest,
    PersonaOnboardingRequest,
    PersonaUpdateRequest,
    ScriptMetadata,
    ScriptRequest,
    TimeRange,
    TrendCategory,
    TrendDetailResponse,
    TrendRequest,
    TrendResponse,
    WebtoonGenerateRequest,
    WebtoonJobResponse,
    WebtoonJobStatusResponse,
    WebtoonRegenerateRequest,
    WebtoonStreamEvent,
)
from app.services.service_function.content_marketing_service import (
    KeywordNotFoundError,
    analyze_persona,
    clear_keyword_cache,
    collect_keywords,
    collect_keywords_stream,
    collect_trends,
    create_persona_from_onboarding,
    generate_metadata,
    generate_script_stream,
    get_chat_history_count,
    get_current_persona,
    get_trend_detail,
    save_persona_feedback,
    search_keyword_news,
    update_current_persona,
)
from app.services.service_function.webtoon_service import (
    regenerate_single_panel,
    run_webtoon_pipeline,
    webtoon_job_manager,
    webtoon_sse_generator,
)
from app.tools.trend.rate_limiter import RateLimitExceededError

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Persona API (v2.0 NEW, §4.1) ──
# Note: Auth Dependency(get_current_user)는 인증 시스템 구현 시 추가.
# 현재는 임시 user_id 사용 (Red Team [심각 1] H-01 플레이스홀더)
TEMP_USER_ID = "temp_user_001"


@router.post("/persona/analyze", response_model=PersonaAnalysisResponse)
async def analyze_persona_endpoint(
    request: PersonaAnalysisRequest,
    db: AsyncSession = Depends(get_db),
) -> PersonaAnalysisResponse:
    """Track 1: 대화 이력 기반 자동 페르소나 분석"""
    # TODO: user_id = Depends(get_current_user).id (Auth 구현 시 교체)
    user_id = TEMP_USER_ID
    persona, insights = await analyze_persona(db, user_id, request)
    return PersonaAnalysisResponse(persona=persona, analysis_insights=insights)


@router.post("/persona/onboarding", response_model=LawyerPersona)
async def onboarding_endpoint(
    request: PersonaOnboardingRequest,
    db: AsyncSession = Depends(get_db),
) -> LawyerPersona:
    """Track 2: 온보딩 결과로 페르소나 생성"""
    user_id = TEMP_USER_ID
    return await create_persona_from_onboarding(db, user_id, request)


@router.get("/persona/current")
async def get_persona_endpoint(
    db: AsyncSession = Depends(get_db),
) -> LawyerPersona | None:
    """현재 페르소나 조회"""
    user_id = TEMP_USER_ID
    return await get_current_persona(db, user_id)


@router.put("/persona/update", response_model=LawyerPersona)
async def update_persona_endpoint(
    request: PersonaUpdateRequest,
    db: AsyncSession = Depends(get_db),
) -> LawyerPersona:
    """페르소나 부분 수정"""
    user_id = TEMP_USER_ID
    result = await update_current_persona(db, user_id, request)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail="페르소나가 존재하지 않습니다. 먼저 페르소나를 생성해주세요.",
        )
    return result


@router.post("/persona/feedback", status_code=201)
async def feedback_endpoint(
    request: PersonaFeedbackRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    """대본 생성 후 피드백 저장"""
    await save_persona_feedback(db, request)
    return {"status": "saved"}


@router.get(
    "/persona/chat-history-count",
    response_model=ChatHistoryCountResponse,
)
async def chat_history_count_endpoint(
    db: AsyncSession = Depends(get_db),
) -> ChatHistoryCountResponse:
    """분석 가능한 대화 이력 건수 반환"""
    user_id = TEMP_USER_ID  # TODO: Auth
    count = await get_chat_history_count(db, user_id)
    return ChatHistoryCountResponse(
        count=count,
        has_sufficient_history=count >= 5,
        oldest_date=None,  # TODO: 실제 조회
    )


# ── Trend API ──


@router.post("/trends", response_model=TrendResponse)
async def get_trends(
    request: TrendRequest,
    db: AsyncSession = Depends(get_db),
) -> TrendResponse:
    """트렌드 이슈 수집 및 스코어링 (v2.0: persona_id로 fitness_score 계산)"""
    return await collect_trends(request, db=db)


@router.get("/trends/{trend_id}", response_model=TrendDetailResponse)
async def get_trend_detail_view(trend_id: str) -> TrendDetailResponse:
    """트렌드 이슈 상세 조회"""
    result = await get_trend_detail(trend_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail="해당 트렌드 이슈를 찾을 수 없습니다. 트렌드를 다시 조회해주세요.",
        )
    return result


# ── Script API ──


@router.post("/script/generate")
async def generate_script(
    request: ScriptRequest,
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """대본 SSE 스트리밍 생성 (v2.0: persona_id → PersonaTone 해석)"""
    return StreamingResponse(
        generate_script_stream(request, db=db),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.post("/script/metadata", response_model=ScriptMetadata)
async def generate_script_metadata(
    request: MetadataRequest,
) -> ScriptMetadata:
    """대본 메타데이터 재생성"""
    return await generate_metadata(request)


# ── Keyword Flow API (v2.1 NEW) ──


@router.get("/keywords/collect/stream")
async def collect_keywords_stream_endpoint(
    max_keywords: int = 10,
    time_range: TimeRange = TimeRange.HOURS_48,
    category: TrendCategory = TrendCategory.ALL,
    force_refresh: bool = False,
    persona_id: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """키워드 수집 SSE 스트리밍 (§7.4)

    단계별 진행 상태를 SSE로 실시간 전송.
    기존 POST /keywords/collect의 스트리밍 버전.
    category: 법률 카테고리 (기본값: all)
    persona_id: 페르소나 ID (선택, 키워드 개인화에 사용)
    """
    user_id = TEMP_USER_ID
    request = KeywordCollectRequest(
        max_keywords=max_keywords,
        time_range=time_range,
        category=category,
        persona_id=persona_id,
    )
    return StreamingResponse(
        collect_keywords_stream(request, user_id=user_id, force_refresh=force_refresh, db=db),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.post("/keywords/collect", response_model=KeywordCollectResponse)
async def collect_keywords_endpoint(
    request: KeywordCollectRequest,
) -> KeywordCollectResponse:
    """커뮤니티 트렌드 키워드 수집 + 4차원 스코어링

    Step 1: Tavily + Naver로 커뮤니티 인기글 수집
    Step 2: LLM으로 키워드 추출 + 점수 산출
    """
    user_id = TEMP_USER_ID
    try:
        return await collect_keywords(request, user_id=user_id)
    except RateLimitExceededError as exc:
        raise HTTPException(
            status_code=429,
            detail=exc.args[0],
            headers={"Retry-After": str(exc.retry_after_seconds)},
        )


@router.delete("/keywords/cache")
async def clear_keywords_cache_endpoint() -> dict[str, int | str]:
    """키워드/뉴스/트렌드 캐시 전체 클리어 (Rate Limit: 5 req/min)"""
    user_id = TEMP_USER_ID
    try:
        cleared = clear_keyword_cache(user_id)
        return {"status": "cleared", **cleared}
    except RateLimitExceededError as exc:
        raise HTTPException(
            status_code=429,
            detail=exc.args[0],
            headers={"Retry-After": str(exc.retry_after_seconds)},
        )


@router.post(
    "/keywords/{keyword_id}/news",
    response_model=KeywordNewsResponse,
)
async def search_keyword_news_endpoint(
    keyword_id: str,
    request: KeywordNewsRequest,
) -> KeywordNewsResponse:
    """선택한 키워드로 뉴스 기사 검색

    캐시된 키워드 목록에서 keyword_id로 키워드를 찾고,
    해당 키워드로 뉴스 소스를 병렬 검색합니다.
    """
    user_id = TEMP_USER_ID
    try:
        return await search_keyword_news(keyword_id, request, user_id=user_id)
    except KeywordNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RateLimitExceededError as exc:
        raise HTTPException(
            status_code=429,
            detail=exc.args[0],
            headers={"Retry-After": str(exc.retry_after_seconds)},
        )


# ── Webtoon Storyboard API (v3.0 NEW) ──


@router.post("/script/webtoon", response_model=WebtoonJobResponse)
async def create_webtoon_job(
    request: WebtoonGenerateRequest,
) -> WebtoonJobResponse:
    """웹툰 스토리보드 Job 생성 → 백그라운드 파이프라인 시작"""
    job_id = webtoon_job_manager.create_job(total_steps=1)
    asyncio.create_task(run_webtoon_pipeline(job_id, request))
    return WebtoonJobResponse(
        job_id=job_id,
        status="accepted",
        estimated_panels=request.panel_count or 10,
    )


@router.get("/script/webtoon/{job_id}/stream")
async def stream_webtoon_progress(job_id: str) -> StreamingResponse:
    """SSE로 패널별 생성 진행률 스트리밍"""
    if not webtoon_job_manager.get_job(job_id):
        raise HTTPException(status_code=404, detail="Job을 찾을 수 없습니다.")
    return StreamingResponse(
        webtoon_sse_generator(job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.get(
    "/script/webtoon/{job_id}",
    response_model=WebtoonJobStatusResponse,
)
async def get_webtoon_job_status(job_id: str) -> WebtoonJobStatusResponse:
    """Job 상태 폴링 조회"""
    job = webtoon_job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job을 찾을 수 없습니다.")
    return WebtoonJobStatusResponse(
        job_id=job_id,
        status=job.status.value,
        progress=job.progress,
        panels=job.result.get("panels", []) if job.result else [],
        error=job.error,
    )


@router.post("/script/webtoon/{job_id}/regenerate")
async def regenerate_webtoon_panel(
    job_id: str,
    request: WebtoonRegenerateRequest,
) -> WebtoonStreamEvent:
    """개별 패널 이미지 재생성"""
    job = webtoon_job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job을 찾을 수 없습니다.")
    return await regenerate_single_panel(job_id, request.panel_number)
