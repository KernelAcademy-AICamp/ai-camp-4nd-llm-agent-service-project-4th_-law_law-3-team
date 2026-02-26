"""
콘텐츠 마케팅 모듈 라우터

Design 문서 Section 4 기반 엔드포인트 구현
v2.0: Persona API 5개 추가 (§4.1)
v2.1: Keyword Flow API 2개 추가 (키워드 수집 + 뉴스 검색)
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.modules.content_marketing.schema import (
    KeywordCollectRequest,
    KeywordCollectResponse,
    KeywordNewsRequest,
    KeywordNewsResponse,
    LawyerPersona,
    MetadataRequest,
    PersonaAnalysisRequest,
    PersonaFeedbackRequest,
    PersonaOnboardingRequest,
    PersonaUpdateRequest,
    ScriptMetadata,
    ScriptRequest,
    TrendDetailResponse,
    TrendRequest,
    TrendResponse,
)
from app.services.service_function.content_marketing_service import (
    KeywordNotFoundError,
    analyze_persona,
    collect_keywords,
    collect_keywords_stream,
    collect_trends,
    create_persona_from_onboarding,
    generate_metadata,
    generate_script_stream,
    get_current_persona,
    get_trend_detail,
    save_persona_feedback,
    search_keyword_news,
    update_current_persona,
)
from app.tools.trend.rate_limiter import RateLimitExceededError

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Persona API (v2.0 NEW, §4.1) ──
# Note: Auth Dependency(get_current_user)는 인증 시스템 구현 시 추가.
# 현재는 임시 user_id 사용 (Red Team [심각 1] H-01 플레이스홀더)
TEMP_USER_ID = "temp_user_001"


@router.post("/persona/analyze", response_model=LawyerPersona)
async def analyze_persona_endpoint(
    request: PersonaAnalysisRequest,
    db: AsyncSession = Depends(get_db),
) -> LawyerPersona:
    """Track 1: 대화 이력 기반 자동 페르소나 분석"""
    # TODO: user_id = Depends(get_current_user).id (Auth 구현 시 교체)
    user_id = TEMP_USER_ID
    return await analyze_persona(db, user_id, request)


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
) -> StreamingResponse:
    """키워드 수집 SSE 스트리밍 (§7.4)

    단계별 진행 상태를 SSE로 실시간 전송.
    기존 POST /keywords/collect의 스트리밍 버전.
    """
    user_id = TEMP_USER_ID
    request = KeywordCollectRequest(max_keywords=max_keywords)
    return StreamingResponse(
        collect_keywords_stream(request, user_id=user_id),
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
