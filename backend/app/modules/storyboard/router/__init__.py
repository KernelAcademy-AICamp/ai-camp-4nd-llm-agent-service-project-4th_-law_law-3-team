"""스토리보드 모듈 - 사건 타임라인 시각화 API"""
import asyncio
import json
import logging
import uuid as _uuid_module
from typing import Any, AsyncGenerator

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from sse_starlette.sse import EventSourceResponse

from app.core.rate_limit import AI_RATE_LIMIT, limiter

from ..schema import (
    AnalyzeBatchResponse,
    AnalyzeImageResponse,
    ExtractTimelineRequest,
    ExtractTimelineResponse,
    GenerateImageRequest,
    GenerateImageResponse,
    GenerateImagesBatchRequest,
    GenerateImagesBatchResponse,
    GenerateVideoRequest,
    GenerateVideoResponse,
    JobStatusResponse,
    MergeTimelineResponse,
    TimelineItem,
    TranscribeResponse,
    ValidateTimelineRequest,
    ValidateTimelineResponse,
)
from ..service import extract_timeline_from_text, validate_timeline_data
from ..service.batch_analyzer import BatchAnalyzer
from ..service.file_validation import FileValidationGate
from ..service.image_generation import generate_image, generate_image_fallback
from ..service.job_manager import (
    job_manager,
    run_batch_image_generation,
)
from ..service.stt import transcribe_audio
from ..service.timeline_merger import TimelineMerger
from ..service.video_generation import generate_video
from ..service.vision import analyze_image

_file_validation_gate = FileValidationGate()
_batch_analyzer = BatchAnalyzer(job_manager)
_timeline_merger = TimelineMerger()

# SEC-11: 인메모리 증거 메타데이터 저장소 (evidence_id → EvidenceFile dict)
_evidence_store: dict[str, dict[str, Any]] = {}

logger = logging.getLogger(__name__)

# 파일 업로드 크기 제한
MAX_AUDIO_SIZE = 25 * 1024 * 1024  # 25MB
MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20MB

router = APIRouter()


@router.post("/extract", response_model=ExtractTimelineResponse)
@limiter.limit(AI_RATE_LIMIT)
async def extract_timeline(request: Request, body: ExtractTimelineRequest) -> ExtractTimelineResponse:
    """
    텍스트에서 타임라인 자동 추출

    OpenAI API를 사용하여 사건 내용에서 시간순 이벤트를 추출합니다.

    - **text**: 사건 내용 텍스트 (최소 10자)
    """
    try:
        result = await extract_timeline_from_text(body.text)
        return result
    except Exception as e:
        logger.error(f"타임라인 추출 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="타임라인 추출 중 오류가 발생했습니다")


@router.post("/validate", response_model=ValidateTimelineResponse)
@limiter.limit(AI_RATE_LIMIT)
async def validate_timeline(request: Request, body: ValidateTimelineRequest) -> ValidateTimelineResponse:
    """
    가져온 JSON 데이터 유효성 검사

    클라이언트에서 파일을 로드한 후 서버에서 스키마 유효성을 검증합니다.
    """
    try:
        is_valid = validate_timeline_data(body.timeline.model_dump())
        return ValidateTimelineResponse(
            valid=is_valid,
            message="유효한 타임라인 데이터입니다" if is_valid else "잘못된 형식입니다",
        )
    except Exception as e:
        logger.warning(f"유효성 검사 실패: {e}")
        raise HTTPException(status_code=400, detail="유효성 검사에 실패했습니다")


@router.post("/transcribe", response_model=TranscribeResponse)
@limiter.limit(AI_RATE_LIMIT)
async def transcribe_audio_endpoint(
    request: Request,
    audio: UploadFile = File(..., description="음성 파일 (wav, mp3, webm, m4a)"),
    language: str = Form(default="ko", description="언어 코드"),
) -> TranscribeResponse:
    """
    음성 파일을 텍스트로 변환 (STT)

    OpenAI Whisper API를 사용하여 음성을 텍스트로 변환합니다.
    """
    if audio.size is not None and audio.size > MAX_AUDIO_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"음성 파일 크기가 제한을 초과했습니다 (최대 {MAX_AUDIO_SIZE // (1024 * 1024)}MB)",
        )

    try:
        text = await transcribe_audio(
            audio_file=audio.file,
            filename=audio.filename or "audio.wav",
            language=language,
        )
        return TranscribeResponse(success=True, text=text)
    except ValueError as e:
        return TranscribeResponse(success=False, error=str(e))
    except Exception as e:
        logger.error(f"음성 변환 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="음성 변환 중 오류가 발생했습니다")


@router.post("/analyze-image", response_model=AnalyzeImageResponse)
@limiter.limit(AI_RATE_LIMIT)
async def analyze_image_endpoint(
    request: Request,
    image: UploadFile = File(..., description="이미지 파일 (jpg, png, gif, webp)"),
    context: str = Form(default="", description="추가 컨텍스트 설명"),
) -> AnalyzeImageResponse:
    """
    이미지 분석을 통한 타임라인 추출

    Gemini Vision API를 사용하여 문서/스크린샷에서 타임라인을 추출합니다.
    """
    if image.size is not None and image.size > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"이미지 파일 크기가 제한을 초과했습니다 (최대 {MAX_IMAGE_SIZE // (1024 * 1024)}MB)",
        )

    try:
        result = await analyze_image(
            image_file=image.file,
            filename=image.filename or "image.png",
            additional_context=context,
        )

        if result["success"]:
            timeline_items = [
                TimelineItem(**item) for item in result["timeline"]
            ]
            return AnalyzeImageResponse(
                success=True,
                timeline=timeline_items,
                summary=result.get("summary"),
            )
        else:
            error_msg = result.get("error") or result.get("message") or "이미지 분석 실패"
            return AnalyzeImageResponse(success=False, error=error_msg)
    except ValueError as e:
        logger.warning(f"타임라인 데이터 변환 실패: {e}")
        return AnalyzeImageResponse(success=False, error="타임라인 데이터 변환에 실패했습니다")
    except Exception as e:
        logger.error(f"이미지 분석 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="이미지 분석 중 오류가 발생했습니다")


@router.post("/generate-image", response_model=GenerateImageResponse)
@limiter.limit(AI_RATE_LIMIT)
async def generate_image_endpoint(request: Request, body: GenerateImageRequest) -> GenerateImageResponse:
    """
    타임라인 항목에 대한 스토리보드 이미지 생성

    Google Gemini 2.0 Flash를 사용하여 스토리보드 스타일 이미지를 생성합니다.
    확장 필드(장소, 시간대, 참여자 역할, 분위기)가 있으면 더 상세한 이미지를 생성합니다.
    """
    try:
        _uuid_module.UUID(body.item_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="item_id가 유효한 UUID 형식이 아닙니다")

    try:
        result = await generate_image(
            item_id=body.item_id,
            title=body.title,
            description=body.description,
            participants=body.participants,
            location=body.location,
            time_of_day=body.time_of_day,
            participants_detailed=body.participants_detailed,
            mood=body.mood,
        )

        if result["success"]:
            return GenerateImageResponse(
                success=True,
                image_url=result["image_url"],
                image_prompt=result["image_prompt"],
            )
        else:
            fallback_result = await generate_image_fallback(
                item_id=body.item_id,
                title=body.title,
                description=body.description,
                participants=body.participants,
                location=body.location,
                time_of_day=body.time_of_day,
                participants_detailed=body.participants_detailed,
                mood=body.mood,
            )
            return GenerateImageResponse(
                success=True,
                image_url=fallback_result["image_url"],
                image_prompt=fallback_result["image_prompt"],
                error="이미지 생성 실패, 플레이스홀더 이미지 사용",
            )
    except Exception as e:
        logger.error(f"이미지 생성 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="이미지 생성 중 오류가 발생했습니다")


@router.post("/generate-images-batch", response_model=GenerateImagesBatchResponse)
@limiter.limit(AI_RATE_LIMIT)
async def generate_images_batch_endpoint(request: Request, body: GenerateImagesBatchRequest) -> GenerateImagesBatchResponse:
    """
    여러 타임라인 항목에 대한 스토리보드 이미지 일괄 생성

    비동기로 처리되며, job_id를 통해 진행 상태를 확인할 수 있습니다.
    """
    if not body.items:
        raise HTTPException(
            status_code=400,
            detail="최소 1개 이상의 타임라인 항목이 필요합니다",
        )

    try:
        # 작업 생성
        job_id = job_manager.create_job(total_steps=len(body.items))

        # 백그라운드에서 실행
        items_dict = [item.model_dump() for item in body.items]
        asyncio.create_task(
            run_batch_image_generation(
                job_id=job_id,
                items=items_dict,
                generate_fn=generate_image,
            )
        )

        return GenerateImagesBatchResponse(success=True, job_id=job_id)
    except Exception as e:
        logger.error(f"일괄 생성 시작 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="일괄 이미지 생성을 시작할 수 없습니다")


@router.get("/jobs/{job_id}/status")
async def get_job_status_sse(job_id: str) -> EventSourceResponse:
    """
    작업 진행 상태 SSE 스트림

    Server-Sent Events를 통해 실시간으로 작업 진행 상태를 전송합니다.
    """
    try:
        _uuid_module.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="job_id가 유효한 UUID 형식이 아닙니다")

    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다")

    async def event_generator() -> AsyncGenerator[dict[str, str], None]:
        async for progress in job_manager.subscribe(job_id):
            yield {
                "event": "progress",
                "data": json.dumps(progress.model_dump(), ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """
    작업 상태 조회 (폴링용)

    SSE 대신 폴링 방식으로 작업 상태를 조회합니다.
    """
    try:
        _uuid_module.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="job_id가 유효한 UUID 형식이 아닙니다")

    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다")

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status.value,
        progress=job.progress,
        current_step=job.current_step,
        total_steps=job.total_steps,
        message=job.message,
        result=job.result,
        error=job.error,
    )


@router.post("/analyze-batch", response_model=AnalyzeBatchResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit(AI_RATE_LIMIT)
async def analyze_batch_endpoint(
    request: Request,
    files: list[UploadFile] = File(..., description="증거 파일 (최대 10개)"),
    context: str = Form(default="", description="추가 컨텍스트"),
    session_id: str = Form(default="", description="세션 ID (증거 소유권 추적)"),
) -> AnalyzeBatchResponse:
    """
    다중 증거 파일 일괄 분석 → 통합 타임라인 생성

    1. FileValidationGate 통과
    2. BatchAnalyzeJob 생성 (job_id 반환)
    3. 백그라운드에서 asyncio.create_task 실행
    4. SSE /jobs/{id}/status 로 진행 상태 확인

    에러 코드:
    - 400: 파일 없음 / 개수 초과
    - 413: 파일 크기 초과
    - 415: 지원하지 않는 파일 형식
    - 422: 파일 검증 실패 (매직 넘버 불일치)
    """
    if not files:
        raise HTTPException(status_code=400, detail="파일을 하나 이상 업로드해야 합니다")

    try:
        validated_files = await _file_validation_gate.validate_and_prepare(
            files, session_id=session_id,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("파일 검증 실패: %s", exc, exc_info=True)
        raise HTTPException(status_code=422, detail="파일 검증 중 오류가 발생했습니다")

    job_id = job_manager.create_job(total_steps=len(validated_files))

    async def _run() -> None:
        try:
            result = await _batch_analyzer.analyze_batch(
                job_id=job_id,
                validated_files=validated_files,
                context=context,
                session_id=session_id,
            )
            # SEC-11: 증거 메타데이터를 인메모리 저장소에 보관
            for ev in result.evidence_files:
                _evidence_store[ev.evidence_id] = ev.model_dump()
            await job_manager.complete_job(
                job_id,
                result=result.model_dump(),
            )
        except Exception as exc:
            logger.error("배치 분석 실패 (job_id=%s): %s", job_id, exc, exc_info=True)
            await job_manager.fail_job(job_id, error=str(exc))

    asyncio.create_task(_run())
    return AnalyzeBatchResponse(success=True, job_id=job_id)


@router.post("/merge", response_model=MergeTimelineResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit(AI_RATE_LIMIT)
async def merge_timeline_endpoint(
    request: Request,
    existing_timeline: str = Form(..., description="기존 타임라인 JSON (MergeTimelineRequest)"),
    files: list[UploadFile] = File(default=[], description="추가 증거 파일"),
    text: str = Form(default="", description="추가 텍스트 입력"),
) -> MergeTimelineResponse:
    """
    기존 타임라인에 새 증거/텍스트 병합

    1. 기존 타임라인 JSON 파싱
    2. 신규 입력 분석 (파일 + 텍스트)
    3. TimelineMerger 3단계 병합 (날짜 후보 → LLM 중복 감지 → 적용)
    4. 병합 결과 + 보고서 반환
    """
    import json

    from ..schema import MergeTimelineRequest

    try:
        request_data = json.loads(existing_timeline)
        merge_request = MergeTimelineRequest(**request_data)
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"기존 타임라인 JSON 파싱 실패: {exc}",
        )

    # 신규 파일 분석
    new_items: list[TimelineItem] = []
    new_evidence = []

    if files:
        try:
            validated_files = await _file_validation_gate.validate_and_prepare(files)
            temp_job_id = job_manager.create_job(total_steps=len(validated_files))
            result = await _batch_analyzer.analyze_batch(
                job_id=temp_job_id,
                validated_files=validated_files,
                context=text,
            )
            new_items = result.timeline_items
            new_evidence = result.evidence_files
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("병합용 파일 분석 실패: %s", exc, exc_info=True)
            return MergeTimelineResponse(
                success=False,
                error=f"신규 파일 분석 실패: {exc}",
            )
    elif text:
        try:
            extract_result = await extract_timeline_from_text(text)
            if extract_result.success:
                new_items = extract_result.timeline
        except Exception as exc:
            logger.error("병합용 텍스트 분석 실패: %s", exc, exc_info=True)

    # TimelineMerger 3단계 병합
    response = await _timeline_merger.merge(
        existing_items=list(merge_request.existing_items),
        new_items=new_items,
        existing_evidence=list(merge_request.existing_evidence),
        new_evidence=new_evidence,
    )
    # SEC-11: 병합 결과의 증거 메타데이터를 인메모리 저장소에 보관
    if response.success:
        for ev in response.merged_evidence:
            _evidence_store[ev.evidence_id] = ev.model_dump()
    return response


@router.get("/evidence/{evidence_id}")
async def get_evidence_file(evidence_id: str) -> dict[str, Any]:
    """
    증거 파일 메타데이터 조회

    SEC-03: UUID v4 형식 검증
    SEC-11: 인메모리 증거 저장소에서 메타데이터 반환
    - evidence_id는 UUID v4 형식 필수
    - 배치 분석/병합 시 자동 저장된 메타데이터 반환
    - 파일 바이너리 스토리지 연동은 Phase 3+
    """
    try:
        _uuid_module.UUID(evidence_id)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="evidence_id가 유효한 UUID v4 형식이 아닙니다",
        )
    evidence = _evidence_store.get(evidence_id)
    if evidence is None:
        raise HTTPException(
            status_code=404,
            detail="증거 파일을 찾을 수 없습니다",
        )
    return evidence


@router.post("/generate-video", response_model=GenerateVideoResponse)
@limiter.limit(AI_RATE_LIMIT)
async def generate_video_endpoint(request: Request, body: GenerateVideoRequest) -> GenerateVideoResponse:
    """
    이미지들을 결합하여 영상 생성

    moviepy를 사용하여 여러 이미지를 30초 영상으로 변환합니다.
    """
    if len(body.image_urls) < 2:
        raise HTTPException(
            status_code=400,
            detail="최소 2개 이상의 이미지가 필요합니다",
        )

    try:
        result = await generate_video(
            timeline_id=body.timeline_id,
            image_urls=body.image_urls,
            duration_per_image=body.duration_per_image,
            transition=body.transition.value,
            transition_duration=body.transition_duration,
            resolution=(body.resolution[0], body.resolution[1]),
        )

        if result["success"]:
            return GenerateVideoResponse(
                success=True,
                video_url=result["video_url"],
                duration=result.get("duration"),
                image_count=result.get("image_count"),
            )
        else:
            return GenerateVideoResponse(
                success=False,
                error=result.get("error", "영상 생성 실패"),
            )
    except Exception as e:
        logger.error(f"영상 생성 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="영상 생성 중 오류가 발생했습니다")
