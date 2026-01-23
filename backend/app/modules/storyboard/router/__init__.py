"""스토리보드 모듈 - 사건 타임라인 시각화 API"""
import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

logger = logging.getLogger(__name__)

from ..schema import (
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
    TimelineItem,
    TranscribeResponse,
    ValidateTimelineRequest,
)
from ..service import extract_timeline_from_text, validate_timeline_data
from ..service.stt import transcribe_audio
from ..service.vision import analyze_image
from ..service.image_generation import generate_image, generate_image_fallback
from ..service.video_generation import generate_video
from ..service.job_manager import (
    job_manager,
    run_batch_image_generation,
    JobStatus,
)

router = APIRouter()


@router.post("/extract", response_model=ExtractTimelineResponse)
async def extract_timeline(request: ExtractTimelineRequest):
    """
    텍스트에서 타임라인 자동 추출

    OpenAI API를 사용하여 사건 내용에서 시간순 이벤트를 추출합니다.

    - **text**: 사건 내용 텍스트 (최소 10자)
    """
    try:
        result = await extract_timeline_from_text(request.text)
        return result
    except Exception as e:
        logger.error(f"타임라인 추출 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="타임라인 추출 중 오류가 발생했습니다")


@router.post("/validate")
async def validate_timeline(request: ValidateTimelineRequest):
    """
    가져온 JSON 데이터 유효성 검사

    클라이언트에서 파일을 로드한 후 서버에서 스키마 유효성을 검증합니다.
    """
    try:
        is_valid = validate_timeline_data(request.timeline.model_dump())
        return {
            "valid": is_valid,
            "message": "유효한 타임라인 데이터입니다" if is_valid else "잘못된 형식입니다",
        }
    except Exception as e:
        logger.warning(f"유효성 검사 실패: {e}")
        raise HTTPException(status_code=400, detail="유효성 검사에 실패했습니다")


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio_endpoint(
    audio: UploadFile = File(..., description="음성 파일 (wav, mp3, webm, m4a)"),
    language: str = Form(default="ko", description="언어 코드"),
):
    """
    음성 파일을 텍스트로 변환 (STT)

    OpenAI Whisper API를 사용하여 음성을 텍스트로 변환합니다.
    """
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
async def analyze_image_endpoint(
    image: UploadFile = File(..., description="이미지 파일 (jpg, png, gif, webp)"),
    context: str = Form(default="", description="추가 컨텍스트 설명"),
):
    """
    이미지 분석을 통한 타임라인 추출

    Gemini Vision API를 사용하여 문서/스크린샷에서 타임라인을 추출합니다.
    """
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
async def generate_image_endpoint(request: GenerateImageRequest):
    """
    타임라인 항목에 대한 스토리보드 이미지 생성

    Google Gemini 2.0 Flash를 사용하여 스토리보드 스타일 이미지를 생성합니다.
    확장 필드(장소, 시간대, 참여자 역할, 분위기)가 있으면 더 상세한 이미지를 생성합니다.
    """
    try:
        result = await generate_image(
            item_id=request.item_id,
            title=request.title,
            description=request.description,
            participants=request.participants,
            location=request.location,
            time_of_day=request.time_of_day,
            participants_detailed=request.participants_detailed,
            mood=request.mood,
        )

        if result["success"]:
            return GenerateImageResponse(
                success=True,
                image_url=result["image_url"],
                image_prompt=result["image_prompt"],
            )
        else:
            fallback_result = await generate_image_fallback(
                item_id=request.item_id,
                title=request.title,
                description=request.description,
                participants=request.participants,
                location=request.location,
                time_of_day=request.time_of_day,
                participants_detailed=request.participants_detailed,
                mood=request.mood,
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
async def generate_images_batch_endpoint(request: GenerateImagesBatchRequest):
    """
    여러 타임라인 항목에 대한 스토리보드 이미지 일괄 생성

    비동기로 처리되며, job_id를 통해 진행 상태를 확인할 수 있습니다.
    """
    if not request.items:
        raise HTTPException(
            status_code=400,
            detail="최소 1개 이상의 타임라인 항목이 필요합니다",
        )

    try:
        # 작업 생성
        job_id = job_manager.create_job(total_steps=len(request.items))

        # 백그라운드에서 실행
        items_dict = [item.model_dump() for item in request.items]
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
async def get_job_status_sse(job_id: str):
    """
    작업 진행 상태 SSE 스트림

    Server-Sent Events를 통해 실시간으로 작업 진행 상태를 전송합니다.
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다")

    async def event_generator():
        async for progress in job_manager.subscribe(job_id):
            yield {
                "event": "progress",
                "data": json.dumps(progress.model_dump(), ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    작업 상태 조회 (폴링용)

    SSE 대신 폴링 방식으로 작업 상태를 조회합니다.
    """
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


@router.post("/generate-video", response_model=GenerateVideoResponse)
async def generate_video_endpoint(request: GenerateVideoRequest):
    """
    이미지들을 결합하여 영상 생성

    moviepy를 사용하여 여러 이미지를 30초 영상으로 변환합니다.
    """
    if len(request.image_urls) < 2:
        raise HTTPException(
            status_code=400,
            detail="최소 2개 이상의 이미지가 필요합니다",
        )

    try:
        result = await generate_video(
            timeline_id=request.timeline_id,
            image_urls=request.image_urls,
            duration_per_image=request.duration_per_image,
            transition=request.transition.value,
            transition_duration=request.transition_duration,
            resolution=tuple(request.resolution),
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
