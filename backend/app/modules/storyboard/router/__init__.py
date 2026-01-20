"""스토리보드 모듈 - 사건 타임라인 시각화 API"""
from fastapi import APIRouter, HTTPException

from ..schema import (
    ExtractTimelineRequest,
    ExtractTimelineResponse,
    ValidateTimelineRequest,
)
from ..service import extract_timeline_from_text, validate_timeline_data

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
        raise HTTPException(status_code=500, detail=f"타임라인 추출 실패: {str(e)}")


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
        raise HTTPException(status_code=400, detail=f"유효성 검사 실패: {str(e)}")
