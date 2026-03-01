"""스토리보드 모듈 - API 요청/응답 스키마"""
from typing import Any

from pydantic import BaseModel, Field

from .models import Participant, TimelineData, TimelineItem, TransitionType


class ExtractTimelineRequest(BaseModel):
    """AI 타임라인 추출 요청"""
    text: str = Field(..., min_length=10, description="사건 내용 텍스트")


class ExtractTimelineResponse(BaseModel):
    """AI 타임라인 추출 응답"""
    success: bool
    timeline: list[TimelineItem]
    summary: str | None = Field(None, description="사건 요약")


class ValidateTimelineRequest(BaseModel):
    """타임라인 유효성 검사 요청"""
    timeline: TimelineData


class ValidateTimelineResponse(BaseModel):
    """타임라인 유효성 검사 응답"""
    valid: bool
    message: str | None = None


# --- 음성 → 텍스트 (STT) ---
class TranscribeResponse(BaseModel):
    """음성 → 텍스트 변환 응답"""
    success: bool
    text: str | None = None
    error: str | None = None


# --- 이미지 분석 (Vision) ---
class AnalyzeImageResponse(BaseModel):
    """이미지 분석 응답"""
    success: bool
    timeline: list[TimelineItem] = Field(default_factory=list)
    summary: str | None = None
    error: str | None = None


# --- 이미지 생성 ---
class GenerateImageRequest(BaseModel):
    """이미지 생성 요청 (스토리보드 스타일 고정)"""
    item_id: str = Field(..., description="타임라인 항목 ID")
    title: str = Field(..., description="이벤트 제목")
    description: str = Field(..., description="이벤트 설명")
    participants: list[str] = Field(default_factory=list, description="관련자 목록")

    # 확장 필드 (이미지 품질 향상용)
    location: str | None = Field(None, description="장소")
    time_of_day: str | None = Field(None, description="시간대")
    participants_detailed: list[Participant] = Field(
        default_factory=list, description="참여자 상세 정보"
    )
    mood: str | None = Field(None, description="장면 분위기")


class GenerateImageResponse(BaseModel):
    """이미지 생성 응답"""
    success: bool
    image_url: str | None = None
    image_prompt: str | None = None
    error: str | None = None


# --- 일괄 이미지 생성 ---
class GenerateImagesBatchRequest(BaseModel):
    """일괄 이미지 생성 요청 (스토리보드 스타일 고정)"""
    items: list[TimelineItem] = Field(..., description="타임라인 항목 목록")


class GenerateImagesBatchResponse(BaseModel):
    """일괄 이미지 생성 응답 (작업 ID 반환)"""
    success: bool
    job_id: str | None = None
    error: str | None = None


# --- 영상 생성 ---
class GenerateVideoRequest(BaseModel):
    """영상 생성 요청"""
    timeline_id: str = Field(..., description="타임라인 ID")
    image_urls: list[str] = Field(..., min_length=2, description="이미지 URL 목록 (최소 2개)")
    duration_per_image: float = Field(default=6.0, description="이미지당 표시 시간 (초)")
    transition: TransitionType = Field(default=TransitionType.FADE, description="전환 효과")
    transition_duration: float = Field(default=1.0, description="전환 효과 시간 (초)")
    resolution: tuple[int, int] = Field(default=(1280, 720), description="영상 해상도")


class GenerateVideoResponse(BaseModel):
    """영상 생성 응답"""
    success: bool
    video_url: str | None = None
    duration: float | None = None
    image_count: int | None = None
    error: str | None = None


# --- 작업 상태 ---
class JobStatusResponse(BaseModel):
    """작업 상태 응답"""
    job_id: str
    status: str
    progress: int
    current_step: int
    total_steps: int
    message: str
    result: dict[str, Any] | None = None
    error: str | None = None
