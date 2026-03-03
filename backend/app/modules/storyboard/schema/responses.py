"""스토리보드 모듈 - API 요청/응답 스키마"""
from typing import Any

from pydantic import BaseModel, Field

from .models import (
    EvidenceFile,
    Participant,
    TimelineData,
    TimelineItem,
    TransitionType,
)


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


# --- 배치 분석 ---
class AnalyzeBatchResponse(BaseModel):
    """배치 분석 응답 (job 생성)"""
    success: bool
    job_id: str | None = None
    error: str | None = None


# --- 병합 ---
class MergeConflict(BaseModel):
    """병합 충돌 정보"""
    existing_item_id: str
    new_item_id: str
    conflict_type: str  # "date_overlap", "content_contradiction"
    description: str


class MergeReport(BaseModel):
    """병합 결과 보고"""
    new_items_added: int = 0
    duplicates_detected: int = 0
    items_updated: int = 0
    conflicts: list[MergeConflict] = Field(default_factory=list)


class MergeTimelineRequest(BaseModel):
    """증분 병합 요청 (기존 타임라인 정보, 신규 파일은 multipart로 수신)"""
    existing_items: list[TimelineItem] = Field(..., description="기존 타임라인 항목")
    existing_evidence: list[EvidenceFile] = Field(default_factory=list, description="기존 증거 파일")


class MergeTimelineResponse(BaseModel):
    """증분 병합 응답"""
    success: bool
    merged_items: list[TimelineItem] = Field(default_factory=list)
    merged_evidence: list[EvidenceFile] = Field(default_factory=list)
    merge_report: MergeReport | None = None
    error: str | None = None


# --- 배치 분석 진행 상태 (SSE 확장) ---
class BatchJobProgress(BaseModel):
    """배치 분석 진행 상태 (SSE event data 확장)"""
    job_id: str
    status: str
    progress: int  # 0~100
    current_file: str | None = None
    current_file_index: int = 0
    total_files: int = 0
    message: str = ""
    result: "BatchAnalysisResult | None" = None
    error: str | None = None


class BatchAnalysisResult(BaseModel):
    """배치 분석 최종 결과"""
    timeline_items: list[TimelineItem]
    evidence_files: list[EvidenceFile]
    topics: list[str]
    summary: str | None = None
    total_files: int
    success_count: int
    failed_files: list[str] = Field(default_factory=list)


# Forward reference 해소
BatchJobProgress.model_rebuild()
