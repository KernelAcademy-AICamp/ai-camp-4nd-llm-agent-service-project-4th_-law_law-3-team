"""스토리보드 모듈 - Pydantic 스키마"""
from enum import Enum
from typing import Any, List, Optional, Tuple

from pydantic import BaseModel, Field


# 전환 효과 타입
class TransitionType(str, Enum):
    FADE = "fade"
    SLIDE = "slide"
    ZOOM = "zoom"
    NONE = "none"


# 입력 모드 타입
class InputMode(str, Enum):
    TEXT = "text"
    VOICE = "voice"
    IMAGE = "image"


# 이미지 상태 타입
class ImageStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# 참여자 역할 타입
class ParticipantRole(str, Enum):
    VICTIM = "victim"           # 피해자
    PERPETRATOR = "perpetrator" # 가해자
    WITNESS = "witness"         # 증인
    BYSTANDER = "bystander"     # 방관자
    AUTHORITY = "authority"     # 공권력
    OTHER = "other"             # 기타


class Participant(BaseModel):
    """참여자 상세 정보"""
    name: str = Field(..., description="이름/호칭")
    role: ParticipantRole = Field(..., description="역할 (피해자/가해자/증인 등)")
    action: Optional[str] = Field(None, description="해당 장면에서의 행동")
    emotion: Optional[str] = Field(None, description="감정 상태")


class TimelineItem(BaseModel):
    """타임라인 개별 항목"""
    id: str = Field(..., description="고유 식별자 (UUID)")
    date: str = Field(..., description="날짜 (YYYY-MM-DD 또는 자유형식)")
    title: str = Field(..., description="이벤트 제목")
    description: str = Field(..., description="이벤트 상세 설명 (하위 호환용)")
    participants: List[str] = Field(default_factory=list, description="관련자 목록 (하위 호환용)")
    order: int = Field(..., description="순서")
    image_url: Optional[str] = Field(None, description="생성된 이미지 URL")
    image_prompt: Optional[str] = Field(None, description="이미지 생성 프롬프트")
    image_status: Optional[ImageStatus] = Field(None, description="이미지 생성 상태")

    # 새 필드 (스토리보드 품질 개선용)
    location: Optional[str] = Field(None, description="장소")
    time_of_day: Optional[str] = Field(None, description="시간대 (아침/낮/저녁/밤)")
    time: Optional[str] = Field(None, description="구체적 시간 (HH:MM 형식 또는 자유형식)")
    scene_number: Optional[int] = Field(None, description="장면 번호")
    description_short: Optional[str] = Field(None, description="한 줄 요약 (50자)")
    description_detailed: Optional[str] = Field(None, description="상세 설명 (300자)")
    participants_detailed: List[Participant] = Field(
        default_factory=list, description="참여자 상세 정보 (역할 포함)"
    )
    key_dialogue: Optional[str] = Field(None, description="핵심 대사/발언")
    legal_significance: Optional[str] = Field(None, description="법적 의미")
    evidence_items: List[str] = Field(default_factory=list, description="관련 증거물")
    mood: Optional[str] = Field(None, description="장면 분위기")


class TimelineData(BaseModel):
    """타임라인 전체 데이터 (JSON 내보내기/가져오기용)"""
    title: str = Field(..., description="타임라인 제목")
    created_at: str = Field(..., description="생성일시")
    updated_at: str = Field(..., description="수정일시")
    items: List[TimelineItem] = Field(default_factory=list)
    original_text: Optional[str] = Field(None, description="원본 입력 텍스트")


class ExtractTimelineRequest(BaseModel):
    """AI 타임라인 추출 요청"""
    text: str = Field(..., min_length=10, description="사건 내용 텍스트")


class ExtractTimelineResponse(BaseModel):
    """AI 타임라인 추출 응답"""
    success: bool
    timeline: List[TimelineItem]
    summary: Optional[str] = Field(None, description="사건 요약")


class ValidateTimelineRequest(BaseModel):
    """타임라인 유효성 검사 요청"""
    timeline: TimelineData


# --- 음성 → 텍스트 (STT) ---
class TranscribeResponse(BaseModel):
    """음성 → 텍스트 변환 응답"""
    success: bool
    text: Optional[str] = None
    error: Optional[str] = None


# --- 이미지 분석 (Vision) ---
class AnalyzeImageResponse(BaseModel):
    """이미지 분석 응답"""
    success: bool
    timeline: List[TimelineItem] = Field(default_factory=list)
    summary: Optional[str] = None
    error: Optional[str] = None


# --- 이미지 생성 ---
class GenerateImageRequest(BaseModel):
    """이미지 생성 요청 (스토리보드 스타일 고정)"""
    item_id: str = Field(..., description="타임라인 항목 ID")
    title: str = Field(..., description="이벤트 제목")
    description: str = Field(..., description="이벤트 설명")
    participants: List[str] = Field(default_factory=list, description="관련자 목록")

    # 확장 필드 (이미지 품질 향상용)
    location: Optional[str] = Field(None, description="장소")
    time_of_day: Optional[str] = Field(None, description="시간대")
    participants_detailed: List[Participant] = Field(
        default_factory=list, description="참여자 상세 정보"
    )
    mood: Optional[str] = Field(None, description="장면 분위기")


class GenerateImageResponse(BaseModel):
    """이미지 생성 응답"""
    success: bool
    image_url: Optional[str] = None
    image_prompt: Optional[str] = None
    error: Optional[str] = None


# --- 일괄 이미지 생성 ---
class GenerateImagesBatchRequest(BaseModel):
    """일괄 이미지 생성 요청 (스토리보드 스타일 고정)"""
    items: List[TimelineItem] = Field(..., description="타임라인 항목 목록")


class GenerateImagesBatchResponse(BaseModel):
    """일괄 이미지 생성 응답 (작업 ID 반환)"""
    success: bool
    job_id: Optional[str] = None
    error: Optional[str] = None


# --- 영상 생성 ---
class GenerateVideoRequest(BaseModel):
    """영상 생성 요청"""
    timeline_id: str = Field(..., description="타임라인 ID")
    image_urls: List[str] = Field(..., min_length=2, description="이미지 URL 목록 (최소 2개)")
    duration_per_image: float = Field(default=6.0, description="이미지당 표시 시간 (초)")
    transition: TransitionType = Field(default=TransitionType.FADE, description="전환 효과")
    transition_duration: float = Field(default=1.0, description="전환 효과 시간 (초)")
    resolution: Tuple[int, int] = Field(default=(1280, 720), description="영상 해상도")


class GenerateVideoResponse(BaseModel):
    """영상 생성 응답"""
    success: bool
    video_url: Optional[str] = None
    duration: Optional[float] = None
    image_count: Optional[int] = None
    error: Optional[str] = None


# --- 작업 상태 ---
class JobStatusResponse(BaseModel):
    """작업 상태 응답"""
    job_id: str
    status: str
    progress: int
    current_step: int
    total_steps: int
    message: str
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
