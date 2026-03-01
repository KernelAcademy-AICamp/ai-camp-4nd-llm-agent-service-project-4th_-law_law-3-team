"""스토리보드 모듈 - 데이터 모델 스키마"""
from enum import Enum

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
    action: str | None = Field(None, description="해당 장면에서의 행동")
    emotion: str | None = Field(None, description="감정 상태")


class TimelineItem(BaseModel):
    """타임라인 개별 항목"""
    id: str = Field(..., description="고유 식별자 (UUID)")
    date: str = Field(..., description="날짜 (YYYY-MM-DD 또는 자유형식)")
    date_raw: str | None = Field(None, description="원본 날짜 표현 (호버 툴팁용)")
    title: str = Field(..., description="이벤트 제목")
    description: str = Field(..., description="이벤트 상세 설명 (하위 호환용)")
    participants: list[str] = Field(default_factory=list, description="관련자 목록 (하위 호환용)")
    order: int = Field(..., description="순서")
    image_url: str | None = Field(None, description="생성된 이미지 URL")
    image_prompt: str | None = Field(None, description="이미지 생성 프롬프트")
    image_status: ImageStatus | None = Field(None, description="이미지 생성 상태")

    # 새 필드 (스토리보드 품질 개선용)
    location: str | None = Field(None, description="장소")
    time_of_day: str | None = Field(None, description="시간대 (아침/낮/저녁/밤)")
    time: str | None = Field(None, description="구체적 시간 (HH:MM 형식 또는 자유형식)")
    scene_number: int | None = Field(None, description="장면 번호")
    description_short: str | None = Field(None, description="한 줄 요약 (50자)")
    description_detailed: str | None = Field(None, description="상세 설명 (300자)")
    participants_detailed: list[Participant] = Field(
        default_factory=list, description="참여자 상세 정보 (역할 포함)"
    )
    key_dialogue: str | None = Field(None, description="핵심 대사/발언")
    legal_significance: str | None = Field(None, description="법적 의미")
    evidence_items: list[str] = Field(default_factory=list, description="관련 증거물")
    mood: str | None = Field(None, description="장면 분위기")


class TimelineData(BaseModel):
    """타임라인 전체 데이터 (JSON 내보내기/가져오기용)"""
    title: str = Field(..., description="타임라인 제목")
    created_at: str = Field(..., description="생성일시")
    updated_at: str = Field(..., description="수정일시")
    items: list[TimelineItem] = Field(default_factory=list)
    original_text: str | None = Field(None, description="원본 입력 텍스트")
