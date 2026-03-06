"""스토리보드 모듈 - 데이터 모델 스키마"""
import uuid
from enum import Enum

from pydantic import BaseModel, Field


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

    # 간트차트 추가 필드
    topic: str | None = Field(None, description="사건 주제 (AI 자동 분류: 폭행, 협박, 금전 등)")
    date_start: str | None = Field(None, description="기간 시작일 (YYYY-MM-DD)")
    date_end: str | None = Field(None, description="기간 종료일 (YYYY-MM-DD)")
    evidence_ids: list[str] = Field(default_factory=list, description="연결된 증거 파일 ID (N:M)")
    confidence: float | None = Field(None, ge=0.0, le=1.0, description="추출 신뢰도 (0.0~1.0)")


class TimelineData(BaseModel):
    """타임라인 전체 데이터 (JSON 내보내기/가져오기용)"""
    title: str = Field(..., description="타임라인 제목")
    created_at: str = Field(..., description="생성일시")
    updated_at: str = Field(..., description="수정일시")
    items: list[TimelineItem] = Field(default_factory=list)
    original_text: str | None = Field(None, description="원본 입력 텍스트")
    evidence_files: list["EvidenceFile"] = Field(default_factory=list, description="증거 파일 목록")
    topics: list[str] = Field(default_factory=list, description="전체 주제 목록 (간트차트 Y축)")


class EvidenceType(str, Enum):
    """증거 파일 유형"""
    KAKAO_TXT = "kakao_txt"
    MESSENGER_SCREENSHOT = "messenger_screenshot"
    VOICE_RECORDING = "voice_recording"
    DOCUMENT = "document"
    PHOTO = "photo"
    TEXT_INPUT = "text_input"
    OTHER = "other"


class EvidenceFile(BaseModel):
    """증거 파일 메타데이터"""
    evidence_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="UUID v4")
    evidence_type: EvidenceType = Field(..., description="증거 유형")
    filename: str = Field(..., description="원본 파일명")
    uploaded_at: str = Field(..., description="업로드 시각 (ISO 8601)")
    file_size_kb: int = Field(..., description="파일 크기 (KB)")
    file_hash: str | None = Field(None, description="SHA-256 해시 (중복 감지용)")
    session_id: str = Field(..., description="소유자 세션 ID (IDOR 방어)")
    extracted_timeline_ids: list[str] = Field(default_factory=list, description="추출된 타임라인 항목 ID")
    tags: list[str] = Field(default_factory=list, description="사용자 정의 태그")
    source_description: str | None = Field(None, description="증거 설명")


# Forward reference 해소
TimelineData.model_rebuild()
