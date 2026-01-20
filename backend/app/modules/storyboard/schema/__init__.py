"""스토리보드 모듈 - Pydantic 스키마"""
from pydantic import BaseModel, Field
from typing import List, Optional


class TimelineItem(BaseModel):
    """타임라인 개별 항목"""
    id: str = Field(..., description="고유 식별자 (UUID)")
    date: str = Field(..., description="날짜 (YYYY-MM-DD 또는 자유형식)")
    title: str = Field(..., description="이벤트 제목")
    description: str = Field(..., description="이벤트 상세 설명")
    participants: List[str] = Field(default_factory=list, description="관련자 목록")
    order: int = Field(..., description="순서")
    image_url: Optional[str] = Field(None, description="생성된 이미지 URL")
    image_prompt: Optional[str] = Field(None, description="이미지 생성 프롬프트")


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
