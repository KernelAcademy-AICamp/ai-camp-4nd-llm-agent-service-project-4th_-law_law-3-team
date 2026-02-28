"""
태그 시스템 스키마

대화 중 수집된 정보를 태그로 분류하여 스토리보드 등에서 활용
"""

from enum import Enum

from pydantic import BaseModel, Field


class TagType(str, Enum):
    """태그 유형"""

    LAWYER = "lawyer"  # 변호사 관련
    PRECEDENT = "precedent"  # 판례 관련
    EVIDENCE = "evidence"  # 증거 (계약서, 사진, 카톡 등)
    TIMELINE = "timeline"  # 시간순 사건
    PARTY = "party"  # 당사자 (피해자/가해자)
    AMOUNT = "amount"  # 금액


class TaggedItem(BaseModel):
    """태그가 붙은 정보 항목"""

    tag_type: TagType
    content: str = Field(max_length=100, description="요약 (100자 이내)")
    source_agent: str = Field(description="추출된 에이전트")
    turn_index: int = Field(ge=0, description="대화 턴 번호")
    date_hint: str | None = Field(default=None, description="날짜 힌트 (YYYY-MM-DD)")
    confidence: float = Field(ge=0.0, le=1.0, default=0.5, description="신뢰도")
