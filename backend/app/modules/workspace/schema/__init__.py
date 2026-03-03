"""워크스페이스 스키마"""

from pydantic import BaseModel


class CaseCreateRequest(BaseModel):
    """사건 생성 요청"""

    case_name: str
    case_type: str | None = None
    conversation_ids: list[str] | None = None


class CaseUpdateRequest(BaseModel):
    """사건 수정 요청"""

    case_name: str | None = None
    case_type: str | None = None
    status: str | None = None


class TimelineRebuildRequest(BaseModel):
    """타임라인 재생성 요청"""

    include_conversations: bool = True
    include_manual: bool = True


class TimelineItemUpdateRequest(BaseModel):
    """타임라인 항목 수정 요청"""

    title: str | None = None
    description: str | None = None
    date_text: str | None = None
    category: str | None = None
