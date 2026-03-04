"""
메시지 스키마

채팅 요청/응답 스키마
"""

from typing import Any

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """채팅 메시지"""

    role: str = "user"  # "user" | "assistant"
    content: str = ""


class ChatRequest(BaseModel):
    """채팅 요청"""

    message: str = Field(max_length=10000)
    user_role: str = "user"
    history: list[ChatMessage] = Field(default_factory=list, max_length=50)
    session_data: dict[str, Any] = {}
    user_location: dict[str, float] | None = None
    agent: str | None = None  # 에이전트 직접 지정 (라우팅 건너뜀)
    conversation_id: str | None = None  # 기존 대화 이어가기
    case_id: str | None = None  # 사건에 연결


class ChatResponse(BaseModel):
    """채팅 응답"""

    response: str
    agent_used: str
    sources: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    session_data: dict[str, Any] = {}
    confidence: float = 1.0
    emotion: str = "neutral"
