"""
메시지 스키마

채팅 요청/응답 스키마
"""

from typing import Any

from pydantic import BaseModel


class ChatMessage(BaseModel):
    """채팅 메시지"""

    role: str = "user"  # "user" | "assistant"
    content: str = ""


class ChatRequest(BaseModel):
    """채팅 요청"""

    message: str
    user_role: str = "user"
    history: list[ChatMessage] = []
    session_data: dict[str, Any] = {}
    user_location: dict[str, float] | None = None
    agent: str | None = None  # 에이전트 직접 지정 (라우팅 건너뜀)


class ChatResponse(BaseModel):
    """채팅 응답"""

    response: str
    agent_used: str
    sources: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    session_data: dict[str, Any] = {}
    confidence: float = 1.0
