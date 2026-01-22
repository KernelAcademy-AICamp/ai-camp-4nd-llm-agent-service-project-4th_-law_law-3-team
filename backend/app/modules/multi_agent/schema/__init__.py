"""
멀티 에이전트 스키마

API 요청/응답 모델 정의
"""

from typing import Any, Literal, Optional

from pydantic import BaseModel

from app.common.agent_base import ActionType


class ChatHistory(BaseModel):
    """대화 기록"""
    role: Literal["user", "assistant"]
    content: str


class MultiAgentChatRequest(BaseModel):
    """멀티 에이전트 챗 요청"""
    message: str
    user_role: Literal["user", "lawyer"] = "user"
    history: list[ChatHistory] = []
    session_data: dict[str, Any] = {}
    user_location: Optional[dict[str, float]] = None  # {latitude, longitude}


class ChatActionResponse(BaseModel):
    """액션 응답"""
    type: ActionType
    label: str
    action: Optional[str] = None
    url: Optional[str] = None
    params: Optional[dict[str, Any]] = None  # navigate params


class MultiAgentChatResponse(BaseModel):
    """멀티 에이전트 챗 응답"""
    response: str
    agent_used: str
    sources: list[dict[str, Any]] = []
    actions: list[ChatActionResponse] = []
    session_data: dict[str, Any] = {}
