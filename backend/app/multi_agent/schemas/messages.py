"""
메시지 스키마

채팅 요청/응답 스키마
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class ChatMessage(BaseModel):
    """채팅 메시지"""
    role: str = "user"  # "user" | "assistant"
    content: str = ""


class ChatRequest(BaseModel):
    """채팅 요청"""
    message: str
    user_role: str = "user"
    history: List[ChatMessage] = []
    session_data: Dict[str, Any] = {}
    user_location: Optional[Dict[str, float]] = None


class ChatResponse(BaseModel):
    """채팅 응답"""
    response: str
    agent_used: str
    sources: List[Dict[str, Any]] = []
    actions: List[Dict[str, Any]] = []
    session_data: Dict[str, Any] = {}
    confidence: float = 1.0
