"""
LangGraph 상태 스키마 및 변환 함수

ChatState TypedDict + ChatRequest/ChatResponse 변환
"""

from typing import Any, Optional

from typing_extensions import TypedDict

from app.multi_agent.schemas.messages import ChatRequest, ChatResponse


class ChatState(TypedDict, total=False):
    """LangGraph 그래프 상태 스키마"""

    # Input (from ChatRequest)
    message: str
    user_role: str
    history: list[dict[str, str]]
    session_data: dict[str, Any]
    user_location: Optional[dict[str, float]]
    agent_override: Optional[str]

    # Routing (set by router_node)
    selected_agent: str
    search_focus: str  # "precedent" | "law" | ""
    routing_confidence: float
    routing_reason: str

    # Output (set by agent nodes)
    response: str
    sources: list[dict[str, Any]]
    actions: list[dict[str, Any]]
    output_session_data: dict[str, Any]
    agent_used: str


def request_to_state(request: ChatRequest) -> dict[str, Any]:
    """ChatRequest -> ChatState 입력 dict 변환

    Args:
        request: FastAPI ChatRequest

    Returns:
        ChatState 호환 dict
    """
    history = [
        {"role": msg.role, "content": msg.content}
        for msg in request.history
    ]

    return {
        "message": request.message,
        "user_role": request.user_role,
        "history": history,
        "session_data": request.session_data,
        "user_location": request.user_location,
        "agent_override": request.agent,
        # 기본값 초기화
        "selected_agent": "",
        "search_focus": "",
        "routing_confidence": 0.0,
        "routing_reason": "",
        "response": "",
        "sources": [],
        "actions": [],
        "output_session_data": {},
        "agent_used": "",
    }


def state_to_response(state: dict[str, Any]) -> ChatResponse:
    """최종 ChatState -> ChatResponse 변환

    Args:
        state: 그래프 실행 결과 상태

    Returns:
        ChatResponse
    """
    return ChatResponse(
        response=state.get("response", ""),
        agent_used=state.get("agent_used", "unknown"),
        sources=state.get("sources", []),
        actions=state.get("actions", []),
        session_data=state.get("output_session_data", {}),
        confidence=state.get("routing_confidence", 1.0),
    )
