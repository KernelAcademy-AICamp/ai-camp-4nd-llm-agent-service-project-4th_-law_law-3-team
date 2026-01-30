"""
멀티 에이전트 API 라우터

/api/multi-agent/* 엔드포인트 정의
"""

from typing import Any

from fastapi import APIRouter

from app.modules.multi_agent.schema import (
    ChatActionResponse,
    MultiAgentChatRequest,
    MultiAgentChatResponse,
)
from app.modules.multi_agent.service import get_orchestrator

router = APIRouter()


@router.post("/chat", response_model=MultiAgentChatResponse)
async def multi_agent_chat(request: MultiAgentChatRequest) -> MultiAgentChatResponse:
    """
    멀티 에이전트 챗 API

    사용자 메시지를 분석하여 적절한 에이전트로 라우팅
    """
    orchestrator = get_orchestrator()

    # 대화 기록 변환
    history = [
        {"role": h.role, "content": h.content}
        for h in request.history
    ] if request.history else None

    # 오케스트레이터 실행
    response, agent_used = await orchestrator.process(
        message=request.message,
        user_role=request.user_role,
        history=history,
        session_data=request.session_data,
        user_location=request.user_location,
    )

    # 응답 변환
    actions = [
        ChatActionResponse(
            type=action.type,
            label=action.label,
            action=action.action,
            url=action.url,
            params=action.params,
        )
        for action in response.actions
    ]

    return MultiAgentChatResponse(
        response=response.message,
        agent_used=agent_used,
        sources=response.sources,
        actions=actions,
        session_data=response.session_data,
    )


@router.get("/agents")
async def list_agents(user_role: str = "user") -> dict[str, Any]:
    """
    사용 가능한 에이전트 목록 반환

    Args:
        user_role: 사용자 역할 (user/lawyer)
    """
    from app.common.agent_router import UserRole, get_available_agents

    try:
        role = UserRole(user_role)
    except ValueError:
        role = UserRole.USER

    agents = get_available_agents(role)

    return {
        "role": role.value,
        "agents": [
            {"type": agent.value, "name": agent.name}
            for agent in agents
        ],
    }


@router.post("/reset")
async def reset_session() -> dict[str, Any]:
    """세션 초기화"""
    return {"status": "ok", "session_data": {}}
