"""
통합 채팅 API 라우터

멀티 에이전트 오케스트레이터를 통한 채팅 처리
"""

import json
import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.multi_agent import ChatRequest, ChatResponse, get_orchestrator

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    채팅 메시지 처리

    멀티 에이전트 오케스트레이터가 메시지를 분석하고
    적절한 에이전트를 선택하여 응답을 생성합니다.
    """
    try:
        orchestrator = get_orchestrator()
        return await orchestrator.process(request)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"채팅 처리 중 오류가 발생했습니다: {str(e)}",
        )


@router.post("/stream")
async def chat_stream(request: ChatRequest) -> EventSourceResponse:
    """
    스트리밍 채팅 메시지 처리 (SSE)

    LLM 응답을 실시간으로 스트리밍합니다.

    이벤트 타입:
    - token: LLM 응답 토큰
    - sources: 참조 자료 (RAG 결과)
    - metadata: 에이전트 정보
    - done: 스트리밍 완료
    - error: 에러 발생
    """
    async def event_generator():
        try:
            orchestrator = get_orchestrator()
            async for event_type, data in orchestrator.process_stream(request):
                yield {
                    "event": event_type,
                    "data": json.dumps(data, ensure_ascii=False),
                }
        except Exception as e:
            logger.exception("스트리밍 채팅 처리 중 오류 발생")
            yield {
                "event": "error",
                "data": json.dumps({"message": str(e)}, ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())


@router.get("/agents")
async def list_agents() -> Dict[str, Any]:
    """
    사용 가능한 에이전트 목록 반환
    """
    from app.multi_agent import get_agent_executor

    executor = get_agent_executor()
    agents = []

    for agent_name in executor.list_agents():
        agent = executor.get_agent(agent_name)
        if agent:
            agents.append(
                {
                    "name": agent.name,
                    "description": agent.description,
                }
            )

    return {"agents": agents}
