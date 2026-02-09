"""
통합 채팅 API 라우터

LangGraph StateGraph를 통한 채팅 처리
"""

import json
import logging
import secrets
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from langgraph.types import Command
from sse_starlette.sse import EventSourceResponse

from app.multi_agent import (
    ChatRequest,
    ChatResponse,
    get_graph,
    request_to_state,
    state_to_response,
)

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)

AGENT_LIST: list[dict[str, str]] = [
    {"name": "legal_search", "description": "판례/법령 검색 기반 법률 상담"},
    {"name": "case_search", "description": "판례 검색"},
    {"name": "law_search", "description": "법령 검색"},
    {"name": "lawyer_finder", "description": "위치 기반 변호사 찾기"},
    {"name": "small_claims", "description": "소액소송 단계별 가이드"},
    {"name": "storyboard", "description": "사건 타임라인 정리"},
    {"name": "lawyer_stats", "description": "변호사 통계"},
    {"name": "law_study", "description": "법학 학습 가이드"},
    {"name": "general", "description": "일반 채팅"},
]


async def _validate_session_secret(
    graph: Any,
    config: dict[str, Any],
    client_secret: str,
) -> None:
    """기존 스레드 재개 시 session_secret을 검증한다.

    Args:
        graph: 컴파일된 LangGraph
        config: thread_id를 포함한 configurable
        client_secret: 클라이언트가 전달한 session_secret

    Raises:
        HTTPException: session_secret이 불일치할 때
    """
    graph_state = await graph.aget_state(config)
    if graph_state.values:
        stored_secret = graph_state.values.get("session_secret", "")
        if stored_secret and stored_secret != client_secret:
            raise HTTPException(status_code=403, detail="Invalid session")


async def _invoke_graph(
    request: ChatRequest,
) -> tuple[dict[str, Any], str, str]:
    """그래프 실행 헬퍼

    interrupt 재개 여부를 판단하고, 적절한 방식으로 그래프를 실행합니다.

    Args:
        request: 채팅 요청

    Returns:
        (그래프 실행 결과, thread_id, session_secret)
    """
    graph = get_graph()
    thread_id = request.session_data.get("thread_id") or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # 세션 시크릿: 클라이언트에서 전달받거나 새로 생성
    session_secret = (
        request.session_data.get("session_secret") or secrets.token_hex(16)
    )

    # interrupt 재개 여부 판단
    if request.session_data.get("thread_id"):
        # 기존 스레드 재개: session_secret 검증
        await _validate_session_secret(
            graph, config, request.session_data.get("session_secret", "")
        )

        graph_state = await graph.aget_state(config)
        if graph_state.tasks and any(t.interrupts for t in graph_state.tasks):
            # interrupt 재개: 사용자 메시지로 resume
            result = await graph.ainvoke(
                Command(resume=request.message), config
            )
            return result, thread_id, session_secret

    # 새 대화 또는 interrupt가 아닌 경우
    state = request_to_state(request)
    state["session_secret"] = session_secret
    result = await graph.ainvoke(state, config)
    return result, thread_id, session_secret


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    채팅 메시지 처리

    LangGraph StateGraph가 메시지를 분석하고
    적절한 에이전트 노드를 선택하여 응답을 생성합니다.
    """
    try:
        result, thread_id, session_secret = await _invoke_graph(request)

        # interrupt 발생 여부 확인
        graph = get_graph()
        config = {"configurable": {"thread_id": thread_id}}
        graph_state = await graph.aget_state(config)

        if graph_state.tasks and any(t.interrupts for t in graph_state.tasks):
            # interrupt 상태 → 중간 응답 반환
            interrupt_data = graph_state.tasks[0].interrupts[0].value
            return ChatResponse(
                response=interrupt_data.get("response", ""),
                agent_used="small_claims",
                sources=interrupt_data.get("sources", []),
                actions=interrupt_data.get("actions", []),
                session_data={
                    "thread_id": thread_id,
                    "session_secret": session_secret,
                },
                confidence=1.0,
            )

        # 정상 완료
        response = state_to_response(result)
        response.session_data["thread_id"] = thread_id
        response.session_data["session_secret"] = session_secret
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("채팅 처리 중 오류 발생")
        raise HTTPException(
            status_code=500,
            detail=f"채팅 처리 중 오류가 발생했습니다: {str(e)}",
        ) from e


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
            graph = get_graph()
            thread_id = request.session_data.get("thread_id") or str(
                uuid.uuid4()
            )
            config = {"configurable": {"thread_id": thread_id}}

            # 세션 시크릿
            session_secret = (
                request.session_data.get("session_secret")
                or secrets.token_hex(16)
            )

            # interrupt 재개 여부 판단
            input_value: dict[str, Any] | Command
            if request.session_data.get("thread_id"):
                # 기존 스레드: session_secret 검증
                await _validate_session_secret(
                    graph,
                    config,
                    request.session_data.get("session_secret", ""),
                )

                graph_state = await graph.aget_state(config)
                if graph_state.tasks and any(
                    t.interrupts for t in graph_state.tasks
                ):
                    input_value = Command(resume=request.message)
                else:
                    input_value = request_to_state(request)
                    input_value["session_secret"] = session_secret
            else:
                input_value = request_to_state(request)
                input_value["session_secret"] = session_secret

            # astream으로 custom 이벤트 수신
            async for chunk in graph.astream(
                input_value, config, stream_mode="custom"
            ):
                yield {
                    "event": chunk.get("event", "token"),
                    "data": json.dumps(
                        chunk.get("data", {}), ensure_ascii=False
                    ),
                }

            # 스트리밍 종료 후 interrupt 확인
            graph_state = await graph.aget_state(config)
            if graph_state.tasks and any(
                t.interrupts for t in graph_state.tasks
            ):
                interrupt_data = graph_state.tasks[0].interrupts[0].value
                # interrupt 응답 텍스트를 token 이벤트로 전송
                response_text = interrupt_data.get("response", "")
                if response_text:
                    yield {
                        "event": "token",
                        "data": json.dumps(
                            {"content": response_text},
                            ensure_ascii=False,
                        ),
                    }
                yield {
                    "event": "metadata",
                    "data": json.dumps(
                        {
                            "agent_used": "small_claims",
                            "actions": interrupt_data.get("actions", []),
                            "session_data": {
                                "thread_id": thread_id,
                                "session_secret": session_secret,
                            },
                        },
                        ensure_ascii=False,
                    ),
                }

            # done 이벤트 (active_agent 포함: metadata 경로 외 백업)
            done_data: dict[str, Any] = {
                "thread_id": thread_id,
                "session_secret": session_secret,
            }
            final_state = await graph.aget_state(config)
            if final_state.values:
                output_sd = final_state.values.get("output_session_data", {})
                if output_sd.get("active_agent"):
                    done_data["active_agent"] = output_sd["active_agent"]
            yield {
                "event": "done",
                "data": json.dumps(done_data, ensure_ascii=False),
            }

        except HTTPException as e:
            yield {
                "event": "error",
                "data": json.dumps(
                    {"message": e.detail}, ensure_ascii=False
                ),
            }
        except Exception as e:
            logger.exception("스트리밍 채팅 처리 중 오류 발생")
            yield {
                "event": "error",
                "data": json.dumps(
                    {"message": str(e)}, ensure_ascii=False
                ),
            }

    return EventSourceResponse(event_generator())


@router.get("/agents")
async def list_agents() -> dict[str, Any]:
    """
    사용 가능한 에이전트 목록 반환
    """
    return {"agents": AGENT_LIST}
