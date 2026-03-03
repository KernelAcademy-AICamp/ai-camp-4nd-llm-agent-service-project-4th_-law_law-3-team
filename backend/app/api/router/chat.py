"""
통합 채팅 API 라우터

LangGraph StateGraph를 통한 채팅 처리.
대화 영속화(chat_conversations/chat_messages)와 크로스 에이전트 태그 서버사이드 누적.
"""

import json
import logging
import secrets
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from sse_starlette.sse import EventSourceResponse

from app.core.database import async_session_factory
from app.core.rate_limit import AI_RATE_LIMIT, limiter
from app.multi_agent import (
    ChatRequest,
    ChatResponse,
    get_graph,
    request_to_state,
    state_to_response,
)
from app.services.workspace.chat_persistence import ChatPersistenceService

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

# 자동 요약 및 분류 주기 (사용자 턴 기준)
_SUMMARIZE_INTERVAL = 10
_CLASSIFY_INTERVAL = 5


async def _persist_after_graph(
    conversation_id: uuid.UUID,
    response_text: str,
    agent_used: str,
    final_state: dict[str, Any],
    user_message: str,
) -> int:
    """그래프 실행 후 대화 영속화 후처리 (태그 누적, 에이전트 업데이트, 자동 분류/요약)

    Returns:
        추출된 태그 수
    """
    tags_extracted = 0
    async with async_session_factory() as db:
        try:
            # 어시스턴트 응답 저장
            await ChatPersistenceService.save_message(
                db, conversation_id, "assistant", response_text, agent_used
            )

            # 에이전트 업데이트
            await ChatPersistenceService.update_last_agent(
                db, conversation_id, agent_used
            )

            # 태그 서버사이드 누적
            output_sd = final_state.get("output_session_data", {})
            new_tags = output_sd.get("tagged_items", [])
            if new_tags:
                await ChatPersistenceService.append_tags(
                    db, conversation_id, new_tags
                )
                tags_extracted = len(new_tags)

            # 메시지 수 기반 자동 분류/요약 (백그라운드, 실패 무시)
            try:
                from sqlalchemy import func as sa_func
                from sqlalchemy import select

                from app.models.chat_conversation import ChatConversation, ChatMessage

                user_msg_count = (
                    await db.execute(
                        select(sa_func.count()).where(
                            ChatMessage.conversation_id == conversation_id,
                            ChatMessage.role == "user",
                        )
                    )
                ).scalar_one()

                if user_msg_count >= 2 and user_msg_count % _CLASSIFY_INTERVAL == 0:
                    # conversation 메타 조회
                    conv_result = await db.execute(
                        select(ChatConversation).where(
                            ChatConversation.id == conversation_id
                        )
                    )
                    conv = conv_result.scalar_one_or_none()
                    if conv:
                        from app.services.workspace.conversation_classifier import (
                            ConversationClassifier,
                        )

                        # 최근 메시지를 dict로 변환
                        recent_msgs_result = await db.execute(
                            select(ChatMessage)
                            .where(ChatMessage.conversation_id == conversation_id)
                            .order_by(ChatMessage.created_at.desc())
                            .limit(20)
                        )
                        recent_msgs = [
                            {"role": m.role, "content": m.content}
                            for m in reversed(list(recent_msgs_result.scalars().all()))
                        ]
                        await ConversationClassifier.auto_classify_if_needed(
                            db,
                            conversation_id,
                            recent_msgs,
                            is_title_manual=conv.is_title_manual,
                            tagged_items=conv.tagged_items,
                        )

                if user_msg_count >= _SUMMARIZE_INTERVAL and user_msg_count % _SUMMARIZE_INTERVAL == 0:
                    from app.services.workspace.structured_summarizer import (
                        StructuredSummarizer,
                    )

                    all_msgs_result = await db.execute(
                        select(ChatMessage)
                        .where(ChatMessage.conversation_id == conversation_id)
                        .order_by(ChatMessage.created_at)
                    )
                    all_msgs = [
                        {"role": m.role, "content": m.content}
                        for m in all_msgs_result.scalars().all()
                    ]
                    conv_result2 = await db.execute(
                        select(ChatConversation.summary).where(
                            ChatConversation.id == conversation_id
                        )
                    )
                    existing_summary = conv_result2.scalar_one_or_none()
                    summary = await StructuredSummarizer.generate_summary(
                        all_msgs, existing_summary
                    )
                    await ChatPersistenceService.update_summary(
                        db, conversation_id, summary
                    )

            except Exception:
                logger.debug("자동 분류/요약 실패 (무시)", exc_info=True)

            await db.commit()
        except Exception:
            await db.rollback()
            logger.exception("대화 영속화 후처리 실패")

    return tags_extracted


@router.post("", response_model=ChatResponse)
@limiter.limit(AI_RATE_LIMIT)
async def chat(request: Request, chat_request: ChatRequest) -> ChatResponse:
    """
    채팅 메시지 처리

    LangGraph StateGraph가 메시지를 분석하고
    적절한 에이전트 노드를 선택하여 응답을 생성합니다.
    """
    try:
        session_token: str = request.state.session_token
        graph = get_graph()

        # thread_id 결정 및 대화 영속화
        async with async_session_factory() as db:
            conversation = await ChatPersistenceService.get_or_create_conversation(
                db,
                session_token,
                chat_request.session_data.get("thread_id") or str(uuid.uuid4()),
                chat_request.conversation_id,
            )
            thread_id = conversation.thread_id
            conv_id = conversation.id

            # 사용자 메시지 저장
            await ChatPersistenceService.save_message(
                db, conv_id, "user", chat_request.message
            )
            await db.commit()

        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        # session_secret: LangGraph 내부용 (응답에는 포함하지 않음)
        session_secret = (
            chat_request.session_data.get("session_secret") or secrets.token_hex(16)
        )

        # interrupt 재개 여부 판단
        if chat_request.session_data.get("thread_id") or chat_request.conversation_id:
            graph_state = await graph.aget_state(config)
            if graph_state.tasks and any(t.interrupts for t in graph_state.tasks):
                result = await graph.ainvoke(
                    Command(resume=chat_request.message), config
                )

                # interrupt 후 또 다른 interrupt 확인
                graph_state2 = await graph.aget_state(config)
                if graph_state2.tasks and any(t.interrupts for t in graph_state2.tasks):
                    interrupt_data = graph_state2.tasks[0].interrupts[0].value
                    response_text = interrupt_data.get("response", "")
                    await _persist_after_graph(
                        conv_id, response_text, "small_claims", result, chat_request.message
                    )
                    return ChatResponse(
                        response=response_text,
                        agent_used="small_claims",
                        sources=interrupt_data.get("sources", []),
                        actions=interrupt_data.get("actions", []),
                        session_data={"thread_id": thread_id},
                        confidence=1.0,
                    )

                # 정상 완료
                response = state_to_response(result)
                response.session_data["thread_id"] = thread_id
                response.session_data["conversation_id"] = str(conv_id)
                await _persist_after_graph(
                    conv_id, response.response, response.agent_used, result, chat_request.message
                )
                return response

        # 새 대화 또는 interrupt가 아닌 경우
        state = request_to_state(chat_request)
        state["session_secret"] = session_secret
        state["conversation_id"] = str(conv_id)
        state["case_id"] = chat_request.case_id

        result = await graph.ainvoke(state, config)

        # interrupt 발생 여부 확인
        graph_state = await graph.aget_state(config)
        if graph_state.tasks and any(t.interrupts for t in graph_state.tasks):
            interrupt_data = graph_state.tasks[0].interrupts[0].value
            response_text = interrupt_data.get("response", "")
            await _persist_after_graph(
                conv_id, response_text, "small_claims", result, chat_request.message
            )
            return ChatResponse(
                response=response_text,
                agent_used="small_claims",
                sources=interrupt_data.get("sources", []),
                actions=interrupt_data.get("actions", []),
                session_data={"thread_id": thread_id},
                confidence=1.0,
            )

        # 정상 완료
        response = state_to_response(result)
        response.session_data["thread_id"] = thread_id
        response.session_data["conversation_id"] = str(conv_id)
        await _persist_after_graph(
            conv_id, response.response, response.agent_used, result, chat_request.message
        )
        return response

    except HTTPException:
        raise
    except Exception:
        logger.exception("채팅 처리 중 오류 발생")
        raise HTTPException(
            status_code=500,
            detail="채팅 처리 중 오류가 발생했습니다.",
        )


@router.post("/stream")
@limiter.limit(AI_RATE_LIMIT)
async def chat_stream(request: Request, chat_request: ChatRequest) -> EventSourceResponse:
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

    async def event_generator() -> Any:
        try:
            session_token: str = request.state.session_token
            graph = get_graph()

            # 대화 영속화: conversation 조회/생성 + 사용자 메시지 저장
            async with async_session_factory() as db:
                conversation = await ChatPersistenceService.get_or_create_conversation(
                    db,
                    session_token,
                    chat_request.session_data.get("thread_id") or str(uuid.uuid4()),
                    chat_request.conversation_id,
                )
                thread_id = conversation.thread_id
                conv_id = conversation.id

                await ChatPersistenceService.save_message(
                    db, conv_id, "user", chat_request.message
                )
                await db.commit()

            config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

            # session_secret: LangGraph 내부용
            session_secret = (
                chat_request.session_data.get("session_secret")
                or secrets.token_hex(16)
            )

            # interrupt 재개 여부 판단
            input_value: dict[str, Any] | Command[str]
            if chat_request.session_data.get("thread_id") or chat_request.conversation_id:
                graph_state = await graph.aget_state(config)
                if graph_state.tasks and any(
                    t.interrupts for t in graph_state.tasks
                ):
                    # interrupt 재개 전 최신 세션 데이터 반영
                    await graph.aupdate_state(
                        config, {"session_data": chat_request.session_data}
                    )
                    input_value = Command(resume=chat_request.message)
                else:
                    input_value = request_to_state(chat_request)
                    input_value["session_secret"] = session_secret
                    input_value["conversation_id"] = str(conv_id)
                    input_value["case_id"] = chat_request.case_id
            else:
                input_value = request_to_state(chat_request)
                input_value["session_secret"] = session_secret
                input_value["conversation_id"] = str(conv_id)
                input_value["case_id"] = chat_request.case_id

            # 응답 텍스트 수집 (DB 저장용)
            collected_response = []

            # astream으로 custom 이벤트 수신
            async for chunk in graph.astream(
                input_value, config, stream_mode="custom"
            ):
                event_type = chunk.get("event", "token")
                yield {
                    "event": event_type,
                    "data": json.dumps(
                        chunk.get("data", {}), ensure_ascii=False
                    ),
                }
                # token 이벤트에서 응답 텍스트 수집
                if event_type == "token":
                    token_content = chunk.get("data", {}).get("content", "")
                    if token_content:
                        collected_response.append(token_content)

            # 스트리밍 종료 후 interrupt 확인
            graph_state = await graph.aget_state(config)
            if graph_state.tasks and any(
                t.interrupts for t in graph_state.tasks
            ):
                interrupt_data = graph_state.tasks[0].interrupts[0].value
                response_text = interrupt_data.get("response", "")
                if response_text:
                    yield {
                        "event": "token",
                        "data": json.dumps(
                            {"content": response_text},
                            ensure_ascii=False,
                        ),
                    }
                    collected_response.append(response_text)

                yield {
                    "event": "metadata",
                    "data": json.dumps(
                        {
                            "agent_used": "small_claims",
                            "actions": interrupt_data.get("actions", []),
                            "session_data": {
                                "thread_id": thread_id,
                                "conversation_id": str(conv_id),
                                "dispute_type": graph_state.values.get("dispute_type"),
                                "step": interrupt_data.get("step") or graph_state.values.get("step"),
                                "claim_amount": graph_state.values.get("claim_amount"),
                            },
                        },
                        ensure_ascii=False,
                    ),
                }

            # 대화 영속화 후처리
            final_state = await graph.aget_state(config)
            final_values = final_state.values or {}
            agent_used = final_values.get("agent_used", "unknown")
            full_response = "".join(collected_response)

            tags_extracted = await _persist_after_graph(
                conv_id, full_response, agent_used, final_values, chat_request.message
            )

            # done 이벤트
            done_data: dict[str, Any] = {
                "thread_id": thread_id,
                "conversation_id": str(conv_id),
                "tags_extracted": tags_extracted,
            }
            if chat_request.case_id:
                done_data["case_id"] = chat_request.case_id
            output_sd = final_values.get("output_session_data", {})
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
        except Exception:
            logger.exception("스트리밍 채팅 처리 중 오류 발생")
            yield {
                "event": "error",
                "data": json.dumps(
                    {"message": "채팅 처리 중 오류가 발생했습니다."},
                    ensure_ascii=False,
                ),
            }

    return EventSourceResponse(event_generator())


@router.get("/agents")
async def list_agents() -> dict[str, Any]:
    """
    사용 가능한 에이전트 목록 반환
    """
    return {"agents": AGENT_LIST}
