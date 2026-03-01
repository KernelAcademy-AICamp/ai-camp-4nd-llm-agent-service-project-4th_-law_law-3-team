"""
LangGraph 노드 함수

router_node + 에이전트 노드 (legal_search, lawyer_finder, storyboard,
lawyer_stats, law_study, simple_chat)
SmallClaims는 별도 subgraph로 분리
"""

import asyncio
import logging
from typing import Any

from langgraph.types import Command, StreamWriter

from app.core.config import settings
from app.multi_agent.agents.base_chat import BaseChatAgent
from app.multi_agent.router import (
    INTENT_OVERRIDE_CONFIDENCE,
    ROLE_AGENTS,
    AgentType,
    RulesRouter,
    UserRole,
    detect_search_type,
)
from app.multi_agent.schemas.plan import AgentPlan
from app.multi_agent.state import ChatState

logger = logging.getLogger(__name__)

# agent_type -> node_name 매핑
AGENT_NODE_MAP: dict[str, str] = {
    # 공통
    "legal_search": "legal_search_node",
    "case_search": "legal_search_node",
    "law_search": "legal_search_node",
    "storyboard": "storyboard_subgraph",
    # 일반인
    "lawyer_finder": "lawyer_finder_node",
    "small_claims": "small_claims_subgraph",
    # 변호사
    "lawyer_stats": "lawyer_stats_node",
    "law_study": "law_study_node",
    # 공통 (체험형)
    "mock_trial": "mock_trial_subgraph",
    # 콘텐츠 마케팅
    "content_marketing": "content_marketing_node",
    # 폴백
    "general": "simple_chat_node",
    # 하위호환 (기존 agent_override 지원)
    "legal_answer": "legal_search_node",
}

# 싱글톤 라우터
_rules_router: RulesRouter | None = None


def _get_rules_router() -> RulesRouter:
    global _rules_router
    if _rules_router is None:
        _rules_router = RulesRouter()
    return _rules_router


# ──────────────────────────────────────────────
# 노드 실행 헬퍼 (#3 보일러플레이트 제거, #5 에러 핸들링, #9 중복 done 제거)
# ──────────────────────────────────────────────

_TIMEOUT_RESPONSE = "응답 시간이 초과되었습니다. 잠시 후 다시 시도해주세요."


async def _run_streaming_node_inner(
    agent: BaseChatAgent,
    state: ChatState,
    writer: StreamWriter,
) -> dict[str, Any]:
    """스트리밍 에이전트 실행 (타임아웃 없이)"""
    full_response = ""
    sources: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    output_session_data: dict[str, Any] = {}

    async for event_type, data in agent.process_stream(
        message=state["message"],
        history=state.get("history"),
        session_data=state.get("session_data"),
        user_location=state.get("user_location"),
    ):
        if event_type == "done":
            continue
        writer({"event": event_type, "data": data})

        if event_type == "token":
            full_response += data.get("content", "")
        elif event_type == "sources":
            sources = data.get("sources", [])
        elif event_type == "metadata":
            actions = data.get("actions", [])
            output_session_data = data.get("session_data", {})

    return {
        "response": full_response,
        "sources": sources,
        "actions": actions,
        "output_session_data": output_session_data,
        "agent_used": agent.name,
    }


async def _run_streaming_node(
    agent: BaseChatAgent,
    state: ChatState,
    writer: StreamWriter,
) -> dict[str, Any]:
    """스트리밍 에이전트 공통 실행 헬퍼 (타임아웃 + 에러 핸들링)

    process_stream()의 이벤트를 writer로 전달하고 결과를 수집한다.
    done 이벤트는 chat.py에서 전송하므로 여기서 필터링한다.
    """
    try:
        return await asyncio.wait_for(
            _run_streaming_node_inner(agent, state, writer),
            timeout=settings.AGENT_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.error(
            "에이전트 타임아웃: %s (%ds)", agent.name, settings.AGENT_TIMEOUT_SECONDS
        )
        writer({
            "event": "error",
            "data": {"message": _TIMEOUT_RESPONSE},
        })
    except Exception:
        logger.exception("에이전트 스트리밍 오류: %s", agent.name)
        writer({
            "event": "error",
            "data": {"message": "처리 중 오류가 발생했습니다."},
        })

    return {
        "response": "",
        "sources": [],
        "actions": [],
        "output_session_data": {},
        "agent_used": agent.name,
    }


async def _run_nonstreaming_node(
    agent: BaseChatAgent,
    state: ChatState,
    writer: StreamWriter,
) -> dict[str, Any]:
    """비스트리밍 에이전트 공통 실행 헬퍼 (타임아웃 + 에러 핸들링)

    process()를 호출한 뒤 결과를 writer로 한꺼번에 전달한다.
    done 이벤트는 chat.py에서 전송하므로 여기서 보내지 않는다.
    """
    try:
        result = await asyncio.wait_for(
            agent.process(
                message=state["message"],
                history=state.get("history"),
                session_data=state.get("session_data"),
                user_location=state.get("user_location"),
            ),
            timeout=settings.AGENT_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.error(
            "에이전트 타임아웃: %s (%ds)", agent.name, settings.AGENT_TIMEOUT_SECONDS
        )
        writer({
            "event": "error",
            "data": {"message": _TIMEOUT_RESPONSE},
        })
        return {
            "response": _TIMEOUT_RESPONSE,
            "sources": [],
            "actions": [],
            "output_session_data": {},
            "agent_used": agent.name,
        }
    except Exception:
        logger.exception("에이전트 처리 오류: %s", agent.name)
        writer({
            "event": "error",
            "data": {"message": "처리 중 오류가 발생했습니다."},
        })
        return {
            "response": "처리 중 오류가 발생했습니다.",
            "sources": [],
            "actions": [],
            "output_session_data": {},
            "agent_used": agent.name,
        }

    writer({"event": "token", "data": {"content": result.message}})
    writer({"event": "sources", "data": {"sources": result.sources}})
    writer({
        "event": "metadata",
        "data": {
            "agent_used": agent.name,
            "actions": result.actions,
            "session_data": result.session_data,
        },
    })

    return {
        "response": result.message,
        "sources": result.sources,
        "actions": result.actions,
        "output_session_data": result.session_data,
        "agent_used": agent.name,
    }


# ──────────────────────────────────────────────
# 라우터 노드
# ──────────────────────────────────────────────


def router_node(state: ChatState) -> Command[str]:
    """라우팅 노드: 메시지를 분석하여 적절한 에이전트 노드로 라우팅

    Args:
        state: 현재 그래프 상태

    Returns:
        Command with goto target node
    """
    message = state["message"]
    user_role = state.get("user_role", "user")
    session_data = state.get("session_data", {})
    agent_override = state.get("agent_override")
    router = _get_rules_router()

    # agent_override ROLE_AGENTS 검증 (#2)
    if agent_override:
        try:
            role_enum = UserRole(user_role)
            override_enum = AgentType(agent_override)
            if override_enum not in ROLE_AGENTS.get(role_enum, []):
                logger.warning(
                    "Agent %s not allowed for role %s, falling back",
                    agent_override,
                    user_role,
                )
                agent_override = None
        except ValueError:
            logger.warning(
                "유효하지 않은 agent_override 값: %s — 검증 스킵",
                agent_override,
            )

    # 규칙 기반 라우팅 (1회만 호출)
    plan: AgentPlan | None = None

    # 강한 명시 의도는 URL agent_override보다 우선
    if agent_override:
        plan = router.route(
            message=message,
            user_role=user_role,
            session_data={},
        )
        if (
            plan.confidence >= INTENT_OVERRIDE_CONFIDENCE
            and plan.agent_type != agent_override
        ):
            logger.info(
                "Agent override released: %s -> %s (confidence: %.2f)",
                agent_override,
                plan.agent_type,
                plan.confidence,
            )
            agent_override = None

    # 에이전트 직접 지정
    if agent_override:
        target_node = AGENT_NODE_MAP.get(agent_override, "legal_search_node")

        # search_focus 결정 (agent_override용)
        search_focus = ""
        if agent_override == "case_search":
            search_focus = "precedent"
        elif agent_override == "law_search":
            search_focus = "law"
        elif agent_override in ("legal_search", "legal_answer"):
            search_focus = detect_search_type(message)

        logger.info(
            "Agent directly specified: %s -> %s", agent_override, target_node
        )
        return Command(
            update={
                "selected_agent": agent_override,
                "search_focus": search_focus,
                "routing_confidence": 1.0,
                "routing_reason": "직접 지정",
            },
            goto=target_node,
        )

    # override가 없었으면 여기서 라우팅
    if plan is None:
        plan = router.route(
            message=message,
            user_role=user_role,
            session_data=session_data,
        )

    # search_focus 결정
    search_focus = ""
    if plan.agent_type in ("legal_search", "case_search", "law_search"):
        if plan.agent_type == "case_search":
            search_focus = "precedent"
        elif plan.agent_type == "law_search":
            search_focus = "law"
        else:
            search_focus = detect_search_type(message)

    target_node = AGENT_NODE_MAP.get(plan.agent_type, "legal_search_node")

    logger.info(
        "Routed to %s -> %s (confidence: %.2f, reason: %s)",
        plan.agent_type,
        target_node,
        plan.confidence,
        plan.reason,
    )

    return Command(
        update={
            "selected_agent": plan.agent_type,
            "search_focus": search_focus,
            "routing_confidence": plan.confidence,
            "routing_reason": plan.reason or "",
        },
        goto=target_node,
    )


# ──────────────────────────────────────────────
# 에이전트 노드
# ──────────────────────────────────────────────


async def legal_search_node(
    state: ChatState, writer: StreamWriter
) -> dict[str, Any]:
    """법률 검색 노드 (RAG 기반 판례/법령 검색)"""
    from app.multi_agent.agents.legal_search_agent import LegalSearchAgent

    focus = state.get("search_focus", "precedent")
    if focus not in ("precedent", "law"):
        focus = "precedent"

    return await _run_streaming_node(LegalSearchAgent(focus=focus), state, writer)


async def lawyer_finder_node(
    state: ChatState, writer: StreamWriter
) -> dict[str, Any]:
    """변호사 찾기 노드"""
    from app.multi_agent.agents.lawyer_finder_agent import LawyerFinderAgent

    return await _run_nonstreaming_node(LawyerFinderAgent(), state, writer)


async def lawyer_stats_node(
    state: ChatState, writer: StreamWriter
) -> dict[str, Any]:
    """변호사 시장 분석 노드 (LLM 스트리밍)"""
    from app.multi_agent.agents.lawyer_stats_agent import LawyerStatsAgent

    return await _run_streaming_node(LawyerStatsAgent(), state, writer)


async def law_study_node(
    state: ChatState, writer: StreamWriter
) -> dict[str, Any]:
    """로스쿨 학습 노드"""
    from app.multi_agent.agents.law_study_agent import LawStudyAgent

    return await _run_streaming_node(LawStudyAgent(), state, writer)


async def content_marketing_node(
    state: ChatState, writer: StreamWriter
) -> dict[str, Any]:
    """콘텐츠 마케팅 노드 (트렌드 분석 + 대본 생성)"""
    from app.multi_agent.agents.content_marketing_agent import ContentMarketingAgent

    return await _run_nonstreaming_node(ContentMarketingAgent(), state, writer)


async def simple_chat_node(
    state: ChatState, writer: StreamWriter
) -> dict[str, Any]:
    """일반 채팅 노드"""
    from app.multi_agent.agents.base_chat import SimpleChatAgent

    return await _run_streaming_node(SimpleChatAgent(), state, writer)
