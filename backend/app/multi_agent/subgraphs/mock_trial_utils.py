"""
모의 법정 유틸리티 및 헬퍼 함수

mock_trial.py에서 분리된 공통 헬퍼/유틸리티 함수 모음입니다.
"""

import html
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from langgraph.types import Command

from app.multi_agent.agents.base_chat import ActionType, ChatAction
from app.multi_agent.subgraphs.mock_trial_agents import CourtAgent
from app.multi_agent.subgraphs.mock_trial_prompts import (
    AGENT_CONFIGS,
    SYSTEM_PROMPTS,
    build_system_prompt,
)
from app.services.service_function.mock_trial_service import (
    build_rag_context,
    search_for_role,
    search_rebuttal,
)

logger = logging.getLogger(__name__)

MAX_LLM_CALLS_PER_SESSION = 50


def _now_iso() -> str:
    """현재 시각 ISO 형식"""
    return datetime.now(tz=timezone.utc).isoformat()


def _record(
    court_record: list[dict[str, Any]],
    stage: str,
    speaker: str,
    content: str,
) -> list[dict[str, Any]]:
    """서기 기록에 엔트리 추가 후 새 리스트 반환"""
    new_record = list(court_record)
    summary = content[:200] if len(content) > 200 else content
    new_record.append({
        "stage": stage,
        "speaker": speaker,
        "content": html.escape(summary),
        "timestamp": _now_iso(),
    })
    return new_record


def _validate_node_input(
    state: Any,
    required_keys: list[str],
) -> None:
    """노드 진입 시 필수 상태 키를 검증합니다 (FR-41).

    Args:
        state: 현재 서브그래프 상태
        required_keys: 필수 키 목록

    Raises:
        ValueError: 필수 키가 누락된 경우
    """
    missing = [k for k in required_keys if not state.get(k)]
    if missing:
        raise ValueError(
            f"MockTrialState 필수 키 누락: {', '.join(missing)}"
        )


def _init_agents(case_type: str) -> dict[str, dict[str, Any]]:
    """사건 유형에 맞는 에이전트 초기화"""
    agents: dict[str, dict[str, Any]] = {}
    for role, config in AGENT_CONFIGS.items():
        base_prompt = SYSTEM_PROMPTS.get((case_type, role), "")
        prompt = build_system_prompt(base_prompt)
        agent = CourtAgent(
            role=role,
            name=config["name"],
            system_prompt=prompt,
            temperature=config["temperature"],
            tools=config.get("tools", []),
        )
        agents[role] = agent.to_state()
    return agents


def _get_agent(state: Any, role: str) -> CourtAgent:
    """state에서 CourtAgent 복원"""
    agents = state.get("agents", {})
    agent_state = agents.get(role, {})
    case_type = state.get("case_type", "criminal")
    prompt = SYSTEM_PROMPTS.get((case_type, role), "")
    temperature = AGENT_CONFIGS.get(role, {}).get("temperature", 0.5)
    return CourtAgent.from_state(agent_state, prompt, temperature)


def _update_agent_in_state(
    agents: dict[str, dict[str, Any]],
    agent: CourtAgent,
) -> dict[str, dict[str, Any]]:
    """에이전트 상태를 갱신한 새 dict 반환"""
    new_agents = dict(agents)
    new_agents[agent.role] = agent.to_state()
    return new_agents


def _get_opponent_role(state: Any) -> str:
    """사용자의 상대 역할 반환"""
    user_role = state.get("user_role", "prosecutor")
    if user_role == "prosecutor":
        return "attorney"
    return "prosecutor"


async def _build_stage_rag_context(
    state: Any, role: str
) -> str:
    """캐시 확인 → 없으면 search_for_role 호출 → 컨텍스트 문자열 반환"""
    cache_key = f"rag_{role}_context"
    cached = str(state.get(cache_key, "") or "")
    if cached:
        return cached

    query = state.get("case_summary", "")
    case_type = state.get("case_type", "criminal")
    cases, articles = await search_for_role(role, query, case_type)
    return build_rag_context(role, cases, articles)


async def _build_rebuttal_context(
    court_record: list[dict[str, Any]],
    case_summary: str,
) -> str:
    """상대방 최근 발언 기반 반박 검색 → 컨텍스트 문자열 반환"""
    if not court_record:
        return ""
    last_entry = court_record[-1]
    opponent_stmt = last_entry.get("content", "")
    if not opponent_stmt:
        return ""
    cases, articles = await search_rebuttal(opponent_stmt, case_summary)
    return build_rag_context("rebuttal", cases, articles)


def _check_rate_limit(state: Any) -> Optional[Command[str]]:
    """LLM 호출 횟수 초과 시 verdict_node로 강제 이동"""
    if state.get("llm_call_count", 0) >= MAX_LLM_CALLS_PER_SESSION:
        logger.warning(
            "Rate limit 초과: %d/%d calls",
            state.get("llm_call_count", 0),
            MAX_LLM_CALLS_PER_SESSION,
        )
        return Command(
            update={
                "response": (
                    "세션당 LLM 호출 횟수 제한(50회)을 초과하여 "
                    "재판을 종결합니다."
                ),
                "speaking_agent": "judge",
                "agent_used": "mock_trial",
            },
            goto="verdict_node",
        )
    return None


def _case_type_actions() -> list[dict[str, Any]]:
    """사건 유형 선택 버튼"""
    return [
        ChatAction(
            type=ActionType.BUTTON,
            label="형사 재판",
            action="case_type_criminal",
        ).model_dump(),
        ChatAction(
            type=ActionType.BUTTON,
            label="민사 재판",
            action="case_type_civil",
        ).model_dump(),
    ]


def _generate_feedback(state: Any) -> str:
    """재판 결과 피드백 생성"""
    user_role = state.get("user_role", "")
    court_record = state.get("court_record", [])
    user_entries = [r for r in court_record if r.get("speaker") == user_role]
    entry_count = len(user_entries)

    feedback_parts = [
        f"총 {entry_count}회 발언하셨습니다.",
    ]

    if entry_count >= 3:
        feedback_parts.append("적극적으로 재판에 참여하셨습니다.")
    elif entry_count >= 1:
        feedback_parts.append("더 적극적으로 주장을 펼치면 좋겠습니다.")
    else:
        feedback_parts.append("발언 기회를 더 활용해보세요.")

    feedback_parts.append(
        "이 모의재판은 교육 목적이며 실제 법률 자문이 아닙니다."
    )
    return "\n".join(feedback_parts)
