"""
모의 법정 민사 전용 노드

민사 재판 단계별 노드 함수:
- pretrial_node: 변론준비 (민사소송법 §258~§268)
- claims_node: 주장/답변 (민사소송법 §256~§257)
- argument_node: 변론 (민사소송법 §134~§148)
- civil_closing_node: 변론종결 (민사소송법 §200)
"""

from typing import TYPE_CHECKING, Any

from langgraph.types import Command, interrupt

from app.multi_agent.agents.base_chat import ActionType, ChatAction
from app.multi_agent.subgraphs.mock_trial_prompts import AGENT_CONFIGS
from app.multi_agent.subgraphs.mock_trial_utils import (
    _build_rebuttal_context,
    _check_rate_limit,
    _get_agent,
    _get_opponent_role,
    _record,
    _update_agent_in_state,
    _validate_node_input,
)

if TYPE_CHECKING:
    from app.multi_agent.subgraphs.mock_trial import MockTrialState


async def pretrial_node(state: "MockTrialState") -> Command[str]:
    """[민사] 변론준비 (민사소송법 §258~§268)

    재판장이 쟁점을 정리하고 증거 목록을 확인합니다.
    """
    _validate_node_input(state, ["case_type", "agents", "case_summary"])
    if (cmd := _check_rate_limit(state)) is not None:
        return cmd
    llm_call_count = state.get("llm_call_count", 0)
    court_record = list(state.get("court_record", []))
    judge = _get_agent(state, "judge")

    judge.update_strategy("쟁점 정리 및 증거 목록 확인")
    response, pretrial_emotion = await judge.generate(
        "pretrial", state.get("case_summary", ""), court_record
    )
    llm_call_count += 1
    court_record = _record(court_record, "pretrial", "judge", response)

    # 자동 진행 (관전)
    interrupt({
        "response": f"[재판장] {response}",
        "speaking_agent": "judge",
        "emotion": pretrial_emotion,
        "stage": "pretrial",
        "references": state.get("rag_references", []),
        "actions": [
            ChatAction(
                type=ActionType.BUTTON,
                label="다음 단계로",
                action="next_stage",
            ).model_dump(),
        ],
    })

    return Command(
        update={
            "stage": "pretrial",
            "court_record": court_record,
            "llm_call_count": llm_call_count,
            "agents": _update_agent_in_state(state.get("agents", {}), judge),
            "response": response,
            "speaking_agent": "judge",
            "emotion": pretrial_emotion,
            "agent_used": "mock_trial",
        },
        goto="claims_node",
    )


async def claims_node(state: "MockTrialState") -> Command[str]:
    """[민사] 주장/답변 (민사소송법 §256~§257)

    사용자 역할에 따라 원고/피고 입력을 받고 상대측 AI가 답변합니다.
    """
    if (cmd := _check_rate_limit(state)) is not None:
        return cmd
    llm_call_count = state.get("llm_call_count", 0)
    court_record = list(state.get("court_record", []))
    user_role = state.get("user_role", "prosecutor")
    agents_state: dict[str, Any] = dict(state.get("agents", {}))

    # RAG 컨텍스트 (evidence_node 캐시)
    rag_pros_ctx = state.get("rag_prosecutor_context", "")
    rag_atty_ctx = state.get("rag_attorney_context", "")
    rag_hints = state.get("rag_user_hints", [])

    if user_role == "prosecutor":
        # 사용자가 원고측 → 청구원인 입력
        interrupt_value = interrupt({
            "response": "원고 측 청구원인을 진술해주세요.",
            "speaking_agent": "judge",
            "stage": "claims",
            "user_hints": rag_hints,
            "references": state.get("rag_references", []),
            "actions": [],
        })
        user_input = str(interrupt_value)
        court_record = _record(court_record, "claims", "prosecutor", user_input)

        # AI 피고측 답변 (RAG 컨텍스트 전달)
        attorney = _get_agent(state, "attorney")
        attorney.update_strategy("원고 청구에 대한 항변 제시")
        opponent_response, claims_emotion = await attorney.generate(
            "claims", state.get("case_summary", ""), court_record,
            rag_context=rag_atty_ctx,
        )
        llm_call_count += 1
        court_record = _record(
            court_record, "claims", "attorney", opponent_response
        )
        agents_state = _update_agent_in_state(agents_state, attorney)
        final_response = f"[피고측] {opponent_response}"
    else:
        # AI 원고측 발언 (RAG 컨텍스트 전달)
        prosecutor = _get_agent(state, "prosecutor")
        prosecutor.update_strategy("청구원인 구체적 입증")
        pros_claim, claims_emotion = await prosecutor.generate(
            "claims", state.get("case_summary", ""), court_record,
            rag_context=rag_pros_ctx,
        )
        llm_call_count += 1
        court_record = _record(
            court_record, "claims", "prosecutor", pros_claim
        )
        agents_state = _update_agent_in_state(agents_state, prosecutor)

        # 사용자(피고측) 답변
        interrupt_value = interrupt({
            "response": (
                f"[원고측] {pros_claim}\n\n"
                "피고 측 답변을 해주세요."
            ),
            "speaking_agent": "prosecutor",
            "emotion": claims_emotion,
            "stage": "claims",
            "user_hints": rag_hints,
            "references": state.get("rag_references", []),
            "actions": [],
        })
        user_input = str(interrupt_value)
        court_record = _record(court_record, "claims", "attorney", user_input)
        final_response = f"[피고측] {user_input}"

    return Command(
        update={
            "stage": "claims",
            "court_record": court_record,
            "llm_call_count": llm_call_count,
            "agents": agents_state,
            "response": final_response,
            "speaking_agent": _get_opponent_role(state),
            "emotion": claims_emotion,
            "agent_used": "mock_trial",
        },
        goto="evidence_node",
    )


async def argument_node(state: "MockTrialState") -> Command[str]:
    """[민사] 변론 (민사소송법 §134~§148) — 2-3 라운드 루프"""
    if (cmd := _check_rate_limit(state)) is not None:
        return cmd
    llm_call_count = state.get("llm_call_count", 0)
    court_record = list(state.get("court_record", []))
    current_round = state.get("current_round", 1)
    max_rounds = state.get("max_rounds", 3)
    agents_state: dict[str, Any] = dict(state.get("agents", {}))

    rag_hints = state.get("rag_user_hints", [])
    interrupt_value = interrupt({
        "response": (
            f"[변론 라운드 {current_round}/{max_rounds}] "
            "주장을 입력하세요."
        ),
        "stage": "argument",
        "user_hints": rag_hints,
        "references": state.get("rag_references", []),
        "actions": [
            ChatAction(
                type=ActionType.BUTTON,
                label="변론 종결 요청",
                action="end_argument",
            ).model_dump(),
        ],
    })

    user_input = str(interrupt_value) if interrupt_value else ""

    if user_input == "end_argument" or current_round >= max_rounds:
        return Command(
            update={
                "stage": "argument",
                "current_round": current_round,
                "court_record": court_record,
                "response": "변론을 종결합니다.",
                "speaking_agent": "judge",
                "agent_used": "mock_trial",
            },
            goto="civil_closing_node",
        )

    # 사용자 발언 기록
    user_role = state.get("user_role", "prosecutor")
    court_record = _record(court_record, "argument", user_role, user_input)

    # 매 라운드 반박용 경량 RAG 검색
    rebuttal_ctx = await _build_rebuttal_context(
        court_record, state.get("case_summary", "")
    )
    # 캐시된 역할별 RAG + 반박 RAG 결합
    opponent_role = _get_opponent_role(state)
    cached_ctx = state.get(f"rag_{opponent_role}_context", "")
    combined_ctx = f"{cached_ctx}\n\n{rebuttal_ctx}".strip()

    # AI 반론 생성 (RAG 컨텍스트 전달)
    opponent = _get_agent(state, opponent_role)
    opponent.update_strategy(f"라운드 {current_round} 반론")
    rebuttal, argument_emotion = await opponent.generate(
        "argument", state.get("case_summary", ""), court_record,
        rag_context=combined_ctx,
    )
    llm_call_count += 1
    court_record = _record(court_record, "argument", opponent_role, rebuttal)
    agents_state = _update_agent_in_state(agents_state, opponent)

    return Command(
        update={
            "stage": "argument",
            "current_round": current_round + 1,
            "court_record": court_record,
            "llm_call_count": llm_call_count,
            "agents": agents_state,
            "response": f"[{AGENT_CONFIGS.get(opponent_role, {}).get('name', opponent_role)}] {rebuttal}",
            "speaking_agent": opponent_role,
            "emotion": argument_emotion,
            "agent_used": "mock_trial",
        },
        goto="argument_node",
    )


async def civil_closing_node(state: "MockTrialState") -> Command[str]:
    """[민사] 변론종결 (민사소송법 §200)

    양측 최종 주장을 정리합니다.
    """
    if (cmd := _check_rate_limit(state)) is not None:
        return cmd
    llm_call_count = state.get("llm_call_count", 0)
    court_record = list(state.get("court_record", []))
    user_role = state.get("user_role", "prosecutor")
    agents_state: dict[str, Any] = dict(state.get("agents", {}))

    # RAG 컨텍스트 + 반박
    rag_hints = state.get("rag_user_hints", [])
    rebuttal_ctx = await _build_rebuttal_context(
        court_record, state.get("case_summary", "")
    )

    # 사용자 최종 주장
    interrupt_value = interrupt({
        "response": "최종 주장을 정리하여 진술해주세요.",
        "speaking_agent": "judge",
        "stage": "closing",
        "user_hints": rag_hints,
        "references": state.get("rag_references", []),
        "actions": [],
    })
    user_stmt = str(interrupt_value)
    court_record = _record(court_record, "closing", user_role, user_stmt)

    # AI 상대측 최종 주장 (캐시 RAG + 반박 결합)
    opponent_role = _get_opponent_role(state)
    cached_ctx = state.get(f"rag_{opponent_role}_context", "")
    combined_ctx = f"{cached_ctx}\n\n{rebuttal_ctx}".strip()
    opponent = _get_agent(state, opponent_role)
    opponent.update_strategy("최종 주장 정리")
    opponent_closing, civil_closing_emotion = await opponent.generate(
        "closing", state.get("case_summary", ""), court_record,
        rag_context=combined_ctx,
    )
    llm_call_count += 1
    court_record = _record(
        court_record, "closing", opponent_role, opponent_closing
    )
    agents_state = _update_agent_in_state(agents_state, opponent)

    return Command(
        update={
            "stage": "closing",
            "court_record": court_record,
            "llm_call_count": llm_call_count,
            "agents": agents_state,
            "response": f"[{AGENT_CONFIGS.get(opponent_role, {}).get('name', opponent_role)}] {opponent_closing}",
            "speaking_agent": opponent_role,
            "emotion": civil_closing_emotion,
            "agent_used": "mock_trial",
        },
        goto="verdict_node",
    )
