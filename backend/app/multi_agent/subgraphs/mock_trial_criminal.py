"""
모의 법정 형사 전용 노드

형사 재판 단계별 노드 함수:
- identity_node: 인정신문 (형사소송법 §284)
- opening_node: 모두진술 (형사소송법 §285~§286)
- examination_node: 피고인신문 (형사소송법 §296-2)
- criminal_closing_node: 최종변론 (형사소송법 §302~§303)
"""

from typing import TYPE_CHECKING, Any

from langgraph.types import Command, interrupt

from app.multi_agent.agents.base_chat import ActionType, ChatAction
from app.multi_agent.subgraphs.mock_trial_utils import (
    _build_rebuttal_context,
    _build_stage_rag_context,
    _check_rate_limit,
    _get_agent,
    _record,
    _update_agent_in_state,
    _validate_node_input,
)

if TYPE_CHECKING:
    from app.multi_agent.subgraphs.mock_trial import MockTrialState


async def identity_node(state: "MockTrialState") -> Command[str]:
    """[형사] 인정신문 (형사소송법 §284)

    재판장이 피고인 인적사항을 확인하고 진술거부권을 고지합니다.
    """
    _validate_node_input(state, ["case_type", "agents", "case_summary"])
    if (cmd := _check_rate_limit(state)) is not None:
        return cmd
    llm_call_count = state.get("llm_call_count", 0)
    court_record = list(state.get("court_record", []))
    judge = _get_agent(state, "judge")

    judge.update_strategy("피고인 인적사항 확인 및 진술거부권 고지")
    response, identity_emotion = await judge.generate(
        "identity", state.get("case_summary", ""), court_record
    )
    llm_call_count += 1
    court_record = _record(court_record, "identity", "judge", response)

    # 자동 진행 (관전) - interrupt로 표시만 하고 다음 단계로
    interrupt({
        "response": f"[재판장] {response}",
        "speaking_agent": "judge",
        "emotion": identity_emotion,
        "stage": "identity",
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
            "stage": "identity",
            "court_record": court_record,
            "llm_call_count": llm_call_count,
            "agents": _update_agent_in_state(state.get("agents", {}), judge),
            "response": response,
            "speaking_agent": "judge",
            "emotion": identity_emotion,
            "agent_used": "mock_trial",
        },
        goto="opening_node",
    )


async def opening_node(state: "MockTrialState") -> Command[str]:
    """[형사] 모두진술 (형사소송법 §285~§286)

    검사 공소사실 요지 진술 후, 피고인/변호인 측 의견을 요청합니다.
    """
    if (cmd := _check_rate_limit(state)) is not None:
        return cmd
    llm_call_count = state.get("llm_call_count", 0)
    court_record = list(state.get("court_record", []))
    user_role = state.get("user_role", "prosecutor")

    # RAG 컨텍스트 (evidence_node에서 캐시)
    rag_pros_ctx = state.get("rag_prosecutor_context", "")
    rag_atty_ctx = state.get("rag_attorney_context", "")
    rag_hints = state.get("rag_user_hints", [])

    # 검사 측 모두진술 (사용자가 검사면 입력 대기, 아니면 AI 생성)
    if user_role == "prosecutor":
        interrupt_value = interrupt({
            "response": "검사 측 모두진술을 해주세요. 공소사실의 요지를 진술하세요.",
            "speaking_agent": "judge",
            "stage": "opening",
            "user_hints": rag_hints,
            "references": state.get("rag_references", []),
            "actions": [],
        })
        pros_stmt = str(interrupt_value)
        court_record = _record(court_record, "opening", "prosecutor", pros_stmt)

        # AI 변호인 반응 (RAG 컨텍스트 전달)
        attorney = _get_agent(state, "attorney")
        attorney.update_strategy("검사 주장에 대한 반박 준비")
        attorney_response, opening_emotion = await attorney.generate(
            "opening", state.get("case_summary", ""), court_record,
            rag_context=rag_atty_ctx,
        )
        llm_call_count += 1
        court_record = _record(
            court_record, "opening", "attorney", attorney_response
        )
        agents = _update_agent_in_state(state.get("agents", {}), attorney)
        final_response = f"[변호인] {attorney_response}"
    else:
        # AI 검사 발언 (RAG 컨텍스트 전달)
        prosecutor = _get_agent(state, "prosecutor")
        prosecutor.update_strategy("공소사실 입증을 위한 모두진술")
        pros_stmt, opening_emotion = await prosecutor.generate(
            "opening", state.get("case_summary", ""), court_record,
            rag_context=rag_pros_ctx,
        )
        llm_call_count += 1
        court_record = _record(court_record, "opening", "prosecutor", pros_stmt)
        agents = _update_agent_in_state(state.get("agents", {}), prosecutor)

        # 사용자(변호인) 입력 대기
        interrupt_value = interrupt({
            "response": (
                f"[검사] {pros_stmt}\n\n"
                "변호인 측 의견을 진술해주세요."
            ),
            "speaking_agent": "prosecutor",
            "emotion": opening_emotion,
            "stage": "opening",
            "user_hints": rag_hints,
            "references": state.get("rag_references", []),
            "actions": [
                ChatAction(
                    type=ActionType.BUTTON,
                    label="인정",
                    action="admit",
                ).model_dump(),
                ChatAction(
                    type=ActionType.BUTTON,
                    label="부인",
                    action="deny",
                ).model_dump(),
            ],
        })
        user_input = str(interrupt_value)
        court_record = _record(court_record, "opening", "attorney", user_input)
        final_response = f"[변호인] {user_input}"

    return Command(
        update={
            "stage": "opening",
            "court_record": court_record,
            "llm_call_count": llm_call_count,
            "agents": agents,
            "response": final_response,
            "speaking_agent": user_role,
            "emotion": opening_emotion,
            "agent_used": "mock_trial",
        },
        goto="evidence_node",
    )


async def examination_node(state: "MockTrialState") -> Command[str]:
    """[형사] 피고인신문 (형사소송법 §296-2)

    검사/변호인이 피고인에게 질문합니다.
    """
    if (cmd := _check_rate_limit(state)) is not None:
        return cmd
    llm_call_count = state.get("llm_call_count", 0)
    court_record = list(state.get("court_record", []))
    user_role = state.get("user_role", "prosecutor")

    # 피고인도 자기 상황 관련 법령 참조
    rag_defendant_ctx = await _build_stage_rag_context(state, "defendant")

    # 피고인 AI 발언 (RAG 컨텍스트 전달)
    defendant = _get_agent(state, "defendant")
    defendant.update_strategy("성실하게 답변, 유리한 사정 강조")
    defendant_stmt, exam_emotion = await defendant.generate(
        "examination", state.get("case_summary", ""), court_record,
        rag_context=rag_defendant_ctx,
    )
    llm_call_count += 1
    court_record = _record(
        court_record, "examination", "defendant", defendant_stmt
    )

    # 사용자 질문 입력
    rag_hints = state.get("rag_user_hints", [])
    interrupt_value = interrupt({
        "response": (
            f"[피고인] {defendant_stmt}\n\n"
            "피고인에게 질문을 하세요."
        ),
        "speaking_agent": "defendant",
        "emotion": exam_emotion,
        "stage": "examination",
        "user_hints": rag_hints,
        "references": state.get("rag_references", []),
        "actions": [
            ChatAction(
                type=ActionType.BUTTON,
                label="질문 완료",
                action="end_examination",
            ).model_dump(),
        ],
    })
    user_input = str(interrupt_value)

    if user_input and user_input != "end_examination":
        court_record = _record(
            court_record, "examination", user_role, user_input
        )
        # 피고인 추가 답변
        defendant_answer, exam_emotion = await defendant.generate(
            "examination", f"질문: {user_input}", court_record,
            rag_context=rag_defendant_ctx,
        )
        llm_call_count += 1
        court_record = _record(
            court_record, "examination", "defendant", defendant_answer
        )

    agents = _update_agent_in_state(state.get("agents", {}), defendant)

    return Command(
        update={
            "stage": "examination",
            "court_record": court_record,
            "llm_call_count": llm_call_count,
            "agents": agents,
            "response": defendant_stmt,
            "speaking_agent": "defendant",
            "emotion": exam_emotion,
            "agent_used": "mock_trial",
        },
        goto="criminal_closing_node",
    )


async def criminal_closing_node(state: "MockTrialState") -> Command[str]:
    """[형사] 최종변론 (형사소송법 §302~§303)

    검사 구형, 변호인 최후변론, 피고인 최후진술
    """
    if (cmd := _check_rate_limit(state)) is not None:
        return cmd
    llm_call_count = state.get("llm_call_count", 0)
    court_record = list(state.get("court_record", []))
    user_role = state.get("user_role", "prosecutor")
    agents_state: dict[str, Any] = dict(state.get("agents", {}))

    # RAG 컨텍스트 (evidence_node 캐시) + 반박 컨텍스트
    rag_pros_ctx = state.get("rag_prosecutor_context", "")
    rag_atty_ctx = state.get("rag_attorney_context", "")
    rag_hints = state.get("rag_user_hints", [])
    rebuttal_ctx = await _build_rebuttal_context(
        court_record, state.get("case_summary", "")
    )

    if user_role == "prosecutor":
        # 사용자(검사) 구형
        interrupt_value = interrupt({
            "response": "최종변론을 해주세요. 구형을 포함하여 의견을 진술하세요.",
            "speaking_agent": "judge",
            "stage": "closing",
            "user_hints": rag_hints,
            "references": state.get("rag_references", []),
            "actions": [],
        })
        user_stmt = str(interrupt_value)
        court_record = _record(court_record, "closing", "prosecutor", user_stmt)

        # AI 변호인 최후변론 (캐시 RAG + 반박 결합)
        combined_ctx = f"{rag_atty_ctx}\n\n{rebuttal_ctx}".strip()
        attorney = _get_agent(state, "attorney")
        attorney.update_strategy("피고인의 정상참작 사유 강조")
        attorney_response, closing_emotion = await attorney.generate(
            "closing", state.get("case_summary", ""), court_record,
            rag_context=combined_ctx,
        )
        llm_call_count += 1
        court_record = _record(
            court_record, "closing", "attorney", attorney_response
        )
        agents_state = _update_agent_in_state(agents_state, attorney)
    else:
        # AI 검사 구형 (캐시 RAG + 반박 결합)
        combined_ctx = f"{rag_pros_ctx}\n\n{rebuttal_ctx}".strip()
        prosecutor = _get_agent(state, "prosecutor")
        prosecutor.update_strategy("양형 기준에 따른 구형")
        pros_closing, closing_emotion = await prosecutor.generate(
            "closing", state.get("case_summary", ""), court_record,
            rag_context=combined_ctx,
        )
        llm_call_count += 1
        court_record = _record(
            court_record, "closing", "prosecutor", pros_closing
        )
        agents_state = _update_agent_in_state(agents_state, prosecutor)

        # 사용자(변호인) 최후변론
        interrupt_value = interrupt({
            "response": (
                f"[검사] {pros_closing}\n\n"
                "변호인의 최후변론을 해주세요."
            ),
            "speaking_agent": "prosecutor",
            "emotion": closing_emotion,
            "stage": "closing",
            "user_hints": rag_hints,
            "references": state.get("rag_references", []),
            "actions": [],
        })
        user_stmt = str(interrupt_value)
        court_record = _record(court_record, "closing", "attorney", user_stmt)

    # 피고인 최후진술
    defendant = _get_agent(state, "defendant")
    defendant.update_strategy("진심 어린 최후진술")
    defendant_stmt, closing_emotion = await defendant.generate(
        "closing", state.get("case_summary", ""), court_record
    )
    llm_call_count += 1
    court_record = _record(
        court_record, "closing", "defendant", defendant_stmt
    )
    agents_state = _update_agent_in_state(agents_state, defendant)

    return Command(
        update={
            "stage": "closing",
            "court_record": court_record,
            "llm_call_count": llm_call_count,
            "agents": agents_state,
            "response": f"[피고인 최후진술] {defendant_stmt}",
            "speaking_agent": "defendant",
            "emotion": closing_emotion,
            "agent_used": "mock_trial",
        },
        goto="verdict_node",
    )
