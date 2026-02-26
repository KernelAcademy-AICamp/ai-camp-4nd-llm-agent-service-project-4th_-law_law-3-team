"""
모의 법정 서브그래프

LangGraph interrupt + Command 패턴으로 형사 6단계 + 민사 6단계 재판을 구현합니다.
small_claims.py 서브그래프 패턴을 따릅니다.

Design 문서 Section 6.3 기반
"""

import html
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from app.multi_agent.agents.base_chat import ActionType, ChatAction
from app.multi_agent.subgraphs.mock_trial_agents import CourtAgent
from app.multi_agent.subgraphs.mock_trial_prompts import (
    AGENT_CONFIGS,
    BURDEN_OF_PROOF_CIVIL,
    BURDEN_OF_PROOF_CRIMINAL,
    STAGE_ESTIMATED_MINUTES,
    SYSTEM_PROMPTS,
    VERDICT_TEMPLATE_CIVIL,
    VERDICT_TEMPLATE_CRIMINAL,
    build_system_prompt,
    sanitize_user_input,
)
from app.services.service_function.mock_trial_service import (
    build_rag_context,
    build_user_hints,
    get_evidence_searcher,
    search_for_role,
    search_for_verdict,
    search_rebuttal,
)

logger = logging.getLogger(__name__)


# ── 상태 정의 ──


class MockTrialState(TypedDict, total=False):
    """모의재판 서브그래프 상태 (SmallClaimsState 패턴 준수)"""

    # 부모 그래프에서 전달
    message: str
    history: list[dict[str, str]]
    session_data: dict[str, Any]

    # 설정 (setup 단계에서 결정)
    case_type: str
    case_category: str
    user_role: str
    case_summary: str

    # 에이전트 상태
    agents: dict[str, dict[str, Any]]

    # 증거
    evidence_cases: list[dict[str, Any]]
    evidence_articles: list[dict[str, Any]]
    selected_evidence: list[str]
    excluded_evidence: list[str]

    # 재판 진행
    stage: str
    current_round: int
    max_rounds: int
    court_record: list[dict[str, Any]]

    # RAG 컨텍스트 (캐시)
    rag_prosecutor_context: str
    rag_attorney_context: str
    rag_user_hints: list[dict[str, Any]]

    # Rate limiting
    llm_call_count: int

    # 출력 (부모 그래프로 전달)
    response: str
    speaking_agent: str
    emotion: str
    actions: list[dict[str, Any]]
    judgment: Optional[str]
    feedback: Optional[str]
    is_complete: bool
    agent_used: str
    output_session_data: dict[str, Any]


# ── 헬퍼 함수 ──


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


def _validate_node_input(state: "MockTrialState", required_keys: list[str]) -> None:
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


def _get_agent(state: MockTrialState, role: str) -> CourtAgent:
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


def _get_opponent_role(state: MockTrialState) -> str:
    """사용자의 상대 역할 반환"""
    user_role = state.get("user_role", "prosecutor")
    if user_role == "prosecutor":
        return "attorney"
    return "prosecutor"


async def _build_stage_rag_context(
    state: MockTrialState, role: str
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


MAX_LLM_CALLS_PER_SESSION = 50


def _check_rate_limit(state: MockTrialState) -> Command[str] | None:
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


def _generate_feedback(state: MockTrialState) -> str:
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


# ── 공통 노드 ──


def setup_node(state: MockTrialState) -> Command[str]:
    """사건 설정 노드 (형사/민사 공통)"""
    interrupt_value = interrupt({
        "response": (
            "모의 법정에 오신 것을 환영합니다.\n\n"
            "사건 유형, 역할, 사건 개요를 입력해주세요.\n\n"
            "**주의: 이 모의재판은 교육 목적이며 실제 법률 자문이 아닙니다.**"
        ),
        "actions": _case_type_actions(),
        "step": "setup",
    })

    # resume 시 사용자 입력 파싱
    user_input = interrupt_value if isinstance(interrupt_value, dict) else {}
    case_type = str(user_input.get("case_type", "criminal"))
    user_role = str(user_input.get("user_role", "prosecutor"))
    case_category = str(user_input.get("case_category", ""))
    case_summary = sanitize_user_input(
        str(user_input.get("case_summary", state.get("message", "")))
    )

    agents = _init_agents(case_type)

    first_stage = "identity_node" if case_type == "criminal" else "pretrial_node"

    return Command(
        update={
            "case_type": case_type,
            "case_category": case_category,
            "user_role": user_role,
            "case_summary": case_summary,
            "agents": agents,
            "stage": "setup",
            "current_round": 1,
            "max_rounds": 3,
            "court_record": [],
            "llm_call_count": 0,
            "evidence_cases": [],
            "evidence_articles": [],
            "selected_evidence": [],
            "excluded_evidence": [],
            "is_complete": False,
            "agent_used": "mock_trial",
            "output_session_data": {"active_agent": "mock_trial"},
        },
        goto=first_stage,
    )


async def evidence_node(state: MockTrialState) -> Command[str]:
    """[공통] 증거조사 (형사: §290~§313 / 민사: §288~§344)"""
    _validate_node_input(state, ["case_type", "agents", "case_summary"])
    if (cmd := _check_rate_limit(state)) is not None:
        return cmd
    llm_call_count = state.get("llm_call_count", 0)
    court_record = list(state.get("court_record", []))

    # RAG 검색: 사건 개요로 판례/법령 검색 (UI 표시용)
    searcher = get_evidence_searcher()
    query = state.get("case_summary", "")
    evidence_cases = await searcher.search_cases(query, limit=5)
    evidence_articles = await searcher.search_articles(query, limit=5)

    # 역할별 RAG 컨텍스트 생성 → state에 캐시
    case_type = state.get("case_type", "criminal")
    pros_cases, pros_articles = await search_for_role(
        "prosecutor", query, case_type
    )
    atty_cases, atty_articles = await search_for_role(
        "attorney", query, case_type
    )
    rag_prosecutor_context = build_rag_context(
        "prosecutor", pros_cases, pros_articles
    )
    rag_attorney_context = build_rag_context(
        "attorney", atty_cases, atty_articles
    )

    # 사용자 힌트 생성 (사용자 역할 기준)
    user_role = state.get("user_role", "prosecutor")
    if user_role == "prosecutor":
        hint_cases, hint_articles = pros_cases, pros_articles
    else:
        hint_cases, hint_articles = atty_cases, atty_articles
    rag_user_hints = build_user_hints(hint_cases, hint_articles)

    # 판사 발언: 증거조사 시작 안내
    judge = _get_agent(state, "judge")
    judge_response, judge_emotion = await judge.generate(
        "evidence", state.get("case_summary", ""), court_record
    )
    llm_call_count += 1
    court_record = _record(court_record, "evidence", "judge", judge_response)

    interrupt_value = interrupt({
        "response": (
            f"[재판장] {judge_response}\n\n"
            "증거를 제출하세요. 판례/법령 검색 결과를 증거로 활용할 수 있습니다."
        ),
        "speaking_agent": "judge",
        "emotion": judge_emotion,
        "stage": "evidence",
        "evidence": {
            "cases": evidence_cases,
            "articles": evidence_articles,
        },
        "user_hints": rag_user_hints,
        "actions": [
            ChatAction(
                type=ActionType.BUTTON,
                label="증거 제출 완료",
                action="submit_evidence",
            ).model_dump(),
        ],
    })

    # interrupt_value 파싱: dict(증거 선택) 또는 str(텍스트 입력)
    selected_ids: list[str] = []
    excluded_ids: list[str] = []
    user_input = ""

    if isinstance(interrupt_value, dict):
        selected_ids = interrupt_value.get("selected_ids", [])
        excluded_ids = interrupt_value.get("excluded_ids", [])
        user_input = str(interrupt_value.get("text", ""))
    elif interrupt_value and str(interrupt_value) != "submit_evidence":
        user_input = str(interrupt_value)

    if user_input:
        court_record = _record(
            court_record, "evidence", state.get("user_role", ""), user_input
        )

    next_node = (
        "examination_node"
        if state.get("case_type") == "criminal"
        else "argument_node"
    )

    return Command(
        update={
            "stage": "evidence",
            "court_record": court_record,
            "llm_call_count": llm_call_count,
            "evidence_cases": evidence_cases,
            "evidence_articles": evidence_articles,
            "selected_evidence": selected_ids,
            "excluded_evidence": excluded_ids,
            "rag_prosecutor_context": rag_prosecutor_context,
            "rag_attorney_context": rag_attorney_context,
            "rag_user_hints": rag_user_hints,
            "agents": _update_agent_in_state(state.get("agents", {}), judge),
            "response": judge_response,
            "speaking_agent": "judge",
            "emotion": judge_emotion,
            "agent_used": "mock_trial",
        },
        goto=next_node,
    )


async def verdict_node(state: MockTrialState) -> Command[str]:
    """[공통] 판결선고"""
    _validate_node_input(state, ["case_type", "agents"])
    court_record = list(state.get("court_record", []))
    judge = _get_agent(state, "judge")

    case_type = state.get("case_type", "criminal")
    verdict_template = (
        VERDICT_TEMPLATE_CRIMINAL if case_type == "criminal"
        else VERDICT_TEMPLATE_CIVIL
    )
    burden_of_proof = (
        BURDEN_OF_PROOF_CRIMINAL if case_type == "criminal"
        else BURDEN_OF_PROOF_CIVIL
    )
    estimated = STAGE_ESTIMATED_MINUTES.get("verdict", 5)

    # 양형 참조용 유사 판례 검색
    verdict_cases = await search_for_verdict(
        state.get("case_summary", ""), case_type
    )
    verdict_rag_context = build_rag_context("judge", verdict_cases, [])

    verdict_context = (
        f"{state.get('case_summary', '')}\n\n"
        f"사건 유형: {case_type}\n"
        f"{burden_of_proof}\n\n"
        f"{verdict_template}\n\n"
        f"예상 소요시간: 약 {estimated}분\n"
        "위 법정 기록과 형식에 따라 판결문을 작성하세요."
    )
    judgment, verdict_emotion = await judge.generate(
        "verdict", verdict_context, court_record,
        rag_context=verdict_rag_context,
    )
    court_record = _record(court_record, "verdict", "judge", judgment)
    feedback = _generate_feedback(state)

    return Command(
        update={
            "stage": "verdict",
            "judgment": judgment,
            "feedback": feedback,
            "is_complete": True,
            "court_record": court_record,
            "agents": _update_agent_in_state(state.get("agents", {}), judge),
            "response": f"[판결]\n{judgment}\n\n[피드백]\n{feedback}",
            "speaking_agent": "judge",
            "emotion": verdict_emotion,
            "agent_used": "mock_trial",
            "output_session_data": {"active_agent": "mock_trial"},
        },
        goto=END,
    )


# ── 형사 전용 노드 ──


async def identity_node(state: MockTrialState) -> Command[str]:
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


async def opening_node(state: MockTrialState) -> Command[str]:
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


async def examination_node(state: MockTrialState) -> Command[str]:
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


async def criminal_closing_node(state: MockTrialState) -> Command[str]:
    """[형사] 최종변론 (형사소송법 §302~§303)

    검사 구형, 변호인 최후변론, 피고인 최후진술
    """
    if (cmd := _check_rate_limit(state)) is not None:
        return cmd
    llm_call_count = state.get("llm_call_count", 0)
    court_record = list(state.get("court_record", []))
    user_role = state.get("user_role", "prosecutor")
    agents_state = dict(state.get("agents", {}))

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


# ── 민사 전용 노드 ──


async def pretrial_node(state: MockTrialState) -> Command[str]:
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


async def claims_node(state: MockTrialState) -> Command[str]:
    """[민사] 주장/답변 (민사소송법 §256~§257)

    사용자 역할에 따라 원고/피고 입력을 받고 상대측 AI가 답변합니다.
    """
    if (cmd := _check_rate_limit(state)) is not None:
        return cmd
    llm_call_count = state.get("llm_call_count", 0)
    court_record = list(state.get("court_record", []))
    user_role = state.get("user_role", "prosecutor")
    agents_state = dict(state.get("agents", {}))

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


async def argument_node(state: MockTrialState) -> Command[str]:
    """[민사] 변론 (민사소송법 §134~§148) — 2-3 라운드 루프"""
    if (cmd := _check_rate_limit(state)) is not None:
        return cmd
    llm_call_count = state.get("llm_call_count", 0)
    court_record = list(state.get("court_record", []))
    current_round = state.get("current_round", 1)
    max_rounds = state.get("max_rounds", 3)
    agents_state = dict(state.get("agents", {}))

    rag_hints = state.get("rag_user_hints", [])
    interrupt_value = interrupt({
        "response": (
            f"[변론 라운드 {current_round}/{max_rounds}] "
            "주장을 입력하세요."
        ),
        "stage": "argument",
        "user_hints": rag_hints,
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


async def civil_closing_node(state: MockTrialState) -> Command[str]:
    """[민사] 변론종결 (민사소송법 §200)

    양측 최종 주장을 정리합니다.
    """
    if (cmd := _check_rate_limit(state)) is not None:
        return cmd
    llm_call_count = state.get("llm_call_count", 0)
    court_record = list(state.get("court_record", []))
    user_role = state.get("user_role", "prosecutor")
    agents_state = dict(state.get("agents", {}))

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


# ── 서브그래프 빌드 ──


def build_mock_trial_subgraph() -> CompiledStateGraph:  # type: ignore[type-arg]
    """모의재판 서브그래프 빌드

    Returns:
        컴파일된 모의재판 서브그래프
    """
    builder = StateGraph(MockTrialState)

    # 공통 노드
    builder.add_node("setup_node", setup_node)
    builder.add_node("evidence_node", evidence_node)
    builder.add_node("verdict_node", verdict_node)

    # 형사 전용 노드
    builder.add_node("identity_node", identity_node)
    builder.add_node("opening_node", opening_node)
    builder.add_node("examination_node", examination_node)
    builder.add_node("criminal_closing_node", criminal_closing_node)

    # 민사 전용 노드
    builder.add_node("pretrial_node", pretrial_node)
    builder.add_node("claims_node", claims_node)
    builder.add_node("argument_node", argument_node)
    builder.add_node("civil_closing_node", civil_closing_node)

    # 엣지
    builder.add_edge(START, "setup_node")
    # setup_node → identity_node / pretrial_node (Command로 분기)
    # 중간 노드들은 모두 Command(goto=...)로 라우팅
    builder.add_edge("verdict_node", END)

    return builder.compile()
