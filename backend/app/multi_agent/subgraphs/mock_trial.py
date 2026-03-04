"""
모의 법정 서브그래프

LangGraph interrupt + Command 패턴으로 형사 6단계 + 민사 6단계 재판을 구현합니다.
small_claims.py 서브그래프 패턴을 따릅니다.

Design 문서 Section 6.3 기반
"""

import logging
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from app.multi_agent.agents.base_chat import ActionType, ChatAction
from app.multi_agent.subgraphs.mock_trial_civil import (
    argument_node,
    civil_closing_node,
    claims_node,
    pretrial_node,
)
from app.multi_agent.subgraphs.mock_trial_criminal import (
    criminal_closing_node,
    examination_node,
    identity_node,
    opening_node,
)
from app.multi_agent.subgraphs.mock_trial_prompts import (
    BURDEN_OF_PROOF_CIVIL,
    BURDEN_OF_PROOF_CRIMINAL,
    STAGE_ESTIMATED_MINUTES,
    VERDICT_TEMPLATE_CIVIL,
    VERDICT_TEMPLATE_CRIMINAL,
    sanitize_user_input,
)
from app.multi_agent.subgraphs.mock_trial_utils import (
    MAX_LLM_CALLS_PER_SESSION,
    _build_rebuttal_context,
    _build_stage_rag_context,
    _case_type_actions,
    _check_rate_limit,
    _generate_feedback,
    _get_agent,
    _get_opponent_role,
    _init_agents,
    _now_iso,
    _record,
    _update_agent_in_state,
    _validate_node_input,
)
from app.services.service_function.mock_trial_service import (
    build_rag_context,
    build_references_payload,
    build_user_hints,
    get_evidence_searcher,
    search_for_role,
    search_for_verdict,
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
    rag_references: list[dict[str, Any]]

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


# ── 공통 노드 ──


async def setup_node(state: MockTrialState) -> Command[str]:
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

    # RAG 검색: 첫 단계부터 참조 판례/법령을 제공하기 위해 setup에서 실행
    rag_prosecutor_context = ""
    rag_attorney_context = ""
    rag_user_hints: list[dict[str, Any]] = []
    rag_references: list[dict[str, Any]] = []

    if case_summary:
        try:
            pros_cases, pros_articles = await search_for_role(
                "prosecutor", case_summary, case_type
            )
            atty_cases, atty_articles = await search_for_role(
                "attorney", case_summary, case_type
            )
            rag_prosecutor_context = build_rag_context(
                "prosecutor", pros_cases, pros_articles
            )
            rag_attorney_context = build_rag_context(
                "attorney", atty_cases, atty_articles
            )

            if user_role == "prosecutor":
                hint_cases, hint_articles = pros_cases, pros_articles
            else:
                hint_cases, hint_articles = atty_cases, atty_articles
            rag_user_hints = build_user_hints(hint_cases, hint_articles)

            rag_references = build_references_payload(
                pros_cases + atty_cases, pros_articles + atty_articles
            )
        except Exception:
            logger.warning("setup_node RAG 검색 실패, 빈 참조로 진행")

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
            "rag_prosecutor_context": rag_prosecutor_context,
            "rag_attorney_context": rag_attorney_context,
            "rag_user_hints": rag_user_hints,
            "rag_references": rag_references,
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

    # 역할별 RAG 컨텍스트: setup_node에서 캐시된 데이터가 있으면 재사용
    case_type = state.get("case_type", "criminal")
    user_role = state.get("user_role", "prosecutor")
    cached_refs = state.get("rag_references", [])

    if cached_refs:
        rag_prosecutor_context = str(state.get("rag_prosecutor_context", ""))
        rag_attorney_context = str(state.get("rag_attorney_context", ""))
        rag_user_hints = list(state.get("rag_user_hints", []))
        rag_references = list(cached_refs)
    else:
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

        if user_role == "prosecutor":
            hint_cases, hint_articles = pros_cases, pros_articles
        else:
            hint_cases, hint_articles = atty_cases, atty_articles
        rag_user_hints = build_user_hints(hint_cases, hint_articles)

        rag_references = build_references_payload(
            pros_cases + atty_cases, pros_articles + atty_articles
        )

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
        "references": rag_references,
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
            "rag_references": rag_references,
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


# ── 하위 호환 re-exports ──
# 다른 파일이 mock_trial에서 직접 import하는 경우를 위해 re-export

__all__ = [
    "MockTrialState",
    "build_mock_trial_subgraph",
    # 공통 노드
    "setup_node",
    "evidence_node",
    "verdict_node",
    # 형사 노드
    "identity_node",
    "opening_node",
    "examination_node",
    "criminal_closing_node",
    # 민사 노드
    "pretrial_node",
    "claims_node",
    "argument_node",
    "civil_closing_node",
    # 유틸리티 (하위 호환)
    "MAX_LLM_CALLS_PER_SESSION",
    "_now_iso",
    "_record",
    "_validate_node_input",
    "_init_agents",
    "_get_agent",
    "_update_agent_in_state",
    "_get_opponent_role",
    "_build_stage_rag_context",
    "_build_rebuttal_context",
    "_check_rate_limit",
    "_case_type_actions",
    "_generate_feedback",
]
