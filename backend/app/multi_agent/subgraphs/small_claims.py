"""
소액소송 interrupt 기반 subgraph

LangGraph interrupt()를 사용하여 multi-turn 대화를 구현합니다.
각 단계에서 사용자 입력을 기다리며, 체크포인터가 중간 상태를 자동 저장합니다.
"""

import logging
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from app.multi_agent.agents.base_chat import ActionType, ChatAction
from app.multi_agent.agents.small_claims_agent import (
    SMALL_CLAIMS_LIMIT,
    STEP_MESSAGES,
    SmallClaimsStep,
    detect_dispute_type,
    extract_amount,
)

logger = logging.getLogger(__name__)


class SmallClaimsState(TypedDict, total=False):
    """소액소송 subgraph 상태"""

    # 부모 그래프에서 전달
    message: str
    history: list[dict[str, str]]
    session_data: dict[str, Any]
    user_location: Optional[dict[str, float]]

    # subgraph 내부 상태
    dispute_type: str
    claim_amount: int
    step: str
    is_complete: bool

    # 출력 (부모 그래프로 전달)
    response: str
    sources: list[dict[str, Any]]
    actions: list[dict[str, Any]]
    output_session_data: dict[str, Any]
    agent_used: str


def _dispute_type_actions() -> list[dict[str, Any]]:
    """분쟁 유형 선택 버튼"""
    return [
        ChatAction(
            type=ActionType.BUTTON,
            label="물품 대금",
            action="dispute_type_goods",
        ).model_dump(),
        ChatAction(
            type=ActionType.BUTTON,
            label="중고거래 사기",
            action="dispute_type_fraud",
        ).model_dump(),
        ChatAction(
            type=ActionType.BUTTON,
            label="임대차 보증금",
            action="dispute_type_deposit",
        ).model_dump(),
    ]


def _evidence_actions() -> list[dict[str, Any]]:
    """증거 단계 액션 버튼"""
    return [
        ChatAction(
            type=ActionType.BUTTON,
            label="내용증명 작성 도움",
            action="draft_demand_letter",
        ).model_dump(),
        ChatAction(
            type=ActionType.BUTTON,
            label="바로 소송 진행",
            action="skip_to_court",
        ).model_dump(),
    ]


def _court_actions() -> list[dict[str, Any]]:
    """소송 단계 액션 버튼"""
    return [
        ChatAction(
            type=ActionType.LINK,
            label="전자소송 바로가기",
            url="https://ecfs.scourt.go.kr",
        ).model_dump(),
        ChatAction(
            type=ActionType.BUTTON,
            label="소장 작성 도움",
            action="draft_complaint",
        ).model_dump(),
        ChatAction(
            type=ActionType.BUTTON,
            label="처음부터 다시",
            action="reset_session",
        ).model_dump(),
    ]


def init_node(state: SmallClaimsState) -> Command[str]:
    """초기 안내 + 분쟁 유형 질문

    메시지에서 분쟁 유형 감지 시 바로 다음 단계로 이동.
    감지 실패 시 interrupt()로 사용자에게 질문.
    """
    message = state["message"]

    # 메시지에서 분쟁 유형 감지 시도
    dispute_type = detect_dispute_type(message)
    if dispute_type:
        response = f"**{dispute_type}** 관련 분쟁이시군요.\n\n"
        response += STEP_MESSAGES[SmallClaimsStep.GATHER_INFO]
        return Command(
            update={
                "dispute_type": dispute_type,
                "step": SmallClaimsStep.GATHER_INFO,
                "response": response,
                "actions": [],
                "agent_used": "small_claims",
                "output_session_data": {"active_agent": "small_claims"},
            },
            goto="gather_info_node",
        )

    # 분쟁 유형 미감지 -> interrupt로 사용자에게 질문
    interrupt_value = interrupt({
        "response": STEP_MESSAGES[SmallClaimsStep.INIT],
        "actions": _dispute_type_actions(),
        "step": SmallClaimsStep.INIT,
    })

    # resume 시 여기서 이어짐
    user_input = str(interrupt_value)
    dispute_type = detect_dispute_type(user_input) or "기타"

    response = f"**{dispute_type}** 관련 분쟁이시군요.\n\n"
    response += STEP_MESSAGES[SmallClaimsStep.GATHER_INFO]

    return Command(
        update={
            "dispute_type": dispute_type,
            "step": SmallClaimsStep.GATHER_INFO,
            "response": response,
            "actions": [],
            "agent_used": "small_claims",
            "output_session_data": {"active_agent": "small_claims"},
        },
        goto="gather_info_node",
    )


def gather_info_node(state: SmallClaimsState) -> Command[str]:
    """금액/상대방 정보 수집"""
    # interrupt로 사용자 입력 대기
    interrupt_value = interrupt({
        "response": STEP_MESSAGES[SmallClaimsStep.GATHER_INFO],
        "actions": [],
        "step": SmallClaimsStep.GATHER_INFO,
    })

    user_input = str(interrupt_value)
    amount = extract_amount(user_input)

    if amount and amount > SMALL_CLAIMS_LIMIT:
        # 한도 초과 안내 후 다시 interrupt
        over_limit_msg = (
            f"청구 금액이 **{amount:,}원**이시군요.\n\n"
            "소액소송은 3,000만원 이하만 가능합니다. "
            "금액이 이를 초과하면 일반 민사소송을 진행해야 합니다.\n\n"
            "그래도 소액소송 범위 내에서 진행하시겠습니까?"
        )
        interrupt_value = interrupt({
            "response": over_limit_msg,
            "actions": [],
            "step": SmallClaimsStep.GATHER_INFO,
        })
        user_input = str(interrupt_value)
        amount = extract_amount(user_input) or amount

    if amount:
        response = (
            f"청구 금액: **{amount:,}원** (소액소송 가능)\n\n"
            "다음은 증거 자료를 정리해야 합니다.\n\n"
            + STEP_MESSAGES[SmallClaimsStep.EVIDENCE]
        )
    else:
        response = (
            "금액 정보를 확인했습니다.\n\n"
            "다음은 증거 자료를 정리해야 합니다.\n\n"
            + STEP_MESSAGES[SmallClaimsStep.EVIDENCE]
        )

    return Command(
        update={
            "claim_amount": amount or 0,
            "step": SmallClaimsStep.EVIDENCE,
            "response": response,
            "actions": [],
            "agent_used": "small_claims",
        },
        goto="evidence_node",
    )


def evidence_node(state: SmallClaimsState) -> Command[str]:
    """증거 자료 안내"""
    interrupt_value = interrupt({
        "response": STEP_MESSAGES[SmallClaimsStep.EVIDENCE],
        "actions": [],
        "step": SmallClaimsStep.EVIDENCE,
    })

    # resume 후 다음 단계로
    _ = interrupt_value  # 사용자 응답 확인

    response = (
        "증거 자료를 확인했습니다.\n\n"
        "다음 단계는 내용증명 발송입니다.\n\n"
        + STEP_MESSAGES[SmallClaimsStep.DEMAND_LETTER]
    )

    return Command(
        update={
            "step": SmallClaimsStep.DEMAND_LETTER,
            "response": response,
            "actions": _evidence_actions(),
            "agent_used": "small_claims",
        },
        goto="demand_letter_node",
    )


def demand_letter_node(state: SmallClaimsState) -> Command[str]:
    """내용증명 안내"""
    interrupt_value = interrupt({
        "response": STEP_MESSAGES[SmallClaimsStep.DEMAND_LETTER],
        "actions": _evidence_actions(),
        "step": SmallClaimsStep.DEMAND_LETTER,
    })

    _ = interrupt_value

    response = (
        "내용증명 발송 후 응답이 없거나 거부당하면, "
        "소송을 제기할 수 있습니다.\n\n"
        + STEP_MESSAGES[SmallClaimsStep.COURT]
    )

    return Command(
        update={
            "step": SmallClaimsStep.COURT,
            "response": response,
            "actions": _court_actions(),
            "agent_used": "small_claims",
        },
        goto="court_node",
    )


def court_node(state: SmallClaimsState) -> dict[str, Any]:
    """소액소송 제기 안내 (최종 단계)"""
    return {
        "response": STEP_MESSAGES[SmallClaimsStep.COURT],
        "actions": _court_actions(),
        "is_complete": True,
        "step": SmallClaimsStep.COMPLETE,
        "agent_used": "small_claims",
        "output_session_data": {"active_agent": "small_claims"},
    }


def build_small_claims_subgraph() -> CompiledStateGraph:
    """소액소송 subgraph 빌드 및 컴파일

    Returns:
        컴파일된 소액소송 subgraph
    """
    builder = StateGraph(SmallClaimsState)

    builder.add_node("init_node", init_node)
    builder.add_node("gather_info_node", gather_info_node)
    builder.add_node("evidence_node", evidence_node)
    builder.add_node("demand_letter_node", demand_letter_node)
    builder.add_node("court_node", court_node)

    builder.add_edge(START, "init_node")
    # init_node ~ demand_letter_node: Command(goto=...)로 라우팅
    builder.add_edge("court_node", END)

    return builder.compile()
