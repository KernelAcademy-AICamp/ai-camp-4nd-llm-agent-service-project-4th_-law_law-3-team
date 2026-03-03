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
    STEP_MESSAGES,
    SmallClaimsStep,
)
from app.services.service_function.small_claims_service import (
    SMALL_CLAIMS_LIMIT,
    detect_dispute_type,
    extract_amount,
    search_for_dispute_type,
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

    # RAG 캐시 (mock_trial의 rag_prosecutor_context 패턴)
    rag_case_context: str  # 판례 컨텍스트 (LLM 응답용)
    rag_law_context: str  # 법령 컨텍스트
    rag_case_sources: list[dict[str, Any]]  # 프론트엔드 소스 전달용

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


def _sync_from_ui_state(state: SmallClaimsState) -> dict[str, Any]:
    """session_data의 wizard_state로부터 내부 상태 동기화"""
    wizard_state = state.get("session_data", {}).get("wizard_state", {})
    if not wizard_state:
        return {}

    updates: dict[str, Any] = {}

    # 분쟁 유형 동기화 (프론트엔드 ID -> 백엔드 한글명)
    ui_dispute_type = wizard_state.get("dispute_type")
    if ui_dispute_type:
        mapping = {
            "product_payment": "물품대금",
            "fraud": "중고거래",
            "deposit": "임대차",
            "service_payment": "용역대금",
            "wage": "임금체불",
        }
        internal_type = mapping.get(ui_dispute_type)
        if internal_type and internal_type != state.get("dispute_type"):
            updates["dispute_type"] = internal_type

    # 청구 금액 동기화
    ui_amount = wizard_state.get("case_info", {}).get("amount")
    if ui_amount and ui_amount != state.get("claim_amount"):
        updates["claim_amount"] = int(ui_amount)

    # 단계 동기화 (프론트엔드 단계 -> 백엔드 단계)
    # 주의: 강제 이동은 위험할 수 있으므로 보조적으로만 사용
    ui_step = wizard_state.get("current_step")
    if ui_step:
        step_mapping = {
            "dispute_type": SmallClaimsStep.INIT,
            "case_info": SmallClaimsStep.GATHER_INFO,
            "evidence": SmallClaimsStep.EVIDENCE,
            "document": SmallClaimsStep.DEMAND_LETTER,
        }
        internal_step = step_mapping.get(ui_step)
        if internal_step and internal_step != state.get("step"):
            # 현재 단계가 초기화 상태거나 명백히 뒤쳐진 경우에만 업데이트
            if not state.get("step") or state.get("step") == SmallClaimsStep.INIT:
                updates["step"] = internal_step

    return updates


def init_node(state: SmallClaimsState) -> Command[str]:
    """초기 안내 + 분쟁 유형 질문"""
    # 1. UI 상태와 동기화
    ui_updates = _sync_from_ui_state(state)
    if ui_updates:
        # UI에서 이미 정보가 있으면 상태 업데이트 후 진행
        dispute_type = ui_updates.get("dispute_type") or state.get("dispute_type")
        if dispute_type:
            response = f"**{dispute_type}** 관련 분쟁이시군요.\n\n"
            response += STEP_MESSAGES[SmallClaimsStep.GATHER_INFO]
            return Command(
                update={
                    **ui_updates,
                    "response": response,
                    "actions": [],
                    "agent_used": "small_claims",
                    "output_session_data": {"active_agent": "small_claims"},
                },
                goto="gather_info_node",
            )

    message = state.get("message", "")
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


async def gather_info_node(state: SmallClaimsState) -> Command[str]:
    """금액/상대방 정보 수집 (RAG 검색 포함)"""
    # 1. UI 상태와 동기화
    ui_updates = _sync_from_ui_state(state)

    # interrupt로 사용자 입력 대기
    interrupt_value = interrupt({
        "response": STEP_MESSAGES[SmallClaimsStep.GATHER_INFO],
        "actions": [],
        "step": SmallClaimsStep.GATHER_INFO,
    })

    user_input = str(interrupt_value)
    amount = extract_amount(user_input)

    # UI에서 전달된 금액이 있고 메시지에서 추출된 금액이 없으면 UI 금액 사용
    if not amount and "claim_amount" in ui_updates:
        amount = ui_updates["claim_amount"]

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

    # 2. RAG 검색 — 금액 추출 직후, 분쟁 유형 기반 판례 검색
    dispute_type = ui_updates.get("dispute_type") or state.get("dispute_type", "기타")
    description = (
        state.get("session_data", {})
        .get("wizard_state", {})
        .get("case_info", {})
        .get("description", "")
    ) or user_input

    rag_case_context = ""
    rag_case_sources: list[dict[str, Any]] = []
    if dispute_type and dispute_type != "기타":
        _, rag_case_context, rag_case_sources = await search_for_dispute_type(
            dispute_type=dispute_type,
            description=description,
        )

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

    # RAG 판례 컨텍스트가 있으면 응답에 관련 판례 요약 추가
    if rag_case_context:
        # 컨텍스트가 너무 길면 앞 300자만 발췌
        preview = rag_case_context[:300].rstrip()
        if len(rag_case_context) > 300:
            preview += "…"
        response += (
            "\n\n---\n**📋 관련 판례 참고:**\n"
            f"{preview}"
        )

    return Command(
        update={
            "claim_amount": amount or 0,
            "step": SmallClaimsStep.EVIDENCE,
            "response": response,
            "actions": [],
            "agent_used": "small_claims",
            "rag_case_context": rag_case_context,
            "rag_law_context": "",
            "rag_case_sources": rag_case_sources,
        },
        goto="evidence_node",
    )


def evidence_node(state: SmallClaimsState) -> Command[str]:
    """증거 자료 안내"""
    # UI 상태와 동기화 (단계 등)
    ui_updates = _sync_from_ui_state(state)

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
            **ui_updates,
            "step": SmallClaimsStep.DEMAND_LETTER,
            "response": response,
            "actions": _evidence_actions(),
            "agent_used": "small_claims",
        },
        goto="demand_letter_node",
    )


def demand_letter_node(state: SmallClaimsState) -> Command[str]:
    """내용증명 안내"""
    # UI 상태와 동기화
    ui_updates = _sync_from_ui_state(state)

    interrupt_value = interrupt({
        "response": STEP_MESSAGES[SmallClaimsStep.DEMAND_LETTER],
        "actions": _evidence_actions(),
        "step": SmallClaimsStep.DEMAND_LETTER,
    })

    user_input = str(interrupt_value)

    # 내용증명 작성 도움 버튼 클릭 시
    if "draft_demand_letter" in user_input or "내용증명" in user_input or "작성" in user_input:
        dispute_type = ui_updates.get("dispute_type") or state.get("dispute_type", "기타")
        claim_amount = ui_updates.get("claim_amount") or state.get("claim_amount", 0)

        draft_response = f"""**내용증명 작성을 도와드리겠습니다.**

📋 **소액소송 서류 작성 페이지**로 이동하여 내용증명을 작성하세요.

**현재 입력된 정보:**
- 분쟁 유형: {dispute_type}
- 청구 금액: {claim_amount:,}원

➡️ 왼쪽 메뉴의 **"소액소송 가이드"**를 클릭하면 서류 작성 페이지로 이동할 수 있습니다.

내용증명을 작성하여 상대방에게 발송하세요. 답변이 없거나 거부당하면 소송을 진행할 수 있습니다.
"""
        return Command(
            update={
                **ui_updates,
                "step": SmallClaimsStep.DEMAND_LETTER,
                "response": draft_response,
                "actions": _court_actions(),
                "agent_used": "small_claims",
            },
            goto="court_node",
        )

    # 일반적인 경우 - 다음 단계로
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
        "sources": state.get("rag_case_sources", []),
        "is_complete": True,
        "step": SmallClaimsStep.COMPLETE,
        "agent_used": "small_claims",
        "output_session_data": {"active_agent": "small_claims"},
    }


def build_small_claims_subgraph() -> CompiledStateGraph[Any, Any, Any, Any]:
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
