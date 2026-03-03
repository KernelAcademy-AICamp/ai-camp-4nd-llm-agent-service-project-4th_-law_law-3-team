"""
스토리보드 interrupt 기반 subgraph

태그 기반 정보 수집 + 지속적 질문으로 충분한 정보를 모은 후
사건 타임라인을 생성한다. small_claims/mock_trial 서브그래프 패턴을 따른다.

흐름: START → collect_node → question_node ←→ (루프) → confirm_node → generate_node → END
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from app.multi_agent.agents.base_chat import ActionType, ChatAction
from app.multi_agent.schemas.tag import TaggedItem, TagType

logger = logging.getLogger(__name__)

MAX_QUESTIONS = 5
MIN_TIMELINE_TAGS = 2
MIN_PARTY_TAGS = 1

# 질문 생성 프롬프트
_QUESTION_PROMPT = """법률 사건 스토리보드 작성을 위해 질문합니다.

수집된 정보:
{collected}

부족한 정보:
{missing}

친근하고 쉬운 말투로 질문 1개를 생성하세요.
구체적 예시를 포함하세요 (예: "정확한 날짜를 기억하시나요? 예: 2024년 3월 5일").
질문만 출력하세요."""

# 타임라인 생성 프롬프트
_GENERATE_PROMPT = """당신은 법률 사건 타임라인 전문가입니다.
수집된 정보를 바탕으로 시간순 타임라인을 작성하세요.

수집된 정보:
{collected_narrative}

태그 출처 요약:
{tag_source_summary}

사용자 최초 요청:
{original_message}

작성 규칙:
1. 날짜/시점이 명시되지 않은 경우 논리적 순서로 배치
2. 각 이벤트는 "- [시점] 내용" 형식
3. 법적으로 중요한 시점(계약일, 이행기, 소멸시효 등) 강조
4. 핵심 쟁점 별도 정리
5. lawyer 태그가 있으면 "법적 전략" 섹션에 변호사/전문분야 정보 포함
6. precedent 태그가 있으면 "관련 판례 근거" 섹션에 판례번호와 요지 포함
7. 마지막에 간단한 법적 조언 추가"""

# 태그 유형별 부족 정보 질문 템플릿
_MISSING_TAG_QUESTIONS: dict[str, str] = {
    "timeline": "사건이 시작된 시점과 주요 경위",
    "party": "관련된 사람 (피해자, 가해자 등)",
    "evidence": "증거가 될 만한 자료 (계약서, 사진, 카톡 등)",
    "amount": "피해 금액이나 청구 금액",
}


class StoryboardState(TypedDict, total=False):
    """스토리보드 subgraph 상태"""

    # 부모 그래프에서 전달
    message: str
    history: list[dict[str, str]]
    session_data: dict[str, Any]
    user_location: Optional[dict[str, float]]

    # 서브그래프 내부 상태
    step: str  # collect, question, confirm, generate
    tagged_items: list[dict[str, Any]]
    questions_asked: int
    collected_narrative: str

    # 출력 (부모 그래프로 전달)
    response: str
    sources: list[dict[str, Any]]
    actions: list[dict[str, Any]]
    output_session_data: dict[str, Any]
    agent_used: str


def _summarize_tags(tagged_items: list[dict[str, Any]]) -> str:
    """태그 항목을 사람이 읽을 수 있는 텍스트로 요약"""
    if not tagged_items:
        return "(수집된 정보 없음)"

    tag_label = {
        "lawyer": "변호사",
        "precedent": "판례",
        "evidence": "증거",
        "timeline": "사건 경위",
        "party": "당사자",
        "amount": "금액",
    }

    lines: list[str] = []
    for item in tagged_items:
        label = tag_label.get(item.get("tag_type", ""), "기타")
        content = item.get("content", "")
        date_hint = item.get("date_hint")
        date_str = f" ({date_hint})" if date_hint else ""
        lines.append(f"- [{label}]{date_str} {content}")

    return "\n".join(lines)


def _summarize_tag_sources(tagged_items: list[dict[str, Any]]) -> str:
    """태그 출처 에이전트별 요약 (예: '법률 검색(3개), 변호사 찾기(2개)')"""
    if not tagged_items:
        return "(태그 출처 없음)"

    agent_label: dict[str, str] = {
        "legal_search": "법률 검색",
        "lawyer_finder": "변호사 찾기",
        "law_study": "법률 학습",
        "workspace": "워크스페이스",
        "storyboard": "스토리보드",
        "general": "일반 대화",
    }

    agent_counts: dict[str, int] = {}
    for item in tagged_items:
        agent = item.get("source_agent", "unknown")
        agent_counts[agent] = agent_counts.get(agent, 0) + 1

    parts: list[str] = []
    for agent, count in sorted(agent_counts.items(), key=lambda x: -x[1]):
        label = agent_label.get(agent, agent)
        parts.append(f"{label}({count}개)")

    return ", ".join(parts)


def _find_missing_tags(tagged_items: list[dict[str, Any]]) -> list[str]:
    """부족한 태그 유형 판별"""
    tag_counts: dict[str, int] = {}
    for item in tagged_items:
        tag_type = item.get("tag_type", "")
        tag_counts[tag_type] = tag_counts.get(tag_type, 0) + 1

    missing: list[str] = []
    if tag_counts.get("timeline", 0) < MIN_TIMELINE_TAGS:
        missing.append("timeline")
    if tag_counts.get("party", 0) < MIN_PARTY_TAGS:
        missing.append("party")
    # evidence, amount는 권장이므로 없으면 추가
    if "evidence" not in tag_counts:
        missing.append("evidence")
    if "amount" not in tag_counts:
        missing.append("amount")
    return missing


def _is_sufficient(tagged_items: list[dict[str, Any]]) -> bool:
    """최소 정보 충족 여부"""
    tag_counts: dict[str, int] = {}
    for item in tagged_items:
        tag_type = item.get("tag_type", "")
        tag_counts[tag_type] = tag_counts.get(tag_type, 0) + 1

    return (
        tag_counts.get("timeline", 0) >= MIN_TIMELINE_TAGS
        and tag_counts.get("party", 0) >= MIN_PARTY_TAGS
    )


def _storyboard_actions() -> list[dict[str, Any]]:
    """스토리보드 페이지 이동 액션"""
    return [
        ChatAction(
            type=ActionType.NAVIGATE,
            label="스토리보드에서 보기",
            action="navigate_storyboard",
            params={"path": "/storyboard"},
        ).model_dump(),
    ]


def _generate_button_actions() -> list[dict[str, Any]]:
    """타임라인 생성 버튼"""
    return [
        ChatAction(
            type=ActionType.BUTTON,
            label="타임라인 생성",
            action="generate_timeline",
        ).model_dump(),
        ChatAction(
            type=ActionType.BUTTON,
            label="정보 추가",
            action="add_more_info",
        ).model_dump(),
    ]


# ──────────────────────────────────────────────
# 노드 함수
# ──────────────────────────────────────────────


def collect_node(state: StoryboardState) -> Command[str]:
    """초기 수집: 기존 태그 로드 + 분기 판단"""
    session_data = state.get("session_data", {})
    existing_tags: list[dict[str, Any]] = session_data.get("tagged_items", [])

    if existing_tags:
        # 기존 대화에서 수집된 정보가 있음
        summary = _summarize_tags(existing_tags)
        source_summary = _summarize_tag_sources(existing_tags)
        missing = _find_missing_tags(existing_tags)
        missing_text = ""
        if missing:
            missing_labels = [
                _MISSING_TAG_QUESTIONS.get(m, m) for m in missing
            ]
            missing_text = (
                "\n\n**아직 부족한 정보:**\n"
                + "\n".join(f"- {label}" for label in missing_labels)
            )
        response = (
            f"이전 대화에서 수집된 정보가 있습니다 (출처: {source_summary}):\n\n"
            f"{summary}"
            f"{missing_text}\n\n"
            "이 정보를 바탕으로 타임라인을 만들까요? "
            "추가 정보가 있으시면 말씀해주세요."
        )

        interrupt_value = interrupt({
            "response": response,
            "actions": _generate_button_actions(),
            "step": "collect",
        })

        user_input = str(interrupt_value)

        # "생성" 버튼 또는 생성 의도 감지
        generate_keywords = ["생성", "만들어", "네", "좋아", "그래", "generate"]
        if any(kw in user_input.lower() for kw in generate_keywords):
            return Command(
                update={
                    "tagged_items": existing_tags,
                    "step": "confirm",
                    "questions_asked": 0,
                    "response": response,
                    "agent_used": "storyboard",
                    "output_session_data": {"active_agent": "storyboard"},
                },
                goto="confirm_node",
            )

        # 추가 정보 제공 → 질문 루프로
        # 사용자 입력에서 태그 추출
        new_tags = _extract_tags_sync(user_input, len(existing_tags))
        merged = existing_tags + [t.model_dump() for t in new_tags] if new_tags else existing_tags

        return Command(
            update={
                "tagged_items": merged,
                "step": "question",
                "questions_asked": 0,
                "response": response,
                "agent_used": "storyboard",
                "output_session_data": {"active_agent": "storyboard"},
            },
            goto="question_node",
        )

    # 태그 없음: 처음부터 정보 수집
    response = (
        "사건 스토리보드를 작성해드리겠습니다.\n\n"
        "사건에 대해 말씀해주세요. 다음 정보가 필요합니다:\n"
        "- **언제**: 사건이 시작된 시점\n"
        "- **누구**: 관련된 사람들\n"
        "- **무엇**: 어떤 일이 있었는지\n"
        "- **증거**: 계약서, 사진, 카톡 등\n\n"
        "편하게 말씀해주세요!"
    )

    interrupt_value = interrupt({
        "response": response,
        "actions": [],
        "step": "collect",
    })

    user_input = str(interrupt_value)
    new_tags = _extract_tags_sync(user_input, 0)
    merged = [t.model_dump() for t in new_tags] if new_tags else []

    return Command(
        update={
            "tagged_items": merged,
            "step": "question",
            "questions_asked": 0,
            "response": response,
            "agent_used": "storyboard",
            "output_session_data": {"active_agent": "storyboard"},
        },
        goto="question_node",
    )


def question_node(state: StoryboardState) -> Command[str]:
    """질문 루프: 부족한 정보에 대해 질문"""
    tagged_items = state.get("tagged_items", [])
    questions_asked = state.get("questions_asked", 0)

    # 종료 조건 체크
    if _is_sufficient(tagged_items) or questions_asked >= MAX_QUESTIONS:
        return Command(
            update={
                "step": "confirm",
                "response": "",
                "agent_used": "storyboard",
                "output_session_data": {"active_agent": "storyboard"},
            },
            goto="confirm_node",
        )

    # 부족한 정보 파악 + LLM 질문 생성
    missing = _find_missing_tags(tagged_items)
    question = _generate_question(tagged_items, missing)

    interrupt_value = interrupt({
        "response": question,
        "actions": [
            ChatAction(
                type=ActionType.BUTTON,
                label="바로 생성하기",
                action="skip_to_generate",
            ).model_dump(),
        ],
        "step": "question",
    })

    user_input = str(interrupt_value)

    # "바로 생성" 의도 감지
    skip_keywords = ["바로 생성", "스킵", "생성해", "만들어", "skip"]
    if any(kw in user_input.lower() for kw in skip_keywords):
        return Command(
            update={
                "step": "confirm",
                "questions_asked": questions_asked + 1,
                "response": "",
                "agent_used": "storyboard",
                "output_session_data": {"active_agent": "storyboard"},
            },
            goto="confirm_node",
        )

    # 사용자 응답에서 태그 추출
    new_tags = _extract_tags_sync(user_input, questions_asked + 1)
    merged = list(tagged_items)
    if new_tags:
        for tag in new_tags:
            tag_dict = tag.model_dump()
            is_dup = any(
                item.get("content") == tag_dict["content"]
                and item.get("tag_type") == tag_dict["tag_type"]
                for item in merged
            )
            if not is_dup:
                merged.append(tag_dict)

    return Command(
        update={
            "tagged_items": merged,
            "questions_asked": questions_asked + 1,
            "response": "",
            "agent_used": "storyboard",
            "output_session_data": {"active_agent": "storyboard"},
        },
        goto="question_node",  # 루프 (종료 조건은 함수 시작에서 체크)
    )


def confirm_node(state: StoryboardState) -> Command[str]:
    """수집 확인: 태그 요약 표시 + 생성 확인"""
    tagged_items = state.get("tagged_items", [])
    summary = _summarize_tags(tagged_items)

    response = (
        "**수집된 정보 요약:**\n\n"
        f"{summary}\n\n"
        "이 정보로 타임라인을 생성할까요?"
    )

    interrupt_value = interrupt({
        "response": response,
        "actions": _generate_button_actions(),
        "step": "confirm",
    })

    user_input = str(interrupt_value)

    # 수정 의도 → 질문 루프로 복귀
    modify_keywords = ["수정", "추가", "변경", "더", "아니"]
    if any(kw in user_input for kw in modify_keywords):
        return Command(
            update={
                "step": "question",
                "response": "",
                "agent_used": "storyboard",
                "output_session_data": {"active_agent": "storyboard"},
            },
            goto="question_node",
        )

    # 생성 진행
    return Command(
        update={
            "step": "generate",
            "collected_narrative": summary,
            "response": "",
            "agent_used": "storyboard",
            "output_session_data": {"active_agent": "storyboard"},
        },
        goto="generate_node",
    )


async def generate_node(state: StoryboardState) -> dict[str, Any]:
    """타임라인 생성: LLM으로 최종 타임라인 생성"""
    collected_narrative = state.get("collected_narrative", "")
    original_message = state.get("message", "")
    tagged_items = state.get("tagged_items", [])

    # LLM으로 타임라인 텍스트 생성
    tag_source_summary = _summarize_tag_sources(tagged_items)
    timeline_text = await _generate_timeline_text(
        collected_narrative, original_message, tag_source_summary,
    )

    response = (
        "**사건 타임라인이 생성되었습니다!**\n\n"
        f"{timeline_text}\n\n"
        "---\n"
        "스토리보드 페이지에서 이미지와 영상을 추가할 수 있습니다."
    )

    # session_data에 타임라인 텍스트와 태그 정보 저장
    output_session_data: dict[str, Any] = {
        "active_agent": "storyboard",
        "tagged_items": tagged_items,
        "generated_timeline_text": timeline_text,
    }

    return {
        "response": response,
        "sources": [],
        "actions": _storyboard_actions(),
        "output_session_data": output_session_data,
        "agent_used": "storyboard",
        "step": "complete",
    }


# ──────────────────────────────────────────────
# 헬퍼 함수
# ──────────────────────────────────────────────


def _extract_tags_sync(
    message: str, turn_index: int
) -> list[TaggedItem]:
    """동기 환경에서 태그 추출 (interrupt resume 컨텍스트)

    서브그래프 노드는 동기 함수이므로, LLM 태그 추출 대신
    규칙 기반으로 간단히 태그를 추출한다.
    """
    if not message or len(message.strip()) < 3:
        return []

    tags: list[TaggedItem] = []

    # 날짜 패턴 감지 → timeline 태그
    import re

    date_patterns = re.findall(
        r"(\d{4}[년./-]\s*\d{1,2}[월./-]\s*\d{1,2}[일.]?)", message
    )
    for date_str in date_patterns:
        tags.append(
            TaggedItem(
                tag_type=TagType.TIMELINE,
                content=message[:100],
                source_agent="storyboard",
                turn_index=turn_index,
                date_hint=_normalize_date(date_str),
                confidence=0.8,
            )
        )

    # 금액 패턴 감지 → amount 태그
    amount_patterns = re.findall(r"(\d[\d,]*)\s*(?:원|만원|억)", message)
    for amount_str in amount_patterns:
        tags.append(
            TaggedItem(
                tag_type=TagType.AMOUNT,
                content=f"금액: {amount_str}",
                source_agent="storyboard",
                turn_index=turn_index,
                confidence=0.8,
            )
        )

    # 당사자 키워드 감지 → party 태그
    party_keywords = [
        "피해자", "가해자", "원고", "피고", "채무자", "채권자",
        "임대인", "임차인", "매도인", "매수인",
    ]
    for kw in party_keywords:
        if kw in message:
            # 키워드 주변 컨텍스트 추출
            idx = message.index(kw)
            context_start = max(0, idx - 20)
            context_end = min(len(message), idx + 30)
            tags.append(
                TaggedItem(
                    tag_type=TagType.PARTY,
                    content=message[context_start:context_end].strip()[:100],
                    source_agent="storyboard",
                    turn_index=turn_index,
                    confidence=0.7,
                )
            )
            break  # 하나만

    # 증거 키워드 감지 → evidence 태그
    evidence_keywords = [
        "계약서", "영수증", "사진", "카톡", "카카오톡", "녹음",
        "녹취", "문자", "이메일", "메일", "CCTV", "진단서",
    ]
    for kw in evidence_keywords:
        if kw in message:
            idx = message.index(kw)
            context_start = max(0, idx - 10)
            context_end = min(len(message), idx + 30)
            tags.append(
                TaggedItem(
                    tag_type=TagType.EVIDENCE,
                    content=message[context_start:context_end].strip()[:100],
                    source_agent="storyboard",
                    turn_index=turn_index,
                    confidence=0.7,
                )
            )
            break  # 하나만

    # 패턴에 안 걸리면 일반 timeline으로 추가 (문장이 충분히 길면)
    if not tags and len(message.strip()) >= 20:
        tags.append(
            TaggedItem(
                tag_type=TagType.TIMELINE,
                content=message[:100],
                source_agent="storyboard",
                turn_index=turn_index,
                confidence=0.5,
            )
        )

    return tags


def _normalize_date(date_str: str) -> str | None:
    """한국어 날짜를 YYYY-MM-DD로 정규화"""
    import re

    nums = re.findall(r"\d+", date_str)
    if len(nums) >= 3:
        year, month, day = nums[0], nums[1], nums[2]
        return f"{year}-{int(month):02d}-{int(day):02d}"
    return None


def _generate_question(
    tagged_items: list[dict[str, Any]],
    missing: list[str],
) -> str:
    """부족한 정보에 대한 질문 생성 (규칙 기반)"""
    if not missing:
        return "추가로 알려주실 내용이 있으신가요?"

    first_missing = missing[0]
    questions = {
        "timeline": (
            "사건이 시작된 시점은 언제인가요? "
            "정확한 날짜를 기억하시면 알려주세요. "
            "(예: 2024년 3월 5일)"
        ),
        "party": (
            "이 사건에 관련된 사람은 누구인가요? "
            "상대방과의 관계도 알려주시면 도움이 됩니다. "
            "(예: 집주인, 직장 상사, 중고거래 판매자 등)"
        ),
        "evidence": (
            "증거가 될 만한 자료가 있으신가요? "
            "(예: 계약서, 카카오톡 대화, 사진, 녹음, 영수증 등)"
        ),
        "amount": (
            "피해 금액이나 청구하려는 금액이 있다면 알려주세요. "
            "(예: 보증금 500만원, 물품 대금 30만원 등)"
        ),
    }

    return questions.get(
        first_missing,
        "사건에 대해 더 알려주실 내용이 있으신가요?",
    )


async def _generate_timeline_text(
    collected_narrative: str,
    original_message: str,
    tag_source_summary: str = "",
) -> str:
    """LLM으로 타임라인 텍스트 생성 (비동기 호출)"""
    try:
        from app.tools.llm import get_chat_model

        model = get_chat_model(temperature=0.3)
        prompt = _GENERATE_PROMPT.format(
            collected_narrative=collected_narrative,
            tag_source_summary=tag_source_summary or "(없음)",
            original_message=original_message[:500],
        )
        response = await model.ainvoke([("user", prompt)])
        return str(response.content)
    except Exception:
        logger.exception("타임라인 LLM 생성 실패, 수집 정보 반환")
        return f"**수집된 정보 기반 타임라인:**\n\n{collected_narrative}"


# ──────────────────────────────────────────────
# 빌드
# ──────────────────────────────────────────────


def build_storyboard_subgraph() -> CompiledStateGraph[Any, Any, Any, Any]:
    """스토리보드 subgraph 빌드 및 컴파일

    Returns:
        컴파일된 스토리보드 subgraph
    """
    builder = StateGraph(StoryboardState)

    builder.add_node("collect_node", collect_node)
    builder.add_node("question_node", question_node)
    builder.add_node("confirm_node", confirm_node)
    builder.add_node("generate_node", generate_node)

    builder.add_edge(START, "collect_node")
    # collect_node ~ confirm_node: Command(goto=...)로 라우팅
    builder.add_edge("generate_node", END)

    return builder.compile()
