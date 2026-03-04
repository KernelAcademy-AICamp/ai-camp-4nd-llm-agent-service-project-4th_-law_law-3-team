"""
법률 검색 에이전트

RAG 기반 판례/법령 + 다중 타입 검색 및 법률 상담 제공.
파이프라인 PRESETS(focus 모드) + format_utils로 정규화.
"""

import logging
import re
from collections.abc import AsyncGenerator
from typing import Any, Literal

from app.multi_agent.agents.base_chat import (
    ActionType,
    BaseChatAgent,
    ChatAction,
    normalize_chunk_content,
)
from app.multi_agent.schemas.plan import AgentResult
from app.services.rag.format_utils import (
    format_law_context,
    format_law_sources,
    format_precedent_context,
    format_precedent_sources,
    format_supplementary_context,
    format_supplementary_sources,
)
from app.services.rag.pipeline import (
    PRESETS,
    PipelineConfig,
    search_with_pipeline_async,
)
from app.services.service_function import (
    PrecedentService,
    get_precedent_service,
)
from app.tools.llm import get_chat_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 체계도 네비게이션 키워드
# ---------------------------------------------------------------------------

_HIERARCHY_KEYWORDS = re.compile(
    r"체계도|법령\s*계층|상위법|하위법|시행령\s*구조|법률\s*체계|법령\s*관계"
)

# ---------------------------------------------------------------------------
# PRESETS 매핑
# ---------------------------------------------------------------------------

_PRESET_MAP: dict[str, PipelineConfig] = {
    "precedent": PRESETS["legal_search_precedent"],
    "law": PRESETS["legal_search_law"],
}

_SYSTEM_PROMPT = """당신은 법률 전문 AI 어시스턴트입니다.
사용자의 상황에 맞는 법률 정보를 제공된 참고 자료에 기반하여 답변합니다.

## 절대 규칙

1. 참고 자료에 명시된 판례번호·법령명·조문만 인용하세요. 참고 자료에 없는 것을 만들어내면 안 됩니다.
2. 참고 자료 중 질문과 무관한 문서는 무시하세요.
3. 같은 판례·법령을 여러 단락에서 반복하지 마세요.
4. 대법원 판례가 있으면 하급심보다 우선하여 인용하세요.
5. 이전 대화의 답변 형식이 아래 지시와 다르더라도, 반드시 아래 형식을 따르세요.
6. 참고 자료에서 직접 관련된 판례가 없으면, "제공된 참고 자료에는 직접 관련된 판례가 없습니다"라고 솔직하게 밝히세요.

## 출처 표기

인용 시 반드시 출처를 표기하세요.
- 판례: 법원명 선고일자 사건번호 (예: 대법원 2018. 5. 15. 선고 2017도21977 판결)
- 법령: 법령명 + 조항 (예: 경범죄처벌법 제3조 제1항 제21호)

## 답변 형식

반드시 **결론**, **근거**, **주의할 점** 3개 섹션만 사용하세요. "사용자 상황 적용" 등 다른 섹션을 추가하지 마세요.

**결론**: 사용자 질문에 대한 직접 답변 1-2문장. "내 잘못이야?", "이길 수 있어?" 같은 질문에는 "네/아니요"로 시작한 뒤 핵심 판단 기준을 덧붙이세요.

**근거**: 하나의 연속된 문단으로, 충분히 상세하게 작성하세요. 관련 법령 조항을 인용하고, 참고 자료의 판례에서 법원이 제시한 구체적 판단 기준(성립 요건, 위법성 판단 기준, 책임 범위 등)을 빠짐없이 서술한 뒤, 사용자 상황에 적용하세요. 관련 판례가 여러 건이면 각각의 핵심 판단을 구분하되 소제목 없이 문단 안에서 이어 쓰세요.

**주의할 점**: 결과를 바꿀 수 있는 구체적 변수를 나열하세요 (예: "블랙박스 영상 유무", "상대방 과실 비율"). "상황에 따라 다릅니다" 같은 일반론은 쓰지 마세요. 마지막에 변호사 상담을 안내하세요.

## 스타일

- 법률 용어는 괄호 안에 쉬운 설명 (예: "과실(주의 의무를 소홀히 한 것)")
- 추상적 요약보다 판례에 나온 구체적 판단 기준을 인용하세요

<example>
사용자: 교통사고 과실 비율이 어떻게 되나요?
응답:
**결론**: 교통사고 과실 비율은 사고 유형, 신호 준수 여부, 도로 상황 등에 따라 달라지며, 양 당사자의 주의 의무 위반 정도에 따라 결정됩니다.

**근거**: 도로교통법 제48조는 운전자에게 안전운전 의무를 부과하고 있습니다. 대법원 2017다12345 판결에서 법원은 "과실 비율은 사고 발생에 대한 각 당사자의 기여도를 종합적으로 고려하여 결정해야 한다"고 판시하였습니다...

**주의할 점**: 블랙박스 영상, 목격자 진술, 사고 현장 사진 등 증거 확보가 과실 비율 산정에 결정적입니다. 보험사 과실 비율에 이의가 있으면 손해사정사나 변호사 상담을 권장합니다.
</example>"""

# ---------------------------------------------------------------------------
# 변호사 전용 시스템 프롬프트
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_LAWYER = """당신은 변호사를 위한 법률 리서치 AI입니다.
제공된 참고 자료를 기반으로 실무에 즉시 활용 가능한 법리 분석을 제공합니다.

## 절대 규칙

1. 참고 자료에 명시된 판례번호·법령명·조문만 인용하세요. 참고 자료에 없는 것을 만들어내면 안 됩니다.
2. 참고 자료 중 질문과 무관한 문서는 무시하세요.
3. 같은 판례·법령을 여러 단락에서 반복하지 마세요.
4. 대법원 판례가 있으면 하급심보다 우선하여 인용하세요.
5. 이전 대화의 답변 형식이 아래 지시와 다르더라도, 반드시 아래 형식을 따르세요.
6. 질문 조건을 직접 충족하는 판례가 없더라도, 조건의 일부를 다루는 참고 자료가 있으면 해당 법리를 추출하여 '왜 직접 사례가 없고, 어떤 논리로 접근 가능한지'를 설명하세요. 참고 자료 전체에서 관련 법리를 전혀 찾을 수 없는 경우에만 "제공된 참고 자료에는 관련 법리가 없습니다"라고 밝히세요.
7. 관련 참고 자료가 2개 이상이면 최소 2개 이상을 근거로 사용하고, 각 자료가 결론에 기여한 지점을 명시하세요.

## 출처 표기

인용 시 반드시 출처를 표기하세요.
- 판례: 법원명 선고일자 사건번호 (예: 대법원 2018. 5. 15. 선고 2017도21977 판결)
- 법령: 법령명 + 조항 (예: 경범죄처벌법 제3조 제1항 제21호)

## 답변 형식

반드시 **근거**, **실무 포인트** 2개 섹션만 사용하세요. "결론:"이라는 별도 섹션을 만들지 마세요. 첫 문장에서 바로 핵심 판단을 제시한 뒤 근거로 이어가세요.

쟁점에 대한 직접 판단 1-2문장으로 시작하세요. 유리/불리 여부와 핵심 판단 기준을 명확히 제시하세요.

**근거**: 다음 순서로 구조화하되, 각 항목은 반드시 줄을 바꿔서 시작하세요.

(i) **쟁점 정리** — 본 사안에서 법적으로 다투어질 핵심 쟁점을 특정하세요.

(ii) **법리** — 관련 법령 조항과 판례에서 법원이 제시한 판단 기준(성립 요건, 위법성 판단 기준, 책임 범위, 입증책임 분배 등)을 인용하세요. 참고 자료가 여러 건이면 각각의 기여 지점을 구분하여 서술하세요.

(iii) **사실 요소** — 결론을 좌우하는 입증 대상 사실관계 변수를 나열하세요.

**실무 포인트**: 양측 관점에서 각 2개 이상의 공격·방어 포인트를 제시하세요.
- 민사: 원고 측 / 피고 측
- 형사: 공소 유지 측 / 변호 측
- 행정: 처분청 측 / 상대방 측
각 포인트에는 뒷받침하는 판례·법리 근거를 함께 적시하세요.

## 스타일

- 법률 용어를 그대로 사용하세요. 일반인 대상 괄호 설명은 불필요합니다.
- 추상적 요약보다 판례의 구체적 판시 사항을 인용하세요.
- 판례 법리를 사안에 포섭할 때 '해당 사안에서는 ~한 점에서 위 법리가 적용될 여지가 있다/없다' 형태로 명시적 포섭을 하세요.

<example>
사용자: 의료과실 소송에서 입증책임은 누구에게 있나요?
응답:
의료과실 소송에서 입증책임은 원칙적으로 환자(원고) 측에 있으나, 판례는 의사의 설명의무 위반 등에서 입증책임을 완화하고 있습니다.

**근거**:
(i) **쟁점 정리** — 의료과실의 인과관계 입증에서 환자 측의 입증 부담 완화 범위

(ii) **법리** — 대법원 2005다5867 판결은 "의료행위는 고도의 전문 지식을 필요로 하므로, 환자 측이 의료행위 과정에서 주의의무 위반과 손해 발생 사이의 인과관계를 의학적으로 완벽히 입증하는 것이 매우 어렵다"고 판시하면서, 일련의 의료 과정과 결과 사이에 일반인의 상식에 바탕을 둔 인과관계가 추정되면 의료진이 무과실을 입증해야 한다고 보았습니다...

**실무 포인트**:
- 원고 측: 진료 기록 감정 신청, 의료 감정 결과를 토대로 주의의무 위반 특정
- 피고 측: 해당 시점 의학적 수준에서 최선의 조치였음을 입증, 설명의무 이행 기록 제시
</example>"""


class LegalSearchAgent(BaseChatAgent):
    """법률 검색 에이전트 (Focus + Supplementary 통합 검색)"""

    def __init__(
        self,
        focus: Literal["precedent", "law"] = "precedent",
        precedent_service: PrecedentService | None = None,
        user_role: str = "user",
    ):
        self.focus = focus
        self._precedent_service = precedent_service
        self.user_role = user_role
        self.config = _PRESET_MAP.get(focus, _PRESET_MAP["precedent"])

    @property
    def precedent_service(self) -> PrecedentService:
        """PrecedentService lazy initialization"""
        if self._precedent_service is None:
            self._precedent_service = get_precedent_service()
        return self._precedent_service

    @property
    def name(self) -> str:
        if self.focus == "law":
            return "law_search"
        return "case_search"

    @property
    def description(self) -> str:
        if self.focus == "law":
            return "RAG 기반 법령 검색 및 법률 상담"
        return "RAG 기반 판례 검색 및 법률 상담"

    @property
    def supports_streaming(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # RAG 검색
    # ------------------------------------------------------------------

    async def _prepare_rag_data(
        self, message: str
    ) -> tuple[str, list[dict[str, Any]]]:
        """파이프라인 1회 호출 → 컨텍스트 + 소스.

        focus 모드 파이프라인이 리라이팅 1회 + focus/supplementary 병렬 검색을
        내부적으로 처리한다.
        """
        result = await search_with_pipeline_async(message, self.config)

        focus_documents = result.documents
        supplementary_documents = result.supplementary_documents or []

        # 판례 상세 조회 (focus=precedent인 경우만)
        precedent_details: dict[str, dict[str, Any]] = {}
        if self.focus == "precedent" and focus_documents:
            source_ids = [
                doc.get("metadata", {}).get("doc_id")
                for doc in focus_documents
                if doc.get("metadata", {}).get("doc_id")
            ]
            if source_ids:
                precedent_details = await self.precedent_service.get_details(
                    source_ids
                )

        # 컨텍스트 구성 (format_utils)
        context = self._build_context(
            focus_documents, precedent_details, supplementary_documents
        )

        # 소스 정보 (format_utils)
        sources = self._format_sources(
            focus_documents, precedent_details, supplementary_documents
        )

        return context, sources

    # ------------------------------------------------------------------
    # 체계도 네비게이션
    # ------------------------------------------------------------------

    def _is_hierarchy_query(self, message: str) -> bool:
        """체계도 관련 쿼리인지 확인."""
        return self.focus == "law" and bool(_HIERARCHY_KEYWORDS.search(message))

    async def _extract_statute_name(self, message: str) -> str | None:
        """메시지에서 법령명 추출 (체계도 키워드 제거 후 핵심 명사)."""
        # 체계도 키워드와 일반적인 접미사 제거
        cleaned = _HIERARCHY_KEYWORDS.sub("", message)
        cleaned = re.sub(r"(보여\s*줘|알려\s*줘|찾아\s*줘|검색|조회|의\s*$)", "", cleaned)
        cleaned = cleaned.strip()
        return cleaned if cleaned else None

    async def _search_statute(
        self, query: str
    ) -> dict[str, str] | None:
        """PgGraphService로 법령 검색. 실패 시 None."""
        try:
            from app.tools.graph.pg_graph_service import get_pg_graph_service

            pg = get_pg_graph_service()
            results = await pg.search_statutes(query, limit=1)
            if results:
                return {
                    "id": results[0]["id"],
                    "name": results[0]["name"],
                }
        except (ValueError, RuntimeError):
            logger.debug("체계도 법령 검색 실패: %s", query, exc_info=True)
        return None

    async def _handle_hierarchy_fast_path(
        self, message: str
    ) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]] | None:
        """체계도 쿼리 fast path: RAG/LLM 없이 직접 응답 + NAVIGATE 액션.

        Returns:
            (response_text, sources, actions) 또는 None (fast path 불가)
        """
        statute_query = await self._extract_statute_name(message)
        if not statute_query:
            return None

        statute = await self._search_statute(statute_query)
        if not statute:
            return None

        params: dict[str, str] = {
            "id": statute["id"],
            "name": statute["name"],
        }
        actions = [
            ChatAction(
                type=ActionType.NAVIGATE,
                label="법령 체계도에서 보기",
                url="/statute-hierarchy",
                params=params,
            ).model_dump()
        ]

        response = (
            f"**{statute['name']}**의 체계도를 확인하실 수 있습니다.\n\n"
            "아래 버튼을 클릭하면 법령 체계도 화면으로 이동합니다."
        )

        return response, [], actions

    async def _build_hierarchy_actions(
        self, message: str, sources: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """체계도 키워드가 있으면 NAVIGATE 액션 생성 (RAG 경유 시 사용)."""
        if not self._is_hierarchy_query(message):
            return []

        # sources에서 첫 번째 법령명 추출
        statute_name: str | None = None
        for src in sources:
            if src.get("doc_type") != "law":
                continue

            law_name = src.get("law_name") or src.get("title") or src.get("case_name")
            if law_name:
                statute_name = law_name
                break

        if not statute_name:
            return []

        # 법령명으로 statute_id를 조회해 액션 param id를 맞추기 시도
        statute: dict[str, str] | None = await self._search_statute(statute_name)
        statute_id = statute["id"] if statute else None
        if statute:
            statute_name = statute["name"]

        params: dict[str, str] = {"name": statute_name}
        if statute_id:
            params["id"] = statute_id

        return [
            ChatAction(
                type=ActionType.NAVIGATE,
                label="법령 체계도에서 보기",
                url="/statute-hierarchy",
                params=params,
            ).model_dump()
        ]

    # ------------------------------------------------------------------
    # process / process_stream
    # ------------------------------------------------------------------

    async def process(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        session_data: dict[str, Any] | None = None,
        user_location: dict[str, float] | None = None,
    ) -> AgentResult:
        """법률 검색 및 응답 생성"""
        # 체계도 fast path: RAG/LLM 건너뛰기
        if self._is_hierarchy_query(message):
            fast = await self._handle_hierarchy_fast_path(message)
            if fast:
                resp_text, sources, actions = fast
                return AgentResult(
                    message=resp_text,
                    sources=sources,
                    actions=actions,
                    session_data={"active_agent": self.name, "focus": self.focus},
                    agent_used=self.name,
                )

        context, sources = await self._prepare_rag_data(message)

        response = await self._generate_response(
            message=message,
            context=context,
            history=history,
        )

        actions = await self._build_hierarchy_actions(message, sources)

        return AgentResult(
            message=response,
            sources=sources,
            actions=actions,
            session_data={"active_agent": self.name, "focus": self.focus},
            agent_used=self.name,
        )

    async def process_stream(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        session_data: dict[str, Any] | None = None,
        user_location: dict[str, float] | None = None,
    ) -> AsyncGenerator[tuple[str, Any], None]:
        """스트리밍 법률 검색 및 응답 생성"""
        # 체계도 fast path: RAG/LLM 건너뛰기
        if self._is_hierarchy_query(message):
            fast = await self._handle_hierarchy_fast_path(message)
            if fast:
                resp_text, sources, actions = fast
                yield ("token", {"content": resp_text})
                yield ("sources", {"sources": sources})
                yield ("metadata", {
                    "agent_used": self.name,
                    "actions": actions,
                    "session_data": {
                        "active_agent": self.name,
                        "focus": self.focus,
                    },
                })
                yield ("done", {})
                return

        context, sources = await self._prepare_rag_data(message)

        # sources를 LLM 스트리밍 전에 먼저 전송 → 프론트 왼쪽 패널 즉시 업데이트
        yield ("sources", {"sources": sources})
        yield ("metadata", {
            "agent_used": self.name,
            "actions": [],
            "session_data": {"active_agent": self.name, "focus": self.focus},
        })

        model = get_chat_model()
        messages = self._build_messages(message, context, history)

        async for chunk in model.astream(messages):
            if chunk.content:
                text = normalize_chunk_content(chunk.content)
                if text:
                    yield ("token", {"content": text})

        actions = await self._build_hierarchy_actions(message, sources)

        # actions는 LLM 완료 후 확정되므로 done에 포함
        yield ("metadata", {
            "agent_used": self.name,
            "actions": actions,
            "session_data": {"active_agent": self.name, "focus": self.focus},
        })
        yield ("done", {})

    # ------------------------------------------------------------------
    # 컨텍스트 / 소스 (format_utils 위임)
    # ------------------------------------------------------------------

    def _build_context(
        self,
        focus_documents: list[dict[str, Any]],
        precedent_details: dict[str, dict[str, Any]],
        supplementary_documents: list[dict[str, Any]],
    ) -> str:
        """Focus + Supplementary 통합 컨텍스트 구성."""
        parts: list[str] = []

        if self.focus == "precedent":
            text = format_precedent_context(focus_documents)
        else:
            text = format_law_context(focus_documents)
        if text:
            parts.append(text)

        sup_text = format_supplementary_context(supplementary_documents)
        if sup_text:
            parts.append(sup_text)

        return "\n\n".join(parts)

    def _format_sources(
        self,
        focus_documents: list[dict[str, Any]],
        precedent_details: dict[str, dict[str, Any]],
        supplementary_documents: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Focus + Supplementary 소스 정보 포맷팅."""
        if self.focus == "precedent":
            focus_sources = format_precedent_sources(
                focus_documents, precedent_details
            )
        else:
            focus_sources = format_law_sources(focus_documents)

        return focus_sources + format_supplementary_sources(
            supplementary_documents
        )

    # ------------------------------------------------------------------
    # LLM 응답
    # ------------------------------------------------------------------

    _MAX_HISTORY_TURNS: int = 3

    def _build_messages(
        self,
        message: str,
        context: str,
        history: list[dict[str, str]] | None = None,
    ) -> list[tuple[str, str]]:
        """시스템 프롬프트 + 히스토리 + 사용자 메시지 구성.

        - 히스토리는 최근 N턴(user+assistant 쌍)만 유지하여 오래된 답변
          형식이 현재 시스템 프롬프트를 덮어쓰지 않도록 한다.
        - 질문을 먼저, 참고 자료를 뒤에 배치하여 LLM 주의력을 질문에
          집중시킨다.
        """
        prompt = (
            _SYSTEM_PROMPT_LAWYER if self.user_role == "lawyer"
            else _SYSTEM_PROMPT
        )
        messages: list[tuple[str, str]] = [("system", prompt)]

        if history:
            # 최근 N턴만 유지 (1턴 = user + assistant 2개)
            recent = history[-(self._MAX_HISTORY_TURNS * 2) :]
            for h in recent:
                messages.append((h.get("role", "user"), h.get("content", "")))

        user_message = f"""사용자 질문: {message}

아래는 검색된 참고 자료입니다. 질문과 무관한 문서가 포함될 수 있으니, 관련 있는 자료만 선별하여 답변하세요.

{context}"""

        messages.append(("user", user_message))
        return messages

    async def _generate_response(
        self,
        message: str,
        context: str,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """LLM 응답 생성 (비동기)"""
        model = get_chat_model()
        messages = self._build_messages(message, context, history)
        response = await model.ainvoke(messages)
        content = response.content
        return content if isinstance(content, str) else str(content)

    def can_handle(self, message: str) -> bool:
        """법률 관련 키워드 확인"""
        keywords = [
            "판례", "사례", "판결", "재판", "법원", "선례",
            "법령", "법률", "조문", "규정", "법조"
        ]
        return any(kw in message for kw in keywords)
