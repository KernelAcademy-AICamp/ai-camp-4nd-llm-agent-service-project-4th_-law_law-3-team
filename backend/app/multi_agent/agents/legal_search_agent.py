"""
법률 검색 에이전트

RAG 기반 판례/법령 + 다중 타입 검색 및 법률 상담 제공.
파이프라인 PRESETS(focus 모드) + format_utils로 정규화.
"""

import logging
import re
from collections.abc import AsyncGenerator
from typing import Any, Literal

from app.multi_agent.agents.base_chat import ActionType, BaseChatAgent, ChatAction
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
사용자의 질문에 대해 제공된 판례, 법령, 그리고 관련 법률 자료를 참고하여 정확하고 이해하기 쉽게 답변해주세요.

답변 시 유의사항:
1. 제공된 판례, 법령, 기타 법률 자료를 근거로 답변하세요
2. 법률 용어는 쉽게 풀어서 설명하세요
3. 일반적인 정보 제공이며, 구체적인 법률 상담은 변호사에게 의뢰하도록 안내하세요
4. 판례 번호와 법령명을 언급할 때는 정확하게 표기하세요
5. 헌재결정례, 행정심판례 등 보충 자료가 있으면 함께 활용하세요"""


class LegalSearchAgent(BaseChatAgent):
    """법률 검색 에이전트 (Focus + Supplementary 통합 검색)"""

    def __init__(
        self,
        focus: Literal["precedent", "law"] = "precedent",
        precedent_service: PrecedentService | None = None,
    ):
        self.focus = focus
        self._precedent_service = precedent_service
        self.config = _PRESET_MAP.get(focus, _PRESET_MAP["precedent"])

    @property
    def precedent_service(self) -> PrecedentService:
        """PrecedentService lazy initialization"""
        if self._precedent_service is None:
            self._precedent_service = get_precedent_service()
        return self._precedent_service

    @property
    def name(self) -> str:
        return "legal_search"

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
        except Exception:
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

        model = get_chat_model()
        messages = self._build_messages(message, context, history)

        async for chunk in model.astream(messages):
            if chunk.content and isinstance(chunk.content, str):
                yield ("token", {"content": chunk.content})

        actions = await self._build_hierarchy_actions(message, sources)

        yield ("sources", {"sources": sources})
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

    def _build_messages(
        self,
        message: str,
        context: str,
        history: list[dict[str, str]] | None = None,
    ) -> list[tuple[str, str]]:
        """시스템 프롬프트 + 히스토리 + 사용자 메시지 구성"""
        messages: list[tuple[str, str]] = [("system", _SYSTEM_PROMPT)]

        if history:
            for h in history:
                messages.append((h.get("role", "user"), h.get("content", "")))

        user_message = f"""참고 자료:
{context}

사용자 질문: {message}"""

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
