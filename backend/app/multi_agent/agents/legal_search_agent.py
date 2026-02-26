"""
법률 검색 에이전트

RAG 기반 판례/법령 + 다중 타입 검색 및 법률 상담 제공.
파이프라인 PRESETS(focus 모드) + format_utils로 정규화.
"""

import logging
from collections.abc import AsyncGenerator
from typing import Any, Literal

from app.multi_agent.agents.base_chat import BaseChatAgent
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
        context, sources = await self._prepare_rag_data(message)

        response = await self._generate_response(
            message=message,
            context=context,
            history=history,
        )

        return AgentResult(
            message=response,
            sources=sources,
            actions=[],
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
        context, sources = await self._prepare_rag_data(message)

        model = get_chat_model()
        messages = self._build_messages(message, context, history)

        async for chunk in model.astream(messages):
            if chunk.content and isinstance(chunk.content, str):
                yield ("token", {"content": chunk.content})

        yield ("sources", {"sources": sources})
        yield ("metadata", {
            "agent_used": self.name,
            "actions": [],
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
