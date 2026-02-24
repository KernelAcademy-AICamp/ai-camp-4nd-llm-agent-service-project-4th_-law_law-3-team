"""
법률 검색 에이전트

RAG 기반 판례/법령 + 다중 타입 검색 및 법률 상담 제공.
Focus(주 타입) + Supplementary(보충 타입) 병렬 검색 구조.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any, Literal

from app.multi_agent.agents.base_chat import BaseChatAgent
from app.multi_agent.schemas.plan import AgentResult
from app.services.rag.pipeline import (
    PipelineConfig,
    search_with_pipeline_async,
)
from app.services.rag.query_rewrite import rewrite_conversational_query
from app.services.service_function import (
    PrecedentService,
    get_precedent_service,
)
from app.tools.llm import get_chat_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Focus + Supplementary 파이프라인 설정
# ---------------------------------------------------------------------------
# enable_rewrite=False: 에이전트 레벨에서 1회만 리라이팅

FOCUS_CONFIG: dict[str, PipelineConfig] = {
    "precedent": PipelineConfig(
        n_results=15, doc_type="precedent",
        enable_rewrite=False, enable_rerank=True, rerank_top_k=5,
    ),
    "law": PipelineConfig(
        n_results=15, doc_type="law",
        enable_rewrite=False, enable_rerank=True, rerank_top_k=5,
    ),
}

SUPPLEMENTARY_CONFIG: dict[str, PipelineConfig] = {
    "precedent": PipelineConfig(
        n_results=7, exclude_doc_types=["판례"],
        enable_rewrite=False, enable_rerank=True, rerank_top_k=3,
    ),
    "law": PipelineConfig(
        n_results=7, exclude_doc_types=["법령"],
        enable_rewrite=False, enable_rerank=True, rerank_top_k=3,
    ),
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

        # Focus/Supplementary 파이프라인 설정
        self.focus_config = FOCUS_CONFIG.get(focus, FOCUS_CONFIG["precedent"])
        self.supplementary_config = SUPPLEMENTARY_CONFIG.get(
            focus, SUPPLEMENTARY_CONFIG["precedent"]
        )

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
        """Focus + Supplementary 병렬 검색 → 컨텍스트 + 소스.

        Returns:
            (context, sources)
        """
        # 1. Focus + Supplementary 병렬 검색
        focus_result, supplementary_result = await asyncio.gather(
            search_with_pipeline_async(message, self.focus_config),
            search_with_pipeline_async(message, self.supplementary_config),
        )

        focus_documents = focus_result.documents
        supplementary_documents = supplementary_result.documents

        # 2. Focus 판례 상세 조회 (focus=precedent인 경우만)
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

        # 3. 컨텍스트 구성
        context = self._build_context(
            focus_documents, precedent_details, supplementary_documents
        )

        # 4. 소스 정보 정리
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
        # 쿼리 리라이팅 1회 (에이전트 레벨)
        search_query = await rewrite_conversational_query(message, history)
        context, sources = await self._prepare_rag_data(search_query)

        # LLM 응답은 원본 message로 생성 (자연스러운 대화)
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
        # 쿼리 리라이팅 1회
        search_query = await rewrite_conversational_query(message, history)
        context, sources = await self._prepare_rag_data(search_query)

        # LLM 스트리밍 응답은 원본 message로 생성
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
    # 컨텍스트 구성
    # ------------------------------------------------------------------

    def _build_context(
        self,
        focus_documents: list[dict[str, Any]],
        precedent_details: dict[str, dict[str, Any]],
        supplementary_documents: list[dict[str, Any]],
    ) -> str:
        """Focus + Supplementary 통합 컨텍스트 구성."""
        context_parts: list[str] = []

        # Focus 컨텍스트
        if self.focus == "precedent":
            context_parts.extend(
                self._build_precedent_context(focus_documents, precedent_details)
            )
        else:
            context_parts.extend(self._build_law_context(focus_documents))

        # Supplementary 컨텍스트
        context_parts.extend(
            self._build_supplementary_context(supplementary_documents)
        )

        return "\n\n".join(context_parts)

    def _build_precedent_context(
        self,
        documents: list[dict[str, Any]],
        details: dict[str, dict[str, Any]],
    ) -> list[str]:
        """판례 컨텍스트 구성"""
        if not documents:
            return []

        parts = ["## 관련 판례"]
        for i, doc in enumerate(documents, 1):
            metadata = doc.get("metadata", {})
            doc_id = metadata.get("doc_id", "")
            case_name = metadata.get("case_name", "")
            case_number = metadata.get("case_number", "")
            content = doc.get("content", "")

            part = f"[판례 {i}] {case_name} ({case_number})\n{content}"

            if doc_id in details:
                detail = details[doc_id]
                if detail.get("ruling"):
                    part += f"\n[주문] {detail['ruling']}"
                if detail.get("reasoning"):
                    part += f"\n[판결요지] {detail['reasoning']}"

            parts.append(part)

        return parts

    def _build_law_context(self, laws: list[dict[str, Any]]) -> list[str]:
        """법령 컨텍스트 구성"""
        if not laws:
            return []

        parts = ["## 관련 법령"]
        for i, doc in enumerate(laws, 1):
            metadata = doc.get("metadata", {})
            law_name = metadata.get("case_name", "") or metadata.get("title", "")
            content = doc.get("content", "")

            part = f"[법령 {i}] {law_name}\n{content}"
            parts.append(part)

        return parts

    def _build_supplementary_context(
        self, documents: list[dict[str, Any]]
    ) -> list[str]:
        """Supplementary 문서 컨텍스트 구성 (다양한 타입)"""
        if not documents:
            return []

        parts = ["## 관련 법률 자료 (보충)"]
        for i, doc in enumerate(documents, 1):
            metadata = doc.get("metadata", {})
            data_type = metadata.get("data_type", "")
            title = metadata.get("case_name", "") or metadata.get("title", "")
            content = doc.get("content", "")

            part = f"[{data_type} {i}] {title}\n{content}"
            parts.append(part)

        return parts

    # ------------------------------------------------------------------
    # 소스 포맷팅
    # ------------------------------------------------------------------

    def _format_sources(
        self,
        focus_documents: list[dict[str, Any]],
        precedent_details: dict[str, dict[str, Any]],
        supplementary_documents: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Focus + Supplementary 소스 정보 포맷팅."""
        focus_sources: list[dict[str, Any]]

        if self.focus == "precedent":
            focus_sources = self._format_precedent_sources(
                focus_documents, precedent_details
            )
        else:
            focus_sources = self._format_law_sources(focus_documents)

        supplementary_sources = self._format_supplementary_sources(
            supplementary_documents
        )

        return focus_sources + supplementary_sources

    def _format_precedent_sources(
        self,
        documents: list[dict[str, Any]],
        details: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """판례 소스 정보 포맷팅"""
        sources: list[dict[str, Any]] = []
        for doc in documents:
            metadata = doc.get("metadata", {})
            doc_id = metadata.get("doc_id", "")
            case_number = metadata.get("case_number", "")

            source_item: dict[str, Any] = {
                "doc_id": doc_id,
                "doc_type": "precedent",
                "case_name": metadata.get("case_name", ""),
                "case_number": case_number,
                "court_name": metadata.get("court_name", ""),
                "similarity": round(doc.get("similarity", 0), 3),
                "content": doc.get("content", ""),
            }

            if doc_id in details:
                detail = details[doc_id]
                if not case_number and detail.get("case_number"):
                    source_item["case_number"] = detail["case_number"]
                if not source_item["case_name"] and detail.get("case_name"):
                    source_item["case_name"] = detail["case_name"]
                source_item["ruling"] = detail.get("ruling", "")
                source_item["claim"] = detail.get("claim", "")
                source_item["reasoning"] = detail.get("reasoning", "")
                source_item["decision_date"] = detail.get("decision_date", "")
                source_item["case_type"] = detail.get("case_type", "")
                source_item["summary"] = detail.get("summary", "")
                source_item["full_reason"] = detail.get("full_reason", "")
                source_item["full_text"] = detail.get("full_text", "")
                source_item["reference_provisions"] = detail.get(
                    "reference_provisions", ""
                )
                source_item["reference_cases"] = detail.get(
                    "reference_cases", ""
                )

            sources.append(source_item)

        return sources

    def _format_law_sources(
        self, laws: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """법령 소스 정보 포맷팅"""
        sources: list[dict[str, Any]] = []
        for doc in laws:
            metadata = doc.get("metadata", {})
            sources.append({
                "doc_id": metadata.get("doc_id", ""),
                "doc_type": "law",
                "law_name": (
                    metadata.get("case_name", "") or metadata.get("title", "")
                ),
                "similarity": round(doc.get("similarity", 0), 3),
                "content": doc.get("content", ""),
            })
        return sources

    def _format_supplementary_sources(
        self, documents: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Supplementary 소스 정보 포맷팅 (다양한 타입)"""
        sources: list[dict[str, Any]] = []
        for doc in documents:
            metadata = doc.get("metadata", {})
            data_type = metadata.get("data_type", "")
            sources.append({
                "doc_id": metadata.get("doc_id", ""),
                "doc_type": data_type,
                "title": (
                    metadata.get("case_name", "") or metadata.get("title", "")
                ),
                "source_name": metadata.get("court_name", ""),
                "date": metadata.get("date", ""),
                "similarity": round(doc.get("similarity", 0), 3),
                "content": doc.get("content", ""),
            })
        return sources

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
