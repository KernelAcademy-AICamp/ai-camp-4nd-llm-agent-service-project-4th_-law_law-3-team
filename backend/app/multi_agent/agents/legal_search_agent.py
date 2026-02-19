"""
법률 검색 에이전트

RAG 기반 판례/법령 검색 및 법률 상담 제공
focus 파라미터로 검색 비율 조절
"""

import logging
from collections.abc import AsyncGenerator
from typing import Any, Literal

from app.multi_agent.agents.base_chat import BaseChatAgent
from app.multi_agent.schemas.plan import AgentResult
from app.services.rag.pipeline import PipelineConfig, search_with_pipeline_async
from app.services.rag.query_rewrite import rewrite_conversational_query
from app.services.service_function import (
    PrecedentService,
    get_precedent_service,
)
from app.tools.llm import get_chat_model

logger = logging.getLogger(__name__)

# focus별 파이프라인 설정 (검색 + 리랭킹)
SEARCH_CONFIG: dict[str, dict[str, PipelineConfig]] = {
    "precedent": {
        "precedent": PipelineConfig(
            n_results=15, doc_type="precedent",
            enable_rerank=True, rerank_top_k=4,
        ),
        "law": PipelineConfig(
            n_results=1, doc_type="law",
        ),
    },
    "law": {
        "precedent": PipelineConfig(
            n_results=1, doc_type="precedent",
        ),
        "law": PipelineConfig(
            n_results=15, doc_type="law",
            enable_rerank=True, rerank_top_k=4,
        ),
    },
}

_SYSTEM_PROMPT = """당신은 법률 전문 AI 어시스턴트입니다.
사용자의 질문에 대해 제공된 판례와 법령을 참고하여 정확하고 이해하기 쉽게 답변해주세요.

답변 시 유의사항:
1. 제공된 판례와 법령을 근거로 답변하세요
2. 법률 용어는 쉽게 풀어서 설명하세요
3. 일반적인 정보 제공이며, 구체적인 법률 상담은 변호사에게 의뢰하도록 안내하세요
4. 판례 번호와 법령명을 언급할 때는 정확하게 표기하세요
5. 판례와 법령을 함께 활용하여 종합적인 답변을 제공하세요"""


class LegalSearchAgent(BaseChatAgent):
    """법률 검색 에이전트 (판례/법령 통합 검색)"""

    def __init__(
        self,
        focus: Literal["precedent", "law"] = "precedent",
        precedent_service: PrecedentService | None = None,
    ):
        self.focus = focus
        self._precedent_service = precedent_service

        # focus에 따른 파이프라인 설정
        configs = SEARCH_CONFIG.get(focus, SEARCH_CONFIG["precedent"])
        self.precedent_config = configs["precedent"]
        self.law_config = configs["law"]

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
        """LegalSearchAgent는 스트리밍 지원"""
        return True

    async def _prepare_rag_data(
        self, message: str
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        dict[str, dict[str, Any]],
        dict[str, dict[str, Any]],
        str,
        list[dict[str, Any]],
    ]:
        """RAG 검색 → 상세조회 → 그래프보강 → 컨텍스트 → 소스 포맷

        Returns:
            (precedent_results, law_results, precedent_details,
             graph_contexts, context, sources)
        """
        # 1. 하이브리드 검색 + 리랭킹 (RAGPipeline)
        precedent_result = await search_with_pipeline_async(
            message, self.precedent_config
        )
        precedent_results = precedent_result.documents

        law_result = await search_with_pipeline_async(
            message, self.law_config
        )
        law_results = law_result.documents

        # 2. 판례 상세 정보 조회
        source_ids = [
            doc.get("metadata", {}).get("doc_id")
            for doc in precedent_results
            if doc.get("metadata", {}).get("doc_id")
        ]

        precedent_details: dict[str, dict[str, Any]] = {}
        if source_ids:
            precedent_details = self.precedent_service.get_details(source_ids)

        # 3. 그래프 컨텍스트 (비활성화 — Neo4j 연결 비용 대비 효과 미미)
        graph_contexts: dict[str, dict[str, Any]] = {}

        # 4. 컨텍스트 구성
        context = self._build_context(
            precedent_results, precedent_details, graph_contexts, law_results
        )

        # 5. 소스 정보 정리
        sources = self._format_sources(
            precedent_results, precedent_details, graph_contexts, law_results
        )

        return (
            precedent_results,
            law_results,
            precedent_details,
            graph_contexts,
            context,
            sources,
        )

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

    async def process(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        session_data: dict[str, Any] | None = None,
        user_location: dict[str, float] | None = None,
    ) -> AgentResult:
        """법률 검색 및 응답 생성"""
        # 대화형 쿼리 리라이팅: RAG 검색에만 적용
        search_query = await rewrite_conversational_query(message, history)
        _, _, _, _, context, sources = await self._prepare_rag_data(search_query)

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

    def _build_context(
        self,
        precedents: list[dict[str, Any]],
        precedent_details: dict[str, dict[str, Any]],
        graph_contexts: dict[str, dict[str, Any]],
        laws: list[dict[str, Any]],
    ) -> str:
        """통합 컨텍스트 구성 (focus에 따라 순서 조절)"""
        context_parts = []

        # focus에 따라 순서 결정
        if self.focus == "law":
            # 법령 먼저
            context_parts.extend(self._build_law_context(laws))
            context_parts.extend(
                self._build_precedent_context(precedents, precedent_details, graph_contexts)
            )
        else:
            # 판례 먼저 (기본)
            context_parts.extend(
                self._build_precedent_context(precedents, precedent_details, graph_contexts)
            )
            context_parts.extend(self._build_law_context(laws))

        return "\n\n".join(context_parts)

    def _build_precedent_context(
        self,
        documents: list[dict[str, Any]],
        details: dict[str, dict[str, Any]],
        graph_contexts: dict[str, dict[str, Any]],
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

            # 상세 정보 추가
            if doc_id in details:
                detail = details[doc_id]
                if detail.get("ruling"):
                    part += f"\n[주문] {detail['ruling']}"
                if detail.get("reasoning"):
                    part += f"\n[판결요지] {detail['reasoning']}"

            # 그래프 컨텍스트 추가 (인용 법령)
            if case_number in graph_contexts:
                graph_ctx = graph_contexts[case_number]
                cited = graph_ctx.get("cited_statutes", [])
                if cited:
                    cited_names = [s.get("name", "") for s in cited[:3]]
                    part += f"\n[인용법령] {', '.join(cited_names)}"

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

    def _format_sources(
        self,
        precedents: list[dict[str, Any]],
        precedent_details: dict[str, dict[str, Any]],
        graph_contexts: dict[str, dict[str, Any]],
        laws: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """소스 정보 포맷팅 (focus에 따라 순서 조절)"""
        precedent_sources = self._format_precedent_sources(
            precedents, precedent_details, graph_contexts
        )
        law_sources = self._format_law_sources(laws)

        # focus에 따라 순서 결정
        if self.focus == "law":
            return law_sources + precedent_sources
        return precedent_sources + law_sources

    def _format_precedent_sources(
        self,
        documents: list[dict[str, Any]],
        details: dict[str, dict[str, Any]],
        graph_contexts: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """판례 소스 정보 포맷팅"""
        sources = []
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

            # 상세 정보 추가
            if doc_id in details:
                detail = details[doc_id]
                source_item["ruling"] = detail.get("ruling", "")
                source_item["claim"] = detail.get("claim", "")
                source_item["reasoning"] = detail.get("reasoning", "")
                source_item["decision_date"] = detail.get("decision_date", "")
                source_item["case_type"] = detail.get("case_type", "")
                source_item["summary"] = detail.get("summary", "")
                source_item["full_reason"] = detail.get("full_reason", "")
                source_item["full_text"] = detail.get("full_text", "")
                source_item["reference_provisions"] = detail.get("reference_provisions", "")
                source_item["reference_cases"] = detail.get("reference_cases", "")

            # 그래프 컨텍스트 추가
            if case_number in graph_contexts:
                graph_ctx = graph_contexts[case_number]
                source_item["cited_statutes"] = graph_ctx.get("cited_statutes", [])
                source_item["similar_cases"] = graph_ctx.get("similar_cases", [])

            sources.append(source_item)

        return sources

    def _format_law_sources(self, laws: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """법령 소스 정보 포맷팅"""
        sources = []
        for doc in laws:
            metadata = doc.get("metadata", {})
            sources.append({
                "doc_id": metadata.get("doc_id", ""),
                "doc_type": "law",
                "law_name": metadata.get("case_name", "") or metadata.get("title", ""),
                "similarity": round(doc.get("similarity", 0), 3),
                "content": doc.get("content", ""),
            })
        return sources

    async def _generate_response(
        self,
        message: str,
        context: str,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """LLM 응답 생성"""
        model = get_chat_model()
        messages = self._build_messages(message, context, history)
        response = model.invoke(messages)
        return response.content

    async def process_stream(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        session_data: dict[str, Any] | None = None,
        user_location: dict[str, float] | None = None,
    ) -> AsyncGenerator[tuple[str, Any], None]:
        """스트리밍 법률 검색 및 응답 생성"""
        # 대화형 쿼리 리라이팅: RAG 검색에만 적용
        search_query = await rewrite_conversational_query(message, history)
        _, _, _, _, context, sources = await self._prepare_rag_data(search_query)

        # LLM 스트리밍 응답은 원본 message로 생성
        model = get_chat_model()
        messages = self._build_messages(message, context, history)

        async for chunk in model.astream(messages):
            if chunk.content:
                yield ("token", {"content": chunk.content})

        yield ("sources", {"sources": sources})
        yield ("metadata", {
            "agent_used": self.name,
            "actions": [],
            "session_data": {"active_agent": self.name, "focus": self.focus},
        })
        yield ("done", {})

    def can_handle(self, message: str) -> bool:
        """법률 관련 키워드 확인"""
        keywords = [
            "판례", "사례", "판결", "재판", "법원", "선례",
            "법령", "법률", "조문", "규정", "법조"
        ]
        return any(kw in message for kw in keywords)
