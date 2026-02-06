"""
로스쿨 학습 에이전트

RAG(법령) + LLM으로 법학 학습 자료 제공
학습용 프롬프트로 법령 내용을 교육적으로 설명
"""

import logging
from collections.abc import AsyncGenerator
from typing import Any

from app.multi_agent.agents.base_chat import BaseChatAgent
from app.multi_agent.schemas.plan import AgentResult
from app.services.rag import search_relevant_documents

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """당신은 법학 교육 전문 AI 튜터입니다.
학습자가 이해하기 쉽도록 법령과 판례를 교육적으로 설명합니다.

답변 규칙:
1. 핵심 개념을 먼저 정의하세요
2. 관련 법령 조문을 인용하며 설명하세요
3. 실제 사례나 예시를 들어 이해를 도우세요
4. 관련된 다른 개념과의 차이점을 설명하세요
5. 시험에 자주 출제되는 포인트가 있다면 강조하세요
6. 마지막에 핵심 정리를 제공하세요"""


class LawStudyAgent(BaseChatAgent):
    """로스쿨 학습 에이전트 (법학 교육)"""

    @property
    def name(self) -> str:
        return "law_study"

    @property
    def description(self) -> str:
        return "법학 학습 자료 제공 및 교육적 설명"

    @property
    def supports_streaming(self) -> bool:
        return True

    async def process(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        session_data: dict[str, Any] | None = None,
        user_location: dict[str, float] | None = None,
    ) -> AgentResult:
        """학습 자료 생성"""
        from app.tools.llm import get_chat_model

        # RAG: 법령 중심 검색
        law_results = search_relevant_documents(
            query=message, n_results=3, doc_type="law",
        )

        # 컨텍스트 구성
        context = self._build_study_context(law_results)

        # LLM 응답
        model = get_chat_model()
        messages: list[tuple[str, str]] = [("system", _SYSTEM_PROMPT)]
        if history:
            for h in history:
                messages.append((h.get("role", "user"), h.get("content", "")))

        user_message = f"참고 법령:\n{context}\n\n학습 질문: {message}"
        messages.append(("user", user_message))

        response = model.invoke(messages)

        sources = self._format_sources(law_results)

        return AgentResult(
            message=response.content,
            sources=sources,
            actions=[],
            session_data={"active_agent": self.name},
            agent_used=self.name,
        )

    async def process_stream(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        session_data: dict[str, Any] | None = None,
        user_location: dict[str, float] | None = None,
    ) -> AsyncGenerator[tuple[str, Any], None]:
        """스트리밍 학습 자료 생성"""
        from app.tools.llm import get_chat_model

        # RAG: 법령 중심 검색
        law_results = search_relevant_documents(
            query=message, n_results=3, doc_type="law",
        )

        context = self._build_study_context(law_results)
        sources = self._format_sources(law_results)

        # LLM 스트리밍
        model = get_chat_model()
        messages: list[tuple[str, str]] = [("system", _SYSTEM_PROMPT)]
        if history:
            for h in history:
                messages.append((h.get("role", "user"), h.get("content", "")))

        user_message = f"참고 법령:\n{context}\n\n학습 질문: {message}"
        messages.append(("user", user_message))

        async for chunk in model.astream(messages):
            if chunk.content:
                yield ("token", {"content": chunk.content})

        yield ("sources", {"sources": sources})
        yield ("metadata", {
            "agent_used": self.name,
            "actions": [],
            "session_data": {"active_agent": self.name},
        })
        yield ("done", {})

    def _build_study_context(self, laws: list[dict[str, Any]]) -> str:
        """학습용 법령 컨텍스트 구성"""
        if not laws:
            return "(관련 법령 없음)"

        parts = []
        for i, doc in enumerate(laws, 1):
            metadata = doc.get("metadata", {})
            law_name = metadata.get("case_name", "") or metadata.get("title", "")
            content = doc.get("content", "")
            parts.append(f"[법령 {i}] {law_name}\n{content}")

        return "\n\n".join(parts)

    def _format_sources(self, laws: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """법령 소스 포맷팅"""
        sources = []
        for doc in laws:
            metadata = doc.get("metadata", {})
            sources.append({
                "doc_type": "law",
                "law_name": metadata.get("case_name", "") or metadata.get("title", ""),
                "similarity": round(doc.get("similarity", 0), 3),
                "content": doc.get("content", ""),
            })
        return sources

    def can_handle(self, message: str) -> bool:
        """학습 관련 키워드 확인"""
        keywords = ["공부", "학습", "시험", "로스쿨", "법학", "문제"]
        return any(kw in message for kw in keywords)
