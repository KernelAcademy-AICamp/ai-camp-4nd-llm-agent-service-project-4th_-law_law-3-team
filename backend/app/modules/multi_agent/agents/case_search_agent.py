"""
판례 검색 에이전트

기존 RAG 기반 chat_service를 래핑하여 판례 검색 기능 제공
"""

from typing import Any

from app.common.agent_base import AgentResponse, BaseAgent
from app.common.chat_service import generate_chat_response


class CaseSearchAgent(BaseAgent):
    """판례 검색 에이전트"""

    @property
    def name(self) -> str:
        return "case_search"

    @property
    def description(self) -> str:
        return "RAG 기반 유사 판례 검색 및 법률 상담"

    async def process(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        session_data: dict[str, Any] | None = None,
        user_location: dict[str, float] | None = None,
    ) -> AgentResponse:
        """기존 RAG 서비스를 활용하여 판례 검색"""
        # 기존 chat_service 호출
        result = generate_chat_response(
            user_message=message,
            chat_history=history,
            n_context_docs=5,
        )

        # 소스 정보 정리
        sources = result.get("sources", [])

        return AgentResponse(
            message=result.get("response", ""),
            sources=sources,
            actions=[],
            session_data={"active_agent": self.name},
        )

    def can_handle(self, message: str) -> bool:
        """판례 관련 키워드 확인"""
        keywords = ["판례", "사례", "판결", "재판", "법원"]
        return any(kw in message for kw in keywords)
