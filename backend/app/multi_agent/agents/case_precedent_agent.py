"""
판례 검색 에이전트

RAG 기반 판례 검색 및 법률 상담 제공
"""

from typing import Any, Optional

from app.multi_agent.agents.base_chat import BaseChatAgent
from app.multi_agent.schemas.plan import AgentResult
from app.services.rag.retrieval import RetrievalService, get_retrieval_service
from app.services.cases.precedent_service import PrecedentService, get_precedent_service
from app.tools.llm import get_chat_model


class CasePrecedentAgent(BaseChatAgent):
    """판례 검색 에이전트"""

    def __init__(
        self,
        retrieval_service: Optional[RetrievalService] = None,
        precedent_service: Optional[PrecedentService] = None,
    ):
        self._retrieval_service = retrieval_service
        self._precedent_service = precedent_service

    @property
    def retrieval_service(self) -> RetrievalService:
        """RetrievalService lazy initialization"""
        if self._retrieval_service is None:
            self._retrieval_service = get_retrieval_service()
        return self._retrieval_service

    @property
    def precedent_service(self) -> PrecedentService:
        """PrecedentService lazy initialization"""
        if self._precedent_service is None:
            self._precedent_service = get_precedent_service()
        return self._precedent_service

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
    ) -> AgentResult:
        """판례 검색 및 응답 생성"""
        # 1. 관련 문서 검색
        search_results = self.retrieval_service.search(
            query=message,
            n_results=5,
            doc_type="precedent",
        )

        # 2. 판례 상세 정보 조회
        source_ids = [
            doc.get("metadata", {}).get("doc_id")
            for doc in search_results
            if doc.get("metadata", {}).get("doc_id")
        ]

        precedent_details = {}
        if source_ids:
            precedent_details = self.precedent_service.get_details(source_ids)

        # 3. 컨텍스트 구성
        context = self._build_precedent_context(search_results, precedent_details)

        # 4. LLM 응답 생성
        response = await self._generate_response(
            message=message,
            context=context,
            history=history,
        )

        # 5. 소스 정보 정리
        sources = self._format_precedent_sources(search_results, precedent_details)

        return AgentResult(
            message=response,
            sources=sources,
            actions=[],
            session_data={"active_agent": self.name},
            agent_used=self.name,
        )

    def _build_precedent_context(
        self,
        documents: list[dict[str, Any]],
        details: dict[str, dict[str, Any]],
    ) -> str:
        """판례 전용 컨텍스트 구성"""
        if not documents:
            return ""

        context_parts = []
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

            context_parts.append(part)

        return "\n\n".join(context_parts)

    def _format_precedent_sources(
        self,
        documents: list[dict[str, Any]],
        details: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """판례 소스 정보 포맷팅"""
        sources = []
        for doc in documents:
            metadata = doc.get("metadata", {})
            doc_id = metadata.get("doc_id", "")

            source_item = {
                "doc_type": "precedent",
                "case_name": metadata.get("case_name", ""),
                "case_number": metadata.get("case_number", ""),
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

            sources.append(source_item)

        return sources

    async def _generate_response(
        self,
        message: str,
        context: str,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """LLM 응답 생성"""
        model = get_chat_model()

        system_prompt = """당신은 법률 전문 AI 어시스턴트입니다.
사용자의 질문에 대해 제공된 판례를 참고하여 정확하고 이해하기 쉽게 답변해주세요.

답변 시 유의사항:
1. 제공된 판례를 근거로 답변하세요
2. 법률 용어는 쉽게 풀어서 설명하세요
3. 일반적인 정보 제공이며, 구체적인 법률 상담은 변호사에게 의뢰하도록 안내하세요
4. 판례 번호를 언급할 때는 정확하게 표기하세요"""

        # 대화 기록 구성
        messages = [("system", system_prompt)]

        if history:
            for h in history:
                messages.append((h.get("role", "user"), h.get("content", "")))

        # 컨텍스트와 함께 질문
        user_message = f"""관련 판례:
{context}

사용자 질문: {message}"""

        messages.append(("user", user_message))

        response = model.invoke(messages)
        return response.content

    def can_handle(self, message: str) -> bool:
        """판례 관련 키워드 확인"""
        keywords = ["판례", "사례", "판결", "재판", "법원", "선례"]
        return any(kw in message for kw in keywords)
