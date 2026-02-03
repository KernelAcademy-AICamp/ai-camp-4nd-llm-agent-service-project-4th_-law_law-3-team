"""
멀티 에이전트 오케스트레이터

라우팅 → 실행 → 응답 변환 관리
"""

import logging
from typing import Any, Optional

from app.core.context import ChatContext, RequestContext
from app.core.state.session_store import SessionStore, get_session_store
from app.multi_agent.router import RouterAgent, get_router_agent
from app.multi_agent.executor import AgentExecutor, get_agent_executor
from app.multi_agent.schemas.messages import ChatRequest, ChatResponse
from app.multi_agent.schemas.plan import AgentResult

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    멀티 에이전트 오케스트레이터

    채팅 요청을 받아 라우팅 → 실행 → 응답 변환
    """

    def __init__(
        self,
        router: Optional[RouterAgent] = None,
        executor: Optional[AgentExecutor] = None,
        session_store: Optional[SessionStore] = None,
    ):
        self._router = router
        self._executor = executor
        self._session_store = session_store

    @property
    def router(self) -> RouterAgent:
        """RouterAgent lazy initialization"""
        if self._router is None:
            self._router = get_router_agent()
        return self._router

    @property
    def executor(self) -> AgentExecutor:
        """AgentExecutor lazy initialization"""
        if self._executor is None:
            self._executor = get_agent_executor()
        return self._executor

    @property
    def session_store(self) -> SessionStore:
        """SessionStore lazy initialization"""
        if self._session_store is None:
            self._session_store = get_session_store()
        return self._session_store

    async def process(self, request: ChatRequest) -> ChatResponse:
        """
        채팅 요청 처리

        Args:
            request: 채팅 요청

        Returns:
            ChatResponse: 채팅 응답
        """
        from app.multi_agent.schemas.plan import AgentPlan

        # 1. 컨텍스트 구성
        context = self._build_context(request)

        # 2. 라우팅 (agent 직접 지정 시 건너뜀)
        if request.agent:
            # 에이전트 직접 지정
            plan = AgentPlan(
                agent_type=request.agent,
                confidence=1.0,
                reason="직접 지정",
            )
            logger.info(f"Agent directly specified: {request.agent}")
        else:
            # 라우터로 에이전트 선택
            plan = self.router.route(context)
            logger.info(
                f"Routed to {plan.agent_type} "
                f"(confidence: {plan.confidence}, reason: {plan.reason})"
            )

        # 3. 에이전트 실행
        result = await self.executor.execute(plan, context)

        # 4. 세션 데이터 저장
        if result.session_data:
            session_id = request.session_data.get("session_id", "default")
            for key, value in result.session_data.items():
                self.session_store.update(session_id, key, value)

        # 5. 응답 변환
        return self._build_response(result)

    def _build_context(self, request: ChatRequest) -> ChatContext:
        """요청에서 ChatContext 구성"""
        # 히스토리 변환
        history = [
            {"role": msg.role, "content": msg.content}
            for msg in request.history
        ]

        return ChatContext(
            message=request.message,
            history=history,
            session_data=request.session_data,
            user_location=request.user_location,
            user_role=request.user_role,
        )

    def _build_response(self, result: AgentResult) -> ChatResponse:
        """AgentResult를 ChatResponse로 변환"""
        return ChatResponse(
            response=result.message,
            agent_used=result.agent_used or "unknown",
            sources=result.sources,
            actions=result.actions,
            session_data=result.session_data,
            confidence=result.confidence,
        )


# 싱글톤 인스턴스
_orchestrator: Optional[Orchestrator] = None


def get_orchestrator() -> Orchestrator:
    """Orchestrator 싱글톤 인스턴스 반환"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator


# 하위 호환성을 위한 별칭
AgentOrchestrator = Orchestrator
