"""
라우터 에이전트

규칙 기반 라우터로 최적의 에이전트 선택
"""

from typing import Any, Dict, Optional

from app.multi_agent.routing.rules_router import RulesRouter
from app.multi_agent.schemas.plan import AgentPlan
from app.core.context import ChatContext


class RouterAgent:
    """
    라우터 에이전트

    규칙 기반 라우팅을 우선 사용하고, 신뢰도가 낮을 경우
    LLM 기반 라우팅으로 보완
    """

    # LLM 라우팅 활성화 임계값
    LLM_ROUTING_THRESHOLD = 0.6

    def __init__(self) -> None:
        self._rules_router = RulesRouter()
        self._llm_router: Optional[Any] = None  # LLMRouter (lazy init)

    def route(
        self,
        context: ChatContext,
    ) -> AgentPlan:
        """
        컨텍스트 기반 에이전트 라우팅

        Args:
            context: 채팅 컨텍스트

        Returns:
            AgentPlan
        """
        # 1. 규칙 기반 라우팅 시도
        plan = self._rules_router.route(
            message=context.message,
            user_role=context.user_role,
            session_data=context.session_data,
        )

        # 2. 신뢰도가 충분하면 반환
        if plan.confidence >= self.LLM_ROUTING_THRESHOLD:
            return plan

        # 3. LLM 라우팅으로 보완 (향후 구현)
        # llm_plan = self._get_llm_router().route(context.message, ...)
        # return llm_plan if llm_plan.confidence > plan.confidence else plan

        return plan

    def route_simple(
        self,
        message: str,
        user_role: str = "user",
        session_data: Optional[Dict[str, Any]] = None,
    ) -> AgentPlan:
        """
        간단한 라우팅 (ChatContext 없이)

        Args:
            message: 사용자 메시지
            user_role: 사용자 역할
            session_data: 세션 데이터

        Returns:
            AgentPlan
        """
        return self._rules_router.route(
            message=message,
            user_role=user_role,
            session_data=session_data,
        )


_router_agent: Optional[RouterAgent] = None


def get_router_agent() -> RouterAgent:
    """RouterAgent 싱글톤 인스턴스 반환"""
    global _router_agent
    if _router_agent is None:
        _router_agent = RouterAgent()
    return _router_agent
