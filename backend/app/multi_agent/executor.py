"""
에이전트 실행기

에이전트 선택 및 실행 담당
"""

import time
from typing import Any, Dict, Optional

from app.multi_agent.agents.base_chat import BaseChatAgent
from app.multi_agent.schemas.plan import AgentPlan, AgentResult
from app.core.context import ChatContext


class AgentExecutor:
    """
    에이전트 실행기

    AgentPlan을 받아 해당 에이전트를 실행하고 결과 반환
    """

    def __init__(self, agents: Dict[str, BaseChatAgent]):
        """
        Args:
            agents: 에이전트 이름 → 에이전트 인스턴스 매핑
        """
        self._agents = agents

    def get_agent(self, agent_type: str) -> Optional[BaseChatAgent]:
        """에이전트 타입으로 에이전트 인스턴스 반환"""
        return self._agents.get(agent_type)

    def list_agents(self) -> list[str]:
        """사용 가능한 에이전트 목록 반환"""
        return list(self._agents.keys())

    async def execute(
        self,
        plan: AgentPlan,
        context: ChatContext,
    ) -> AgentResult:
        """
        에이전트 실행

        Args:
            plan: 에이전트 실행 계획
            context: 채팅 컨텍스트

        Returns:
            AgentResult: 에이전트 실행 결과
        """
        start_time = time.time()

        # 에이전트 선택
        agent = self.get_agent(plan.agent_type)

        if agent is None:
            # 기본 에이전트로 폴백
            default_agent = self._get_default_agent()
            if default_agent is None:
                return AgentResult(
                    message="에이전트를 찾을 수 없습니다.",
                    sources=[],
                    actions=[],
                    session_data={},
                    agent_used="unknown",
                    confidence=0.0,
                )
            agent = default_agent

        # 에이전트 실행
        result = await agent.process(
            message=context.message,
            history=context.history,
            session_data=context.session_data,
            user_location=context.user_location,
        )

        # 처리 시간 기록
        processing_time_ms = (time.time() - start_time) * 1000
        result.processing_time_ms = processing_time_ms
        result.agent_used = agent.name
        result.confidence = plan.confidence

        return result

    def _get_default_agent(self) -> Optional[BaseChatAgent]:
        """기본 에이전트 반환"""
        # case_search를 기본으로 사용
        if "case_search" in self._agents:
            return self._agents["case_search"]
        # 없으면 첫 번째 에이전트
        if self._agents:
            return next(iter(self._agents.values()))
        return None


# 싱글톤 인스턴스
_executor: Optional[AgentExecutor] = None


def get_agent_executor() -> AgentExecutor:
    """AgentExecutor 싱글톤 인스턴스 반환"""
    global _executor
    if _executor is None:
        from app.multi_agent.agents import (
            CasePrecedentAgent,
            LawyerFinderAgent,
            SmallClaimsAgent,
            SimpleChatAgent,
        )

        agents: Dict[str, BaseChatAgent] = {
            "case_search": CasePrecedentAgent(),
            "lawyer_finder": LawyerFinderAgent(),
            "small_claims": SmallClaimsAgent(),
            "general": SimpleChatAgent(),
        }
        _executor = AgentExecutor(agents)
    return _executor
