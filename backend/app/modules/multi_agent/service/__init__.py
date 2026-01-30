"""
멀티 에이전트 오케스트레이터

에이전트 선택 및 실행 관리
"""

from typing import Any, Optional

from app.common.agent_base import AgentResponse, BaseAgent
from app.common.agent_router import AgentType, UserRole, detect_intent
from app.modules.multi_agent.agents import (
    CaseSearchAgent,
    LawyerFinderAgent,
    SmallClaimsAgent,
)


class AgentOrchestrator:
    """
    에이전트 오케스트레이터

    Intent 감지 → 에이전트 선택 → 실행 → 응답 반환
    """

    def __init__(self) -> None:
        # 에이전트 인스턴스 생성
        self._agents: dict[AgentType, BaseAgent] = {
            AgentType.CASE_SEARCH: CaseSearchAgent(),
            AgentType.LAWYER_FINDER: LawyerFinderAgent(),
            AgentType.SMALL_CLAIMS: SmallClaimsAgent(),
        }

    def get_agent(self, agent_type: AgentType) -> Optional[BaseAgent]:
        """에이전트 타입으로 에이전트 인스턴스 반환"""
        return self._agents.get(agent_type)

    async def process(
        self,
        message: str,
        user_role: str = "user",
        history: Optional[list[dict[str, str]]] = None,
        session_data: Optional[dict[str, Any]] = None,
        user_location: Optional[dict[str, float]] = None,
    ) -> tuple[AgentResponse, str]:
        """
        메시지 처리

        Args:
            message: 사용자 메시지
            user_role: 사용자 역할 ("user" or "lawyer")
            history: 대화 기록
            session_data: 세션 데이터
            user_location: 사용자 위치

        Returns:
            (AgentResponse, agent_name): 에이전트 응답과 사용된 에이전트 이름
        """
        session_data = session_data or {}

        # 1. UserRole 변환
        try:
            role = UserRole(user_role)
        except ValueError:
            role = UserRole.USER

        # 2. Intent 감지
        agent_type = detect_intent(message, role, session_data)

        # 3. 에이전트 선택
        agent = self.get_agent(agent_type)

        # 에이전트가 없으면 기본 판례 검색
        if agent is None:
            agent = self._agents[AgentType.CASE_SEARCH]
            agent_type = AgentType.CASE_SEARCH

        # 4. 에이전트 실행
        response = await agent.process(
            message=message,
            history=history,
            session_data=session_data,
            user_location=user_location,
        )

        return response, agent_type.value


# 싱글톤 인스턴스
_orchestrator: Optional[AgentOrchestrator] = None


def get_orchestrator() -> AgentOrchestrator:
    """오케스트레이터 싱글톤 인스턴스 반환"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
    return _orchestrator
