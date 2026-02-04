"""
에이전트 실행기

에이전트 선택 및 실행 담당
"""

import time
from collections.abc import AsyncGenerator
from typing import Any, Dict, Optional

from app.core.context import ChatContext
from app.multi_agent.agents.base_chat import BaseChatAgent
from app.multi_agent.schemas.plan import AgentPlan, AgentResult


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

    async def execute_stream(
        self,
        plan: AgentPlan,
        context: ChatContext,
    ) -> AsyncGenerator[tuple[str, Any], None]:
        """
        스트리밍 에이전트 실행

        Args:
            plan: 에이전트 실행 계획
            context: 채팅 컨텍스트

        Yields:
            (event_type, data) 튜플
        """
        # 에이전트 선택
        agent = self.get_agent(plan.agent_type)

        if agent is None:
            # 기본 에이전트로 폴백
            default_agent = self._get_default_agent()
            if default_agent is None:
                yield ("error", {"message": "에이전트를 찾을 수 없습니다."})
                return
            agent = default_agent

        # 스트리밍 지원 여부에 따라 분기
        if agent.supports_streaming:
            # 스트리밍 지원: process_stream 사용
            async for event_type, data in agent.process_stream(
                message=context.message,
                history=context.history,
                session_data=context.session_data,
                user_location=context.user_location,
            ):
                yield (event_type, data)
        else:
            # 스트리밍 미지원: 일반 process 후 단일 청크
            result = await agent.process(
                message=context.message,
                history=context.history,
                session_data=context.session_data,
                user_location=context.user_location,
            )
            # 토큰 먼저, sources는 나중에
            yield ("token", {"content": result.message})
            yield ("sources", {"sources": result.sources})
            yield ("metadata", {
                "agent_used": result.agent_used or agent.name,
                "actions": result.actions,
                "session_data": result.session_data,
            })
            yield ("done", {})

    def _get_default_agent(self) -> Optional[BaseChatAgent]:
        """기본 에이전트 반환"""
        # legal_answer를 기본으로 사용
        if "legal_answer" in self._agents:
            return self._agents["legal_answer"]
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
            LawyerFinderAgent,
            LegalAnswerAgent,
            SimpleChatAgent,
            SmallClaimsAgent,
        )

        agents: Dict[str, BaseChatAgent] = {
            # 판례 위주 검색 (기존 case_search 대체)
            "legal_answer": LegalAnswerAgent(focus="precedent"),
            "case_search": LegalAnswerAgent(focus="precedent"),  # 하위 호환
            # 법령 위주 검색
            "law_search": LegalAnswerAgent(focus="law"),
            # 기타 에이전트
            "lawyer_finder": LawyerFinderAgent(),
            "small_claims": SmallClaimsAgent(),
            "general": SimpleChatAgent(),
        }
        _executor = AgentExecutor(agents)
    return _executor
