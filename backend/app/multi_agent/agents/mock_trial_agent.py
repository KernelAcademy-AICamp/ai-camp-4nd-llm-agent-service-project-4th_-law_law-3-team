"""
모의 법정 에이전트

BaseChatAgent를 상속하여 모의재판 기능을 제공합니다.
실제 재판 진행은 mock_trial 서브그래프에서 처리하며,
이 에이전트는 라우팅 및 초기 안내를 담당합니다.
"""

from typing import Any

from app.multi_agent.agents.base_chat import BaseChatAgent
from app.multi_agent.schemas.plan import AgentResult


class MockTrialAgent(BaseChatAgent):
    """모의 법정 에이전트"""

    @property
    def name(self) -> str:
        return "mock_trial"

    @property
    def description(self) -> str:
        return "모의재판 시뮬레이션 (형사/민사 재판 절차 체험)"

    async def process(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        session_data: dict[str, Any] | None = None,
        user_location: dict[str, float] | None = None,
    ) -> AgentResult:
        """모의재판 안내 메시지 반환

        서브그래프로 라우팅되므로 이 메서드는 폴백용입니다.
        """
        return AgentResult(
            message=(
                "모의 법정에 오신 것을 환영합니다.\n\n"
                "형사 재판 또는 민사 재판을 선택하고, "
                "역할(검사/변호사)을 정한 후 사건 개요를 입력하면 "
                "AI 에이전트들과 함께 재판을 진행할 수 있습니다.\n\n"
                "**주의: 이 모의재판은 교육 목적이며 실제 법률 자문이 아닙니다.**"
            ),
            sources=[],
            actions=[],
            session_data={"active_agent": self.name},
            agent_used=self.name,
        )

    def can_handle(self, message: str) -> bool:
        """모의재판 관련 키워드 확인"""
        keywords = [
            "모의재판", "모의 재판", "모의법정", "모의 법정",
            "재판 시뮬", "법정 체험", "법정 시뮬",
            "재판 연습", "재판 게임", "법정 게임",
        ]
        return any(kw in message for kw in keywords)
