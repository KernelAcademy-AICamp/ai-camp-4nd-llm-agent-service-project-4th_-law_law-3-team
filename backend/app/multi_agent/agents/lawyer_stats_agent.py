"""
변호사 통계 에이전트

변호사 통계 대시보드 페이지로 안내
RAG/LLM 미사용, 네비게이션 에이전트
"""

from typing import Any

from app.multi_agent.agents.base_chat import ActionType, BaseChatAgent, ChatAction
from app.multi_agent.schemas.plan import AgentResult


class LawyerStatsAgent(BaseChatAgent):
    """변호사 통계 안내 에이전트"""

    @property
    def name(self) -> str:
        return "lawyer_stats"

    @property
    def description(self) -> str:
        return "변호사 통계 대시보드 안내"

    async def process(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        session_data: dict[str, Any] | None = None,
        user_location: dict[str, float] | None = None,
    ) -> AgentResult:
        """변호사 통계 페이지로 안내"""
        msg = (
            "변호사 통계 대시보드에서 지역별/전문분야별 변호사 현황을 확인하실 수 있습니다.\n\n"
            "제공 정보:\n"
            "- 지역별 변호사 수 및 인구 대비 밀도\n"
            "- 전문분야별 변호사 분포\n"
            "- 지역 x 전문분야 교차 분석\n"
            "- 향후 예측 (2030/2035/2040년)\n\n"
            "통계 대시보드로 이동합니다..."
        )

        return AgentResult(
            message=msg,
            sources=[],
            actions=[
                ChatAction(
                    type=ActionType.NAVIGATE,
                    label="변호사 통계 보기",
                    url="/lawyer-stats",
                ).model_dump(),
            ],
            session_data={"active_agent": self.name},
            agent_used=self.name,
        )

    def can_handle(self, message: str) -> bool:
        """변호사 통계 관련 키워드 확인"""
        keywords = ["변호사 통계", "변호사 현황", "변호사 분포", "지역별 변호사"]
        return any(kw in message for kw in keywords)
