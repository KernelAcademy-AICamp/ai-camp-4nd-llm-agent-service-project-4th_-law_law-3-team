"""콘텐츠 마케팅 에이전트 (채팅 위젯 연동)

트렌드 분석 + 대본 생성을 하나의 에이전트로 통합.
메시지 내용에 따라 트렌드 분석 또는 대본 생성 모드를 자동 판별합니다.

v2.0: Legal Gate 스코어 표시, ScriptGenerator 직접 호출, PersonaTone 지원
"""

import logging
from typing import Any

from app.modules.content_marketing.schema import (
    PersonaType,
    ScriptDuration,
    ScriptRequest,
    TrendRequest,
)
from app.multi_agent.agents.base_chat import BaseChatAgent
from app.multi_agent.schemas.plan import AgentResult
from app.services.service_function.content_marketing_service import collect_trends

logger = logging.getLogger(__name__)

_SCRIPT_KEYWORDS = frozenset({"대본", "스크립트", "영상", "유튜브", "콘텐츠"})


class ContentMarketingAgent(BaseChatAgent):
    """채팅 위젯용 콘텐츠 마케팅 에이전트

    v2.0: Legal Gate 스코어 표시, ScriptGenerator 직접 호출
    """

    @property
    def name(self) -> str:
        return "content_marketing"

    @property
    def description(self) -> str:
        return "실시간 법률 트렌드 분석 및 유튜브 대본 생성을 수행합니다."

    async def process(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        session_data: dict[str, Any] | None = None,
        user_location: dict[str, float] | None = None,
    ) -> AgentResult:
        """메시지 분석 후 트렌드 조회 또는 대본 생성 실행"""
        message_lower = message.lower().replace(" ", "")

        if any(kw in message_lower for kw in _SCRIPT_KEYWORDS):
            return await self._handle_script(message)
        return await self._handle_trend()

    async def _handle_trend(self) -> AgentResult:
        """트렌드 조회 모드 (v2.0: Legal Gate + 5차원 스코어)"""
        request = TrendRequest(limit=5)
        response = await collect_trends(request)

        parts: list[str] = ["## 최근 법률 트렌드 이슈 TOP 5\n"]
        for i, issue in enumerate(response.trends, 1):
            # 기본 정보
            entry = (
                f"### {i}위. {issue.title} (점수: {issue.score:.1f}/100)\n"
                f"{issue.summary}\n\n"
            )

            # v2.0 Legal Gate 정보
            if issue.score_detail is not None:
                detail = issue.score_detail
                gate_status = "통과" if detail.legal_gate_passed else "미달"
                entry += (
                    f"**법적 분석 지표:** "
                    f"법적쟁점화 {detail.legal_score:.2f} | "
                    f"논란도 {detail.controversy_score:.2f} | "
                    f"확산도 {detail.spread_score:.2f} | "
                    f"Legal Gate {gate_status}\n\n"
                )

            # 적합도 라벨
            if issue.fitness_label:
                entry += f"**채널 적합도:** {issue.fitness_label}\n\n"

            # 핵심 쟁점
            entry += "**핵심 쟁점:**\n"
            entry += "\n".join(f"- {p}" for p in issue.key_points)
            entry += "\n"

            parts.append(entry)

        return AgentResult(
            message="\n".join(parts),
            agent_used="content_marketing",
            session_data={"active_agent": "content_marketing"},
        )

    async def _handle_script(self, message: str) -> AgentResult:
        """대본 생성 모드 (채팅에서는 전체 응답 한번에, ScriptGenerator 직접 호출)"""
        from app.tools.script.generator import ScriptGenerator

        request = ScriptRequest(
            topic=message,
            persona=PersonaType.PROFESSIONAL,
            duration=ScriptDuration.MEDIUM,
        )

        generator = ScriptGenerator()
        sections: list[str] = []

        async for event in generator.generate_stream(request):
            if event.event == "content":
                sections.append(event.content)

        return AgentResult(
            message="".join(sections) if sections else "대본을 생성할 수 없습니다.",
            agent_used="content_marketing",
            session_data={"active_agent": "content_marketing"},
        )
