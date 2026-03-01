"""
워크스페이스 에이전트

사건 조회, 타임라인 재생성 등 워크스페이스 관련 대화 처리
"""

import uuid
from typing import Any

from app.multi_agent.agents.base_chat import BaseChatAgent
from app.multi_agent.schemas.plan import AgentResult


class WorkspaceAgent(BaseChatAgent):
    """워크스페이스 에이전트 — 사건 관리 대화 처리"""

    @property
    def name(self) -> str:
        return "workspace"

    @property
    def description(self) -> str:
        return "사건 워크스페이스 관리 (조회, 타임라인)"

    async def process(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        session_data: dict[str, Any] | None = None,
        user_location: dict[str, float] | None = None,
    ) -> AgentResult:
        session_data = session_data or {}
        session_token: str | None = session_data.get("session_token")

        if not session_token:
            return AgentResult(
                message="세션 정보가 없어 사건을 조회할 수 없습니다. 페이지를 새로고침해주세요.",
                sources=[],
                actions=[],
                session_data={"active_agent": self.name},
                agent_used=self.name,
            )

        message_lower = message.replace(" ", "").lower()

        # 타임라인 재생성 의도
        if "타임라인" in message_lower and ("재생성" in message_lower or "다시" in message_lower):
            return await self._handle_timeline_rebuild(session_data, session_token)

        # 사건 목록 조회 의도
        if any(kw in message_lower for kw in ("사건목록", "내사건", "워크스페이스", "사건조회")):
            return await self._handle_list_cases(session_token)

        # 특정 사건 상세 조회 (case_id가 세션에 있을 때)
        case_id = session_data.get("active_case_id")
        if case_id and any(kw in message_lower for kw in ("진행상황", "사건정리", "상세")):
            return await self._handle_case_detail(case_id, session_token)

        # 기본: 사건 목록 표시
        return await self._handle_list_cases(session_token)

    async def _handle_list_cases(self, session_token: str) -> AgentResult:
        """사건 목록 조회"""
        from app.core.database import async_session_factory
        from app.services.workspace.workspace_case_service import WorkspaceCaseService

        async with async_session_factory() as db:
            items, total = await WorkspaceCaseService.list_cases(
                db, session_token, status="active", page=1, page_size=10
            )

        if not items:
            return AgentResult(
                message="등록된 사건이 없습니다. 워크스페이스 페이지에서 새 사건을 생성해보세요.",
                sources=[],
                actions=[],
                session_data={"active_agent": self.name},
                agent_used=self.name,
            )

        lines = [f"**내 사건 목록** (총 {total}건)\n"]
        for item in items:
            case_type_str = f" [{item['case_type']}]" if item.get("case_type") else ""
            conv_count = item.get("conversation_count", 0)
            lines.append(
                f"- **{item['case_name']}**{case_type_str} "
                f"(대화 {conv_count}개, 태그 {item.get('tag_count', 0)}개)"
            )

        return AgentResult(
            message="\n".join(lines),
            sources=[],
            actions=[],
            session_data={"active_agent": self.name},
            agent_used=self.name,
        )

    async def _handle_case_detail(
        self, case_id: str, session_token: str
    ) -> AgentResult:
        """사건 상세 조회"""
        from app.core.database import async_session_factory
        from app.services.workspace.workspace_case_service import WorkspaceCaseService

        async with async_session_factory() as db:
            detail = await WorkspaceCaseService.get_case_detail(
                db, uuid.UUID(case_id), session_token
            )

        if not detail:
            return AgentResult(
                message="해당 사건을 찾을 수 없습니다.",
                sources=[],
                actions=[],
                session_data={"active_agent": self.name},
                agent_used=self.name,
            )

        lines = [f"**{detail['case_name']}**\n"]
        if detail.get("case_type"):
            lines.append(f"- 유형: {detail['case_type']}")
        lines.append(f"- 상태: {detail['status']}")

        timeline = detail.get("timeline", [])
        if timeline:
            lines.append(f"\n**타임라인** ({len(timeline)}건)")
            for t in timeline[:5]:
                date_str = t.get("date_text", "")
                lines.append(f"  - [{date_str}] {t['title']}")
            if len(timeline) > 5:
                lines.append(f"  ... 외 {len(timeline) - 5}건")

        tags = detail.get("tagged_items", [])
        if tags:
            lines.append(f"\n수집된 태그: {len(tags)}개")

        return AgentResult(
            message="\n".join(lines),
            sources=[],
            actions=[],
            session_data={
                "active_agent": self.name,
                "active_case_id": case_id,
            },
            agent_used=self.name,
        )

    async def _handle_timeline_rebuild(
        self, session_data: dict[str, Any], session_token: str
    ) -> AgentResult:
        """타임라인 재생성"""
        case_id = session_data.get("active_case_id")
        if not case_id:
            return AgentResult(
                message="타임라인을 재생성할 사건이 지정되지 않았습니다. "
                "먼저 사건 목록에서 사건을 선택해주세요.",
                sources=[],
                actions=[],
                session_data={"active_agent": self.name},
                agent_used=self.name,
            )

        from app.core.database import async_session_factory
        from app.services.workspace.timeline_engine import TimelineEngine
        from app.services.workspace.workspace_case_service import WorkspaceCaseService

        async with async_session_factory() as db:
            detail = await WorkspaceCaseService.get_case_detail(
                db, uuid.UUID(case_id), session_token
            )
            if not detail:
                return AgentResult(
                    message="해당 사건을 찾을 수 없습니다.",
                    sources=[],
                    actions=[],
                    session_data={"active_agent": self.name},
                    agent_used=self.name,
                )

            all_tags: list[dict[str, Any]] = detail.get("tagged_items", []) or []
            items = await TimelineEngine.rebuild(
                db, uuid.UUID(case_id), all_tags, preserve_manual=True
            )
            await db.commit()

        lines = [f"**{detail['case_name']}** 타임라인을 재생성했습니다.\n"]
        lines.append(f"총 {len(items)}개 항목이 생성되었습니다.\n")
        for t in items[:5]:
            date_str = t.get("date_text", "")
            lines.append(f"- [{date_str}] {t['title']}")
        if len(items) > 5:
            lines.append(f"... 외 {len(items) - 5}건")

        return AgentResult(
            message="\n".join(lines),
            sources=[],
            actions=[],
            session_data={
                "active_agent": self.name,
                "active_case_id": case_id,
            },
            agent_used=self.name,
        )
