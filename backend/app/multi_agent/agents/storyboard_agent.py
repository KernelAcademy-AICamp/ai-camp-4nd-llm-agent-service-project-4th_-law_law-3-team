"""
스토리보드 에이전트

LLM 기반 사건 타임라인 생성
사용자가 입력한 사건 내용을 시간순으로 정리
"""

import logging
from collections.abc import AsyncGenerator
from typing import Any

from app.multi_agent.agents.base_chat import BaseChatAgent, normalize_chunk_content
from app.multi_agent.schemas.plan import AgentResult

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """당신은 법률 사건 타임라인 전문가입니다.
사용자가 설명하는 사건 내용을 분석하여 시간순으로 정리된 타임라인을 작성합니다.

작성 규칙:
1. 날짜/시점이 명시되지 않은 경우 논리적 순서로 배치하세요
2. 각 이벤트는 "- [시점] 내용" 형식으로 작성하세요
3. 법적으로 중요한 시점(계약일, 이행기, 소멸시효 등)을 강조하세요
4. 핵심 쟁점을 별도로 정리하세요
5. 마지막에 간단한 법적 조언이나 주의사항을 추가하세요

<example>
사용자: 작년 3월에 전세 계약했는데 집주인이 보증금을 안 돌려줘요. 6월에 계약 만료됐고 8월에 내용증명 보냈어요.
응답:
## 사건 타임라인

- [2024년 3월] 전세 계약 체결
- [2025년 6월] 임대차 계약 만료 → **보증금 반환 의무 발생 시점**
- [2025년 8월] 임대인에게 내용증명 발송

## 핵심 쟁점
- 임대차보증금 반환 청구권 (민법 제312조, 주택임대차보호법)
- 임차권등기명령 신청 가능 여부

## 주의사항
- 보증금 반환 청구의 소멸시효는 10년이나, 조속한 법적 조치가 권장됩니다.
- 임차권등기명령을 신청하면 이사 후에도 대항력과 우선변제권을 유지할 수 있습니다.
</example>"""


class StoryboardAgent(BaseChatAgent):
    """스토리보드 에이전트 (사건 타임라인 생성)"""

    @property
    def name(self) -> str:
        return "storyboard"

    @property
    def description(self) -> str:
        return "사건 내용을 시간순 타임라인으로 정리"

    @property
    def supports_streaming(self) -> bool:
        return True

    async def process(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        session_data: dict[str, Any] | None = None,
        user_location: dict[str, float] | None = None,
    ) -> AgentResult:
        """사건 타임라인 생성"""
        from app.tools.llm import get_chat_model

        model = get_chat_model()

        messages: list[tuple[str, str]] = [("system", _SYSTEM_PROMPT)]
        if history:
            for h in history:
                messages.append((h.get("role", "user"), h.get("content", "")))
        messages.append(("user", message))

        response = await model.ainvoke(messages)

        return AgentResult(
            message=str(response.content),
            sources=[],
            actions=[],
            session_data={"active_agent": self.name},
            agent_used=self.name,
        )

    async def process_stream(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        session_data: dict[str, Any] | None = None,
        user_location: dict[str, float] | None = None,
    ) -> AsyncGenerator[tuple[str, Any], None]:
        """스트리밍 타임라인 생성"""
        from app.tools.llm import get_chat_model

        model = get_chat_model()

        messages: list[tuple[str, str]] = [("system", _SYSTEM_PROMPT)]
        if history:
            for h in history:
                messages.append((h.get("role", "user"), h.get("content", "")))
        messages.append(("user", message))

        async for chunk in model.astream(messages):
            if chunk.content:
                text = normalize_chunk_content(chunk.content)
                if text:
                    yield ("token", {"content": text})

        yield ("sources", {"sources": []})
        yield ("metadata", {
            "agent_used": self.name,
            "actions": [],
            "session_data": {"active_agent": self.name},
        })
        yield ("done", {})

    def can_handle(self, message: str) -> bool:
        """타임라인 관련 키워드 확인"""
        keywords = ["타임라인", "스토리보드", "사건 정리", "시간순", "사건 경위"]
        return any(kw in message for kw in keywords)
