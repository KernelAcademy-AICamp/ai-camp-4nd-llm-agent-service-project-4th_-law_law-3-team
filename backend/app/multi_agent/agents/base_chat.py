"""
채팅 에이전트 베이스 클래스

모든 에이전트가 상속받는 추상 클래스와 기본 구현
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel

from app.multi_agent.schemas.plan import AgentResult


class ActionType(str, Enum):
    """액션 버튼 타입"""

    BUTTON = "button"
    LINK = "link"
    REQUEST_LOCATION = "request_location"
    NAVIGATE = "navigate"


class ChatAction(BaseModel):
    """챗봇 응답에 포함되는 액션 버튼"""

    type: ActionType
    label: str
    action: Optional[str] = None
    url: Optional[str] = None
    params: Optional[dict[str, Any]] = None


class BaseChatAgent(ABC):
    """
    채팅 에이전트 베이스 클래스

    모든 에이전트는 이 클래스를 상속받아 구현합니다.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """에이전트 이름"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """에이전트 설명"""
        pass

    @abstractmethod
    async def process(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        session_data: dict[str, Any] | None = None,
        user_location: dict[str, float] | None = None,
    ) -> AgentResult:
        """
        메시지 처리

        Args:
            message: 사용자 메시지
            history: 대화 기록
            session_data: 세션 데이터 (에이전트별 상태 저장)
            user_location: 사용자 위치 {latitude, longitude}

        Returns:
            AgentResult: 에이전트 응답
        """
        pass

    def can_handle(self, message: str) -> bool:
        """
        이 에이전트가 메시지를 처리할 수 있는지 확인

        서브클래스에서 오버라이드하여 키워드 매칭 등 구현 가능
        """
        return False

    @property
    def supports_streaming(self) -> bool:
        """에이전트가 스트리밍을 지원하는지 여부"""
        return False

    async def process_stream(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        session_data: dict[str, Any] | None = None,
        user_location: dict[str, float] | None = None,
    ) -> AsyncGenerator[tuple[str, Any], None]:
        """
        스트리밍 메시지 처리

        기본 구현: 스트리밍 미지원 시 단일 청크 반환

        Args:
            message: 사용자 메시지
            history: 대화 기록
            session_data: 세션 데이터
            user_location: 사용자 위치

        Yields:
            (event_type, data) 튜플
            - token: 응답 토큰
            - sources: 참조 자료
            - metadata: 에이전트 정보
            - done: 완료 신호
        """
        result = await self.process(message, history, session_data, user_location)
        # 토큰 먼저, sources는 나중에 (큰 데이터 청킹 문제 방지)
        yield ("token", {"content": result.message})
        yield ("sources", {"sources": result.sources})
        yield ("metadata", {
            "agent_used": result.agent_used,
            "actions": result.actions,
            "session_data": result.session_data,
        })
        yield ("done", {})


_SIMPLE_CHAT_SYSTEM_PROMPT = """당신은 법률 서비스 플랫폼의 AI 어시스턴트입니다.
사용자의 질문에 친절하고 정확하게 답변하세요.

사용자가 법률 분쟁에 대해 상담 중이면, 답변 마지막에 자연스럽게 아래 중 1가지를 질문하여 사건 정보를 수집하세요:
- 사건 발생 시점이나 경위
- 관련 당사자 (상대방 관계)
- 증거 자료 보유 여부
- 피해 금액이나 청구 규모
이미 충분한 정보가 제공된 경우에는 질문하지 않아도 됩니다.

<example>
사용자: 계약서 없이 돈을 빌려줬는데 안 갚아요
응답: 계약서가 없더라도 금전 대여 사실을 입증할 수 있다면 법적 청구가 가능합니다. 카카오톡 대화, 계좌 이체 내역, 증인 등이 증거가 될 수 있습니다. 금액에 따라 소액소송(3,000만 원 이하)이나 지급명령 신청도 고려해 볼 수 있습니다. 혹시 돈을 빌려준 시점이 언제쯤인지 알 수 있을까요?
</example>

<example>
사용자: 민법이 뭐예요?
응답: 민법은 개인 간의 법률관계를 규율하는 기본 법률입니다. 크게 총칙, 물권, 채권, 친족, 상속의 5편으로 구성되어 있으며, 계약·손해배상·소유권·상속 등 일상생활에서 발생하는 대부분의 법률 문제를 다룹니다.
</example>"""


class SimpleChatAgent(BaseChatAgent):
    """
    단순 LLM 채팅 에이전트

    RAG 없이 LLM만으로 응답 생성
    """

    def __init__(self, agent_name: str = "general", agent_description: str = "일반 채팅"):
        self._name = agent_name
        self._description = agent_description

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def supports_streaming(self) -> bool:
        """SimpleChatAgent는 스트리밍 지원"""
        return True

    async def process(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        session_data: dict[str, Any] | None = None,
        user_location: dict[str, float] | None = None,
    ) -> AgentResult:
        """단순 LLM 응답 생성"""
        from app.tools.llm import get_chat_model

        model = get_chat_model()

        # 대화 기록 구성
        messages: list[tuple[str, str]] = [("system", _SIMPLE_CHAT_SYSTEM_PROMPT)]
        if history:
            for h in history:
                messages.append((h.get("role", "user"), h.get("content", "")))

        messages.append(("user", message))

        # LLM 호출 (비동기)
        response = await model.ainvoke(messages)
        content = response.content
        text = content if isinstance(content, str) else str(content)

        return AgentResult(
            message=text,
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
        """스트리밍 LLM 응답 생성"""
        from app.tools.llm import get_chat_model

        model = get_chat_model()

        # 대화 기록 구성
        messages: list[tuple[str, str]] = [("system", _SIMPLE_CHAT_SYSTEM_PROMPT)]
        if history:
            for h in history:
                messages.append((h.get("role", "user"), h.get("content", "")))

        messages.append(("user", message))

        # LLM 스트리밍 호출
        async for chunk in model.astream(messages):
            if chunk.content:
                yield ("token", {"content": chunk.content})

        # sources 전송 (토큰 스트리밍 완료 후)
        yield ("sources", {"sources": []})

        # 메타데이터 전송
        yield ("metadata", {
            "agent_used": self.name,
            "actions": [],
            "session_data": {"active_agent": self.name},
        })
        yield ("done", {})
