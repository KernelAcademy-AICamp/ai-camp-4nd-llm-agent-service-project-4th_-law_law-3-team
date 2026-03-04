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


def normalize_chunk_content(content: Any) -> str:
    """LLM 청크의 content를 문자열로 정규화.

    Google Gemini 등은 content가 list[dict] 형태로 올 수 있으므로
    프로바이더에 관계없이 항상 str을 반환합니다.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


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
이미 충분한 정보가 제공된 경우에는 질문하지 않아도 됩니다."""


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
                text = normalize_chunk_content(chunk.content)
                if text:
                    yield ("token", {"content": text})

        # sources 전송 (토큰 스트리밍 완료 후)
        yield ("sources", {"sources": []})

        # 메타데이터 전송
        yield ("metadata", {
            "agent_used": self.name,
            "actions": [],
            "session_data": {"active_agent": self.name},
        })
        yield ("done", {})
