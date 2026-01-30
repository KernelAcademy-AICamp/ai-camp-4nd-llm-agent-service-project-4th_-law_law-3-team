"""
채팅 에이전트 베이스 클래스

모든 에이전트가 상속받는 추상 클래스와 기본 구현
"""

from abc import ABC, abstractmethod
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
        messages = []
        if history:
            for h in history:
                messages.append((h.get("role", "user"), h.get("content", "")))

        messages.append(("user", message))

        # LLM 호출
        response = model.invoke(messages)

        return AgentResult(
            message=response.content,
            sources=[],
            actions=[],
            session_data={"active_agent": self.name},
            agent_used=self.name,
        )
