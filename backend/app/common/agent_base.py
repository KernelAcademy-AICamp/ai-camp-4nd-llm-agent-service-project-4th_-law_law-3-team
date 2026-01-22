"""
에이전트 베이스 클래스 및 응답 모델

모든 에이전트가 상속받는 추상 클래스와 공통 응답 형식 정의
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel


class ActionType(str, Enum):
    """액션 버튼 타입"""
    BUTTON = "button"
    LINK = "link"
    REQUEST_LOCATION = "request_location"
    NAVIGATE = "navigate"  # 페이지 이동 (쿼리 파라미터 포함)


class ChatAction(BaseModel):
    """챗봇 응답에 포함되는 액션 버튼"""
    type: ActionType
    label: str
    action: Optional[str] = None  # button action identifier
    url: Optional[str] = None  # link URL
    params: Optional[dict[str, Any]] = None  # navigate params (for NAVIGATE type)


class AgentResponse(BaseModel):
    """에이전트 응답 모델"""
    message: str
    sources: list[dict[str, Any]] = []
    actions: list[ChatAction] = []
    session_data: dict[str, Any] = {}


class BaseAgent(ABC):
    """
    에이전트 베이스 클래스

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
    ) -> AgentResponse:
        """
        메시지 처리

        Args:
            message: 사용자 메시지
            history: 대화 기록
            session_data: 세션 데이터 (에이전트별 상태 저장)
            user_location: 사용자 위치 {latitude, longitude}

        Returns:
            AgentResponse: 에이전트 응답
        """
        pass

    def can_handle(self, message: str) -> bool:
        """
        이 에이전트가 메시지를 처리할 수 있는지 확인

        서브클래스에서 오버라이드하여 키워드 매칭 등 구현 가능
        """
        return False
