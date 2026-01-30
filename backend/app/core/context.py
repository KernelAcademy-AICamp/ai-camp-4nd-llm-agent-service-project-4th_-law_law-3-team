"""
요청/세션 컨텍스트 정의

FastAPI 요청 처리 시 사용되는 컨텍스트 객체
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid


@dataclass
class RequestContext:
    """
    HTTP 요청 컨텍스트

    각 요청마다 생성되어 요청 추적 및 로깅에 사용
    """
    user_id: Optional[str] = None
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_role: str = "user"
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환 (로깅용)"""
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "user_role": self.user_role,
            "trace_id": self.trace_id,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ChatContext:
    """
    채팅 요청 컨텍스트

    에이전트 처리에 필요한 모든 정보를 담는 컨테이너
    """
    message: str
    history: List[Dict[str, str]] = field(default_factory=list)
    session_data: Dict[str, Any] = field(default_factory=dict)
    user_location: Optional[Dict[str, float]] = None
    user_role: str = "user"
    request_context: Optional[RequestContext] = None

    @property
    def active_agent(self) -> Optional[str]:
        """현재 활성 에이전트"""
        return self.session_data.get("active_agent")
