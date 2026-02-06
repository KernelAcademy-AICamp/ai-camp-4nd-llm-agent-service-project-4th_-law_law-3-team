"""
에이전트 플랜 및 결과 스키마

라우팅 결과와 에이전트 실행 결과를 담는 데이터 클래스
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentPlan:
    """에이전트 실행 계획

    RulesRouter가 생성하여 router_node의 Command.update로 전달
    """

    agent_type: str
    use_rag: bool = True
    confidence: float = 1.0
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "agent_type": self.agent_type,
            "use_rag": self.use_rag,
            "confidence": self.confidence,
            "reason": self.reason,
            "metadata": self.metadata,
        }


@dataclass
class AgentResult:
    """에이전트 실행 결과

    각 에이전트 노드가 반환하여 state_to_response()로 변환
    """

    message: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    session_data: dict[str, Any] = field(default_factory=dict)
    agent_used: str | None = None
    confidence: float = 1.0
    processing_time_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "message": self.message,
            "sources": self.sources,
            "actions": self.actions,
            "session_data": self.session_data,
            "agent_used": self.agent_used,
            "confidence": self.confidence,
            "processing_time_ms": self.processing_time_ms,
        }
