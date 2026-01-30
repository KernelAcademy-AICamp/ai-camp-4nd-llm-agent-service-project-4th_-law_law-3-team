"""
에이전트 플랜 및 결과 스키마

라우팅 결과와 에이전트 실행 결과를 담는 데이터 클래스
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentPlan:
    """
    에이전트 실행 계획

    라우터가 생성하여 Executor에 전달
    """
    agent_type: str
    use_rag: bool = True
    confidence: float = 1.0
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
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
    """
    에이전트 실행 결과

    에이전트가 반환하여 Orchestrator가 응답으로 변환
    """
    message: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    session_data: Dict[str, Any] = field(default_factory=dict)
    agent_used: Optional[str] = None
    confidence: float = 1.0
    processing_time_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
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
