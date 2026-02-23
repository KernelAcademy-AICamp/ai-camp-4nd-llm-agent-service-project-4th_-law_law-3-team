"""
RAG 트레이스 수집 및 저장

파이프라인 각 단계의 중간 결과를 인메모리로 저장하여
디버깅/모니터링에 활용. 서버 재시작 시 초기화.
"""

from __future__ import annotations

import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class RagStepResult:
    """RAG 파이프라인 개별 단계 결과."""

    step_name: str  # "query_rewrite" | "vector_search" | "keyword_search" | "rrf_fusion" | "rerank" | "final_context"
    documents: list[dict[str, Any]]  # 요약된 결과 (source_id, title, score)
    count: int
    time_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """직렬화용 딕셔너리 변환."""
        return {
            "step_name": self.step_name,
            "documents": self.documents,
            "count": self.count,
            "time_ms": round(self.time_ms, 1),
            "metadata": self.metadata,
        }


@dataclass
class RagTrace:
    """RAG 파이프라인 전체 트레이스."""

    trace_id: str
    timestamp: datetime
    query: str  # 원본 질문
    search_query: str  # 리라이팅된 검색 쿼리
    agent_used: str
    response_full: str  # LLM 응답 전문
    steps: list[RagStepResult] = field(default_factory=list)
    pipeline_config: dict[str, Any] = field(default_factory=dict)
    total_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """직렬화용 딕셔너리 변환."""
        return {
            "trace_id": self.trace_id,
            "timestamp": self.timestamp.isoformat(),
            "query": self.query,
            "search_query": self.search_query,
            "agent_used": self.agent_used,
            "response_full": self.response_full,
            "steps": [s.to_dict() for s in self.steps],
            "pipeline_config": self.pipeline_config,
            "total_time_ms": round(self.total_time_ms, 1),
        }

    def to_summary(self) -> dict[str, Any]:
        """목록 표시용 요약 딕셔너리."""
        return {
            "trace_id": self.trace_id,
            "timestamp": self.timestamp.isoformat(),
            "query": self.query,
            "agent_used": self.agent_used,
            "total_time_ms": round(self.total_time_ms, 1),
            "step_count": len(self.steps),
        }


class RagTraceStore:
    """인메모리 트레이스 스토어 (thread-safe)."""

    def __init__(self, max_size: int = 50) -> None:
        self._traces: deque[RagTrace] = deque(maxlen=max_size)
        self._lock = threading.Lock()

    def add(self, trace: RagTrace) -> None:
        """트레이스 추가."""
        with self._lock:
            self._traces.appendleft(trace)

    def list_all(self) -> list[RagTrace]:
        """전체 트레이스 목록 (최신순)."""
        with self._lock:
            return list(self._traces)

    def get(self, trace_id: str) -> RagTrace | None:
        """ID로 트레이스 조회."""
        with self._lock:
            for trace in self._traces:
                if trace.trace_id == trace_id:
                    return trace
            return None

    def clear(self) -> None:
        """전체 트레이스 삭제."""
        with self._lock:
            self._traces.clear()


def generate_trace_id() -> str:
    """고유 트레이스 ID 생성."""
    return uuid.uuid4().hex[:12]


def now_utc() -> datetime:
    """현재 UTC 시간."""
    return datetime.now(timezone.utc)


def summarize_doc(doc: dict[str, Any], *, include_content: bool = False) -> dict[str, Any]:
    """문서를 트레이스 저장용으로 요약."""
    meta = doc.get("metadata", {})
    result: dict[str, Any] = {
        "source_id": meta.get("doc_id", ""),
        "title": meta.get("case_name", "") or meta.get("title", ""),
        "data_type": meta.get("data_type", ""),
        "similarity": round(doc.get("similarity", 0), 4),
    }
    if "rerank_score" in doc:
        result["rerank_score"] = round(doc.get("rerank_score", 0), 4)
    if include_content:
        content = doc.get("content", "")
        result["content"] = content[:500] if content else ""
    return result


# 모듈 레벨 싱글턴
trace_store = RagTraceStore(max_size=50)
