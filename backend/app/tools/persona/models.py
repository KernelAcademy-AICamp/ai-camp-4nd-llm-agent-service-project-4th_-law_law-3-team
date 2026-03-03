"""페르소나 내부 데이터 모델 (Pydantic 외부 스키마와 분리)

Design 문서 §3.6 기반 — ChatMetadata, PersonaExtractionResult, ScoredIssueV2, ScriptContext
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.modules.content_marketing.schema import AnalysisInsights, LawyerPersona


@dataclass
class ChatMetadata:
    """대화 이력 메타데이터 (PII 미포함, Red Team [심각 3] 반영)"""

    total_conversations: int
    agent_type_distribution: dict[str, int]
    top_legal_keywords: list[tuple[str, int]]
    category_distribution: dict[str, float]
    avg_message_length: float
    date_range_days: int


@dataclass
class PersonaExtractionResult:
    """LLM 페르소나 추출 결과 (검증 전)"""

    specialty_areas: list[str]
    focus_topics: list[str]
    confidence: float
    raw_response: str


@dataclass
class AnalysisResult:
    """페르소나 분석 결과 (persona + insights)"""

    persona: LawyerPersona
    insights: AnalysisInsights


@dataclass
class ScoredIssueV2:
    """v2.0 스코어링 완료된 이슈"""

    id: str
    title: str
    raw_items: list[object]
    mention_score: float
    legal_score: float
    controversy_score: float
    spread_score: float
    fitness_score: float
    legal_stage: str
    legal_gate_passed: bool
    combined_score: float
    gate_rejection_reason: str | None = None
    category: str = "all"


@dataclass
class ScriptContext:
    """프롬프트 체인 결과 컨텍스트"""

    issues: list[dict[str, object]]
    laws_by_issue: dict[str, list[object]]
    cases_by_issue: dict[str, list[object]]
    cross_validated: bool
    chain_latency_ms: dict[str, int] = field(default_factory=dict)
