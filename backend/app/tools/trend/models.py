"""트렌드 내부 데이터 모델 (Pydantic 외부 스키마와 분리)

v2.0: ScoredIssueV2, UnifiedAnalysisResult 추가
"""

from dataclasses import dataclass, field
from datetime import datetime

from app.modules.content_marketing.schema import (
    TrendCategory,
    TrendResponse,
    TrendSource,
)


@dataclass
class RawTrendItem:
    """소스별 수집된 원시 트렌드 항목

    v3: engagement 메트릭, 수렴 점수, 초기 신호 감지 필드 추가
    """

    title: str
    url: str
    snippet: str
    source: TrendSource
    published_at: datetime | None = None
    mention_count: int = 0
    raw_data: dict[str, object] = field(default_factory=dict)
    # v3 engagement 메트릭
    view_count: int | None = None
    comment_count: int | None = None
    like_count: int | None = None
    engagement_velocity: float | None = None
    is_shorts: bool = False
    # v3 수렴/신호 분석
    convergence_score: float = 0.0
    z_score: float | None = None
    is_early_signal: bool = False
    merged_sources: list[TrendSource] = field(default_factory=list)


@dataclass
class ScoredIssue:
    """스코어링 완료된 이슈 (v1.0 내부 처리용, 하위호환)"""

    id: str
    title: str
    raw_items: list[RawTrendItem]
    mention_score: float
    legal_relevance_score: float
    combined_score: float
    category: TrendCategory = TrendCategory.ALL


@dataclass
class ScoredIssueV2:
    """v2.0 스코어링 완료된 이슈 (5차원 + Legal Gate)"""

    id: str
    title: str
    raw_items: list[RawTrendItem]
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
class UnifiedAnalysisResult:
    """통합 LLM 분석 결과 (LegalGateScorer 내부용)"""

    legal_score: float
    legal_stage: str
    controversy_score: float
    category: str


@dataclass
class CommunityTopic:
    """Stage 1+2 중간 결과물: 커뮤니티 수집 + 키워드 추출"""

    raw_items: list[RawTrendItem]
    extracted_keywords: list[str]
    context_summary: str


@dataclass
class TrendCacheEntry:
    """트렌드 캐시 엔트리"""

    key: str
    response: TrendResponse
    created_at: datetime
    ttl_seconds: int = 86400


# ── Keyword Flow 내부 모델 (v2.1 NEW) ──


@dataclass
class KeywordScore:
    """키워드 4차원 점수 (virality, social_impact, legal_relevance, content_fitness)"""

    virality: float
    social_impact: float
    legal_relevance: float
    content_fitness: float


@dataclass
class ScoredKeyword:
    """스코어링 완료된 키워드"""

    id: str
    keyword: str
    context: str
    source_posts: list[RawTrendItem]
    scores: KeywordScore
    total_score: float
    rank: int
    score_reason: str = ""
    confidence: float = 0.0
