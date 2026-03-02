"""
콘텐츠 마케팅 Pydantic 스키마

Design 문서 Section 3 기반 요청/응답 모델 정의
v2.0: 페르소나 시스템, Legal Gate 스코어링, 프롬프트 체인 추가
"""

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

# ── Enum ──


class TrendSource(str, Enum):
    """트렌드 데이터 소스"""

    TAVILY = "tavily"
    NAVER = "naver"
    PERPLEXITY = "perplexity"
    GOOGLE_TRENDS = "google_trends"
    YOUTUBE = "youtube"
    NEWSDATA = "newsdata"
    NEWSAPI = "newsapi"


class TrendCategory(str, Enum):
    """트렌드 카테고리 (법률 분류)"""

    ALL = "all"
    CRIMINAL = "criminal"
    CIVIL = "civil"
    LABOR = "labor"
    FAMILY = "family"
    ADMINISTRATIVE = "administrative"
    CORPORATE = "corporate"
    IP = "ip"


class TimeRange(str, Enum):
    """트렌드 조회 기간"""

    HOURS_48 = "48h"
    DAYS_7 = "7d"
    DAYS_14 = "14d"
    DAYS_30 = "30d"


class PersonaType(str, Enum):
    """대본 페르소나 타입 (v1.0 하위호환, deprecated → PersonaTone 사용)"""

    PROFESSIONAL = "professional"
    CASUAL = "casual"


class PersonaTone(str, Enum):
    """영상 톤 (v2.0, PersonaType 대체)"""

    PROFESSIONAL = "professional"
    CASUAL = "casual"
    STORYTELLING = "storytelling"
    EDUCATIONAL = "educational"


class TargetAudience(str, Enum):
    """타겟 시청자"""

    GENERAL_PUBLIC = "general_public"
    BUSINESS = "business"
    LEGAL_STUDENT = "legal_student"
    LEGAL_PROFESSIONAL = "legal_professional"


class ChannelStyle(str, Enum):
    """채널 스타일"""

    EXPERT = "expert"
    CASUAL_FRIENDLY = "casual_friendly"
    STORYTELLING = "storytelling"
    LECTURE = "lecture"


class LegalStage(str, Enum):
    """법적 단계"""

    LITIGATION = "litigation"
    LEGISLATION = "legislation"
    PROSECUTION = "prosecution"
    DISPUTE = "dispute"
    MENTION = "mention"


class ScriptDuration(int, Enum):
    """대본 길이 (분)"""

    SHORT = 5
    MEDIUM = 10
    LONG = 15


class SectionType(str, Enum):
    """대본 섹션 타입"""

    HOOKING = "hooking"
    ANALYSIS = "analysis"
    ADVICE_CTA = "advice_cta"


# ── PersonaType → PersonaTone 매핑 (하위호환) ──

PERSONA_TYPE_TO_TONE: dict[PersonaType, PersonaTone] = {
    PersonaType.PROFESSIONAL: PersonaTone.PROFESSIONAL,
    PersonaType.CASUAL: PersonaTone.CASUAL,
}


# ── Persona Schema (v2.0 NEW, §3.1) ──


class LawyerPersona(BaseModel):
    """변호사 페르소나"""

    id: str
    user_id: str
    specialty_areas: list[TrendCategory]
    focus_topics: list[str] = Field(default_factory=list, max_length=5)
    preferred_tone: PersonaTone = PersonaTone.PROFESSIONAL
    target_audience: TargetAudience = TargetAudience.GENERAL_PUBLIC
    channel_style: ChannelStyle | None = None
    source: Literal["passive", "active"]
    confidence: float = Field(default=1.0, ge=0, le=1)
    created_at: datetime
    updated_at: datetime


class PersonaAnalysisRequest(BaseModel):
    """Track 1: 자동 분석 요청 (user_id는 Auth Dependency에서 추출)"""

    max_history: int = Field(default=100, ge=10, le=500)
    days_back: int = Field(default=30, ge=7, le=90)


class PersonaOnboardingRequest(BaseModel):
    """Track 2: 온보딩 결과 (user_id는 Auth Dependency에서 추출)"""

    specialty_areas: list[TrendCategory] = Field(min_length=1, max_length=3)
    target_audience: TargetAudience
    preferred_tone: PersonaTone
    channel_style: ChannelStyle | None = None
    focus_topics: list[str] = Field(default_factory=list, max_length=5)


class PersonaUpdateRequest(BaseModel):
    """페르소나 부분 수정"""

    specialty_areas: list[TrendCategory] | None = None
    focus_topics: list[str] | None = None
    preferred_tone: PersonaTone | None = None
    target_audience: TargetAudience | None = None
    channel_style: ChannelStyle | None = None


class PersonaFeedbackRequest(BaseModel):
    """대본 생성 후 피드백"""

    persona_id: str
    script_id: str | None = None
    rating: int = Field(ge=1, le=5)
    feedback_type: Literal[
        "tone_mismatch",
        "specialty_mismatch",
        "audience_mismatch",
        "other",
    ] | None = None
    feedback_text: str | None = None


# ── Persona Analysis Response (v3.0 NEW, persona-ux-redesign) ──


class EvidenceSnippet(BaseModel):
    """분석 근거 대화 발췌"""

    text: str
    category: TrendCategory
    date: datetime


class AnalysisInsights(BaseModel):
    """AI 분석 인사이트 (persona/analyze 응답 확장)"""

    area_scores: dict[str, float]  # { "criminal": 0.78, "civil": 0.62 }
    summary: str
    total_conversations_analyzed: int
    analysis_period_days: int
    evidence_snippets: list[EvidenceSnippet] = Field(
        default_factory=list, max_length=3
    )


class PersonaAnalysisResponse(BaseModel):
    """Track 1 분석 응답 (기존 LawyerPersona + insights)"""

    persona: LawyerPersona
    analysis_insights: AnalysisInsights


class ChatHistoryCountResponse(BaseModel):
    """대화 이력 건수 응답"""

    count: int
    has_sufficient_history: bool = False  # count >= 5
    oldest_date: datetime | None = None


# ── Scoring Schema (v2.0 ENHANCED, §3.2) ──


class TrendScoreDetail(BaseModel):
    """트렌드 세부 점수 (v2.0 5차원)"""

    mention_score: float = Field(ge=0, le=1)
    legal_score: float = Field(ge=0, le=1)
    controversy_score: float = Field(ge=0, le=1)
    spread_score: float = Field(ge=0, le=1)
    fitness_score: float = Field(ge=0, le=1)
    legal_stage: LegalStage
    legal_gate_passed: bool
    gate_rejection_reason: str | None = None
    combined_score: float = Field(ge=0, le=100)


# ── Trend Request / Response ──


class TrendRequest(BaseModel):
    """트렌드 조회 요청"""

    time_range: TimeRange = TimeRange.HOURS_48
    category: TrendCategory = TrendCategory.ALL
    limit: int = Field(default=10, ge=1, le=30)
    query: str | None = Field(
        default=None,
        description="키워드 필터 (없으면 전체 트렌드)",
    )
    persona_id: str | None = Field(
        default=None,
        description="페르소나 ID (fitness_score 계산용, v2.0)",
    )


class SourceArticle(BaseModel):
    """개별 소스 기사/게시글"""

    title: str
    url: str
    source: TrendSource
    published_at: datetime | None = None
    snippet: str = ""


class RelatedLaw(BaseModel):
    """관련 법령 정보"""

    law_id: str
    law_name: str
    relevance_score: float = Field(ge=0, le=1)


class RelatedCase(BaseModel):
    """관련 판례 정보"""

    case_id: str
    case_number: str
    case_name: str
    relevance_score: float = Field(ge=0, le=1)


class TrendIssue(BaseModel):
    """트렌드 이슈 (대시보드 카드)"""

    id: str
    title: str
    summary: str
    key_points: list[str] = Field(min_length=1, max_length=5)
    score: float = Field(ge=0, le=100)
    mention_score: float = Field(ge=0, le=1)
    legal_relevance_score: float = Field(ge=0, le=1)
    category: TrendCategory
    score_detail: TrendScoreDetail | None = None
    fitness_label: str | None = None
    sources: list[TrendSource]
    source_articles: list[SourceArticle]
    related_laws: list[RelatedLaw]
    related_cases: list[RelatedCase]
    collected_at: datetime


class TrendResponse(BaseModel):
    """트렌드 조회 응답"""

    trends: list[TrendIssue]
    total_count: int
    collected_at: datetime
    sources_used: list[TrendSource]
    cache_hit: bool = False


class TrendDetailResponse(BaseModel):
    """트렌드 상세 조회 응답"""

    issue: TrendIssue
    source_articles: list[SourceArticle]
    related_laws_detail: list[dict[str, object]]
    related_cases_detail: list[dict[str, object]]


# ── Script Request / Response ──


class NewsArticleForScript(BaseModel):
    """대본 생성용 경량 뉴스 기사"""

    title: str
    snippet: str = ""
    source: str = ""
    published_at: datetime | None = None


class ScriptRequest(BaseModel):
    """대본 생성 요청 (v2.0: persona_id 추가, persona 필드는 하위호환 유지)"""

    topic: str = Field(
        min_length=2,
        max_length=500,
        description="주제 (직접 입력 또는 트렌드 요약)",
    )
    trend_id: str | None = Field(
        default=None,
        description="연결된 트렌드 ID (트렌드에서 선택 시)",
    )
    persona: PersonaType = PersonaType.PROFESSIONAL
    persona_id: str | None = Field(
        default=None,
        description="페르소나 ID (v2.0, persona 필드보다 우선)",
    )
    duration: ScriptDuration = ScriptDuration.MEDIUM
    related_laws: list[str] = Field(default_factory=list)
    related_cases: list[str] = Field(default_factory=list)
    news_articles: list[NewsArticleForScript] | None = Field(
        default=None,
        description="선택된 뉴스 기사 (최대 10개)",
        max_length=10,
    )

    @model_validator(mode="after")
    def strip_topic(self) -> "ScriptRequest":
        """topic 공백 제거 후 min_length 재검증 (API 직접 호출 시 공백-only 방어)"""
        self.topic = self.topic.strip()
        if len(self.topic) < 2:
            msg = "topic은 공백 제거 후 최소 2자 이상이어야 합니다."
            raise ValueError(msg)
        return self


class MetadataRequest(BaseModel):
    """메타데이터 생성 요청"""

    script_content: str = Field(min_length=100)
    topic: str
    persona: PersonaType = PersonaType.PROFESSIONAL


class Citation(BaseModel):
    """인용 출처"""

    source_type: str
    source_id: str
    source_name: str
    relevant_text: str = ""


class ScriptSection(BaseModel):
    """대본 섹션"""

    section_type: SectionType
    title: str
    content: str
    citations: list[Citation]
    word_count: int


class ScriptMetadata(BaseModel):
    """영상 메타데이터"""

    description: str
    tags: list[str]
    cta_text: str
    hashtags: list[str]


class ScriptResult(BaseModel):
    """대본 생성 결과"""

    id: str
    topic: str
    persona: PersonaType
    sections: list[ScriptSection]
    metadata: ScriptMetadata
    all_citations: list[Citation]
    word_count: int
    estimated_duration: int
    disclaimer: str = "본 콘텐츠는 AI가 생성한 것으로, 법률 자문이 아닙니다."
    created_at: datetime


class ScriptStreamEvent(BaseModel):
    """SSE 스트리밍 이벤트 (v2.0: stage_update 이벤트 추가)"""

    event: str
    section: SectionType | None = None
    content: str = ""
    metadata: ScriptMetadata | None = None
    error: str | None = None
    stage: str | None = None
    status: str | None = None
    detail: str | None = None


# ── Keyword Flow Schema (v2.1 NEW) ──


class KeywordCollectRequest(BaseModel):
    """키워드 수집 요청"""

    time_range: TimeRange = TimeRange.HOURS_48
    category: TrendCategory = TrendCategory.ALL
    community_domains: list[str] | None = Field(
        default=None,
        description="커뮤니티 도메인 목록 (None이면 기본 도메인 사용). 화이트리스트 도메인만 허용됨.",
    )
    max_keywords: int = Field(default=10, ge=1, le=30)
    persona_id: str | None = Field(
        default=None,
        description="페르소나 ID (향후 개인화 스코어링용)",
    )

    @model_validator(mode="after")
    def validate_community_domains(self) -> "KeywordCollectRequest":
        """community_domains를 화이트리스트와 교집합 필터링 (SSRF 방어)"""
        if self.community_domains is not None:
            from app.core.config import settings

            allowed = set(settings.KEYWORD_COMMUNITY_DOMAINS)
            self.community_domains = [d for d in self.community_domains if d in allowed]
            if not self.community_domains:
                self.community_domains = None
        return self


class KeywordScoreSchema(BaseModel):
    """키워드 점수 (v3: convergence 추가)"""

    virality: float = Field(ge=0, le=1)
    social_impact: float = Field(ge=0, le=1)
    legal_relevance: float = Field(ge=0, le=1)
    content_fitness: float = Field(ge=0, le=1)
    convergence: float = Field(default=0.0, ge=0, le=1)


class KeywordItem(BaseModel):
    """키워드 단일 항목 (v3: convergence_score, is_early_signal 추가)"""

    id: str
    keyword: str
    context: str
    scores: KeywordScoreSchema
    total_score: float = Field(ge=0, le=100)
    rank: int
    score_reason: str = ""
    confidence: float = Field(default=0.0, ge=0, le=1)
    convergence_score: float | None = None
    is_early_signal: bool = False


class SourceFailInfo(BaseModel):
    """소스 실패 정보 (사용자 친화적 메시지만 포함, FR-09)"""

    source_name: str  # "youtube", "google_trends", etc.
    error_type: str  # "timeout", "auth", "rate_limit", "network", "parse", "unknown"
    error_message: str | None = None  # sanitize된 메시지 (기술 스택 미노출)


class KeywordCollectResponse(BaseModel):
    """키워드 수집 응답"""

    keywords: list[KeywordItem]
    total_count: int
    collected_at: datetime
    sources_used: list[str]
    sources_failed: list[SourceFailInfo] = Field(default_factory=list)
    cache_hit: bool = False
    prompt_version: str = "1.0"
    model_version: str = ""


class KeywordStreamEvent(BaseModel):
    """키워드 수집 SSE 스트리밍 이벤트 (§7.4)"""

    step: str  # tavily_start, tavily_done, llm_start, llm_done, scoring, cache_hit, done, error
    progress: int = 0  # 0~100
    message: str = ""
    count: int | None = None
    data: KeywordCollectResponse | None = None
    error: str | None = None


class KeywordNewsRequest(BaseModel):
    """키워드 뉴스 검색 요청"""

    max_results: int = Field(default=10, ge=1, le=30)


class NewsArticle(BaseModel):
    """뉴스 기사 (v2: engagement/convergence 5차원 스코어링)"""

    title: str
    url: str
    source: str
    published_at: datetime | None = None
    snippet: str = ""
    related_laws: list[str] = Field(default_factory=list)
    legal_issue_label: str | None = None
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0, description="키워드 관련도")
    legal_score: float = Field(default=0.0, ge=0.0, le=1.0, description="법적 관련도")
    recency_score: float = Field(default=0.0, ge=0.0, le=1.0, description="최신성")
    total_score: float = Field(default=0.0, ge=0.0, le=100.0, description="종합 점수")
    # v2 engagement 메트릭
    engagement_score: float = Field(default=0.0, ge=0.0, le=1.0, description="참여도 점수")
    convergence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="수렴 점수")
    view_count: int | None = None
    comment_count: int | None = None
    is_early_signal: bool = False
    source_weight: float = Field(default=0.5, ge=0.0, le=2.0, description="소스 권위도 가중치")
    score_breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="차원별 점수 분해 (예: relevance, legal, recency, engagement, convergence)",
    )


class RelatedLawBrief(BaseModel):
    """경량 관련 법령 정보 (RAG 기반)"""

    law_name: str
    issue_label: str = ""


class KeywordNewsResponse(BaseModel):
    """키워드 뉴스 검색 응답"""

    keyword_id: str
    keyword: str
    articles: list[NewsArticle]
    related_laws: list[RelatedLawBrief] = Field(default_factory=list)
    total_count: int
    sources_used: list[str] = Field(default_factory=list)
    sources_failed: list[SourceFailInfo] = Field(default_factory=list)
    searched_at: datetime


# ── Webtoon Storyboard Schema (스토리보드 자동 생성) ──


class WebtoonSceneType(str, Enum):
    """웹툰 씬 타입 (8종)"""

    HOOK_SHOCK = "hook_shock"
    HOOK_QUESTION = "hook_question"
    LEGAL_EXPLANATION = "legal_explanation"
    CASE_EXAMPLE = "case_example"
    CONFLICT_DRAMA = "conflict_drama"
    DOCUMENT_CLOSEUP = "document_closeup"
    LAWYER_ADVICE = "lawyer_advice"
    CTA_SUBSCRIBE = "cta_subscribe"


class ImageStatus(str, Enum):
    """이미지 생성 상태"""

    PENDING = "pending"
    GENERATING = "generating"
    RETRYING = "retrying"
    COMPLETED = "completed"
    ERROR = "error"


class WebtoonPanel(BaseModel):
    """웹툰 패널 (장면 분할 결과 + 이미지 생성 결과)"""

    panel_number: int = Field(ge=1, le=14)
    section: SectionType
    scene_type: WebtoonSceneType
    script_excerpt: str = Field(max_length=500)
    scene_description: str = Field(max_length=300)
    location: str = Field(max_length=100)
    time_of_day: str = Field(max_length=50)
    characters: list[str] = Field(default_factory=list, max_length=5)
    emotion: str = Field(max_length=50)
    visual_focus: str = Field(max_length=100)
    camera_angle: str = Field(max_length=50)
    legal_keyword: str = Field(max_length=100)
    image_prompt: str | None = None
    image_url: str | None = None
    image_status: ImageStatus = ImageStatus.PENDING
    error_message: str | None = None
    model_version: str | None = None
    prompt_version: str | None = None
    generation_cost_ms: int | None = None
    safety_flags: list[str] = Field(default_factory=list)


class WebtoonGenerateRequest(BaseModel):
    """웹툰 스토리보드 생성 요청"""

    topic: str = Field(min_length=2, max_length=500)
    sections: dict[str, str]
    persona: PersonaType = PersonaType.PROFESSIONAL
    persona_id: str | None = None
    panel_count: int | None = Field(default=None, ge=4, le=14)

    @model_validator(mode="after")
    def validate_topic_and_sections(self) -> "WebtoonGenerateRequest":
        """topic 공백 제거 + 필수 섹션 키 검증"""
        self.topic = self.topic.strip()
        if len(self.topic) < 2:
            msg = "topic은 공백 제거 후 최소 2자 이상이어야 합니다."
            raise ValueError(msg)
        required = {"hooking", "analysis", "advice_cta"}
        if not required.issubset(self.sections.keys()):
            missing = required - self.sections.keys()
            msg = f"필수 섹션 누락: {missing}"
            raise ValueError(msg)
        for key in required:
            if not self.sections[key].strip():
                msg = f"섹션 '{key}'의 내용이 비어있습니다."
                raise ValueError(msg)
        return self


class WebtoonJobResponse(BaseModel):
    """웹툰 Job 생성 응답"""

    job_id: str
    status: str = "accepted"
    estimated_panels: int


class WebtoonStreamEvent(BaseModel):
    """SSE 스트리밍 이벤트"""

    event: str
    panel_number: int | None = None
    total_panels: int | None = None
    section: str | None = None
    caption: str | None = None
    scene_description: str | None = None
    image_url: str | None = None
    error: str | None = None


class WebtoonJobStatusResponse(BaseModel):
    """웹툰 Job 상태 폴링 응답"""

    job_id: str
    status: str
    progress: int
    panels: list[WebtoonPanel] = Field(default_factory=list)
    error: str | None = None


class WebtoonRegenerateRequest(BaseModel):
    """개별 패널 재생성 요청"""

    panel_number: int = Field(ge=1, le=14)
