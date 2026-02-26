/** 콘텐츠 마케팅 타입 정의 (Backend snake_case 그대로 사용)
 * v2.0: 페르소나, Legal Gate, 프롬프트 체인 타입 추가
 */

// ── Enum Types ──

export type TrendSource = 'tavily' | 'naver' | 'perplexity' | 'google_trends' | 'youtube' | 'newsdata' | 'newsapi'
export type TrendCategory = 'all' | 'criminal' | 'civil' | 'labor' | 'family' | 'administrative' | 'corporate' | 'ip'
export type TimeRange = '48h' | '7d' | '30d'
export type PersonaType = 'professional' | 'casual'
export type ScriptDuration = 5 | 10 | 15
export type SectionType = 'hooking' | 'analysis' | 'advice_cta'

// v2.0 Enum Types
export type PersonaTone = 'professional' | 'casual' | 'storytelling' | 'educational'
export type TargetAudience = 'general_public' | 'business' | 'legal_student' | 'legal_professional'
export type ChannelStyle = 'expert' | 'casual_friendly' | 'storytelling' | 'lecture'
export type LegalStage = 'litigation' | 'legislation' | 'prosecution' | 'dispute' | 'mention'
export type PersonaSource = 'passive' | 'active'

// ── Persona Types (v2.0 NEW, §3.5) ──

export interface LawyerPersona {
  id: string
  user_id: string
  specialty_areas: TrendCategory[]
  focus_topics: string[]
  preferred_tone: PersonaTone
  target_audience: TargetAudience
  channel_style: ChannelStyle | null
  source: PersonaSource
  confidence: number
  created_at: string
  updated_at: string
}

export interface PersonaAnalysisRequest {
  max_history?: number
  days_back?: number
}

export interface PersonaOnboardingRequest {
  specialty_areas: TrendCategory[]
  target_audience: TargetAudience
  preferred_tone: PersonaTone
  channel_style?: ChannelStyle | null
  focus_topics?: string[]
}

export interface PersonaUpdateRequest {
  specialty_areas?: TrendCategory[]
  focus_topics?: string[]
  preferred_tone?: PersonaTone
  target_audience?: TargetAudience
  channel_style?: ChannelStyle | null
}

export interface PersonaFeedbackRequest {
  persona_id: string
  script_id?: string | null
  rating: number
  feedback_type?: 'tone_mismatch' | 'specialty_mismatch' | 'audience_mismatch' | 'other' | null
  feedback_text?: string | null
}

// ── Scoring Types (v2.0 ENHANCED, §3.5) ──

export interface TrendScoreDetail {
  mention_score: number
  legal_score: number
  controversy_score: number
  spread_score: number
  fitness_score: number
  legal_stage: LegalStage
  legal_gate_passed: boolean
  gate_rejection_reason: string | null
  combined_score: number
}

// ── Trend Types ──

export interface TrendRequest {
  time_range: TimeRange
  category: TrendCategory
  limit: number
  query: string | null
  persona_id?: string | null
}

export interface SourceArticle {
  title: string
  url: string
  source: TrendSource
  published_at: string | null
  snippet: string
}

export interface RelatedLaw {
  law_id: string
  law_name: string
  relevance_score: number
}

export interface RelatedCase {
  case_id: string
  case_number: string
  case_name: string
  relevance_score: number
}

export interface TrendIssue {
  id: string
  title: string
  summary: string
  key_points: string[]
  score: number
  mention_score: number
  legal_relevance_score: number
  category: TrendCategory
  score_detail: TrendScoreDetail | null
  fitness_label: string | null
  sources: TrendSource[]
  source_articles: SourceArticle[]
  related_laws: RelatedLaw[]
  related_cases: RelatedCase[]
  collected_at: string
}

export interface TrendResponse {
  trends: TrendIssue[]
  total_count: number
  collected_at: string
  sources_used: TrendSource[]
  cache_hit: boolean
}

export interface TrendDetailResponse {
  issue: TrendIssue
  source_articles: SourceArticle[]
  related_laws_detail: Record<string, unknown>[]
  related_cases_detail: Record<string, unknown>[]
}

// ── Script Types ──

export interface ScriptRequest {
  topic: string
  trend_id: string | null
  persona: PersonaType
  persona_id?: string | null
  duration: ScriptDuration
  related_laws: string[]
  related_cases: string[]
}

export interface MetadataRequest {
  script_content: string
  topic: string
  persona: PersonaType
}

export interface ScriptMetadata {
  description: string
  tags: string[]
  cta_text: string
  hashtags: string[]
}

export interface ScriptStreamEvent {
  event: 'section_start' | 'content' | 'section_end' | 'metadata' | 'done' | 'error' | 'stage_update'
  section: SectionType | null
  content: string
  metadata: ScriptMetadata | null
  error: string | null
  stage: string | null
  status: string | null
  detail: string | null
}

// ── UI State Types ──

export interface TrendFilters {
  time_range: TimeRange
  category: TrendCategory
}

export interface ScriptGeneratorState {
  topic: string
  trend_id: string | null
  persona: PersonaType
  persona_id: string | null
  duration: ScriptDuration
  is_generating: boolean
  sections: Record<SectionType, string>
  metadata: ScriptMetadata | null
}

// ── Keyword Stream Event (§7.4) ──

export interface KeywordStreamEvent {
  step: 'tavily_start' | 'tavily_done' | 'llm_start' | 'llm_done' | 'scoring' | 'done' | 'error'
  progress: number
  message: string
  count: number | null
  data: KeywordCollectResponse | null
  error: string | null
}

// ── Keyword Flow Types (v2.1 NEW) ──

export interface KeywordCollectRequest {
  time_range?: TimeRange
  category?: TrendCategory
  community_domains?: string[] | null
  max_keywords?: number
  persona_id?: string | null
}

export interface KeywordScoreSchema {
  virality: number
  social_impact: number
  legal_relevance: number
  content_fitness: number
}

export interface KeywordItem {
  id: string
  keyword: string
  context: string
  scores: KeywordScoreSchema
  total_score: number
  rank: number
  score_reason: string
  confidence: number
}

export interface KeywordCollectResponse {
  keywords: KeywordItem[]
  total_count: number
  collected_at: string
  sources_used: string[]
  cache_hit: boolean
  prompt_version: string
  model_version: string
}

export interface KeywordNewsRequest {
  max_results?: number
}

export interface NewsArticle {
  title: string
  url: string
  source: string
  published_at: string | null
  snippet: string
  related_laws: string[]
  legal_issue_label: string | null
}

export interface RelatedLawBrief {
  law_name: string
  issue_label: string
}

export interface KeywordNewsResponse {
  keyword_id: string
  keyword: string
  articles: NewsArticle[]
  related_laws: RelatedLawBrief[]
  total_count: number
  sources_used: string[]
  searched_at: string
}

// v2.0 UI State Types
export type OnboardingStep = 1 | 2 | 3 | 4

export interface OnboardingState {
  step: OnboardingStep
  specialty_areas: TrendCategory[]
  target_audience: TargetAudience | null
  preferred_tone: PersonaTone | null
  channel_style: ChannelStyle | null
  focus_topics: string[]
}
