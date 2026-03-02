/** 콘텐츠 마케팅 타입 정의 (Backend snake_case 그대로 사용)
 * v2.0: 페르소나, Legal Gate, 프롬프트 체인 타입 추가
 */

// ── Enum Types ──

export type TrendSource = 'tavily' | 'naver' | 'perplexity' | 'google_trends' | 'youtube' | 'newsdata' | 'newsapi'
export type TrendCategory = 'all' | 'criminal' | 'civil' | 'labor' | 'family' | 'administrative' | 'corporate' | 'ip'
export type TimeRange = '48h' | '7d' | '14d' | '30d'
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
  news_articles?: NewsArticleForScript[] | null
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
  step: 'tavily_start' | 'tavily_done' | 'llm_start' | 'llm_done' | 'scoring' | 'cache_hit' | 'done' | 'error'
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
  convergence: number
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
  convergence_score: number | null
  is_early_signal: boolean
}

export interface SourceFailInfo {
  source_name: string
  error_type: string
  error_message: string | null
}

export interface KeywordCollectResponse {
  keywords: KeywordItem[]
  total_count: number
  collected_at: string
  sources_used: string[]
  sources_failed: SourceFailInfo[]
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
  relevance_score: number
  legal_score: number
  recency_score: number
  total_score: number
  // v2 engagement 메트릭
  engagement_score: number
  convergence_score: number
  view_count: number | null
  comment_count: number | null
  is_early_signal: boolean
  source_weight: number
  score_breakdown: Record<string, number>
}

export interface NewsArticleForScript {
  title: string
  snippet: string
  source: string
  published_at: string | null
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
  sources_failed: SourceFailInfo[]
  searched_at: string
}

// ── Persona Analysis Response Types (v3.0 NEW — persona-ux-redesign) ──

/** 분석 근거 발췌 */
export interface EvidenceSnippet {
  text: string
  category: TrendCategory
  date: string
}

/** AI 분석 인사이트 */
export interface AnalysisInsights {
  area_scores: Record<string, number>
  summary: string
  total_conversations_analyzed: number
  analysis_period_days: number
  evidence_snippets: EvidenceSnippet[]
}

/** Track 1 분석 API 확장 응답 */
export interface PersonaAnalysisResponse {
  persona: LawyerPersona
  analysis_insights: AnalysisInsights
}

/** 대화 이력 건수 응답 */
export interface ChatHistoryCountResponse {
  count: number
  has_sufficient_history: boolean
  oldest_date: string | null
}

/** 표준 에러 응답 */
export interface PersonaApiError {
  error_code: string
  detail: string
  retryable: boolean
}

// ── GateScreen 상태머신 (v3.0 NEW) ──

export type GateScreen =
  | 'loading'
  | 'welcome'
  | 'track1_analyzing'
  | 'track1_review'
  | 'track2_onboarding'
  | 'ready'

/** PersonaGate 상태 */
export interface PersonaGateState {
  screen: GateScreen
  persona: LawyerPersona | null
  analysisInsights: AnalysisInsights | null
  draftOnboarding: Partial<OnboardingState> | null
  chatHistoryCount: number | null
  error: string | null
  isAnalyzing: boolean
  isSaving: boolean
}

/** PersonaGate 액션 (useReducer) */
export type GateAction =
  | { type: 'LOAD_START' }
  | { type: 'LOAD_SUCCESS'; persona: LawyerPersona | null; chatHistoryCount: number }
  | { type: 'LOAD_ERROR'; error: string }
  | { type: 'START_ANALYSIS' }
  | { type: 'ANALYSIS_SUCCESS'; persona: LawyerPersona; insights: AnalysisInsights }
  | { type: 'ANALYSIS_FAIL'; error: string; redirectScreen?: GateScreen }
  | { type: 'SAVE_START' }
  | { type: 'SAVE_SUCCESS'; persona: LawyerPersona }
  | { type: 'SAVE_ERROR'; error: string }
  | { type: 'SET_SCREEN'; screen: GateScreen }
  | { type: 'SET_DRAFT'; draft: Partial<OnboardingState> }
  | { type: 'CLEAR_ERROR' }
  | { type: 'SKIP_SETUP' }

// ── 상수 (v3.0 NEW) ──

/** 전문분야별 추천 키워드 */
export const SPECIALTY_KEYWORDS: Record<string, string[]> = {
  criminal: ['음주운전', '사기', '폭행', '성범죄', '마약'],
  civil: ['손해배상', '부동산', '계약해지', '대여금', '명예훼손'],
  labor: ['부당해고', '산업재해', '임금체불', '직장내 괴롭힘'],
  family: ['이혼', '양육권', '재산분할', '상속', '가사조정'],
  administrative: ['행정소송', '인허가', '징계처분', '과태료'],
  corporate: ['M&A', '회사법', '금융/증권', '공정거래', '지배구조'],
  ip: ['특허침해', '상표등록', '저작권', '영업비밀', '디자인권'],
}

/** 톤 미리보기 텍스트 */
export const TONE_PREVIEW_TEXT: Record<PersonaTone, string> = {
  professional: '대법원 2024다12345 판결에 따르면, 해당 사안은 민법 제750조 불법행위에 해당하며...',
  casual: '쉽게 말하면 이런 상황이에요. 비유를 들어볼게요. 여러분이 가게에서...',
  storytelling: '어느 날, 한 직장인에게 갑작스러운 통보가 날아왔습니다. "내일부터 나오지 마세요"...',
  educational: 'Step 1. 먼저 이 개념부터 이해해야 합니다. 불법행위란 타인의 권리를 위법하게...',
}

// v3.0 UI State Types (4스텝 → 3스텝)
export type OnboardingStep = 1 | 2 | 3

export interface OnboardingState {
  step: OnboardingStep
  specialty_areas: TrendCategory[]
  target_audience: TargetAudience | null
  preferred_tone: PersonaTone | null
  channel_style: ChannelStyle | null
  focus_topics: string[]
}

// ── Webtoon Storyboard Types (v3.0) ──

export type WebtoonSceneType =
  | 'hook_shock' | 'hook_question'
  | 'legal_explanation' | 'case_example' | 'conflict_drama' | 'document_closeup'
  | 'lawyer_advice' | 'cta_subscribe'

export type WebtoonImageStatus = 'pending' | 'generating' | 'retrying' | 'completed' | 'error'

export interface WebtoonPanel {
  panel_number: number
  section: SectionType
  scene_type: WebtoonSceneType
  script_excerpt: string
  scene_description: string
  location: string
  time_of_day: string
  characters: string[]
  emotion: string
  visual_focus: string
  camera_angle: string
  legal_keyword: string
  image_prompt: string | null
  image_url: string | null
  image_status: WebtoonImageStatus
  error_message: string | null
  model_version: string | null
  prompt_version: string | null
  generation_cost_ms: number | null
  safety_flags: string[]
}

export interface WebtoonGenerateRequest {
  topic: string
  sections: Record<string, string>
  persona: PersonaType
  persona_id?: string | null
  panel_count?: number | null
}

export interface WebtoonJobResponse {
  job_id: string
  status: string
  estimated_panels: number
}

export interface WebtoonStreamEvent {
  event: 'scene_split_start' | 'scene_split_done' | 'panel_start' | 'panel_complete' | 'panel_failed' | 'all_done' | 'error'
  panel_number: number | null
  total_panels: number | null
  section: string | null
  caption: string | null
  scene_description: string | null
  image_url: string | null
  error: string | null
}

export interface WebtoonJobStatus {
  job_id: string
  status: 'pending' | 'processing' | 'completed' | 'failed'
  progress: number
  panels: WebtoonPanel[]
  error: string | null
}
