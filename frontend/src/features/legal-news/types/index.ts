/** 법률 뉴스 타입 정의 (Backend snake_case 그대로 사용) */

// ── 소스 타입 ──

export type NewsSource = 'lawtimes' | 'naver'

// ── 기사 목록용 요약 ──

export interface NewsArticleSummary {
  id: string
  title: string
  source: NewsSource
  publisher: string
  published_at: string | null
  summary_one_liner: string
  section: string | null
  tags: string[] | null
}

// ── 기사 상세 ──

export interface NewsArticleResponse {
  id: string
  title: string
  source: NewsSource
  publisher: string
  published_at: string | null
  collected_at: string
  url: string
  author: string | null
  section: string | null
  tags: string[] | null
  cleaned_text: string
  summary_one_liner: string
  summary_issues: string[] | null
  summary_laws: string[] | null
  summary_cases: string[] | null
  summary_institutions: string[] | null
  summary_implications: string[] | null
  disclaimer: string
  schema_version: string
}

// ── 목록 응답 ──

export interface NewsListResponse {
  items: NewsArticleSummary[]
  total: number
  page: number
  page_size: number
  has_next: boolean
}

// ── 검색 요청 ──

export interface NewsSearchRequest {
  query: string
  limit?: number
  source?: NewsSource | null
}

// ── 검색 결과 항목 ──

export interface NewsSearchResult {
  chunk_id: string
  doc_id: string
  title: string
  chunk_text: string
  chunk_type: string
  source: NewsSource
  publisher: string
  url: string
  published_at: string | null
  rerank_score: number | null
}

// ── 검색 응답 ──

export interface NewsSearchResponse {
  results: NewsSearchResult[]
  query: string
  total: number
}

// ── UI 상태 타입 ──

export interface NewsListFilters {
  source: NewsSource | null
  published_date: string | null
  page: number
  page_size: number
}

export type NewsTab = 'list' | 'search'

// ── 통계 타입 ──

export interface DailyStatItem {
  date: string
  source: string
  count: number
}

export interface NewsStatsDaily {
  items: DailyStatItem[]
  total: number
  period_days: number
}

export interface CategoryStatItem {
  category: string
  count: number
}

export interface NewsCategoryStats {
  items: CategoryStatItem[]
  total: number
  period_days: number | null
}

// ── RAG 기여도 ──

export interface MainRagSourceItem {
  table_name: string
  label: string
  count: number
}

export interface AssistRagSourceItem {
  source: string
  label: string
  count: number
  indexed_count: number
}

export interface RagContributionStats {
  main_rag_total: number
  main_rag_sources: MainRagSourceItem[]
  assist_rag_total: number
  assist_rag_sources: AssistRagSourceItem[]
  assist_contribution_percent: number
}
