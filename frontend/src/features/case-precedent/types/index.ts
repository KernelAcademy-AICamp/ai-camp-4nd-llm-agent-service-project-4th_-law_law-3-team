export interface PrecedentItem {
  id: string
  doc_id?: string
  case_name: string
  case_number: string
  doc_type: string
  court?: string
  date?: string
  summary: string
  similarity: number
}

export interface PrecedentListResponse {
  keyword: string
  total: number
  precedents: PrecedentItem[]
}

export interface PrecedentDetail {
  id: string
  doc_id?: string
  case_name: string
  case_number: string
  doc_type: string
  court?: string
  date?: string
  content: string
  summary: string
  // 판례 상세 필드 (PostgreSQL)
  ruling?: string  // 주문
  claim?: string  // 청구취지
  reasoning?: string  // 판결요지
  full_reason?: string  // 이유
  full_text?: string  // 전문
  reference_provisions?: string  // 참조조문
  reference_cases?: string  // 참조판례
  court_name?: string  // 법원명
  decision_date?: string  // 선고일
  // 법령용 필드 (aiReferences에서 법령 데이터 흐름 지원)
  law_name?: string
  law_type?: string
  article_number?: string
  article_title?: string
  ministry?: string
  // 그래프 보강 정보
  cited_statutes?: string[]
  similar_cases?: string[]
}

export interface ChatSource {
  doc_id?: string
  case_name?: string
  case_number?: string
  doc_type: string
  similarity: number
  summary?: string       // 판시사항 (핵심 쟁점)
  content?: string
  // 판례 상세 필드 (역할별 차등 표시용)
  ruling?: string        // 주문
  claim?: string         // 청구취지
  reasoning?: string     // 판결요지
  full_reason?: string   // 이유 (전체)
  full_text?: string     // 판례내용 (전문)
  reference_provisions?: string  // 참조조문
  reference_cases?: string       // 참조판례
  court_name?: string    // 법원명 (대법원, 헌법재판소 등)
  decision_date?: string // 선고일
  case_type?: string     // 사건유형 (민사/형사/행정)
  // 법령용 필드
  law_name?: string
  law_type?: string
  article_number?: string   // 조문번호
  article_title?: string    // 조문제목
  ministry?: string         // 소관부처
  // 그래프 보강 정보
  cited_statutes?: string[]
  similar_cases?: string[]
}

export interface AIQuestionResponse {
  answer: string
  sources: ChatSource[]
}

export interface SearchFilters {
  keyword: string
  docType?: string
  court?: string
  limit: number
}

export type DocType = 'precedent' | 'constitutional' | ''
export type Court = '대법원' | '고등법원' | '지방법원' | '헌법재판소' | ''

// 법령 전문 타입
export interface LawArticleItem {
  article_number: string
  article_title?: string
  article_content: string
}

export interface LawFullText {
  law_id: string
  law_name: string
  law_type?: string
  ministry?: string
  ai_summary?: string
  supplementary?: string
  articles: LawArticleItem[]
  total_articles: number
  enforcement_date?: string
  promulgation_date?: string
  promulgation_no?: string
}

// 법령 인용 판례 타입
export interface CitingCaseItem {
  serial_number?: string
  case_number?: string
  case_name?: string
  decision_date?: string
  court_name?: string
}

export interface CitingCasesResponse {
  statute_id: string
  total: number
  cases: CitingCaseItem[]
}

// 법령 계층도 타입
export * from './hierarchy'
