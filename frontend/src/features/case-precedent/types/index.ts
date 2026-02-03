export interface PrecedentItem {
  id: string
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
  case_name: string
  case_number: string
  doc_type: string
  court?: string
  date?: string
  content: string
  summary: string
}

export interface ChatSource {
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
  court_name?: string    // 법원명 (대법원, 헌법재판소 등)
  decision_date?: string // 선고일
  case_type?: string     // 사건유형 (민사/형사/행정)
  // 법령용 필드
  law_name?: string
  law_type?: string
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

// 법령 계층도 타입
export * from './hierarchy'
