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
  summary?: string
  content?: string
  // 법령용 필드
  law_name?: string
  law_type?: string
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
