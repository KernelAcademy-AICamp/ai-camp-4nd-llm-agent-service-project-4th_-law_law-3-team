/** 시험 문제 파일 메타데이터 */
export interface ExamFile {
  category: string
  year: number
  session: number
  filename: string
  title: string
}

export interface ExamListResponse {
  exams: ExamFile[]
  total: number
}

export interface ExamContentResponse {
  category: string
  year: number
  session: number
  title: string
  content: string
  total_chars: number
}

export interface ReferenceSearchRequest {
  query: string
  doc_type?: string | null
  n_results?: number
}

export interface ReferenceSearchResult {
  id: string
  doc_type: string
  title: string
  case_number?: string | null
  summary: string
  similarity: number
}

export interface ReferenceSearchResponse {
  query: string
  results: ReferenceSearchResult[]
}

export interface AnswerFeedbackRequest {
  answer_text: string
}

export interface AnswerFeedbackResponse {
  feedback: string
}

/** 카테고리 표시 정보 */
export const CATEGORY_MAP: Record<string, { label: string; color: string }> = {
  CIVIL: { label: '민사법', color: 'blue' },
  CRIMINAL: { label: '형사법', color: 'red' },
  PUBLIC: { label: '공법', color: 'green' },
}

export const CATEGORY_ORDER = ['CIVIL', 'CRIMINAL', 'PUBLIC'] as const
