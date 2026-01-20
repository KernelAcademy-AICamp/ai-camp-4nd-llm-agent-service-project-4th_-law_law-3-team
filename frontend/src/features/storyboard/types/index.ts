// 타임라인 개별 항목
export interface TimelineItem {
  id: string
  date: string
  title: string
  description: string
  participants: string[]
  order: number
  imageUrl?: string
  imagePrompt?: string
}

// 타임라인 전체 데이터 (JSON 내보내기/가져오기용)
export interface TimelineData {
  title: string
  created_at: string
  updated_at: string
  items: TimelineItem[]
  original_text?: string
}

// AI 추출 요청
export interface ExtractTimelineRequest {
  text: string
}

// AI 추출 응답
export interface ExtractTimelineResponse {
  success: boolean
  timeline: TimelineItem[]
  summary?: string
}

// 유효성 검사 응답
export interface ValidateTimelineResponse {
  valid: boolean
  message: string
}

// 편집 모드
export type EditMode = 'view' | 'edit'
