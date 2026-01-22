// 전환 효과 타입
export type TransitionType = 'fade' | 'slide' | 'zoom' | 'none'

// 입력 모드 타입
export type InputMode = 'text' | 'voice' | 'image'

// 이미지 상태 타입
export type ImageStatus = 'pending' | 'processing' | 'completed' | 'failed'

// 참여자 역할 타입
export type ParticipantRole =
  | 'victim'       // 피해자
  | 'perpetrator'  // 가해자
  | 'witness'      // 증인
  | 'bystander'    // 방관자
  | 'authority'    // 공권력
  | 'other'        // 기타

// 참여자 상세 정보
export interface Participant {
  name: string                 // 이름/호칭
  role: ParticipantRole        // 역할
  action?: string              // 해당 장면에서의 행동
  emotion?: string             // 감정 상태
}

// 역할별 라벨 및 색상 매핑
export const PARTICIPANT_ROLE_CONFIG: Record<ParticipantRole, { label: string; color: string; bgColor: string }> = {
  victim: { label: '피해자', color: 'text-blue-400', bgColor: 'bg-blue-500/20' },
  perpetrator: { label: '가해자', color: 'text-red-400', bgColor: 'bg-red-500/20' },
  witness: { label: '증인', color: 'text-amber-400', bgColor: 'bg-amber-500/20' },
  bystander: { label: '방관자', color: 'text-slate-400', bgColor: 'bg-slate-500/20' },
  authority: { label: '공권력', color: 'text-purple-400', bgColor: 'bg-purple-500/20' },
  other: { label: '기타', color: 'text-slate-300', bgColor: 'bg-slate-600/20' },
}

// 타임라인 개별 항목
export interface TimelineItem {
  id: string
  date: string
  title: string
  description: string            // 하위 호환용
  participants: string[]         // 하위 호환용
  order: number
  imageUrl?: string
  imagePrompt?: string
  imageStatus?: ImageStatus

  // 새 필드 (스토리보드 품질 개선용)
  location?: string              // 장소
  timeOfDay?: string             // 시간대 (아침/낮/저녁/밤)
  time?: string                  // 구체적 시간 (HH:MM 또는 자유형식)
  sceneNumber?: number           // 장면 번호
  descriptionShort?: string      // 한 줄 요약 (50자)
  descriptionDetailed?: string   // 상세 설명 (300자)
  participantsDetailed?: Participant[] // 참여자 상세 정보 (역할 포함)
  keyDialogue?: string           // 핵심 대사/발언
  legalSignificance?: string     // 법적 의미
  evidenceItems?: string[]       // 관련 증거물
  mood?: string                  // 장면 분위기
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

// 음성 → 텍스트 응답
export interface TranscribeResponse {
  success: boolean
  text?: string
  error?: string
}

// 이미지 분석 응답
export interface AnalyzeImageResponse {
  success: boolean
  timeline: TimelineItem[]
  summary?: string
  error?: string
}

// 이미지 생성 요청 (스토리보드 스타일 고정)
export interface GenerateImageRequest {
  item_id: string
  title: string
  description: string
  participants: string[]
  // 확장 필드 (이미지 품질 향상용)
  location?: string
  time_of_day?: string
  participants_detailed?: Participant[]
  mood?: string
}

// 이미지 생성 응답
export interface GenerateImageResponse {
  success: boolean
  image_url?: string
  image_prompt?: string
  error?: string
}

// 일괄 이미지 생성 요청 (스토리보드 스타일 고정)
export interface GenerateImagesBatchRequest {
  items: TimelineItem[]
}

// 일괄 이미지 생성 응답
export interface GenerateImagesBatchResponse {
  success: boolean
  job_id?: string
  error?: string
}

// 영상 생성 요청
export interface GenerateVideoRequest {
  timeline_id: string
  image_urls: string[]
  duration_per_image?: number
  transition?: TransitionType
  transition_duration?: number
  resolution?: [number, number]
}

// 영상 생성 응답
export interface GenerateVideoResponse {
  success: boolean
  video_url?: string
  duration?: number
  image_count?: number
  error?: string
}

// 작업 상태 응답
export interface JobStatusResponse {
  job_id: string
  status: 'pending' | 'processing' | 'completed' | 'failed'
  progress: number
  current_step: number
  total_steps: number
  message: string
  result?: {
    generated: Array<{
      item_id: string
      image_url: string
      image_prompt: string
    }>
    failed: string[]
    total: number
    success_count: number
  }
  error?: string
}

// 편집 모드
export type EditMode = 'view' | 'edit'

// 영상 생성 설정
export interface VideoSettings {
  durationPerImage: number
  transition: TransitionType
  transitionDuration: number
  resolution: [number, number]
}

// 전환 효과 옵션
export const TRANSITION_OPTIONS: { value: TransitionType; label: string }[] = [
  { value: 'fade', label: '페이드' },
  { value: 'slide', label: '슬라이드' },
  { value: 'zoom', label: '줌' },
  { value: 'none', label: '없음' },
]
