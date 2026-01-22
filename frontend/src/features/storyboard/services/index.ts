import { api, endpoints } from '@/lib/api'
import type {
  AnalyzeImageResponse,
  ExtractTimelineResponse,
  GenerateImageRequest,
  GenerateImageResponse,
  GenerateImagesBatchResponse,
  GenerateVideoRequest,
  GenerateVideoResponse,
  JobStatusResponse,
  Participant,
  ParticipantRole,
  TimelineData,
  TimelineItem,
  TranscribeResponse,
  ValidateTimelineResponse,
} from '../types'

const BASE_URL = endpoints.storyboard

/**
 * API 응답 참여자(snake_case)를 프론트엔드 형식(camelCase)으로 변환
 */
function transformParticipant(p: Record<string, unknown>): Participant {
  return {
    name: p.name as string,
    role: p.role as ParticipantRole,
    action: p.action as string | undefined,
    emotion: p.emotion as string | undefined,
  }
}

/**
 * API 응답 타임라인 항목(snake_case)을 프론트엔드 형식(camelCase)으로 변환
 * null 값은 undefined로 변환
 */
function transformTimelineItem(item: Record<string, unknown>): TimelineItem {
  const rawParticipantsDetailed = item.participants_detailed
  const participantsDetailed = Array.isArray(rawParticipantsDetailed)
    ? rawParticipantsDetailed.map(transformParticipant)
    : undefined

  const rawEvidenceItems = item.evidence_items
  const evidenceItems = Array.isArray(rawEvidenceItems) ? rawEvidenceItems as string[] : undefined

  const rawParticipants = item.participants
  const participants = Array.isArray(rawParticipants) ? rawParticipants as string[] : []

  return {
    id: (item.id as string) || '',
    date: (item.date as string) || '날짜 미상',
    title: (item.title as string) || '제목 없음',
    description: (item.description as string) || '',
    participants,
    order: typeof item.order === 'number' ? item.order : 0,
    imageUrl: item.image_url as string | undefined ?? undefined,
    imagePrompt: item.image_prompt as string | undefined ?? undefined,
    imageStatus: item.image_status as TimelineItem['imageStatus'] ?? undefined,
    location: item.location as string | undefined ?? undefined,
    timeOfDay: item.time_of_day as string | undefined ?? undefined,
    time: item.time as string | undefined ?? undefined,
    sceneNumber: typeof item.scene_number === 'number' ? item.scene_number : undefined,
    descriptionShort: item.description_short as string | undefined ?? undefined,
    descriptionDetailed: item.description_detailed as string | undefined ?? undefined,
    participantsDetailed,
    keyDialogue: item.key_dialogue as string | undefined ?? undefined,
    legalSignificance: item.legal_significance as string | undefined ?? undefined,
    evidenceItems,
    mood: item.mood as string | undefined ?? undefined,
  }
}

export const storyboardService = {
  /**
   * 텍스트에서 타임라인 자동 추출
   */
  extractTimeline: async (text: string): Promise<ExtractTimelineResponse> => {
    const response = await api.post<{ success: boolean; timeline: Record<string, unknown>[]; summary?: string }>(
      `${BASE_URL}/extract`,
      { text }
    )

    const { success, timeline, summary } = response.data

    if (!success || !timeline) {
      return { success: false, timeline: [], summary: undefined }
    }

    return {
      success,
      timeline: timeline.map(transformTimelineItem),
      summary,
    }
  },

  /**
   * JSON 데이터 유효성 검사
   */
  validateTimeline: async (
    timeline: TimelineData
  ): Promise<ValidateTimelineResponse> => {
    const response = await api.post<ValidateTimelineResponse>(
      `${BASE_URL}/validate`,
      { timeline }
    )
    return response.data
  },

  /**
   * 음성 파일을 텍스트로 변환 (STT)
   */
  transcribeAudio: async (
    audioFile: File,
    language: string = 'ko'
  ): Promise<TranscribeResponse> => {
    const formData = new FormData()
    formData.append('audio', audioFile)
    formData.append('language', language)

    const response = await api.post<TranscribeResponse>(
      `${BASE_URL}/transcribe`,
      formData,
      {
        headers: { 'Content-Type': 'multipart/form-data' },
      }
    )
    return response.data
  },

  /**
   * 이미지 분석을 통한 타임라인 추출
   */
  analyzeImage: async (
    imageFile: File,
    context: string = ''
  ): Promise<AnalyzeImageResponse> => {
    const formData = new FormData()
    formData.append('image', imageFile)
    formData.append('context', context)

    const response = await api.post<{ success: boolean; timeline: Record<string, unknown>[]; summary?: string; error?: string }>(
      `${BASE_URL}/analyze-image`,
      formData,
      {
        headers: { 'Content-Type': 'multipart/form-data' },
      }
    )

    const { success, timeline, summary, error } = response.data

    if (!success || !timeline) {
      return { success: false, timeline: [], summary: undefined, error }
    }

    return {
      success,
      timeline: timeline.map(transformTimelineItem),
      summary,
      error,
    }
  },

  /**
   * 타임라인 항목에 대한 스토리보드 이미지 생성
   * 확장 필드(장소, 시간대, 참여자 역할, 분위기)가 있으면 더 상세한 이미지를 생성합니다.
   */
  generateImage: async (item: TimelineItem): Promise<GenerateImageResponse> => {
    const participantsDetailedSnake = item.participantsDetailed?.map(p => ({
      name: p.name,
      role: p.role,
      action: p.action,
      emotion: p.emotion,
    }))

    const request = {
      item_id: item.id,
      title: item.title,
      description: item.descriptionDetailed || item.description,
      participants: item.participants,
      location: item.location,
      time_of_day: item.timeOfDay,
      participants_detailed: participantsDetailedSnake,
      mood: item.mood,
    }

    const response = await api.post<GenerateImageResponse>(
      `${BASE_URL}/generate-image`,
      request
    )
    return response.data
  },

  /**
   * 여러 타임라인 항목에 대한 스토리보드 이미지 일괄 생성
   */
  generateImagesBatch: async (
    items: TimelineItem[]
  ): Promise<GenerateImagesBatchResponse> => {
    const response = await api.post<GenerateImagesBatchResponse>(
      `${BASE_URL}/generate-images-batch`,
      { items }
    )
    return response.data
  },

  /**
   * 작업 상태 조회 (폴링)
   */
  getJobStatus: async (jobId: string): Promise<JobStatusResponse> => {
    const response = await api.get<JobStatusResponse>(
      `${BASE_URL}/jobs/${jobId}`
    )
    return response.data
  },


  /**
   * 이미지들을 결합하여 영상 생성
   */
  generateVideo: async (
    request: GenerateVideoRequest
  ): Promise<GenerateVideoResponse> => {
    const response = await api.post<GenerateVideoResponse>(
      `${BASE_URL}/generate-video`,
      request
    )
    return response.data
  },
}

/**
 * 폴링으로 작업 진행 상태 구독
 */
export function subscribeToJobStatus(
  jobId: string,
  onProgress: (status: JobStatusResponse) => void,
  onComplete: (result: JobStatusResponse) => void,
  onError: (error: string) => void
): () => void {
  let isActive = true
  const POLL_INTERVAL = 1000 // 1초마다 폴링

  const poll = async () => {
    if (!isActive) return

    try {
      const status = await storyboardService.getJobStatus(jobId)
      onProgress(status)

      if (status.status === 'completed') {
        onComplete(status)
        isActive = false
      } else if (status.status === 'failed') {
        onError(status.error || '작업 실패')
        isActive = false
      } else {
        // 계속 폴링
        setTimeout(poll, POLL_INTERVAL)
      }
    } catch (error) {
      console.error('Polling error:', error)
      onError('상태 조회 실패')
      isActive = false
    }
  }

  // 첫 폴링 시작
  poll()

  // 정리 함수 반환
  return () => {
    isActive = false
  }
}
