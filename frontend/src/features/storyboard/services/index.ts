import { api, endpoints } from '@/lib/api'
import type {
  AnalyzeImageResponse,
  ExtractTimelineResponse,
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

const VALID_PARTICIPANT_ROLES: ParticipantRole[] = [
  'victim', 'perpetrator', 'witness', 'bystander', 'authority', 'other',
]

const VALID_IMAGE_STATUSES: TimelineItem['imageStatus'][] = ['pending', 'processing', 'completed', 'failed']

function toStringOrUndefined(value: unknown): string | undefined {
  return typeof value === 'string' ? value : undefined
}

function toParticipantRole(value: unknown): ParticipantRole {
  if (typeof value === 'string' && (VALID_PARTICIPANT_ROLES as string[]).includes(value)) {
    return value as ParticipantRole
  }
  return 'other'
}

function toImageStatus(value: unknown): TimelineItem['imageStatus'] {
  if (typeof value === 'string' && (VALID_IMAGE_STATUSES as string[]).includes(value)) {
    return value as TimelineItem['imageStatus']
  }
  return undefined
}

/**
 * API 응답 참여자(snake_case)를 프론트엔드 형식(camelCase)으로 변환
 */
function transformParticipant(p: Record<string, unknown>): Participant {
  return {
    name: typeof p.name === 'string' ? p.name : '',
    role: toParticipantRole(p.role),
    action: toStringOrUndefined(p.action),
    emotion: toStringOrUndefined(p.emotion),
  }
}

/**
 * API 응답 타임라인 항목(snake_case)을 프론트엔드 형식(camelCase)으로 변환
 * null 값은 undefined로 변환
 */
function transformTimelineItem(item: Record<string, unknown>): TimelineItem {
  const rawParticipantsDetailed = item.participants_detailed
  const participantsDetailed = Array.isArray(rawParticipantsDetailed)
    ? rawParticipantsDetailed.filter((p): p is Record<string, unknown> => typeof p === 'object' && p !== null).map(transformParticipant)
    : undefined

  const rawEvidenceItems = item.evidence_items
  const evidenceItems = Array.isArray(rawEvidenceItems)
    ? rawEvidenceItems.filter((e): e is string => typeof e === 'string')
    : undefined

  const rawParticipants = item.participants
  const participants = Array.isArray(rawParticipants)
    ? rawParticipants.filter((p): p is string => typeof p === 'string')
    : []

  return {
    id: typeof item.id === 'string' ? item.id : '',
    date: typeof item.date === 'string' ? item.date : '날짜 미상',
    title: typeof item.title === 'string' ? item.title : '제목 없음',
    description: typeof item.description === 'string' ? item.description : '',
    participants,
    order: typeof item.order === 'number' ? item.order : 0,
    imageUrl: toStringOrUndefined(item.image_url),
    imagePrompt: toStringOrUndefined(item.image_prompt),
    imageStatus: toImageStatus(item.image_status),
    location: toStringOrUndefined(item.location),
    timeOfDay: toStringOrUndefined(item.time_of_day),
    time: toStringOrUndefined(item.time),
    sceneNumber: typeof item.scene_number === 'number' ? item.scene_number : undefined,
    descriptionShort: toStringOrUndefined(item.description_short),
    descriptionDetailed: toStringOrUndefined(item.description_detailed),
    participantsDetailed,
    keyDialogue: toStringOrUndefined(item.key_dialogue),
    legalSignificance: toStringOrUndefined(item.legal_significance),
    evidenceItems,
    mood: toStringOrUndefined(item.mood),
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
   * 프론트엔드 camelCase → 백엔드 snake_case 변환
   */
  generateImagesBatch: async (
    items: TimelineItem[]
  ): Promise<GenerateImagesBatchResponse> => {
    const snakeCaseItems = items.map((item) => ({
      id: item.id,
      date: item.date,
      title: item.title,
      description: item.description,
      participants: item.participants,
      order: item.order,
      image_url: item.imageUrl,
      image_prompt: item.imagePrompt,
      image_status: item.imageStatus,
      location: item.location,
      time_of_day: item.timeOfDay,
      time: item.time,
      scene_number: item.sceneNumber,
      description_short: item.descriptionShort,
      description_detailed: item.descriptionDetailed,
      participants_detailed: item.participantsDetailed?.map((p) => ({
        name: p.name,
        role: p.role,
        action: p.action,
        emotion: p.emotion,
      })),
      key_dialogue: item.keyDialogue,
      legal_significance: item.legalSignificance,
      evidence_items: item.evidenceItems,
      mood: item.mood,
    }))

    const response = await api.post<GenerateImagesBatchResponse>(
      `${BASE_URL}/generate-images-batch`,
      { items: snakeCaseItems }
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
