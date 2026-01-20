import { api, endpoints } from '@/lib/api'
import type {
  ExtractTimelineResponse,
  TimelineData,
  ValidateTimelineResponse,
} from '../types'

export const storyboardService = {
  /**
   * 텍스트에서 타임라인 자동 추출
   */
  extractTimeline: async (text: string): Promise<ExtractTimelineResponse> => {
    const response = await api.post<ExtractTimelineResponse>(
      `${endpoints.storyboard}/extract`,
      { text }
    )
    return response.data
  },

  /**
   * JSON 데이터 유효성 검사
   */
  validateTimeline: async (
    timeline: TimelineData
  ): Promise<ValidateTimelineResponse> => {
    const response = await api.post<ValidateTimelineResponse>(
      `${endpoints.storyboard}/validate`,
      { timeline }
    )
    return response.data
  },
}
