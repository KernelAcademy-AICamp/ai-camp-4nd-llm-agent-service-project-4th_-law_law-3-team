import { api, endpoints } from '@/lib/api'
import type {
  AnswerFeedbackRequest,
  AnswerFeedbackResponse,
  ExamContentResponse,
  ExamListResponse,
  ReferenceSearchRequest,
  ReferenceSearchResponse,
} from '../types'

export const lawStudyService = {
  /** 시험 문제 목록 조회 */
  getExamList: async (category?: string): Promise<ExamListResponse> => {
    const params = new URLSearchParams()
    if (category) params.append('category', category)
    const response = await api.get<ExamListResponse>(
      `${endpoints.lawStudy}/exams?${params}`,
    )
    return response.data
  },

  /** 시험 문제 전문 조회 */
  getExamContent: async (
    category: string,
    session: number,
  ): Promise<ExamContentResponse> => {
    const response = await api.get<ExamContentResponse>(
      `${endpoints.lawStudy}/exams/${category}/${session}`,
    )
    return response.data
  },

  /** 판례/법령 참조 검색 */
  searchReferences: async (
    body: ReferenceSearchRequest,
  ): Promise<ReferenceSearchResponse> => {
    const response = await api.post<ReferenceSearchResponse>(
      `${endpoints.lawStudy}/reference/search`,
      body,
    )
    return response.data
  },

  /** AI 답안 피드백 요청 */
  getAnswerFeedback: async (
    category: string,
    session: number,
    body: AnswerFeedbackRequest,
  ): Promise<AnswerFeedbackResponse> => {
    const response = await api.post<AnswerFeedbackResponse>(
      `${endpoints.lawStudy}/exams/${category}/${session}/feedback`,
      body,
    )
    return response.data
  },
}
