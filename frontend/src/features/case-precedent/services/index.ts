import { api, endpoints } from '@/lib/api'
import type {
  PrecedentListResponse,
  PrecedentDetail,
  AIQuestionResponse,
  SearchFilters,
} from '../types'

export const casePrecedentService = {
  analyzeCase: async (description: string) => {
    const response = await api.post(`${endpoints.casePrecedent}/analyze`, {
      description,
    })
    return response.data
  },

  searchPrecedents: async (filters: SearchFilters): Promise<PrecedentListResponse> => {
    const params = new URLSearchParams({ keyword: filters.keyword })
    if (filters.docType) params.append('doc_type', filters.docType)
    if (filters.court) params.append('court', filters.court)
    if (filters.limit) params.append('limit', filters.limit.toString())

    const response = await api.get(`${endpoints.casePrecedent}/precedents?${params}`)
    return response.data
  },

  getPrecedentDetail: async (precedentId: string): Promise<PrecedentDetail> => {
    const response = await api.get(`${endpoints.casePrecedent}/precedents/${precedentId}`)
    return response.data
  },

  askAboutPrecedent: async (precedentId: string, question: string): Promise<AIQuestionResponse> => {
    const response = await api.post(`${endpoints.casePrecedent}/precedents/${precedentId}/ask`, {
      question,
    })
    return response.data
  },
}
