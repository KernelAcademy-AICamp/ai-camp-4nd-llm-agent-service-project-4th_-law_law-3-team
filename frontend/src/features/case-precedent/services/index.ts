import { api, endpoints } from '@/lib/api'
import type {
  PrecedentListResponse,
  PrecedentDetail,
  AIQuestionResponse,
  SearchFilters,
  StatuteSearchResponse,
  StatuteHierarchyResponse,
  StatuteChildrenResponse,
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

  // 법령 계층도 API
  searchStatutes: async (query: string, limit: number = 10): Promise<StatuteSearchResponse> => {
    const params = new URLSearchParams({ query, limit: limit.toString() })
    const response = await api.get(`${endpoints.casePrecedent}/statutes/search?${params}`)
    return response.data
  },

  getStatuteHierarchy: async (statuteId: string): Promise<StatuteHierarchyResponse> => {
    const response = await api.get(`${endpoints.casePrecedent}/statutes/hierarchy/${statuteId}`)
    return response.data
  },

  getStatuteChildren: async (statuteId: string, limit: number = 20): Promise<StatuteChildrenResponse> => {
    const params = new URLSearchParams({ limit: limit.toString() })
    const response = await api.get(`${endpoints.casePrecedent}/statutes/${statuteId}/children?${params}`)
    return response.data
  },

  getStatuteGraph: async (centerId?: string, limit: number = 100): Promise<{ nodes: GraphNode[]; links: GraphLink[] }> => {
    const params = new URLSearchParams({ limit: limit.toString() })
    if (centerId) params.append('center_id', centerId)
    const response = await api.get(`${endpoints.casePrecedent}/statutes/graph?${params}`)
    return response.data
  },
}

export interface GraphNode {
  id: string
  name: string
  type: string
  abbreviation?: string
  citation_count: number
}

export interface GraphLink {
  source: string
  target: string
  relation: string
}
