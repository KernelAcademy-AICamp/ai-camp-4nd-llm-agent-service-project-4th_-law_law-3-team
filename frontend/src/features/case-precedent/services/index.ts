import { api, endpoints } from '@/lib/api'
import type {
  PrecedentListResponse,
  PrecedentDetail,
  AIQuestionResponse,
  SearchFilters,
  LawFullText,
  CitingCasesResponse,
  FilteredPrecedentListResponse,
  FilteredLawListResponse,
  LawFilterOptions,
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

  // 법령 전문 조회 API
  getLawFullText: async (lawId: string): Promise<LawFullText> => {
    const response = await api.get(`${endpoints.casePrecedent}/laws/${lawId}/full-text`)
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

  getCitingCases: async (statuteId: string, limit: number = 10): Promise<CitingCasesResponse> => {
    const params = new URLSearchParams({ limit: limit.toString() })
    const response = await api.get(`${endpoints.casePrecedent}/statutes/${statuteId}/citing-cases?${params}`)
    return response.data
  },

  // 판례 필터 검색 API (PostgreSQL 직접 쿼리)
  filterPrecedents: async (params: {
    keyword?: string
    case_type?: string
    date_from?: string
    date_to?: string
    sort?: string
    offset?: number
    limit?: number
  }, signal?: AbortSignal): Promise<FilteredPrecedentListResponse> => {
    const searchParams = new URLSearchParams()
    if (params.keyword) searchParams.append('keyword', params.keyword)
    if (params.case_type) searchParams.append('case_type', params.case_type)
    if (params.date_from) searchParams.append('date_from', params.date_from)
    if (params.date_to) searchParams.append('date_to', params.date_to)
    if (params.sort) searchParams.append('sort', params.sort)
    if (params.offset !== undefined) searchParams.append('offset', params.offset.toString())
    if (params.limit !== undefined) searchParams.append('limit', params.limit.toString())
    const response = await api.get(`${endpoints.casePrecedent}/precedents/filter?${searchParams}`, { signal })
    return response.data
  },

  getCaseTypes: async (): Promise<string[]> => {
    const response = await api.get(`${endpoints.casePrecedent}/precedents/case-types`)
    return response.data.case_types
  },

  getStatuteGraph: async (centerId?: string, limit: number = 100): Promise<{ nodes: GraphNode[]; links: GraphLink[] }> => {
    const params = new URLSearchParams({ limit: limit.toString() })
    if (centerId) params.append('center_id', centerId)
    const response = await api.get(`${endpoints.casePrecedent}/statutes/graph?${params}`)
    return response.data
  },

  // 법령 필터 검색 API (PostgreSQL 직접 쿼리, BM25+ILIKE)
  filterLaws: async (params: {
    keyword?: string
    law_type?: string
    ministry?: string
    promulgation_from?: string
    promulgation_to?: string
    enforcement_from?: string
    enforcement_to?: string
    sort?: string
    offset?: number
    limit?: number
  }, signal?: AbortSignal): Promise<FilteredLawListResponse> => {
    const searchParams = new URLSearchParams()
    if (params.keyword) searchParams.append('keyword', params.keyword)
    if (params.law_type) searchParams.append('law_type', params.law_type)
    if (params.ministry) searchParams.append('ministry', params.ministry)
    if (params.promulgation_from) searchParams.append('promulgation_from', params.promulgation_from)
    if (params.promulgation_to) searchParams.append('promulgation_to', params.promulgation_to)
    if (params.enforcement_from) searchParams.append('enforcement_from', params.enforcement_from)
    if (params.enforcement_to) searchParams.append('enforcement_to', params.enforcement_to)
    if (params.sort) searchParams.append('sort', params.sort)
    if (params.offset !== undefined) searchParams.append('offset', params.offset.toString())
    if (params.limit !== undefined) searchParams.append('limit', params.limit.toString())
    const response = await api.get(`${endpoints.casePrecedent}/laws/filter?${searchParams}`, { signal })
    return response.data
  },

  getLawFilterOptions: async (): Promise<LawFilterOptions> => {
    const response = await api.get(`${endpoints.casePrecedent}/laws/filter-options`)
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
