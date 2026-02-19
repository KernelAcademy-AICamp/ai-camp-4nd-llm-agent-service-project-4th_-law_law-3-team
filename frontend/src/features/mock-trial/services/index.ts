/** 모의 법정 API 서비스 */

import { api, endpoints } from '@/lib/api'
import type { CaseType, EvidenceItem, StageInfo } from '../types'

interface CaseTypesResponse {
  case_types: {
    id: string
    name: string
    categories: { id: string; name: string; description: string }[]
  }[]
}

interface RolesResponse {
  case_type: string
  roles: { id: string; name: string; description: string }[]
}

interface SearchEvidenceResponse {
  cases: EvidenceItem[]
  articles: EvidenceItem[]
}

interface StageInfoResponse {
  case_type: string
  stages: StageInfo[]
}

export const mockTrialService = {
  getCaseTypes: async (): Promise<CaseTypesResponse> => {
    const response = await api.get(`${endpoints.mockTrial}/case-types`)
    return response.data as CaseTypesResponse
  },

  getRoles: async (caseType: CaseType): Promise<RolesResponse> => {
    const response = await api.get(`${endpoints.mockTrial}/roles/${caseType}`)
    return response.data as RolesResponse
  },

  searchEvidence: async (
    query: string,
    searchType: 'all' | 'cases' | 'articles' = 'all',
    limit: number = 5
  ): Promise<SearchEvidenceResponse> => {
    const response = await api.post(`${endpoints.mockTrial}/search-evidence`, {
      query,
      search_type: searchType,
      limit,
    })
    return response.data as SearchEvidenceResponse
  },

  getStageInfo: async (caseType: CaseType): Promise<StageInfoResponse> => {
    const response = await api.get(
      `${endpoints.mockTrial}/stage-info/${caseType}`
    )
    return response.data as StageInfoResponse
  },
}
