import { api, endpoints } from '@/lib/api'

export const casePrecedentService = {
  analyzeCase: async (description: string) => {
    const response = await api.post(`${endpoints.casePrecedent}/analyze`, {
      description,
    })
    return response.data
  },

  searchPrecedents: async (keyword: string, category?: string, limit?: number) => {
    const params = new URLSearchParams({ keyword })
    if (category) params.append('category', category)
    if (limit) params.append('limit', limit.toString())

    const response = await api.get(`${endpoints.casePrecedent}/precedents?${params}`)
    return response.data
  },

  getPrecedentDetail: async (precedentId: string) => {
    const response = await api.get(`${endpoints.casePrecedent}/precedents/${precedentId}`)
    return response.data
  },
}
