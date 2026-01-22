import { api, endpoints } from '@/lib/api'
import type {
  NearbySearchResponse,
  SearchResponse,
  Lawyer,
  StatsResponse,
  CategoriesResponse,
} from '../types'

export const lawyerFinderService = {
  /**
   * 전문분야 12대분류 목록 조회
   */
  getCategories: async (): Promise<CategoriesResponse> => {
    const response = await api.get(`${endpoints.lawyerFinder}/categories`)
    return response.data
  },

  /**
   * 주변 변호사 검색
   */
  getNearbyLawyers: async (
    latitude: number,
    longitude: number,
    radius: number = 5000,
    limit: number = 50,
    category?: string
  ): Promise<NearbySearchResponse> => {
    const params = new URLSearchParams({
      latitude: latitude.toString(),
      longitude: longitude.toString(),
      radius: radius.toString(),
      limit: limit.toString(),
    })
    if (category) params.append('category', category)
    const response = await api.get(`${endpoints.lawyerFinder}/nearby?${params}`)
    return response.data
  },

  /**
   * 변호사 상세 정보 조회
   */
  getLawyerDetail: async (lawyerId: number): Promise<Lawyer> => {
    const response = await api.get(`${endpoints.lawyerFinder}/${lawyerId}`)
    return response.data
  },

  /**
   * 이름/사무소/지역/전문분야 카테고리로 검색 (선택적 위치 필터 포함)
   */
  searchLawyers: async (
    params: {
      name?: string
      office?: string
      district?: string
      category?: string
      latitude?: number
      longitude?: number
      radius?: number
      limit?: number
    }
  ): Promise<SearchResponse> => {
    const searchParams = new URLSearchParams()
    if (params.name) searchParams.append('name', params.name)
    if (params.office) searchParams.append('office', params.office)
    if (params.district) searchParams.append('district', params.district)
    if (params.category) searchParams.append('category', params.category)
    if (params.latitude !== undefined) searchParams.append('latitude', params.latitude.toString())
    if (params.longitude !== undefined) searchParams.append('longitude', params.longitude.toString())
    if (params.radius !== undefined) searchParams.append('radius', params.radius.toString())
    if (params.limit) searchParams.append('limit', params.limit.toString())

    const response = await api.get(`${endpoints.lawyerFinder}/search?${searchParams}`)
    return response.data
  },

  /**
   * 통계 정보 조회
   */
  getStats: async (): Promise<StatsResponse> => {
    const response = await api.get(`${endpoints.lawyerFinder}/stats`)
    return response.data
  },
}
