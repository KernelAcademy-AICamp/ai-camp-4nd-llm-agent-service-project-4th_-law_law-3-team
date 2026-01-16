import { api, endpoints } from '@/lib/api'

export const lawyerFinderService = {
  getNearbyLawyers: async (
    latitude: number,
    longitude: number,
    radius?: number,
    specialty?: string
  ) => {
    const params = new URLSearchParams({
      latitude: latitude.toString(),
      longitude: longitude.toString(),
      ...(radius && { radius: radius.toString() }),
      ...(specialty && { specialty }),
    })
    const response = await api.get(`${endpoints.lawyerFinder}/nearby?${params}`)
    return response.data
  },

  getLawyerDetail: async (lawyerId: number) => {
    const response = await api.get(`${endpoints.lawyerFinder}/${lawyerId}`)
    return response.data
  },
}
