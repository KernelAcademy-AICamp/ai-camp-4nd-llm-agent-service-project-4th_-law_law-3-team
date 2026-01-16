import { api, endpoints } from '@/lib/api'

export const reviewPriceService = {
  getLawyerReviews: async (lawyerId: number, sortBy?: string, limit?: number) => {
    const params = new URLSearchParams()
    if (sortBy) params.append('sort_by', sortBy)
    if (limit) params.append('limit', limit.toString())

    const response = await api.get(
      `${endpoints.reviewPrice}/lawyers/${lawyerId}/reviews?${params}`
    )
    return response.data
  },

  createReview: async (
    lawyerId: number,
    rating: number,
    content: string,
    price: number
  ) => {
    const response = await api.post(
      `${endpoints.reviewPrice}/lawyers/${lawyerId}/reviews`,
      { rating, content, price }
    )
    return response.data
  },

  getPriceComparison: async (category: string, region?: string) => {
    const params = new URLSearchParams({ category })
    if (region) params.append('region', region)

    const response = await api.get(`${endpoints.reviewPrice}/price-comparison?${params}`)
    return response.data
  },
}
