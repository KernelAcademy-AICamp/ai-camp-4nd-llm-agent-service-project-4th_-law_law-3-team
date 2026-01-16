import { api, endpoints } from '@/lib/api'

export const lawStudyService = {
  getStudyCases: async (subject?: string, difficulty?: string, limit?: number) => {
    const params = new URLSearchParams()
    if (subject) params.append('subject', subject)
    if (difficulty) params.append('difficulty', difficulty)
    if (limit) params.append('limit', limit.toString())

    const response = await api.get(`${endpoints.lawStudy}/cases?${params}`)
    return response.data
  },

  getCaseSummary: async (caseId: string) => {
    const response = await api.get(`${endpoints.lawStudy}/cases/${caseId}/summary`)
    return response.data
  },

  generateQuiz: async (subject: string, count?: number) => {
    const response = await api.post(`${endpoints.lawStudy}/quiz/generate`, {
      subject,
      count: count || 10,
    })
    return response.data
  },

  submitQuiz: async (quizId: string, answers: Record<string, string>) => {
    const response = await api.post(`${endpoints.lawStudy}/quiz/submit`, {
      quiz_id: quizId,
      answers,
    })
    return response.data
  },
}
