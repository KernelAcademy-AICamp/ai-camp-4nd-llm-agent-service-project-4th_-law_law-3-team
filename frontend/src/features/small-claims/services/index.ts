import { api, endpoints } from '@/lib/api'
import type {
  EvidenceChecklistResponse,
  CaseInfo,
  DocumentResponse,
  RelatedCasesResponse,
  DocumentType,
  DisputeType,
} from '../types'

export const smallClaimsService = {
  startInterview: async (caseType: string) => {
    const response = await api.post(`${endpoints.smallClaims}/interview/start`, {
      case_type: caseType,
    })
    return response.data
  },

  submitAnswer: async (sessionId: string, answer: string) => {
    const response = await api.post(
      `${endpoints.smallClaims}/interview/${sessionId}/answer`,
      { answer }
    )
    return response.data
  },

  generateDocuments: async (sessionId: string, documentTypes: string[]) => {
    const response = await api.post(`${endpoints.smallClaims}/documents/generate`, {
      session_id: sessionId,
      document_types: documentTypes,
    })
    return response.data
  },

  uploadEvidence: async (sessionId: string, files: File[], evidenceType: string) => {
    const formData = new FormData()
    formData.append('session_id', sessionId)
    formData.append('evidence_type', evidenceType)
    files.forEach((file) => formData.append('files', file))

    const response = await api.post(`${endpoints.smallClaims}/evidence/upload`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    return response.data
  },

  organizeEvidence: async (sessionId: string) => {
    const response = await api.post(`${endpoints.smallClaims}/evidence/${sessionId}/organize`)
    return response.data
  },

  getLawsuitGuide: async (caseType: string) => {
    const response = await api.get(`${endpoints.smallClaims}/guide/${caseType}`)
    return response.data
  },

  // New endpoints
  getEvidenceChecklist: async (disputeType: DisputeType): Promise<EvidenceChecklistResponse> => {
    const response = await api.get(`${endpoints.smallClaims}/evidence-checklist/${disputeType}`)
    return response.data
  },

  getDisputeTypes: async () => {
    const response = await api.get(`${endpoints.smallClaims}/dispute-types`)
    return response.data
  },

  generateDocument: async (
    documentType: DocumentType,
    caseInfo: CaseInfo
  ): Promise<DocumentResponse> => {
    const response = await api.post(`${endpoints.smallClaims}/generate-document`, {
      document_type: documentType,
      case_info: caseInfo,
    })
    return response.data
  },

  getRelatedCases: async (disputeType: DisputeType): Promise<RelatedCasesResponse> => {
    const response = await api.get(`${endpoints.smallClaims}/related-cases/${disputeType}`)
    return response.data
  },
}
