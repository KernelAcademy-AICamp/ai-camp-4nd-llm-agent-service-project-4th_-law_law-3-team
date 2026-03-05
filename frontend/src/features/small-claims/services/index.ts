import { api, endpoints } from '@/lib/api'
import type {
  EvidenceChecklistResponse,
  CaseInfo,
  DocumentResponse,
  DocumentRegenerateRequest,
  DocumentRegenerateResponse,
  RelatedCasesResponse,
  DocumentType,
  DisputeType,
  UploadedFile,
} from '../types'

export interface EvidenceUploadResponse {
  uploaded_files: UploadedFile[]
}

export const smallClaimsService = {
  uploadEvidence: async (
    files: File[],
    evidenceItemId: string,
    sessionId: string = 'default'
  ): Promise<EvidenceUploadResponse> => {
    const formData = new FormData()
    files.forEach((file) => formData.append('files', file))

    const response = await api.post(
      `${endpoints.smallClaims}/evidence/upload?evidence_item_id=${encodeURIComponent(evidenceItemId)}&session_id=${encodeURIComponent(sessionId)}`,
      formData,
      { headers: { 'Content-Type': 'multipart/form-data' } }
    )
    return response.data
  },

  getEvidenceChecklist: async (disputeType: DisputeType): Promise<EvidenceChecklistResponse> => {
    const response = await api.get(`${endpoints.smallClaims}/evidence-checklist/${disputeType}`)
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

  regenerateDocument: async (
    request: DocumentRegenerateRequest
  ): Promise<DocumentRegenerateResponse> => {
    const response = await api.post(`${endpoints.smallClaims}/regenerate-document`, request)
    return response.data
  },
}
