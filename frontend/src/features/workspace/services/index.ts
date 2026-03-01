/**
 * 워크스페이스 API 서비스
 */

import { api, endpoints } from '@/lib/api'
import type {
  CaseCreateRequest,
  CaseUpdateRequest,
  ConversationDetail,
  ConversationListItem,
  ConversationUpdateRequest,
  PaginatedResponse,
  TimelineItem,
  TimelineItemUpdateRequest,
  TimelineRebuildRequest,
  WorkspaceCase,
  WorkspaceCaseDetail,
} from '../types'

// ── 사건 API ──

export async function createCase(data: CaseCreateRequest): Promise<WorkspaceCase> {
  const res = await api.post(`${endpoints.workspace}/cases`, data)
  return res.data
}

export async function listCases(params?: {
  status?: string
  search?: string
  page?: number
  page_size?: number
}): Promise<PaginatedResponse<WorkspaceCase>> {
  const res = await api.get(`${endpoints.workspace}/cases`, { params })
  return res.data
}

export async function getCase(caseId: string): Promise<WorkspaceCaseDetail> {
  const res = await api.get(`${endpoints.workspace}/cases/${caseId}`)
  return res.data
}

export async function updateCase(
  caseId: string,
  data: CaseUpdateRequest,
): Promise<WorkspaceCase> {
  const res = await api.patch(`${endpoints.workspace}/cases/${caseId}`, data)
  return res.data
}

export async function deleteCase(caseId: string): Promise<void> {
  await api.delete(`${endpoints.workspace}/cases/${caseId}`)
}

// ── 타임라인 API ──

export async function getTimeline(
  caseId: string,
): Promise<{ items: TimelineItem[]; total: number }> {
  const res = await api.get(`${endpoints.workspace}/cases/${caseId}/timeline`)
  return res.data
}

export async function rebuildTimeline(
  caseId: string,
  data?: TimelineRebuildRequest,
): Promise<{ items: TimelineItem[]; total: number }> {
  const res = await api.post(
    `${endpoints.workspace}/cases/${caseId}/timeline/rebuild`,
    data ?? {},
  )
  return res.data
}

export async function updateTimelineItem(
  caseId: string,
  itemId: string,
  data: TimelineItemUpdateRequest,
): Promise<TimelineItem> {
  const res = await api.patch(
    `${endpoints.workspace}/cases/${caseId}/timeline/items/${itemId}`,
    data,
  )
  return res.data
}

// ── 사건 내보내기 ──

export function exportCaseUrl(
  caseId: string,
  format: 'json' | 'txt' = 'json',
  include: 'all' | 'timeline' | 'conversations' = 'all',
): string {
  return `/api${endpoints.workspace}/cases/${caseId}/export?format=${format}&include=${include}`
}

// ── 대화 API ──

export async function listConversations(params?: {
  case_id?: string
  search?: string
  page?: number
  page_size?: number
}): Promise<PaginatedResponse<ConversationListItem>> {
  const res = await api.get(endpoints.chatConversations, { params })
  return res.data
}

export async function getConversation(
  conversationId: string,
): Promise<ConversationDetail> {
  const res = await api.get(`${endpoints.chatConversations}/${conversationId}`)
  return res.data
}

export async function updateConversation(
  conversationId: string,
  data: ConversationUpdateRequest,
): Promise<ConversationDetail> {
  const res = await api.patch(
    `${endpoints.chatConversations}/${conversationId}`,
    data,
  )
  return res.data
}
