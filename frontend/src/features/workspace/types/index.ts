/**
 * 워크스페이스 타입 정의
 */

// ── 사건(Case) ──

export interface WorkspaceCase {
  id: string
  case_name: string
  case_type: string | null
  status: string
  tagged_items: TaggedItem[]
  created_at: string | null
  updated_at?: string | null
}

export interface WorkspaceCaseDetail extends WorkspaceCase {
  timeline: TimelineItem[]
  conversations: ConversationSummary[]
}

export interface CaseCreateRequest {
  case_name: string
  case_type?: string | null
  conversation_ids?: string[]
}

export interface CaseUpdateRequest {
  case_name?: string
  case_type?: string
  status?: string
}

// ── 타임라인 ──

export interface TimelineItem {
  id: string
  title: string
  description: string | null
  date_text: string | null
  date_normalized: string | null
  category: string | null
  source_type: string
  confidence: number
  source_conversation_id: string | null
  created_at: string | null
}

export interface TimelineRebuildRequest {
  include_conversations?: boolean
  include_manual?: boolean
}

export interface TimelineItemUpdateRequest {
  title?: string
  description?: string
  date_text?: string
  category?: string
}

// ── 대화(Conversation) ──

export interface ConversationListItem {
  id: string
  title: string | null
  case_type: string | null
  customer_name: string | null
  last_agent: string | null
  tag_count: number
  message_count: number
  created_at: string | null
  updated_at: string | null
}

export interface ConversationDetail {
  id: string
  thread_id: string
  title: string | null
  case_id: string | null
  case_type: string | null
  customer_name: string | null
  is_title_manual: boolean
  summary: Record<string, unknown> | null
  tagged_items: TaggedItem[]
  last_agent: string | null
  messages: ConversationMessage[]
  created_at: string | null
  updated_at: string | null
}

export interface ConversationMessage {
  role: 'user' | 'assistant'
  content: string
  agent_type: string | null
  created_at: string | null
}

export interface ConversationSummary {
  id: string
  title: string | null
  message_count: number
}

export interface ConversationUpdateRequest {
  title?: string
  case_id?: string
  customer_name?: string
}

// ── 태그 ──

export interface TaggedItem {
  type: string
  value: string
  label?: string
  confidence?: number
  source?: string
  [key: string]: unknown
}

// ── 페이지네이션 ──

export interface PaginatedResponse<T> {
  items: T[]
  total: number
  page: number
  page_size: number
}

// ── 태그 유형별 색상 ──

export const TAG_TYPE_COLORS: Record<string, string> = {
  date: 'bg-blue-100 text-blue-800',
  amount: 'bg-green-100 text-green-800',
  party: 'bg-purple-100 text-purple-800',
  evidence: 'bg-yellow-100 text-yellow-800',
  legal_term: 'bg-red-100 text-red-800',
  location: 'bg-orange-100 text-orange-800',
}
