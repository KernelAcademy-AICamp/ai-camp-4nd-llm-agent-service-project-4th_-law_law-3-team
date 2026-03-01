'use client'

import { useState, useEffect, useCallback } from 'react'
import { useParams, useRouter } from 'next/navigation'
import { useUI } from '@/context/UIContext'
import { useChat } from '@/context/ChatContext'
import {
  getCase,
  updateCase,
  deleteCase,
  getTimeline,
  rebuildTimeline,
  exportCaseUrl,
} from '@/features/workspace/services'
import type {
  WorkspaceCaseDetail,
  TimelineItem,
} from '@/features/workspace/types'
import { TAG_TYPE_COLORS } from '@/features/workspace/types'

type Tab = 'summary' | 'tags' | 'timeline' | 'conversations'

const TAB_LABELS: Record<Tab, string> = {
  summary: '요약',
  tags: '태그',
  timeline: '타임라인',
  conversations: '대화',
}

const STATUS_LABELS: Record<string, { label: string; color: string }> = {
  open: { label: '진행 중', color: 'bg-green-100 text-green-800' },
  closed: { label: '종결', color: 'bg-gray-100 text-gray-600' },
  archived: { label: '보관', color: 'bg-yellow-100 text-yellow-800' },
}

export default function CaseDetailPage() {
  const params = useParams()
  const router = useRouter()
  const { isChatOpen, chatMode } = useUI()
  const { setConversationId, setCaseId: setChatCaseId } = useChat()

  const caseId = params.caseId as string

  const [caseData, setCaseData] = useState<WorkspaceCaseDetail | null>(null)
  const [timeline, setTimeline] = useState<TimelineItem[]>([])
  const [activeTab, setActiveTab] = useState<Tab>('summary')
  const [isLoading, setIsLoading] = useState(true)
  const [isEditing, setIsEditing] = useState(false)
  const [editName, setEditName] = useState('')
  const [isRebuilding, setIsRebuilding] = useState(false)

  const fetchCase = useCallback(async () => {
    setIsLoading(true)
    try {
      const data = await getCase(caseId)
      setCaseData(data)
      setTimeline(data.timeline ?? [])
    } catch (error) {
      console.error('사건 조회 실패:', error)
    } finally {
      setIsLoading(false)
    }
  }, [caseId])

  useEffect(() => {
    fetchCase()
  }, [fetchCase])

  const handleSaveName = async () => {
    if (!editName.trim() || !caseData) return
    try {
      const updated = await updateCase(caseId, { case_name: editName.trim() })
      setCaseData({ ...caseData, case_name: updated.case_name })
      setIsEditing(false)
    } catch (error) {
      console.error('사건 이름 수정 실패:', error)
    }
  }

  const handleDelete = async () => {
    if (!confirm('이 사건을 삭제하시겠습니까? 복구할 수 없습니다.')) return
    try {
      await deleteCase(caseId)
      router.push('/workspace')
    } catch (error) {
      console.error('사건 삭제 실패:', error)
    }
  }

  const handleStatusChange = async (status: string) => {
    if (!caseData) return
    try {
      const updated = await updateCase(caseId, { status })
      setCaseData({ ...caseData, status: updated.status })
    } catch (error) {
      console.error('상태 변경 실패:', error)
    }
  }

  const handleRebuildTimeline = async () => {
    setIsRebuilding(true)
    try {
      const result = await rebuildTimeline(caseId)
      setTimeline(result.items)
    } catch (error) {
      console.error('타임라인 재생성 실패:', error)
    } finally {
      setIsRebuilding(false)
    }
  }

  const handleLoadTimeline = async () => {
    try {
      const result = await getTimeline(caseId)
      setTimeline(result.items)
    } catch (error) {
      console.error('타임라인 로드 실패:', error)
    }
  }

  const handleContinueConversation = (conversationId: string) => {
    setConversationId(conversationId)
    setChatCaseId(caseId)
    router.push('/')
  }

  const handleOpenStoryboard = () => {
    setChatCaseId(caseId)
    router.push('/storyboard')
  }

  if (isLoading) {
    return (
      <div className="h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center text-gray-500">
          <div className="animate-spin inline-block w-6 h-6 border-2 border-gray-300 border-t-blue-500 rounded-full" />
          <p className="mt-2 text-sm">사건 정보를 불러오는 중...</p>
        </div>
      </div>
    )
  }

  if (!caseData) {
    return (
      <div className="h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center text-gray-500">
          <p className="text-lg">사건을 찾을 수 없습니다</p>
          <button
            onClick={() => router.push('/workspace')}
            className="mt-4 px-4 py-2 text-sm text-blue-600 hover:bg-blue-50 rounded-lg"
          >
            목록으로 돌아가기
          </button>
        </div>
      </div>
    )
  }

  return (
    <div
      className={`h-screen flex flex-col bg-gray-50 transition-all duration-500 ${
        isChatOpen && chatMode === 'split' ? 'w-1/2' : 'w-full'
      }`}
    >
      {/* 헤더 */}
      <header className="bg-white border-b px-6 py-4">
        <div className="max-w-5xl mx-auto">
          <div className="flex items-center gap-2 text-sm text-gray-500 mb-2">
            <button
              onClick={() => router.push('/workspace')}
              className="hover:text-blue-600 transition-colors"
            >
              워크스페이스
            </button>
            <span>/</span>
            <span className="text-gray-900">{caseData.case_name}</span>
          </div>
          <div className="flex items-start justify-between">
            <div className="flex-1 min-w-0">
              {isEditing ? (
                <div className="flex items-center gap-2">
                  <input
                    type="text"
                    value={editName}
                    onChange={(e) => setEditName(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleSaveName()}
                    autoFocus
                    className="text-xl font-bold text-gray-900 border-b-2 border-blue-500 outline-none bg-transparent"
                  />
                  <button
                    onClick={handleSaveName}
                    className="text-sm text-blue-600 hover:text-blue-800"
                  >
                    저장
                  </button>
                  <button
                    onClick={() => setIsEditing(false)}
                    className="text-sm text-gray-500 hover:text-gray-700"
                  >
                    취소
                  </button>
                </div>
              ) : (
                <h1
                  className="text-xl font-bold text-gray-900 cursor-pointer hover:text-blue-600 transition-colors"
                  onClick={() => {
                    setEditName(caseData.case_name)
                    setIsEditing(true)
                  }}
                  title="클릭하여 이름 수정"
                >
                  {caseData.case_name}
                </h1>
              )}
              <div className="flex items-center gap-3 mt-1.5 text-xs text-gray-500">
                {caseData.status && (
                  <span
                    className={`inline-flex items-center px-2 py-0.5 rounded-full font-medium ${
                      STATUS_LABELS[caseData.status]?.color ?? 'bg-gray-100 text-gray-600'
                    }`}
                  >
                    {STATUS_LABELS[caseData.status]?.label ?? caseData.status}
                  </span>
                )}
                {caseData.case_type && (
                  <span className="inline-flex items-center px-2 py-0.5 rounded-full bg-indigo-50 text-indigo-700">
                    {caseData.case_type}
                  </span>
                )}
                {caseData.created_at && (
                  <span>
                    생성:{' '}
                    {new Date(caseData.created_at).toLocaleDateString('ko-KR', {
                      year: 'numeric',
                      month: 'short',
                      day: 'numeric',
                    })}
                  </span>
                )}
              </div>
            </div>
            <div className="flex items-center gap-2 ml-4">
              <select
                value={caseData.status}
                onChange={(e) => handleStatusChange(e.target.value)}
                className="px-2 py-1.5 text-xs border rounded-lg bg-white"
              >
                <option value="open">진행 중</option>
                <option value="closed">종결</option>
                <option value="archived">보관</option>
              </select>
              <a
                href={exportCaseUrl(caseId)}
                download
                className="px-3 py-1.5 text-xs border rounded-lg hover:bg-gray-50 transition-colors"
              >
                내보내기
              </a>
              <button
                onClick={handleDelete}
                className="px-3 py-1.5 text-xs text-red-600 border border-red-200 rounded-lg hover:bg-red-50 transition-colors"
              >
                삭제
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* 탭 */}
      <div className="bg-white border-b px-6">
        <div className="max-w-5xl mx-auto flex gap-0">
          {(['summary', 'tags', 'timeline', 'conversations'] as Tab[]).map((tab) => (
            <button
              key={tab}
              onClick={() => {
                setActiveTab(tab)
                if (tab === 'timeline' && timeline.length === 0) {
                  handleLoadTimeline()
                }
              }}
              className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
                activeTab === tab
                  ? 'border-blue-600 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              {TAB_LABELS[tab]}
              {tab === 'tags' && caseData.tagged_items.length > 0 && (
                <span className="ml-1 text-xs text-gray-400">
                  ({caseData.tagged_items.length})
                </span>
              )}
              {tab === 'conversations' && caseData.conversations.length > 0 && (
                <span className="ml-1 text-xs text-gray-400">
                  ({caseData.conversations.length})
                </span>
              )}
              {tab === 'timeline' && timeline.length > 0 && (
                <span className="ml-1 text-xs text-gray-400">
                  ({timeline.length})
                </span>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* 탭 콘텐츠 */}
      <div className="flex-1 overflow-y-auto px-6 py-4">
        <div className="max-w-5xl mx-auto">
          {activeTab === 'summary' && (
            <SummaryTab caseData={caseData} timeline={timeline} />
          )}
          {activeTab === 'tags' && <TagsTab caseData={caseData} />}
          {activeTab === 'timeline' && (
            <TimelineTab
              timeline={timeline}
              isRebuilding={isRebuilding}
              onRebuild={handleRebuildTimeline}
              onOpenStoryboard={handleOpenStoryboard}
            />
          )}
          {activeTab === 'conversations' && (
            <ConversationsTab
              caseData={caseData}
              onContinue={handleContinueConversation}
            />
          )}
        </div>
      </div>
    </div>
  )
}

// ── 요약 탭 ──

function SummaryTab({
  caseData,
  timeline,
}: {
  caseData: WorkspaceCaseDetail
  timeline: TimelineItem[]
}) {
  const tagsByType: Record<string, number> = {}
  for (const tag of caseData.tagged_items) {
    tagsByType[tag.type] = (tagsByType[tag.type] || 0) + 1
  }

  return (
    <div className="space-y-6">
      {/* 통계 카드 */}
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-white rounded-lg border p-4">
          <p className="text-sm text-gray-500">대화</p>
          <p className="text-2xl font-bold text-gray-900 mt-1">
            {caseData.conversations.length}
          </p>
        </div>
        <div className="bg-white rounded-lg border p-4">
          <p className="text-sm text-gray-500">태그</p>
          <p className="text-2xl font-bold text-gray-900 mt-1">
            {caseData.tagged_items.length}
          </p>
        </div>
        <div className="bg-white rounded-lg border p-4">
          <p className="text-sm text-gray-500">타임라인</p>
          <p className="text-2xl font-bold text-gray-900 mt-1">
            {timeline.length}
          </p>
        </div>
      </div>

      {/* 태그 유형 분포 */}
      {Object.keys(tagsByType).length > 0 && (
        <div className="bg-white rounded-lg border p-4">
          <h3 className="text-sm font-medium text-gray-700 mb-3">태그 유형 분포</h3>
          <div className="flex flex-wrap gap-2">
            {Object.entries(tagsByType).map(([type, count]) => (
              <span
                key={type}
                className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium ${
                  TAG_TYPE_COLORS[type] ?? 'bg-gray-100 text-gray-700'
                }`}
              >
                {type}: {count}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* 최근 대화 */}
      {caseData.conversations.length > 0 && (
        <div className="bg-white rounded-lg border p-4">
          <h3 className="text-sm font-medium text-gray-700 mb-3">연결된 대화</h3>
          <div className="space-y-2">
            {caseData.conversations.map((conv) => (
              <div
                key={conv.id}
                className="flex items-center justify-between py-1.5 text-sm"
              >
                <span className="text-gray-900 truncate">
                  {conv.title || '제목 없는 대화'}
                </span>
                <span className="text-xs text-gray-500 shrink-0 ml-2">
                  {conv.message_count}개 메시지
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// ── 태그 탭 ──

function TagsTab({ caseData }: { caseData: WorkspaceCaseDetail }) {
  if (caseData.tagged_items.length === 0) {
    return (
      <div className="text-center py-12 text-gray-500">
        <p className="text-lg">추출된 태그가 없습니다</p>
        <p className="text-sm mt-1">대화를 시작하면 자동으로 태그가 수집됩니다</p>
      </div>
    )
  }

  const grouped: Record<string, typeof caseData.tagged_items> = {}
  for (const tag of caseData.tagged_items) {
    const type = tag.type || 'other'
    if (!grouped[type]) grouped[type] = []
    grouped[type].push(tag)
  }

  return (
    <div className="space-y-4">
      {Object.entries(grouped).map(([type, tags]) => (
        <div key={type} className="bg-white rounded-lg border p-4">
          <h3 className="text-sm font-medium text-gray-700 mb-3 flex items-center gap-2">
            <span
              className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs ${
                TAG_TYPE_COLORS[type] ?? 'bg-gray-100 text-gray-700'
              }`}
            >
              {type}
            </span>
            <span className="text-gray-400 text-xs">{tags.length}개</span>
          </h3>
          <div className="flex flex-wrap gap-2">
            {tags.map((tag, i) => (
              <div
                key={i}
                className="bg-gray-50 rounded-lg px-3 py-2 text-sm"
              >
                <span className="font-medium text-gray-900">
                  {tag.label || tag.value}
                </span>
                {tag.confidence !== undefined && tag.confidence < 1 && (
                  <span className="ml-1 text-xs text-gray-400">
                    ({Math.round(tag.confidence * 100)}%)
                  </span>
                )}
                {tag.source && (
                  <span className="ml-1 text-xs text-gray-400">
                    - {tag.source}
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}

// ── 타임라인 탭 ──

function TimelineTab({
  timeline,
  isRebuilding,
  onRebuild,
  onOpenStoryboard,
}: {
  timeline: TimelineItem[]
  isRebuilding: boolean
  onRebuild: () => void
  onOpenStoryboard: () => void
}) {
  const sorted = [...timeline].sort((a, b) => {
    if (a.date_normalized && b.date_normalized) {
      return a.date_normalized.localeCompare(b.date_normalized)
    }
    if (a.date_normalized) return -1
    if (b.date_normalized) return 1
    return 0
  })

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium text-gray-700">
          타임라인 ({timeline.length}개 항목)
        </h3>
        <div className="flex items-center gap-2">
          <button
            onClick={onOpenStoryboard}
            className="px-3 py-1.5 text-xs font-medium text-indigo-600 border border-indigo-200 rounded-lg hover:bg-indigo-50 transition-colors"
          >
            스토리보드 AI로 생성
          </button>
          <button
            onClick={onRebuild}
            disabled={isRebuilding}
            className="px-3 py-1.5 text-xs font-medium text-blue-600 border border-blue-200 rounded-lg hover:bg-blue-50 disabled:opacity-50 transition-colors"
          >
            {isRebuilding ? '재생성 중...' : '타임라인 재생성'}
          </button>
        </div>
      </div>

      {sorted.length === 0 ? (
        <div className="text-center py-12 text-gray-500">
          <p className="text-lg">타임라인 항목이 없습니다</p>
          <p className="text-sm mt-1">
            &ldquo;스토리보드 AI로 생성&rdquo;으로 대화 내용에서 타임라인을 추출하거나,
            &ldquo;타임라인 재생성&rdquo;으로 태그에서 타임라인을 만들어 보세요
          </p>
        </div>
      ) : (
        <div className="relative pl-6 space-y-4">
          {/* 세로 선 */}
          <div className="absolute left-2 top-0 bottom-0 w-0.5 bg-gray-200" />

          {sorted.map((item) => (
            <div key={item.id} className="relative">
              {/* 점 */}
              <div
                className={`absolute -left-4 top-1 w-3 h-3 rounded-full border-2 border-white ${
                  item.source_type === 'manual'
                    ? 'bg-blue-500'
                    : item.confidence >= 0.8
                      ? 'bg-green-500'
                      : item.confidence >= 0.5
                        ? 'bg-yellow-500'
                        : 'bg-red-500'
                }`}
              />
              <div className="bg-white rounded-lg border p-3">
                <div className="flex items-start justify-between">
                  <div>
                    <p className="font-medium text-gray-900 text-sm">
                      {item.title}
                    </p>
                    {item.description && (
                      <p className="text-xs text-gray-500 mt-1">
                        {item.description}
                      </p>
                    )}
                  </div>
                  <div className="text-right shrink-0 ml-4">
                    {item.date_text && (
                      <p className="text-xs font-medium text-gray-700">
                        {item.date_text}
                      </p>
                    )}
                    {item.category && (
                      <span className="inline-flex items-center px-1.5 py-0.5 rounded text-xs bg-gray-100 text-gray-600 mt-1">
                        {item.category}
                      </span>
                    )}
                  </div>
                </div>
                <div className="flex items-center gap-2 mt-2 text-xs text-gray-400">
                  <span>
                    신뢰도: {Math.round(item.confidence * 100)}%
                  </span>
                  <span>|</span>
                  <span>{item.source_type}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ── 대화 탭 ──

function ConversationsTab({
  caseData,
  onContinue,
}: {
  caseData: WorkspaceCaseDetail
  onContinue: (conversationId: string) => void
}) {
  if (caseData.conversations.length === 0) {
    return (
      <div className="text-center py-12 text-gray-500">
        <p className="text-lg">연결된 대화가 없습니다</p>
        <p className="text-sm mt-1">
          채팅에서 이 사건과 연결된 대화를 시작해보세요
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-3">
      {caseData.conversations.map((conv) => (
        <div
          key={conv.id}
          className="bg-white rounded-lg border border-gray-200 p-4 flex items-center justify-between"
        >
          <div>
            <h4 className="font-medium text-gray-900 text-sm">
              {conv.title || '제목 없는 대화'}
            </h4>
            <p className="text-xs text-gray-500 mt-1">
              {conv.message_count}개 메시지
            </p>
          </div>
          <button
            onClick={() => onContinue(conv.id)}
            className="px-3 py-1.5 text-sm font-medium text-blue-600 bg-blue-50 rounded-lg hover:bg-blue-100 transition-colors shrink-0"
          >
            이어가기
          </button>
        </div>
      ))}
    </div>
  )
}
