'use client'

import { useState, useEffect, useCallback } from 'react'
import { useParams, useRouter, useSearchParams } from 'next/navigation'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { ArrowLeft } from 'lucide-react'
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
import { CaseSummaryTab } from '@/features/workspace/components/CaseSummaryTab'
import { CaseTagsTab } from '@/features/workspace/components/CaseTagsTab'
import { CaseTimelineTab } from '@/features/workspace/components/CaseTimelineTab'
import { CaseConversationsTab } from '@/features/workspace/components/CaseConversationsTab'

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
  const searchParams = useSearchParams()
  const queryClient = useQueryClient()
  const { isChatOpen, chatMode } = useUI()
  const { setConversationId, setCaseId: setChatCaseId } = useChat()

  const caseId = params.caseId as string
  const isDemoParam = process.env.NODE_ENV === 'development' && searchParams.get('demo') === '1'

  const [timeline, setTimeline] = useState<TimelineItem[]>([])
  const [activeTab, setActiveTab] = useState<Tab>('summary')
  const [isEditing, setIsEditing] = useState(false)
  const [editName, setEditName] = useState('')
  const [isRebuilding, setIsRebuilding] = useState(false)
  const [isDemoMode, setIsDemoMode] = useState(false)

  const { data: caseData, isLoading } = useQuery({
    queryKey: ['workspace', 'case', caseId, isDemoParam ? 'demo' : 'live'],
    queryFn: async () => {
      if (isDemoParam) {
        const { getDemoCaseDetail } = require('@/features/workspace/demo/demo-data')
        const demo = getDemoCaseDetail(caseId)
        if (demo) return demo as WorkspaceCaseDetail
      }
      return getCase(caseId)
    },
  })

  // caseData에서 timeline 초기화 + 데모 모드 동기화
  useEffect(() => {
    if (caseData) {
      setTimeline(caseData.timeline ?? [])
      setIsDemoMode(isDemoParam)
    }
  }, [caseData, isDemoParam])

  const loadDemoData = useCallback(() => {
    const { getDemoCaseDetail } = require('@/features/workspace/demo/demo-data')
    const demo = getDemoCaseDetail(caseId)
    if (demo) {
      queryClient.setQueryData(
        ['workspace', 'case', caseId, isDemoParam ? 'demo' : 'live'],
        demo as WorkspaceCaseDetail,
      )
      setIsDemoMode(true)
    }
  }, [caseId, queryClient, isDemoParam])

  const handleSaveName = async () => {
    if (!editName.trim() || !caseData) return
    try {
      const updated = await updateCase(caseId, { case_name: editName.trim() })
      queryClient.setQueryData(
        ['workspace', 'case', caseId, isDemoParam ? 'demo' : 'live'],
        { ...caseData, case_name: updated.case_name },
      )
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
      queryClient.setQueryData(
        ['workspace', 'case', caseId, isDemoParam ? 'demo' : 'live'],
        { ...caseData, status: updated.status },
      )
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
              onClick={() => router.push(isDemoMode ? '/workspace?demo=1' : '/workspace')}
              className="flex items-center gap-1 hover:text-blue-600 transition-colors"
            >
              <ArrowLeft size={14} />
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
              {process.env.NODE_ENV === 'development' && !isDemoMode && (
                <button
                  onClick={loadDemoData}
                  className="px-3 py-1.5 text-xs font-medium text-gray-500 bg-gray-100 rounded-lg hover:bg-gray-200 transition-colors"
                >
                  [DEV] 더미
                </button>
              )}
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
            <CaseSummaryTab caseData={caseData} timeline={timeline} />
          )}
          {activeTab === 'tags' && <CaseTagsTab caseData={caseData} />}
          {activeTab === 'timeline' && (
            <CaseTimelineTab
              timeline={timeline}
              isRebuilding={isRebuilding}
              onRebuild={handleRebuildTimeline}
              onOpenStoryboard={handleOpenStoryboard}
            />
          )}
          {activeTab === 'conversations' && (
            <CaseConversationsTab
              caseData={caseData}
              onContinue={handleContinueConversation}
            />
          )}
        </div>
      </div>
    </div>
  )
}
