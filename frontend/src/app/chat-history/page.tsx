'use client'

import { useState, useEffect, useCallback } from 'react'
import { useRouter } from 'next/navigation'
import { useUI } from '@/context/UIContext'
import { useChat } from '@/context/ChatContext'
import { listConversations } from '@/features/workspace/services'
import type { ConversationListItem, PaginatedResponse } from '@/features/workspace/types'

const AGENT_LABELS: Record<string, string> = {
  legal_search: '판례/법령 검색',
  case_search: '판례 검색',
  law_search: '법령 검색',
  lawyer_finder: '변호사 찾기',
  small_claims: '소액소송',
  storyboard: '스토리보드',
  lawyer_stats: '변호사 통계',
  law_study: '로스쿨 학습',
  mock_trial: '모의 법정',
  content_marketing: '콘텐츠 마케팅',
  workspace: '워크스페이스',
  simple_chat: '일반 상담',
}

export default function ChatHistoryPage() {
  const router = useRouter()
  const { isChatOpen, chatMode } = useUI()
  const { setConversationId } = useChat()

  const [data, setData] = useState<PaginatedResponse<ConversationListItem> | null>(null)
  const [search, setSearch] = useState('')
  const [page, setPage] = useState(1)
  const [isLoading, setIsLoading] = useState(true)

  const fetchData = useCallback(async () => {
    setIsLoading(true)
    try {
      const result = await listConversations({
        search: search || undefined,
        page,
        page_size: 20,
      })
      setData(result)
    } catch (error) {
      console.error('대화 목록 조회 실패:', error)
    } finally {
      setIsLoading(false)
    }
  }, [search, page])

  useEffect(() => {
    fetchData()
  }, [fetchData])

  const handleContinue = (conversationId: string) => {
    setConversationId(conversationId)
    router.push('/')
  }

  const totalPages = data ? Math.ceil(data.total / data.page_size) : 0

  return (
    <div
      className={`h-screen flex flex-col bg-gray-50 transition-all duration-500 ${
        isChatOpen && chatMode === 'split' ? 'w-1/2' : 'w-full'
      }`}
    >
      {/* 헤더 */}
      <header className="bg-white border-b px-6 py-4">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-xl font-bold text-gray-900">대화 기록</h1>
          <p className="text-sm text-gray-500 mt-1">
            이전 법률 상담 대화를 검색하고 이어갈 수 있습니다
          </p>
        </div>
      </header>

      {/* 검색 */}
      <div className="bg-white border-b px-6 py-3">
        <div className="max-w-4xl mx-auto">
          <input
            type="text"
            placeholder="대화 제목 또는 내용 검색..."
            value={search}
            onChange={(e) => {
              setSearch(e.target.value)
              setPage(1)
            }}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
          />
        </div>
      </div>

      {/* 대화 목록 */}
      <div className="flex-1 overflow-y-auto px-6 py-4">
        <div className="max-w-4xl mx-auto space-y-3">
          {isLoading ? (
            <div className="text-center py-12 text-gray-500">
              <div className="animate-spin inline-block w-6 h-6 border-2 border-gray-300 border-t-blue-500 rounded-full" />
              <p className="mt-2 text-sm">대화 목록을 불러오는 중...</p>
            </div>
          ) : !data || data.items.length === 0 ? (
            <div className="text-center py-12 text-gray-500">
              <p className="text-lg">대화 기록이 없습니다</p>
              <p className="text-sm mt-1">채팅을 시작하면 여기에 기록됩니다</p>
            </div>
          ) : (
            data.items.map((conv) => (
              <div
                key={conv.id}
                className="bg-white rounded-lg border border-gray-200 p-4 hover:border-blue-300 hover:shadow-sm transition-all"
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1 min-w-0">
                    <h3 className="font-medium text-gray-900 truncate">
                      {conv.title || '제목 없는 대화'}
                    </h3>
                    <div className="flex items-center gap-3 mt-1.5 text-xs text-gray-500">
                      {conv.last_agent && (
                        <span className="inline-flex items-center px-2 py-0.5 rounded-full bg-blue-50 text-blue-700">
                          {AGENT_LABELS[conv.last_agent] || conv.last_agent}
                        </span>
                      )}
                      <span>{conv.message_count}개 메시지</span>
                      {conv.tag_count > 0 && (
                        <span>{conv.tag_count}개 태그</span>
                      )}
                      {conv.created_at && (
                        <span>
                          {new Date(conv.created_at).toLocaleDateString('ko-KR', {
                            year: 'numeric',
                            month: 'short',
                            day: 'numeric',
                          })}
                        </span>
                      )}
                    </div>
                  </div>
                  <button
                    onClick={() => handleContinue(conv.id)}
                    className="ml-4 px-3 py-1.5 text-sm font-medium text-blue-600 bg-blue-50 rounded-lg hover:bg-blue-100 transition-colors shrink-0"
                  >
                    이어가기
                  </button>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* 페이지네이션 */}
      {totalPages > 1 && (
        <div className="bg-white border-t px-6 py-3">
          <div className="max-w-4xl mx-auto flex items-center justify-between">
            <button
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page <= 1}
              className="px-3 py-1.5 text-sm border rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50"
            >
              이전
            </button>
            <span className="text-sm text-gray-600">
              {page} / {totalPages} 페이지 (총 {data?.total}건)
            </span>
            <button
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              disabled={page >= totalPages}
              className="px-3 py-1.5 text-sm border rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50"
            >
              다음
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
