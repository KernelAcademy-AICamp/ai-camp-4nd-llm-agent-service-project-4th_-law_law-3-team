'use client'

import { useState, useEffect, useCallback } from 'react'
import { useRouter } from 'next/navigation'
import { BackButton } from '@/components/ui/BackButton'
import { useUI } from '@/context/UIContext'
import { listCases, createCase } from '@/features/workspace/services'
import type { WorkspaceCase, PaginatedResponse } from '@/features/workspace/types'
import { TAG_TYPE_COLORS } from '@/features/workspace/types'

const STATUS_LABELS: Record<string, { label: string; color: string }> = {
  open: { label: '진행 중', color: 'bg-green-100 text-green-800' },
  closed: { label: '종결', color: 'bg-gray-100 text-gray-600' },
  archived: { label: '보관', color: 'bg-yellow-100 text-yellow-800' },
}

export default function WorkspacePage() {
  const router = useRouter()
  const { isChatOpen, chatMode } = useUI()

  const [data, setData] = useState<PaginatedResponse<WorkspaceCase> | null>(null)
  const [search, setSearch] = useState('')
  const [statusFilter, setStatusFilter] = useState<string>('')
  const [page, setPage] = useState(1)
  const [isLoading, setIsLoading] = useState(true)
  const [isCreating, setIsCreating] = useState(false)
  const [newCaseName, setNewCaseName] = useState('')

  const fetchData = useCallback(async () => {
    setIsLoading(true)
    try {
      const result = await listCases({
        search: search || undefined,
        status: statusFilter || undefined,
        page,
        page_size: 20,
      })
      setData(result)
    } catch (error) {
      console.error('사건 목록 조회 실패:', error)
    } finally {
      setIsLoading(false)
    }
  }, [search, statusFilter, page])

  useEffect(() => {
    if (
      process.env.NODE_ENV === 'development' &&
      typeof window !== 'undefined' &&
      new URLSearchParams(window.location.search).get('demo') === '1'
    ) {
      const { getDemoCaseList } = require('@/features/workspace/demo/demo-data')
      setData(getDemoCaseList())
      setIsLoading(false)
    } else {
      fetchData()
    }
  }, [fetchData])

  const handleCreate = async () => {
    if (!newCaseName.trim()) return
    try {
      const created = await createCase({ case_name: newCaseName.trim() })
      setNewCaseName('')
      setIsCreating(false)
      router.push(`/workspace/${created.id}`)
    } catch (error) {
      console.error('사건 생성 실패:', error)
    }
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
        <div className="max-w-5xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <BackButton />
            <div>
              <h1 className="text-xl font-bold text-gray-900">사건 워크스페이스</h1>
              <p className="text-sm text-gray-500 mt-1">
                사건별 대화, 태그, 타임라인을 한곳에서 관리합니다
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {process.env.NODE_ENV === 'development' && (
              <button
                onClick={() => {
                  const { getDemoCaseList } = require('@/features/workspace/demo/demo-data')
                  setData(getDemoCaseList())
                  setIsLoading(false)
                }}
                className="px-3 py-2 text-xs font-medium text-gray-500 bg-gray-100 rounded-lg hover:bg-gray-200 transition-colors"
              >
                [DEV] 더미 데이터
              </button>
            )}
            <button
              onClick={() => setIsCreating(true)}
              className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 transition-colors"
            >
              + 새 사건
            </button>
          </div>
        </div>
      </header>

      {/* 사건 생성 모달 */}
      {isCreating && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30">
          <div className="bg-white rounded-xl shadow-lg p-6 w-full max-w-md mx-4">
            <h2 className="text-lg font-bold text-gray-900 mb-4">새 사건 만들기</h2>
            <input
              type="text"
              placeholder="사건명을 입력하세요"
              value={newCaseName}
              onChange={(e) => setNewCaseName(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleCreate()}
              autoFocus
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
            />
            <div className="flex justify-end gap-2 mt-4">
              <button
                onClick={() => {
                  setIsCreating(false)
                  setNewCaseName('')
                }}
                className="px-4 py-2 text-sm text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
              >
                취소
              </button>
              <button
                onClick={handleCreate}
                disabled={!newCaseName.trim()}
                className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                생성
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 필터 */}
      <div className="bg-white border-b px-6 py-3">
        <div className="max-w-5xl mx-auto flex gap-3">
          <input
            type="text"
            placeholder="사건명 검색..."
            value={search}
            onChange={(e) => {
              setSearch(e.target.value)
              setPage(1)
            }}
            className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
          />
          <select
            value={statusFilter}
            onChange={(e) => {
              setStatusFilter(e.target.value)
              setPage(1)
            }}
            className="px-3 py-2 border border-gray-300 rounded-lg text-sm bg-white"
          >
            <option value="">전체 상태</option>
            <option value="open">진행 중</option>
            <option value="closed">종결</option>
            <option value="archived">보관</option>
          </select>
        </div>
      </div>

      {/* 사건 목록 */}
      <div className="flex-1 overflow-y-auto px-6 py-4">
        <div className="max-w-5xl mx-auto space-y-3">
          {isLoading ? (
            <div className="text-center py-12 text-gray-500">
              <div className="animate-spin inline-block w-6 h-6 border-2 border-gray-300 border-t-blue-500 rounded-full" />
              <p className="mt-2 text-sm">사건 목록을 불러오는 중...</p>
            </div>
          ) : !data || data.items.length === 0 ? (
            <div className="text-center py-12 text-gray-500">
              <p className="text-lg">등록된 사건이 없습니다</p>
              <p className="text-sm mt-1">
                &ldquo;새 사건&rdquo; 버튼을 눌러 사건을 만들어 보세요
              </p>
            </div>
          ) : (
            data.items.map((c) => (
              <button
                key={c.id}
                onClick={() => router.push(
                  c.id.startsWith('demo-')
                    ? `/workspace/${c.id}?demo=1`
                    : `/workspace/${c.id}`
                )}
                className="w-full text-left bg-white rounded-lg border border-gray-200 p-4 hover:border-blue-300 hover:shadow-sm transition-all"
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <h3 className="font-medium text-gray-900 truncate">
                        {c.case_name}
                      </h3>
                      {c.status && (
                        <span
                          className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${
                            STATUS_LABELS[c.status]?.color ?? 'bg-gray-100 text-gray-600'
                          }`}
                        >
                          {STATUS_LABELS[c.status]?.label ?? c.status}
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-3 mt-2 text-xs text-gray-500">
                      {c.case_type && (
                        <span className="inline-flex items-center px-2 py-0.5 rounded-full bg-indigo-50 text-indigo-700">
                          {c.case_type}
                        </span>
                      )}
                      {c.tagged_items.length > 0 && (
                        <span>{c.tagged_items.length}개 태그</span>
                      )}
                      {c.created_at && (
                        <span>
                          {new Date(c.created_at).toLocaleDateString('ko-KR', {
                            year: 'numeric',
                            month: 'short',
                            day: 'numeric',
                          })}
                        </span>
                      )}
                    </div>
                    {/* 태그 프리뷰 */}
                    {c.tagged_items.length > 0 && (
                      <div className="flex flex-wrap gap-1 mt-2">
                        {c.tagged_items.slice(0, 5).map((tag, i) => (
                          <span
                            key={i}
                            className={`inline-flex items-center px-1.5 py-0.5 rounded text-xs ${
                              TAG_TYPE_COLORS[tag.type] ?? 'bg-gray-100 text-gray-700'
                            }`}
                          >
                            {tag.label || tag.value}
                          </span>
                        ))}
                        {c.tagged_items.length > 5 && (
                          <span className="text-xs text-gray-400">
                            +{c.tagged_items.length - 5}
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                  <span className="ml-4 text-gray-400 shrink-0">&rsaquo;</span>
                </div>
              </button>
            ))
          )}
        </div>
      </div>

      {/* 페이지네이션 */}
      {totalPages > 1 && (
        <div className="bg-white border-t px-6 py-3">
          <div className="max-w-5xl mx-auto flex items-center justify-between">
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
