'use client'

import { useCallback, useMemo, useState } from 'react'
import type { NewsArticleSummary, NewsSource } from '../types'
import { useNewsList } from '../hooks/useNewsList'
import { useNewsDetail } from '../hooks/useNewsDetail'
import { SOURCE_OPTIONS } from '../utils/constants'
import { NewsCard } from './NewsCard'
import { NewsDetailPanel } from './NewsDetailPanel'

type SortMode = 'latest' | 'oldest'

export function NewsListPanel() {
  const [sortMode, setSortMode] = useState<SortMode>('latest')
  const {
    items,
    total,
    hasNext,
    loading,
    error,
    filters,
    setFilters,
  } = useNewsList()

  const {
    selectedArticle,
    detailLoading,
    detailError,
    selectArticle,
    clearSelection,
  } = useNewsDetail()

  const handleSourceChange = useCallback((value: string) => {
    setFilters({ source: (value || null) as NewsSource | null, page: 1 })
  }, [setFilters])

  const handleDateChange = useCallback((date: string) => {
    setFilters({ published_date: date || null, page: 1 })
  }, [setFilters])

  const handlePageChange = useCallback((page: number) => {
    setFilters({ page })
  }, [setFilters])

  const sortedItems = useMemo(() => {
    if (sortMode === 'latest') return items
    return [...items].sort((a, b) => {
      if (!a.published_at && !b.published_at) return 0
      if (!a.published_at) return 1
      if (!b.published_at) return -1
      return new Date(a.published_at).getTime() - new Date(b.published_at).getTime()
    })
  }, [items, sortMode])

  const currentPage = filters.page
  const totalPages = Math.ceil(total / filters.page_size) || 1

  return (
    <div className="relative">
      {/* 필터 */}
      <div className="flex items-center gap-3 mb-4">
        <select
          value={filters.source ?? ''}
          onChange={(e) => handleSourceChange(e.target.value)}
          className="px-3 py-2 text-sm border border-gray-300 rounded-lg bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          {SOURCE_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>{opt.label}</option>
          ))}
        </select>

        <input
          type="date"
          value={filters.published_date ?? ''}
          onChange={(e) => handleDateChange(e.target.value)}
          className="px-3 py-2 text-sm border border-gray-300 rounded-lg bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
        />

        <div className="flex items-center gap-1 ml-auto">
          <button
            onClick={() => setSortMode('latest')}
            className={`px-2.5 py-1.5 text-xs font-medium rounded-l-md border transition-colors ${
              sortMode === 'latest'
                ? 'bg-blue-600 text-white border-blue-600'
                : 'bg-white text-gray-600 border-gray-300 hover:bg-gray-50'
            }`}
          >
            최신순
          </button>
          <button
            onClick={() => setSortMode('oldest')}
            className={`px-2.5 py-1.5 text-xs font-medium rounded-r-md border border-l-0 transition-colors ${
              sortMode === 'oldest'
                ? 'bg-blue-600 text-white border-blue-600'
                : 'bg-white text-gray-600 border-gray-300 hover:bg-gray-50'
            }`}
          >
            오래된순
          </button>
        </div>

        <span className="text-sm text-gray-500">
          총 {total.toLocaleString()}건
        </span>
      </div>

      {/* 로딩 */}
      {loading && (
        <div className="flex items-center justify-center h-40">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600" />
        </div>
      )}

      {/* 에러 */}
      {error && !loading && (
        <div className="text-sm text-red-600 bg-red-50 rounded-lg p-4 mb-4">
          {error}
        </div>
      )}

      {/* 카드 그리드 */}
      {!loading && !error && items.length > 0 && (
        <div className="flex flex-col gap-3 mb-6">
          {sortedItems.map((article: NewsArticleSummary) => (
            <NewsCard
              key={article.id}
              article={article}
              isSelected={selectedArticle?.id === article.id}
              onClick={selectArticle}
            />
          ))}
        </div>
      )}

      {/* 빈 상태 */}
      {!loading && !error && items.length === 0 && (
        <div className="text-center text-gray-500 py-12">
          <p className="text-lg mb-1">뉴스가 없습니다</p>
          <p className="text-sm">다른 필터 조건을 시도해보세요</p>
        </div>
      )}

      {/* 페이지네이션 */}
      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-3">
          <button
            onClick={() => handlePageChange(currentPage - 1)}
            disabled={currentPage <= 1}
            className="px-3 py-1.5 text-sm border border-gray-300 rounded-lg disabled:opacity-40 disabled:cursor-not-allowed hover:bg-gray-50 transition-colors"
          >
            이전
          </button>
          <span className="text-sm text-gray-600">
            {currentPage} / {totalPages}
          </span>
          <button
            onClick={() => handlePageChange(currentPage + 1)}
            disabled={!hasNext}
            className="px-3 py-1.5 text-sm border border-gray-300 rounded-lg disabled:opacity-40 disabled:cursor-not-allowed hover:bg-gray-50 transition-colors"
          >
            다음
          </button>
        </div>
      )}

      {/* 상세 패널 (Portal로 body에 렌더링) */}
      <NewsDetailPanel
        article={selectedArticle}
        loading={detailLoading}
        error={detailError}
        onClose={clearSelection}
      />
    </div>
  )
}
