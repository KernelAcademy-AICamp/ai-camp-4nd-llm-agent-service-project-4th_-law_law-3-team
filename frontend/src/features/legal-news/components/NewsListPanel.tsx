'use client'

import { useCallback, useEffect } from 'react'
import type { NewsArticleSummary, NewsSource } from '../types'
import { useNewsList } from '../hooks/useNewsList'
import { useNewsDetail } from '../hooks/useNewsDetail'
import { SOURCE_OPTIONS } from '../utils/constants'
import { NewsCard } from './NewsCard'
import { NewsDetailPanel } from './NewsDetailPanel'

export function NewsListPanel() {
  const {
    items,
    total,
    hasNext,
    loading,
    error,
    filters,
    setFilters,
    loadList,
  } = useNewsList()

  const {
    selectedArticle,
    detailLoading,
    detailError,
    selectArticle,
    clearSelection,
  } = useNewsDetail()

  useEffect(() => {
    loadList()
  }, [loadList])

  const handleSourceChange = useCallback((value: string) => {
    const newSource = (value || null) as NewsSource | null
    setFilters({ source: newSource, page: 1 })
    loadList({ source: newSource, page: 1, published_date: filters.published_date, page_size: filters.page_size })
  }, [setFilters, loadList, filters.published_date, filters.page_size])

  const handleDateChange = useCallback((date: string) => {
    const newDate = date || null
    setFilters({ published_date: newDate, page: 1 })
    loadList({ published_date: newDate, page: 1, source: filters.source, page_size: filters.page_size })
  }, [setFilters, loadList, filters.source, filters.page_size])

  const handlePageChange = useCallback((page: number) => {
    setFilters({ page })
    loadList({ ...filters, page })
  }, [setFilters, loadList, filters])

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

        <span className="text-sm text-gray-500 ml-auto">
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
          {items.map((article: NewsArticleSummary) => (
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

      {/* 상세 패널 */}
      <NewsDetailPanel
        article={selectedArticle}
        loading={detailLoading}
        error={detailError}
        onClose={clearSelection}
      />

      {/* 오버레이 */}
      {(selectedArticle || detailLoading || detailError) && (
        <div
          className="fixed inset-0 bg-black/20 z-40"
          onClick={clearSelection}
        />
      )}
    </div>
  )
}
