'use client'

import { useCallback } from 'react'
import { Search } from 'lucide-react'
import type { NewsSource } from '../types'
import { useNewsSearch } from '../hooks/useNewsSearch'
import { useNewsDetail } from '../hooks/useNewsDetail'
import { SOURCE_OPTIONS, LIMIT_OPTIONS } from '../utils/constants'
import { SearchResultCard } from './SearchResultCard'
import { NewsDetailPanel } from './NewsDetailPanel'

export function SearchPanel() {
  const {
    query,
    setQuery,
    sourceFilter,
    setSourceFilter,
    limit,
    setLimit,
    results,
    totalResults,
    searching,
    searchError,
    search,
  } = useNewsSearch()

  const {
    selectedArticle,
    detailLoading,
    detailError,
    selectArticle,
    clearSelection,
  } = useNewsDetail()

  const handleSubmit = useCallback((e: React.FormEvent) => {
    e.preventDefault()
    search()
  }, [search])

  const handleResultClick = useCallback((docId: string) => {
    selectArticle(docId)
  }, [selectArticle])

  return (
    <div className="relative">
      {/* 검색 폼 */}
      <form onSubmit={handleSubmit} className="mb-4">
        <div className="flex gap-2">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" size={16} />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="법률 뉴스 검색어를 입력하세요 (2자 이상)..."
              className="w-full pl-9 pr-4 py-2.5 text-sm border border-gray-300 rounded-lg bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <button
            type="submit"
            disabled={searching}
            className="px-4 py-2.5 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {searching ? '검색 중...' : '검색'}
          </button>
        </div>
      </form>

      {/* 필터 */}
      <div className="flex items-center gap-3 mb-4">
        <select
          value={sourceFilter ?? ''}
          onChange={(e) => setSourceFilter((e.target.value || null) as NewsSource | null)}
          className="px-3 py-2 text-sm border border-gray-300 rounded-lg bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          {SOURCE_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>{opt.label}</option>
          ))}
        </select>

        <select
          value={limit}
          onChange={(e) => setLimit(Number(e.target.value))}
          className="px-3 py-2 text-sm border border-gray-300 rounded-lg bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          {LIMIT_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>{opt.label}</option>
          ))}
        </select>

        {totalResults > 0 && (
          <span className="text-sm text-gray-500 ml-auto">
            검색 결과 {totalResults}건
          </span>
        )}
      </div>

      {/* 로딩 */}
      {searching && (
        <div className="flex items-center justify-center h-40">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600" />
        </div>
      )}

      {/* 에러 */}
      {searchError && !searching && (
        <div className="text-sm text-red-600 bg-red-50 rounded-lg p-4 mb-4">
          {searchError}
        </div>
      )}

      {/* 결과 목록 */}
      {!searching && !searchError && results.length > 0 && (
        <div className="space-y-3">
          {results.map((result) => (
            <SearchResultCard
              key={result.chunk_id}
              result={result}
              onClick={handleResultClick}
            />
          ))}
        </div>
      )}

      {/* 초기 상태 */}
      {!searching && !searchError && results.length === 0 && !query && (
        <div className="text-center text-gray-500 py-12">
          <Search className="mx-auto mb-3 text-gray-300" size={40} />
          <p className="text-lg mb-1">법률 뉴스를 검색해보세요</p>
          <p className="text-sm">하이브리드 검색 (Vector + FTS + 리랭커)</p>
        </div>
      )}

      {/* 검색했으나 결과 없음 */}
      {!searching && !searchError && results.length === 0 && query && totalResults === 0 && (
        <div className="text-center text-gray-500 py-12">
          <p className="text-lg mb-1">검색 결과가 없습니다</p>
          <p className="text-sm">다른 검색어를 시도해보세요</p>
        </div>
      )}

      {/* 상세 패널 */}
      <NewsDetailPanel
        article={selectedArticle}
        loading={detailLoading}
        error={detailError}
        onClose={clearSelection}
      />
    </div>
  )
}
