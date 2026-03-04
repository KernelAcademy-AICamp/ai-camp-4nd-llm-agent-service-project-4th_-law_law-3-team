'use client'

import type { ReactNode } from 'react'
import type { FilteredLawItem, SortOrder } from '../types'

interface FilteredLawResultListProps {
  laws: FilteredLawItem[]
  total: number
  selectedId: string | null
  onSelect: (id: string) => void
  hasMore: boolean
  onLoadMore: () => void
  isLoading: boolean
  error: string | null
  hasSearched: boolean
  highlightKeyword?: string
  sortOrder?: SortOrder
  onSortChange?: (sort: SortOrder) => void
}

function highlightText(text: string, keyword: string): ReactNode {
  if (!keyword) return text
  const escaped = keyword.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const parts = text.split(new RegExp(`(${escaped})`, 'gi'))
  if (parts.length === 1) return text
  return parts.map((part, i) =>
    part.toLowerCase() === keyword.toLowerCase()
      ? <mark key={i} className="bg-yellow-200 text-yellow-900 rounded-sm px-0.5">{part}</mark>
      : part
  )
}

function getLawTypeColor(lawType: string | null): string {
  const colors: Record<string, string> = {
    '법률': 'bg-blue-100 text-blue-700',
    '시행령': 'bg-green-100 text-green-700',
    '시행규칙': 'bg-purple-100 text-purple-700',
    '대통령령': 'bg-yellow-100 text-yellow-700',
    '총리령': 'bg-orange-100 text-orange-700',
    '부령': 'bg-red-100 text-red-700',
  }
  return colors[lawType || ''] || 'bg-gray-100 text-gray-700'
}

export function FilteredLawResultList({
  laws,
  total,
  selectedId,
  onSelect,
  hasMore,
  onLoadMore,
  isLoading,
  error,
  hasSearched,
  highlightKeyword = '',
  sortOrder = 'relevance',
  onSortChange,
}: FilteredLawResultListProps) {
  if (error) {
    return (
      <div className="p-4 text-center text-red-500 text-sm">
        {error}
      </div>
    )
  }

  if (!hasSearched) {
    return (
      <div className="p-6 text-center text-gray-400">
        <svg className="w-12 h-12 mx-auto mb-3 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
        </svg>
        <p className="text-sm">필터를 설정하고 검색하세요</p>
      </div>
    )
  }

  if (laws.length === 0 && !isLoading) {
    return (
      <div className="p-6 text-center text-gray-400">
        <p className="text-sm">검색 결과가 없습니다</p>
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-y-auto">
      {/* 결과 수 + 정렬 */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-gray-100">
        <span className="text-xs text-gray-500">총 {total.toLocaleString()}건</span>
        {onSortChange && (
          <div className="flex gap-1 text-xs">
            <button
              onClick={() => onSortChange('relevance')}
              className={`px-2 py-0.5 rounded ${
                sortOrder === 'relevance'
                  ? 'text-blue-600 font-medium'
                  : 'text-gray-400 hover:text-gray-600'
              }`}
            >
              정확도순
            </button>
            <span className="text-gray-300">|</span>
            <button
              onClick={() => onSortChange('latest')}
              className={`px-2 py-0.5 rounded ${
                sortOrder === 'latest'
                  ? 'text-blue-600 font-medium'
                  : 'text-gray-400 hover:text-gray-600'
              }`}
            >
              최신순
            </button>
          </div>
        )}
      </div>

      {/* 결과 목록 */}
      <div className="space-y-1 p-2">
        {laws.map((item) => (
          <button
            key={item.id}
            onClick={() => onSelect(item.id)}
            className={`w-full text-left p-3 rounded-lg border transition-all ${
              selectedId === item.id
                ? 'border-blue-500 bg-blue-50 shadow-sm'
                : 'border-gray-200 bg-white hover:border-gray-300 hover:shadow-sm'
            }`}
          >
            <div className="flex items-start gap-2">
              {/* 법령유형 뱃지 */}
              {item.law_type && (
                <span className={`shrink-0 px-2 py-0.5 text-xs rounded-full ${getLawTypeColor(item.law_type)}`}>
                  {item.law_type}
                </span>
              )}
              <div className="flex-1 min-w-0">
                <h3 className="text-sm font-medium text-gray-900 line-clamp-1">
                  {highlightText(item.law_name, highlightKeyword)}
                </h3>
                <div className="flex items-center gap-2 mt-1">
                  {item.ministry && (
                    <span className="text-xs text-gray-500">{item.ministry}</span>
                  )}
                  {item.abbreviation && item.abbreviation !== item.law_name && (
                    <span className="text-xs text-gray-400">
                      ({highlightText(item.abbreviation, highlightKeyword)})
                    </span>
                  )}
                </div>
                {item.enforcement_date && (
                  <p className="text-xs text-gray-400 mt-0.5">시행일: {item.enforcement_date}</p>
                )}
                {item.ai_summary && (
                  <p className="text-xs text-gray-500 mt-1 line-clamp-2">
                    {item.ai_summary}
                  </p>
                )}
              </div>
            </div>
          </button>
        ))}
      </div>

      {/* 더 보기 */}
      {hasMore && (
        <div className="p-3">
          <button
            onClick={onLoadMore}
            disabled={isLoading}
            className="w-full py-2 text-sm text-blue-600 border border-blue-200 rounded-lg hover:bg-blue-50 disabled:opacity-50 transition-colors"
          >
            {isLoading ? '불러오는 중...' : '더 보기'}
          </button>
        </div>
      )}
    </div>
  )
}
