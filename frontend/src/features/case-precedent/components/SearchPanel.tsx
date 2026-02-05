'use client'

import { useState, useMemo, useCallback, KeyboardEvent } from 'react'
import type { PrecedentItem, PrecedentDetail, SearchFilters } from '../types'
import { CaseCard } from './CaseCard'

interface SearchPanelProps {
  results: PrecedentItem[]
  totalResults: number
  isSearching: boolean
  error: string | null
  filters: SearchFilters
  selectedCaseId: string | null
  selectedCase?: PrecedentDetail | null
  onFilterChange: (filters: Partial<SearchFilters>) => void
  onSearch: () => void
  onCaseSelect: (id: string) => void
}

const DOC_TYPE_OPTIONS = [
  { value: '', label: '전체 문서' },
  { value: 'precedent', label: '판례' },
  { value: 'constitutional', label: '헌재결정' },
]

const COURT_OPTIONS = [
  { value: '', label: '전체 법원' },
  { value: '대법원', label: '대법원' },
  { value: '고등법원', label: '고등법원' },
  { value: '지방법원', label: '지방법원' },
  { value: '헌법재판소', label: '헌법재판소' },
]

export function SearchPanel({
  results,
  totalResults,
  isSearching,
  error,
  filters,
  selectedCaseId,
  selectedCase,
  onFilterChange,
  onSearch,
  onCaseSelect,
}: SearchPanelProps) {
  const [isFilterOpen, setIsFilterOpen] = useState(false)
  const [isProvisionsOpen, setIsProvisionsOpen] = useState(true)

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      onSearch()
    }
  }

  // Memoized case select handler to prevent CaseCard re-renders
  const handleCaseSelect = useCallback((id: string) => {
    onCaseSelect(id)
  }, [onCaseSelect])

  const provisions = useMemo(() =>
    selectedCase?.reference_provisions
      ? selectedCase.reference_provisions
          .split(',')
          .map((s) => s.trim())
          .filter(Boolean)
      : []
  , [selectedCase?.reference_provisions])

  return (
    <div className="w-96 bg-white border-r border-gray-200 flex flex-col h-full">
      {/* Results Header */}
      <div className="px-4 py-2 bg-gray-50 border-b text-sm text-gray-600">
        {isSearching ? (
          <span>검색 중...</span>
        ) : results.length > 0 ? (
          <span>관련 문서</span>
        ) : filters.keyword ? (
          <span>검색 결과가 없습니다</span>
        ) : (
          <span>검색어를 입력하세요</span>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="px-4 py-3 bg-red-50 border-b border-red-100 text-sm text-red-600">
          {error}
        </div>
      )}

      {/* Results List */}
      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {isSearching ? (
          <div className="flex items-center justify-center h-32">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600" />
          </div>
        ) : results.length > 0 ? (
          results.map((case_) => (
            <CaseCard
              key={case_.id}
              case_={case_}
              selected={selectedCaseId === case_.id}
              onSelect={handleCaseSelect}
            />
          ))
        ) : filters.keyword ? (
          <div className="text-center text-gray-500 py-8">
            <svg
              className="w-12 h-12 mx-auto mb-3 text-gray-300"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            <p>검색 결과가 없습니다</p>
            <p className="text-xs mt-1">다른 키워드로 검색해보세요</p>
          </div>
        ) : (
          <div className="text-center text-gray-500 py-8">
            <svg
              className="w-12 h-12 mx-auto mb-3 text-gray-300"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
              />
            </svg>
            <p>검색어를 입력하여</p>
            <p>관련 판례를 찾아보세요</p>
          </div>
        )}
      </div>

      {/* 참조 조문 */}
      {provisions.length > 0 && (
        <div className="border-t border-gray-200">
          <button
            onClick={() => setIsProvisionsOpen(!isProvisionsOpen)}
            className="w-full flex items-center gap-2 px-4 py-3 hover:bg-gray-50 transition-colors"
          >
            <svg className="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25" />
            </svg>
            <span className="text-sm font-medium text-gray-700">참조 조문</span>
            <span className="text-xs text-gray-400">{provisions.length}</span>
            <svg
              className={`w-4 h-4 text-gray-400 ml-auto transition-transform ${isProvisionsOpen ? 'rotate-180' : ''}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
            </svg>
          </button>
          {isProvisionsOpen && (
            <ul className="px-4 pb-3 space-y-1 max-h-48 overflow-y-auto">
              {provisions.map((provision, idx) => (
                <li
                  key={idx}
                  className="text-sm text-gray-700 py-1.5 px-3 rounded hover:bg-gray-50 cursor-default"
                >
                  {provision}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  )
}
