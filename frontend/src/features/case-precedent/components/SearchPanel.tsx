'use client'

import { useState, KeyboardEvent } from 'react'
import type { PrecedentItem, SearchFilters } from '../types'
import { CaseCard } from './CaseCard'

interface SearchPanelProps {
  results: PrecedentItem[]
  totalResults: number
  isSearching: boolean
  error: string | null
  filters: SearchFilters
  selectedCaseId: string | null
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
  onFilterChange,
  onSearch,
  onCaseSelect,
}: SearchPanelProps) {
  const [isFilterOpen, setIsFilterOpen] = useState(false)

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      onSearch()
    }
  }

  return (
    <div className="w-96 bg-white border-r border-gray-200 flex flex-col h-full">
      {/* Search Header */}
      <div className="p-4 border-b border-gray-200">
        <div className="relative">
          <input
            type="text"
            value={filters.keyword}
            onChange={(e) => onFilterChange({ keyword: e.target.value })}
            onKeyDown={handleKeyDown}
            placeholder="판례 검색어 입력..."
            className="w-full pl-10 pr-4 py-2.5 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
          <svg
            className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
            />
          </svg>
        </div>

        <div className="mt-3 flex gap-2">
          <button
            onClick={onSearch}
            disabled={isSearching || !filters.keyword.trim()}
            className="flex-1 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition"
          >
            {isSearching ? '검색 중...' : '검색'}
          </button>
          <button
            onClick={() => setIsFilterOpen(!isFilterOpen)}
            className={`px-3 py-2 border rounded-lg text-sm transition ${
              isFilterOpen
                ? 'border-blue-500 text-blue-600 bg-blue-50'
                : 'border-gray-300 text-gray-600 hover:bg-gray-50'
            }`}
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z"
              />
            </svg>
          </button>
        </div>

        {/* Filters */}
        {isFilterOpen && (
          <div className="mt-3 p-3 bg-gray-50 rounded-lg space-y-3">
            <div>
              <label className="block text-xs text-gray-500 mb-1">문서 유형</label>
              <select
                value={filters.docType || ''}
                onChange={(e) => onFilterChange({ docType: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {DOC_TYPE_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-xs text-gray-500 mb-1">법원</label>
              <select
                value={filters.court || ''}
                onChange={(e) => onFilterChange({ court: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {COURT_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </div>
          </div>
        )}
      </div>

      {/* Results Header */}
      <div className="px-4 py-2 bg-gray-50 border-b text-sm text-gray-600">
        {isSearching ? (
          <span>검색 중...</span>
        ) : results.length > 0 ? (
          <span>
            검색 결과 <strong className="text-gray-900">{totalResults}</strong>건
          </span>
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
              onClick={() => onCaseSelect(case_.id)}
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
    </div>
  )
}
