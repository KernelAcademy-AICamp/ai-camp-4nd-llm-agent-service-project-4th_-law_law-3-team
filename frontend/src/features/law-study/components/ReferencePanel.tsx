'use client'

import { useState } from 'react'
import { Search, Loader2, Scale, BookOpen } from 'lucide-react'
import type { ReferenceSearchResult } from '../types'

interface ReferencePanelProps {
  results: ReferenceSearchResult[]
  isSearching: boolean
  onSearch: (query: string, docType?: string | null) => void
}

const DOC_TYPE_OPTIONS = [
  { value: null, label: '전체' },
  { value: 'precedent', label: '판례' },
  { value: 'law', label: '법령' },
] as const

const DOC_TYPE_BADGE: Record<string, { label: string; color: string; icon: typeof Scale }> = {
  precedent: { label: '판례', color: 'bg-purple-100 text-purple-700', icon: Scale },
  law: { label: '법령', color: 'bg-emerald-100 text-emerald-700', icon: BookOpen },
}

export function ReferencePanel({ results, isSearching, onSearch }: ReferencePanelProps) {
  const [query, setQuery] = useState('')
  const [docType, setDocType] = useState<string | null>(null)

  const handleSearch = () => {
    if (query.trim()) {
      onSearch(query.trim(), docType)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') handleSearch()
  }

  return (
    <div className="flex flex-col h-full">
      <div className="p-3 border-b border-gray-200">
        <h3 className="text-sm font-semibold text-gray-700 mb-2">판례/법령 검색</h3>

        {/* 검색 입력 */}
        <div className="flex gap-1.5">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="검색어 입력..."
            className="flex-1 px-3 py-1.5 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            onClick={handleSearch}
            disabled={!query.trim() || isSearching}
            className="px-3 py-1.5 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors"
          >
            {isSearching ? <Loader2 size={16} className="animate-spin" /> : <Search size={16} />}
          </button>
        </div>

        {/* 문서 유형 필터 */}
        <div className="flex gap-1 mt-2">
          {DOC_TYPE_OPTIONS.map((opt) => (
            <button
              key={opt.value ?? 'all'}
              onClick={() => setDocType(opt.value)}
              className={`px-2 py-1 text-xs rounded-full transition-colors ${
                docType === opt.value
                  ? 'bg-blue-100 text-blue-700 font-medium'
                  : 'bg-gray-100 text-gray-500 hover:bg-gray-200'
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {/* 검색 결과 */}
      <div className="flex-1 overflow-y-auto p-2 space-y-2">
        {results.length === 0 && !isSearching && (
          <div className="flex flex-col items-center justify-center py-12 text-gray-400 text-sm">
            <Search size={32} className="mb-2" />
            <p>판례·법령을 검색하세요</p>
          </div>
        )}

        {results.map((result) => {
          const badge = DOC_TYPE_BADGE[result.doc_type] ?? {
            label: result.doc_type,
            color: 'bg-gray-100 text-gray-600',
            icon: BookOpen,
          }
          const BadgeIcon = badge.icon
          return (
            <div
              key={result.id}
              className="p-3 border border-gray-200 rounded-lg hover:border-gray-300 transition-colors"
            >
              <div className="flex items-center gap-2 mb-1">
                <span className={`inline-flex items-center gap-1 px-1.5 py-0.5 text-xs rounded ${badge.color}`}>
                  <BadgeIcon size={10} />
                  {badge.label}
                </span>
                <span className="text-xs text-gray-400">
                  유사도 {(result.similarity * 100).toFixed(1)}%
                </span>
              </div>
              <h4 className="text-sm font-medium text-gray-900 line-clamp-2">
                {result.title}
              </h4>
              {result.case_number && (
                <p className="text-xs text-gray-500 mt-0.5">{result.case_number}</p>
              )}
              <p className="text-xs text-gray-600 mt-1 line-clamp-3">
                {result.summary}
              </p>
            </div>
          )
        })}
      </div>
    </div>
  )
}
