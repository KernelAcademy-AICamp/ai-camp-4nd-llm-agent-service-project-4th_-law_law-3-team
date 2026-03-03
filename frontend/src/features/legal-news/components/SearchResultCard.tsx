'use client'

import type { NewsSearchResult } from '../types'
import { formatDate } from '../utils/formatDate'
import { NewsSourceBadge } from './NewsSourceBadge'

interface SearchResultCardProps {
  result: NewsSearchResult
  onClick: (docId: string) => void
}

export function SearchResultCard({ result, onClick }: SearchResultCardProps) {
  return (
    <button
      onClick={() => onClick(result.doc_id)}
      className="w-full text-left p-4 rounded-lg border border-gray-200 bg-white hover:border-gray-300 hover:shadow-sm transition-colors"
    >
      <div className="flex items-center gap-2 mb-2">
        <NewsSourceBadge source={result.source} />
        <span className="text-xs text-gray-400">{formatDate(result.published_at)}</span>
        <span className="text-xs text-gray-400">{result.chunk_type}</span>
        {result.rerank_score !== null && (
          <span className="ml-auto text-xs font-mono text-blue-600">
            {result.rerank_score.toFixed(3)}
          </span>
        )}
      </div>

      <h3 className="text-sm font-semibold text-gray-900 mb-1 line-clamp-2">
        {result.title}
      </h3>

      <p className="text-xs text-gray-500 line-clamp-3">
        {result.chunk_text}
      </p>

      <div className="mt-2 text-xs text-gray-400">
        {result.publisher}
      </div>
    </button>
  )
}
