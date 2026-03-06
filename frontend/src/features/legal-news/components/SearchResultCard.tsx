'use client'

import type { NewsSearchResult } from '../types'
import { formatDate } from '../utils/formatDate'
import { NewsSourceBadge } from './NewsSourceBadge'

interface SearchResultCardProps {
  result: NewsSearchResult
  onClick: (articleId: string) => void
}

export function SearchResultCard({ result, onClick }: SearchResultCardProps) {
  return (
    <button
      onClick={() => onClick(result.id)}
      className="w-full text-left p-4 rounded-lg border border-gray-200 bg-white hover:border-gray-300 hover:shadow-sm transition-colors"
    >
      <div className="flex items-center gap-2 mb-2">
        <NewsSourceBadge source={result.source} />
        <span className="text-xs text-gray-400">{formatDate(result.published_at)}</span>
        {result.section && (
          <span className="text-xs text-gray-400">{result.section}</span>
        )}
      </div>

      <h3 className="text-sm font-semibold text-gray-900 mb-1 line-clamp-2">
        {result.title}
      </h3>

      <p className="text-xs text-gray-500 line-clamp-3">
        {result.summary_one_liner}
      </p>

      {result.tags && result.tags.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1">
          {result.tags.slice(0, 5).map((tag) => (
            <span key={tag} className="text-xs px-1.5 py-0.5 bg-gray-100 text-gray-500 rounded">
              {tag}
            </span>
          ))}
        </div>
      )}

      <div className="mt-2 text-xs text-gray-400">
        {result.publisher}
      </div>
    </button>
  )
}
