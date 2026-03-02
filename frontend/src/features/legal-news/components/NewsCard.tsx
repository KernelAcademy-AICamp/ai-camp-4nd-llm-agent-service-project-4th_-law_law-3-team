'use client'

import type { NewsArticleSummary } from '../types'
import { formatDate } from '../utils/formatDate'
import { NewsSourceBadge } from './NewsSourceBadge'
import { TagList } from './TagList'

interface NewsCardProps {
  article: NewsArticleSummary
  isSelected: boolean
  onClick: (id: string) => void
}

export function NewsCard({ article, isSelected, onClick }: NewsCardProps) {
  return (
    <button
      onClick={() => onClick(article.id)}
      className={`w-full text-left p-4 rounded-lg border transition-colors ${
        isSelected
          ? 'border-blue-500 bg-blue-50'
          : 'border-gray-200 bg-white hover:border-gray-300 hover:shadow-sm'
      }`}
    >
      <div className="flex items-center gap-2 mb-2">
        <NewsSourceBadge source={article.source} />
        <span className="text-xs text-gray-400">{formatDate(article.published_at)}</span>
        {article.section && (
          <span className="text-xs text-gray-400">{article.section}</span>
        )}
      </div>

      <h3 className="text-sm font-semibold text-gray-900 mb-1 line-clamp-2">
        {article.title}
      </h3>

      <p className="text-xs text-gray-500 mb-2 line-clamp-2">
        {article.summary_one_liner}
      </p>

      <div className="flex items-center justify-between">
        <TagList tags={article.tags} maxVisible={3} />
        <span className="text-xs text-gray-400 shrink-0 ml-2">{article.publisher}</span>
      </div>
    </button>
  )
}
