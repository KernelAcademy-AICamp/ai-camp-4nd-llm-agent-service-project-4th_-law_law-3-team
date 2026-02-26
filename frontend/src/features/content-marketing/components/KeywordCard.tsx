'use client'

import type { KeywordItem } from '../types'

interface ScoreBarProps {
  label: string
  value: number
  color: string
}

function ScoreBar({ label, value, color }: ScoreBarProps) {
  const percentage = Math.round(value * 100)
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-16 text-gray-500 shrink-0">{label}</span>
      <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full ${color}`}
          style={{ width: `${percentage}%` }}
        />
      </div>
      <span className="w-8 text-right text-gray-600 font-mono">{percentage}</span>
    </div>
  )
}

interface KeywordCardProps {
  keyword: KeywordItem
  onSearchNews: (keyword: KeywordItem) => void
}

export function KeywordCard({ keyword, onSearchNews }: KeywordCardProps) {
  return (
    <div className="bg-white border border-gray-200 rounded-xl p-4 hover:shadow-md transition-shadow">
      {/* 헤더 */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="flex items-center justify-center w-6 h-6 bg-blue-100 text-blue-700 text-xs font-bold rounded-full">
            {keyword.rank}
          </span>
          <h3 className="font-semibold text-gray-900">{keyword.keyword}</h3>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="text-lg font-bold text-blue-600">
            {keyword.total_score.toFixed(1)}
          </span>
          <span className="text-xs text-gray-400">점</span>
        </div>
      </div>

      {/* 맥락 */}
      {keyword.context && (
        <p className="text-sm text-gray-600 mb-3 line-clamp-2">{keyword.context}</p>
      )}

      {/* 4차원 점수 바 */}
      <div className="space-y-1.5 mb-3">
        <ScoreBar label="바이럴" value={keyword.scores.virality} color="bg-orange-400" />
        <ScoreBar label="사회영향" value={keyword.scores.social_impact} color="bg-red-400" />
        <ScoreBar label="법적연관" value={keyword.scores.legal_relevance} color="bg-blue-500" />
        <ScoreBar label="콘텐츠" value={keyword.scores.content_fitness} color="bg-green-400" />
      </div>

      {/* 점수 근거 */}
      {keyword.score_reason && (
        <p className="text-xs text-gray-500 mb-3 bg-gray-50 rounded-lg px-3 py-2">
          {keyword.score_reason}
        </p>
      )}

      {/* 뉴스 검색 버튼 */}
      <button
        onClick={() => onSearchNews(keyword)}
        className="w-full py-2 px-4 bg-blue-50 text-blue-700 text-sm font-medium rounded-lg hover:bg-blue-100 transition-colors"
      >
        뉴스 검색
      </button>
    </div>
  )
}
