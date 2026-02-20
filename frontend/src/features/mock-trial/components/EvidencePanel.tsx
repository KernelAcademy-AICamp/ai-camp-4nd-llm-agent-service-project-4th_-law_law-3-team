'use client'

import type { EvidenceItem } from '../types'

interface EvidencePanelProps {
  cases: EvidenceItem[]
  articles: EvidenceItem[]
  selectedIds: Set<string>
  onToggle: (id: string) => void
  onSubmit: () => void
  isLoading: boolean
}

function EvidenceCard({
  item,
  isSelected,
  onToggle,
}: {
  item: EvidenceItem
  isSelected: boolean
  onToggle: () => void
}) {
  const scorePercent = Math.round(item.relevance_score * 100)

  return (
    <button
      onClick={onToggle}
      className={`w-full text-left p-3 rounded-lg border-2 transition-colors ${
        isSelected
          ? 'border-blue-500 bg-blue-50'
          : 'border-gray-200 hover:border-gray-300'
      }`}
    >
      <div className="flex items-start justify-between">
        <h4 className="text-sm font-medium text-gray-800 flex-1 pr-2">
          {item.title}
        </h4>
        <span className="text-xs px-2 py-0.5 rounded-full bg-gray-100 text-gray-500 whitespace-nowrap">
          {scorePercent}%
        </span>
      </div>
      <p className="text-xs text-gray-500 mt-1 line-clamp-2">{item.summary}</p>
    </button>
  )
}

export function EvidencePanel({
  cases,
  articles,
  selectedIds,
  onToggle,
  onSubmit,
  isLoading,
}: EvidencePanelProps) {
  if (isLoading) {
    return (
      <div className="p-4 text-center text-sm text-gray-400 animate-pulse">
        증거 검색 중...
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full overflow-y-auto">
      {/* 판례 */}
      {cases.length > 0 && (
        <div className="p-3">
          <h3 className="text-xs font-semibold text-gray-500 uppercase mb-2">
            관련 판례 ({cases.length}건)
          </h3>
          <div className="space-y-2">
            {cases.map((item) => (
              <EvidenceCard
                key={item.id}
                item={item}
                isSelected={selectedIds.has(item.id)}
                onToggle={() => onToggle(item.id)}
              />
            ))}
          </div>
        </div>
      )}

      {/* 법령 */}
      {articles.length > 0 && (
        <div className="p-3">
          <h3 className="text-xs font-semibold text-gray-500 uppercase mb-2">
            관련 법령 ({articles.length}건)
          </h3>
          <div className="space-y-2">
            {articles.map((item) => (
              <EvidenceCard
                key={item.id}
                item={item}
                isSelected={selectedIds.has(item.id)}
                onToggle={() => onToggle(item.id)}
              />
            ))}
          </div>
        </div>
      )}

      {cases.length === 0 && articles.length === 0 && (
        <div className="p-4 text-center text-sm text-gray-400">
          아직 검색된 증거가 없습니다
        </div>
      )}

      {/* 제출 버튼 */}
      {(cases.length > 0 || articles.length > 0) && (
        <div className="p-3 border-t border-gray-100">
          <button
            onClick={onSubmit}
            className="w-full py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 transition-colors"
          >
            증거 제출 ({selectedIds.size}건 선택)
          </button>
        </div>
      )}
    </div>
  )
}
