'use client'

import type { RagContributionStats } from '../types'

interface RagContributionCardProps {
  stats: RagContributionStats | null
  loading: boolean
  error?: string | null
}

function formatCount(count: number): string {
  return count.toLocaleString()
}

export function RagContributionCard({ stats, loading, error }: RagContributionCardProps) {
  if (error) {
    return (
      <div className="bg-white rounded-xl border border-gray-200 p-4 shadow-sm">
        <h3 className="text-sm font-semibold text-gray-800 mb-2">RAG 데이터 기여도</h3>
        <div className="text-xs text-gray-400 text-center py-4">
          일시적으로 조회할 수 없습니다
        </div>
      </div>
    )
  }

  if (loading || !stats) {
    return (
      <div className="bg-white rounded-xl border border-gray-200 p-4 shadow-sm">
        <h3 className="text-sm font-semibold text-gray-800 mb-2">RAG 데이터 기여도</h3>
        <div className="flex items-center justify-center h-32">
          <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600" />
        </div>
      </div>
    )
  }

  const { main_rag_total, main_rag_sources, assist_rag_total, assist_rag_sources, assist_contribution_percent } = stats
  const maxMainCount = main_rag_sources.length > 0 ? main_rag_sources[0].count : 1

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-4 shadow-sm">
      <h3 className="text-sm font-semibold text-gray-800 mb-3">RAG 데이터 기여도</h3>

      {/* Main RAG 섹션 */}
      <div className="mb-4">
        <div className="flex items-center gap-1.5 mb-2">
          <span className="w-2 h-2 rounded-full bg-blue-500 shrink-0" />
          <span className="text-xs font-semibold text-gray-700">Main RAG</span>
          <span className="text-xs text-gray-400 ml-auto">{formatCount(main_rag_total)}건</span>
        </div>
        <div className="max-h-48 overflow-y-auto space-y-1.5 pr-1">
          {main_rag_sources.map((src) => {
            const barWidth = maxMainCount > 0 ? (src.count / maxMainCount) * 100 : 0
            return (
              <div key={src.table_name} className="flex items-center gap-2 text-xs">
                <span className="text-gray-600 w-28 shrink-0 truncate" title={src.label}>
                  {src.label}
                </span>
                <div className="flex-1 bg-gray-100 rounded-full h-1.5">
                  <div
                    className="bg-blue-400 h-1.5 rounded-full transition-all"
                    style={{ width: `${Math.min(barWidth, 100)}%` }}
                  />
                </div>
                <span className="text-gray-500 w-16 text-right shrink-0 tabular-nums">
                  {formatCount(src.count)}
                </span>
              </div>
            )
          })}
        </div>
      </div>

      {/* Assist RAG 섹션 */}
      <div className="mb-4">
        <div className="flex items-center gap-1.5 mb-2">
          <span className="w-2 h-2 rounded-full bg-green-500 shrink-0" />
          <span className="text-xs font-semibold text-gray-700">Assist RAG</span>
          <span className="text-xs text-gray-400 ml-auto">{formatCount(assist_rag_total)}건</span>
        </div>
        <div className="space-y-2">
          {assist_rag_sources.map((src) => (
            <div key={src.source} className="text-xs">
              <div className="flex items-center justify-between mb-0.5">
                <span className="text-gray-600">{src.label}</span>
                <span className="text-gray-500 tabular-nums">
                  {formatCount(src.count)}건
                  <span className="text-gray-400 ml-1">
                    (임베딩 {formatCount(src.indexed_count)})
                  </span>
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* 비율 바 */}
      <div className="border-t border-gray-100 pt-3">
        <div className="flex items-center justify-between text-xs mb-1">
          <span className="text-gray-600">Assist RAG 기여 비율</span>
          <span className="font-semibold text-blue-600">
            {assist_contribution_percent.toFixed(2)}%
          </span>
        </div>
        <div className="w-full bg-gray-100 rounded-full h-2.5">
          <div
            className="bg-gradient-to-r from-blue-500 to-green-500 h-2.5 rounded-full transition-all"
            style={{ width: `${Math.min(assist_contribution_percent, 100)}%` }}
          />
        </div>
        <p className="text-xs text-gray-400 mt-2 text-center">
          뉴스 {formatCount(assist_rag_total)}건이 Main RAG {formatCount(main_rag_total)}건을 보완
        </p>
      </div>
    </div>
  )
}
