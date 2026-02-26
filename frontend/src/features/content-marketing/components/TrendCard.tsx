'use client'

import type { TrendIssue, TrendSource } from '../types'
import { ScoreBar } from './ScoreBar'

const SOURCE_LABELS: Record<TrendSource, string> = {
  tavily: 'Tavily',
  naver: 'Naver',
  perplexity: 'Perplexity',
  google_trends: 'Google',
  youtube: 'YouTube',
  newsdata: 'NewsData',
  newsapi: 'NewsAPI',
}

interface TrendCardProps {
  issue: TrendIssue
  rank: number
  onSelect: (issue: TrendIssue) => void
  onGenerateScript: (issue: TrendIssue) => void
}

export function TrendCard({ issue, rank, onSelect, onGenerateScript }: TrendCardProps) {
  const rankIcon = rank <= 3 ? ['', '1', '2', '3'][rank] : `${rank}`

  return (
    <div className="bg-white border border-gray-200 rounded-xl p-5 hover:shadow-md transition-shadow">
      {/* 헤더 */}
      <div className="flex items-start gap-3 mb-3">
        <span className="text-2xl font-bold text-blue-600 shrink-0">{rankIcon}위</span>
        <h3
          className="text-base font-semibold text-gray-900 line-clamp-2 cursor-pointer hover:text-blue-600 transition-colors"
          onClick={() => onSelect(issue)}
        >
          {issue.title}
        </h3>
      </div>

      {/* 종합 점수 */}
      <div className="mb-3 text-sm text-gray-600">
        종합점수: <span className="font-bold text-gray-900">{issue.score.toFixed(1)}</span> / 100
      </div>

      {/* v2.0 Legal Gate 배지 + fitness 라벨 */}
      {issue.score_detail && (
        <div className="flex items-center gap-2 mb-2">
          <span
            className={`px-2 py-0.5 text-xs font-medium rounded-full ${
              issue.score_detail.legal_gate_passed
                ? 'bg-green-100 text-green-700'
                : 'bg-red-100 text-red-600'
            }`}
          >
            Legal Gate {issue.score_detail.legal_gate_passed ? '통과' : '미달'}
          </span>
          {issue.fitness_label && (
            <span className="px-2 py-0.5 text-xs bg-purple-100 text-purple-700 rounded-full">
              {issue.fitness_label}
            </span>
          )}
        </div>
      )}

      {/* 스코어 바 */}
      <div className="space-y-1.5 mb-4">
        {issue.score_detail ? (
          <>
            <ScoreBar score={issue.score_detail.legal_score} label="법적쟁점화" />
            <ScoreBar score={issue.score_detail.controversy_score} label="논란도" />
            <ScoreBar score={issue.score_detail.spread_score} label="확산도" />
          </>
        ) : (
          <>
            <ScoreBar score={issue.mention_score} label="언급량" />
            <ScoreBar score={issue.legal_relevance_score} label="법적해석" />
          </>
        )}
      </div>

      {/* 핵심 쟁점 */}
      <div className="mb-4">
        <p className="text-xs font-medium text-gray-500 mb-1">핵심 쟁점:</p>
        <ol className="space-y-0.5">
          {issue.key_points.slice(0, 3).map((point, index) => (
            <li key={index} className="text-sm text-gray-700 line-clamp-1">
              {index + 1}. {point}
            </li>
          ))}
        </ol>
      </div>

      {/* 소스 태그 */}
      <div className="flex flex-wrap gap-1 mb-4">
        {issue.sources.map((source) => (
          <span
            key={source}
            className="px-2 py-0.5 text-xs bg-gray-100 text-gray-600 rounded-full"
          >
            {SOURCE_LABELS[source]}
          </span>
        ))}
      </div>

      {/* 액션 버튼 */}
      <div className="flex gap-2">
        <button
          onClick={() => onSelect(issue)}
          className="flex-1 px-3 py-2 text-sm font-medium text-gray-700 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
        >
          상세보기
        </button>
        <button
          onClick={() => onGenerateScript(issue)}
          className="flex-1 px-3 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 transition-colors"
        >
          대본 생성
        </button>
      </div>
    </div>
  )
}
