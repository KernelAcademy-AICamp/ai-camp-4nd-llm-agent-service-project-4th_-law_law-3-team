'use client'

import { useMemo, useState } from 'react'
import type { NewsArticle, RelatedLawBrief, SourceFailInfo } from '../types'

type SortMode = 'total' | 'engagement' | 'recency' | 'convergence'

const SORT_OPTIONS: { value: SortMode; label: string }[] = [
  { value: 'total', label: '종합순' },
  { value: 'engagement', label: '참여도순' },
  { value: 'recency', label: '최신순' },
  { value: 'convergence', label: '수렴도순' },
]

const NEWS_ERROR_TYPE_LABELS: Record<string, string> = {
  timeout: '시간 초과',
  auth: '인증 오류',
  rate_limit: '할당량 초과',
  network: '네트워크 오류',
  parse: '응답 파싱 오류',
  unknown: '알 수 없는 오류',
}

interface ScoreBarProps {
  label: string
  value: number
  color: string
}

function ScoreBar({ label, value, color }: ScoreBarProps) {
  const percentage = Math.round(value * 100)
  return (
    <div className="flex items-center gap-1.5 text-xs">
      <span className="w-12 text-gray-400 shrink-0">{label}</span>
      <div className="flex-1 h-1.5 bg-gray-100 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full ${color}`}
          style={{ width: `${percentage}%` }}
        />
      </div>
      <span className="w-6 text-right text-gray-500 font-mono text-[10px]">{percentage}</span>
    </div>
  )
}

function getSourceBadgeStyle(weight: number): string {
  if (weight >= 1.0) return 'text-blue-700 bg-blue-100'
  if (weight >= 0.7) return 'text-blue-600 bg-blue-50'
  return 'text-gray-600 bg-gray-100'
}

function getScoreBadgeStyle(score: number): string {
  if (score >= 70) return 'bg-green-100 text-green-700'
  if (score >= 40) return 'bg-yellow-100 text-yellow-700'
  return 'bg-gray-100 text-gray-500'
}

function formatDate(dateStr: string | null): string {
  if (!dateStr) return ''
  try {
    return new Date(dateStr).toLocaleDateString('ko-KR', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    })
  } catch {
    return ''
  }
}

function formatCount(count: number): string {
  if (count >= 10000) return `${(count / 10000).toFixed(1)}만`
  if (count >= 1000) return `${(count / 1000).toFixed(1)}천`
  return String(count)
}

function EngagementIcon({ type }: { type: 'view' | 'comment' }) {
  if (type === 'view') {
    return (
      <svg className="w-3 h-3 inline-block mr-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
      </svg>
    )
  }
  return (
    <svg className="w-3 h-3 inline-block mr-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
    </svg>
  )
}

interface KeywordNewsListProps {
  keyword: string
  articles: NewsArticle[]
  relatedLaws: RelatedLawBrief[]
  sourcesUsed: string[]
  sourcesFailed: SourceFailInfo[]
  totalCount: number
  selectedArticles: NewsArticle[]
  onToggleArticle: (article: NewsArticle) => void
  onBack: () => void
  onGenerateScript: (keyword: string) => void
  onGenerateWithSelected: () => void
}

export function KeywordNewsList({
  keyword,
  articles,
  relatedLaws,
  sourcesUsed,
  sourcesFailed,
  totalCount,
  selectedArticles,
  onToggleArticle,
  onBack,
  onGenerateScript,
  onGenerateWithSelected,
}: KeywordNewsListProps) {
  const selectedCount = selectedArticles.length
  const [sortMode, setSortMode] = useState<SortMode>('total')

  const sortedArticles = useMemo(() => {
    const sorted = [...articles]
    switch (sortMode) {
      case 'engagement':
        sorted.sort((a, b) => b.engagement_score - a.engagement_score)
        break
      case 'recency':
        sorted.sort((a, b) => b.recency_score - a.recency_score)
        break
      case 'convergence':
        sorted.sort((a, b) => b.convergence_score - a.convergence_score)
        break
      default:
        sorted.sort((a, b) => b.total_score - a.total_score)
        break
    }
    return sorted
  }, [articles, sortMode])

  return (
    <div className="space-y-4">
      {/* 헤더 */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button
            onClick={onBack}
            className="p-1.5 text-gray-400 hover:text-gray-700 rounded-lg hover:bg-gray-100 transition-colors"
            aria-label="뒤로가기"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
          </button>
          <div>
            <h2 className="text-lg font-semibold text-gray-900">
              &ldquo;{keyword}&rdquo; 뉴스
            </h2>
            <p className="text-sm text-gray-500">
              {totalCount}건의 기사
              {sourcesUsed.length > 0 && (
                <span className="ml-2 text-gray-400">
                  ({sourcesUsed.join(', ')})
                </span>
              )}
              {sourcesFailed.length > 0 && (
                <span className="ml-1 text-red-400">
                  · 실패 {sourcesFailed.length}개
                </span>
              )}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <select
            value={sortMode}
            onChange={(e) => setSortMode(e.target.value as SortMode)}
            className="px-3 py-2 text-sm border border-gray-300 rounded-lg bg-white text-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          >
            {SORT_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
          {selectedCount > 0 && (
            <button
              onClick={onGenerateWithSelected}
              className="px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-700 transition-colors"
            >
              선택한 뉴스로 대본 생성 ({selectedCount})
            </button>
          )}
          <button
            onClick={() => onGenerateScript(keyword)}
            className="px-4 py-2 bg-green-600 text-white text-sm font-medium rounded-lg hover:bg-green-700 transition-colors"
          >
            대본 생성하기
          </button>
        </div>
      </div>

      {/* 뉴스 소스 실패 알림 */}
      {sourcesFailed.length > 0 && (
        <div className="bg-amber-50 border border-amber-200 rounded-lg px-4 py-3">
          <p className="text-xs font-medium text-amber-700 mb-1.5">
            일부 뉴스 소스에서 검색에 실패했습니다 — 표시된 결과는 나머지 소스에서 수집한 것입니다
          </p>
          <div className="flex flex-wrap gap-1.5">
            {sourcesFailed.map((fail) => {
              const errorLabel = NEWS_ERROR_TYPE_LABELS[fail.error_type] ?? fail.error_type
              const tooltipText = fail.error_message
                ? `${errorLabel}: ${fail.error_message}`
                : errorLabel
              return (
                <span
                  key={fail.source_name}
                  className="inline-flex items-center gap-1 text-xs bg-white text-amber-700 border border-amber-200 px-2 py-0.5 rounded-full cursor-help"
                  title={tooltipText}
                >
                  <span className="w-1.5 h-1.5 bg-amber-400 rounded-full" />
                  {fail.source_name}
                  <span className="text-amber-500">({errorLabel})</span>
                </span>
              )
            })}
          </div>
        </div>
      )}

      {/* 관련 법률 (RAG enrichment) */}
      {relatedLaws.length > 0 && (
        <div className="bg-indigo-50 border border-indigo-100 rounded-lg px-4 py-3">
          <p className="text-xs font-medium text-indigo-700 mb-2">관련 법률</p>
          <div className="flex flex-wrap gap-2">
            {relatedLaws.map((law) => (
              <span
                key={law.law_name}
                className="inline-flex items-center gap-1 text-xs bg-white text-indigo-700 border border-indigo-200 px-2.5 py-1 rounded-full"
              >
                {law.law_name}
                {law.issue_label && (
                  <span className="text-indigo-400">({law.issue_label})</span>
                )}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* 뉴스 리스트 */}
      {sortedArticles.length > 0 ? (
        <div className="space-y-3">
          {sortedArticles.map((article, index) => {
            const isSelected = selectedArticles.some((a) => a.url === article.url)
            const showScore = article.total_score > 0

            return (
              <div
                key={`${article.url}-${index}`}
                className={`bg-white border rounded-lg p-4 transition-all ${
                  isSelected
                    ? 'border-blue-400 ring-1 ring-blue-200'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
              >
                <div className="flex items-start gap-3">
                  {/* 체크박스 */}
                  <label className="flex items-center shrink-0 mt-0.5 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={isSelected}
                      onChange={() => onToggleArticle(article)}
                      className="w-4 h-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                  </label>

                  {/* 기사 내용 */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-start justify-between gap-3">
                      <h3 className="text-sm font-medium text-gray-900 line-clamp-2">
                        {article.title}
                      </h3>
                      <div className="flex items-center gap-2 shrink-0">
                        {/* 점수 뱃지 */}
                        {showScore && (
                          <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${getScoreBadgeStyle(article.total_score)}`}>
                            {article.total_score.toFixed(0)}
                          </span>
                        )}
                        {/* 외부 링크 */}
                        <a
                          href={article.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="p-1 text-gray-300 hover:text-blue-500 transition-colors"
                          onClick={(e) => e.stopPropagation()}
                          aria-label="기사 열기"
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                          </svg>
                        </a>
                      </div>
                    </div>

                    {article.snippet && (
                      <p className="text-xs text-gray-500 mt-1.5 line-clamp-2">
                        {article.snippet}
                      </p>
                    )}

                    <div className="flex items-center gap-2 mt-2 flex-wrap">
                      <span className={`text-xs font-medium px-2 py-0.5 rounded ${getSourceBadgeStyle(article.source_weight)}`}>
                        {article.source}
                      </span>
                      {article.published_at && (
                        <span className="text-xs text-gray-400">
                          {formatDate(article.published_at)}
                        </span>
                      )}
                      {article.legal_issue_label && (
                        <span className="text-xs text-purple-600 bg-purple-50 px-2 py-0.5 rounded">
                          {article.legal_issue_label}
                        </span>
                      )}
                      {article.view_count != null && article.view_count > 0 && (
                        <span className="text-xs text-gray-500 bg-gray-50 px-2 py-0.5 rounded" title="조회수">
                          <EngagementIcon type="view" /> {formatCount(article.view_count)}
                        </span>
                      )}
                      {article.comment_count != null && article.comment_count > 0 && (
                        <span className="text-xs text-gray-500 bg-gray-50 px-2 py-0.5 rounded" title="댓글수">
                          <EngagementIcon type="comment" /> {formatCount(article.comment_count)}
                        </span>
                      )}
                      {article.is_early_signal && (
                        <span className="text-xs text-orange-600 bg-orange-50 px-2 py-0.5 rounded font-medium">
                          초기 신호
                        </span>
                      )}
                    </div>

                    {article.related_laws.length > 0 && (
                      <div className="flex flex-wrap gap-1 mt-2">
                        {article.related_laws.map((law) => (
                          <span
                            key={law}
                            className="text-xs text-gray-500 bg-gray-100 px-2 py-0.5 rounded"
                          >
                            {law}
                          </span>
                        ))}
                      </div>
                    )}

                    {/* 세부 점수 바 (5차원) */}
                    {showScore && (
                      <div className="mt-2.5 space-y-1">
                        <ScoreBar label="관련도" value={article.relevance_score} color="bg-blue-500" />
                        <ScoreBar label="법적" value={article.legal_score} color="bg-purple-500" />
                        <ScoreBar label="최신성" value={article.recency_score} color="bg-green-500" />
                        <ScoreBar label="참여도" value={article.engagement_score} color="bg-amber-500" />
                        <ScoreBar label="수렴" value={article.convergence_score} color="bg-teal-500" />
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      ) : (
        <div className="text-center py-12 text-gray-400">
          <p className="text-sm">검색 결과가 없습니다.</p>
          <p className="text-xs mt-1">다른 키워드를 선택해 보세요.</p>
        </div>
      )}
    </div>
  )
}
