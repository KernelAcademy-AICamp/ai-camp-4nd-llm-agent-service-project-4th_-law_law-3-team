'use client'

import type { NewsArticle, RelatedLawBrief } from '../types'

interface KeywordNewsListProps {
  keyword: string
  articles: NewsArticle[]
  relatedLaws: RelatedLawBrief[]
  sourcesUsed: string[]
  totalCount: number
  onBack: () => void
  onGenerateScript: (keyword: string) => void
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

export function KeywordNewsList({
  keyword,
  articles,
  relatedLaws,
  sourcesUsed,
  totalCount,
  onBack,
  onGenerateScript,
}: KeywordNewsListProps) {
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
            </p>
          </div>
        </div>
        <button
          onClick={() => onGenerateScript(keyword)}
          className="px-4 py-2 bg-green-600 text-white text-sm font-medium rounded-lg hover:bg-green-700 transition-colors"
        >
          대본 생성하기
        </button>
      </div>

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
      {articles.length > 0 ? (
        <div className="space-y-3">
          {articles.map((article, index) => (
            <a
              key={`${article.url}-${index}`}
              href={article.url}
              target="_blank"
              rel="noopener noreferrer"
              className="block bg-white border border-gray-200 rounded-lg p-4 hover:border-blue-300 hover:shadow-sm transition-all"
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex-1 min-w-0">
                  <h3 className="text-sm font-medium text-gray-900 line-clamp-2">
                    {article.title}
                  </h3>
                  {article.snippet && (
                    <p className="text-xs text-gray-500 mt-1.5 line-clamp-2">
                      {article.snippet}
                    </p>
                  )}
                  <div className="flex items-center gap-2 mt-2">
                    <span className="text-xs text-blue-600 font-medium bg-blue-50 px-2 py-0.5 rounded">
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
                </div>
                <svg className="w-4 h-4 text-gray-300 shrink-0 mt-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                </svg>
              </div>
            </a>
          ))}
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
