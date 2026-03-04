'use client'

import { useEffect } from 'react'
import { X, ExternalLink } from 'lucide-react'
import type { NewsArticleResponse } from '../types'
import { formatDate } from '../utils/formatDate'
import { isSafeUrl } from '../utils/url'
import { NewsSourceBadge } from './NewsSourceBadge'
import { TagList } from './TagList'

interface NewsDetailPanelProps {
  article: NewsArticleResponse | null
  loading: boolean
  error: string | null
  onClose: () => void
}

function SummarySection({ title, items }: { title: string; items: string[] | null }) {
  if (!items || items.length === 0) return null
  return (
    <div className="mb-4">
      <h4 className="text-sm font-semibold text-gray-700 mb-1">{title}</h4>
      <ul className="space-y-1">
        {items.map((item, index) => (
          <li key={`news-${index}`} className="text-sm text-gray-600 flex items-start gap-1.5">
            <span className="text-gray-400 mt-0.5 shrink-0">-</span>
            <span>{item}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

export function NewsDetailPanel({ article, loading, error, onClose }: NewsDetailPanelProps) {
  // ESC 키로 패널 닫기
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose()
      }
    }

    if (article || loading || error) {
      document.addEventListener('keydown', handleKeyDown)
    }

    return () => {
      document.removeEventListener('keydown', handleKeyDown)
    }
  }, [article, loading, error, onClose])

  if (!article && !loading && !error) return null

  const articleUrl = article?.url && isSafeUrl(article.url) ? article.url : null

  return (
    <div
      className="fixed inset-y-0 right-0 w-full max-w-lg bg-white shadow-xl border-l border-gray-200 z-50 flex flex-col"
      role="dialog"
      aria-label="기사 상세"
    >
      {/* 헤더 */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 shrink-0">
        <h3 className="text-sm font-semibold text-gray-700">기사 상세</h3>
        <button
          onClick={onClose}
          className="p-1 text-gray-400 hover:text-gray-600 rounded transition-colors"
          aria-label="닫기"
        >
          <X size={18} />
        </button>
      </div>

      {/* 본문 */}
      <div className="flex-1 overflow-y-auto p-4">
        {loading && (
          <div className="flex items-center justify-center h-32">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600" />
          </div>
        )}

        {error && (
          <div className="text-sm text-red-600 bg-red-50 rounded p-3">
            {error}
          </div>
        )}

        {article && (
          <div className="space-y-4">
            {/* 메타 정보 */}
            <div className="flex items-center gap-2 flex-wrap">
              <NewsSourceBadge source={article.source} />
              <span className="text-xs text-gray-500">{formatDate(article.published_at)}</span>
              {article.author && (
                <span className="text-xs text-gray-500">{article.author}</span>
              )}
              {article.section && (
                <span className="text-xs text-gray-400">{article.section}</span>
              )}
            </div>

            {/* 제목 */}
            <h2 className="text-lg font-bold text-gray-900">{article.title}</h2>

            {/* 한줄 요약 */}
            <div className="bg-blue-50 rounded-lg p-3">
              <p className="text-sm font-medium text-blue-800">{article.summary_one_liner}</p>
            </div>

            {/* 태그 */}
            <TagList tags={article.tags} />

            {/* AI 요약 섹션 */}
            <SummarySection title="핵심 이슈" items={article.summary_issues} />
            <SummarySection title="관련 법령" items={article.summary_laws} />
            <SummarySection title="관련 판례" items={article.summary_cases} />
            <SummarySection title="관련 기관" items={article.summary_institutions} />
            <SummarySection title="시사점" items={article.summary_implications} />

            {/* 본문 */}
            <div>
              <h4 className="text-sm font-semibold text-gray-700 mb-2">본문</h4>
              <div className="text-sm text-gray-600 whitespace-pre-wrap leading-relaxed">
                {article.cleaned_text}
              </div>
            </div>

            {/* 면책 고지 */}
            <div className="bg-amber-50 rounded-lg p-3 text-xs text-amber-700">
              {article.disclaimer}
            </div>

            {/* 원문 링크 (URL 프로토콜 검증 통과 시만 표시) */}
            {articleUrl && (
              <a
                href={articleUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1 text-sm text-blue-600 hover:text-blue-800 transition-colors"
              >
                <ExternalLink size={14} />
                원문 보기
              </a>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
