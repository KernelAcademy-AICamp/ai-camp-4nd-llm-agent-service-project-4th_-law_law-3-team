'use client'

import { useState } from 'react'
import type { RelatedCaseItem } from '../types'

interface RelatedCasesProps {
  cases: RelatedCaseItem[]
  isLoading: boolean
  disputeType: string | null
  onClose?: () => void
}

export function RelatedCases({ cases, isLoading, disputeType, onClose }: RelatedCasesProps) {
  if (!disputeType) {
    return null
  }

  return (
    <div className="w-full h-full bg-white border-r border-gray-200 flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <h3 className="font-semibold text-gray-900 flex items-center gap-2">
            <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"
              />
            </svg>
            유사 판례
          </h3>
          {onClose && (
            <button
              onClick={onClose}
              className="p-1 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded transition-colors"
              aria-label="유사 판례 패널 접기"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
            </button>
          )}
        </div>
        <p className="text-xs text-gray-500 mt-1">비슷한 사건의 판결을 참고해보세요</p>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4">
        {isLoading ? (
          <div className="flex items-center justify-center h-32">
            <div className="text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-2" />
              <p className="text-sm text-gray-500">판례 검색 중...</p>
            </div>
          </div>
        ) : cases.length > 0 ? (
          <div className="space-y-3">
            {cases.map((caseItem) => (
              <CaseCard key={caseItem.id} caseItem={caseItem} />
            ))}
          </div>
        ) : (
          <div className="text-center text-gray-500 py-8">
            <svg
              className="w-10 h-10 mx-auto mb-2 text-gray-300"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
              />
            </svg>
            <p className="text-sm">유사 판례를 찾지 못했습니다</p>
          </div>
        )}
      </div>

      {/* Info */}
      {cases.length > 0 && (
        <div className="p-4 border-t border-gray-200 bg-gray-50">
          <p className="text-xs text-gray-500">{cases[0].relevance}</p>
        </div>
      )}
    </div>
  )
}

const DOC_TYPE_BADGE: Record<string, { label: string; color: string }> = {
  판례: { label: '판례', color: 'bg-blue-100 text-blue-700' },
  특별행정심판: { label: '특별행정심판', color: 'bg-purple-100 text-purple-700' },
  행정심판례: { label: '행정심판', color: 'bg-indigo-100 text-indigo-700' },
  헌재결정례: { label: '헌재결정', color: 'bg-rose-100 text-rose-700' },
}

function CaseCard({ caseItem }: { caseItem: RelatedCaseItem }) {
  const [isExpanded, setIsExpanded] = useState(false)
  const hasDetails = Boolean(caseItem.ruling || caseItem.reasoning)
  const badge = caseItem.doc_type ? DOC_TYPE_BADGE[caseItem.doc_type] : undefined

  return (
    <div className="p-3 bg-gray-50 rounded-lg border border-gray-200 hover:border-gray-300 transition">
      <div className="flex items-start gap-1.5">
        {badge && (
          <span className={`shrink-0 mt-0.5 px-1.5 py-0.5 rounded text-[10px] font-medium ${badge.color}`}>
            {badge.label}
          </span>
        )}
        <h4 className="font-medium text-sm text-gray-900 line-clamp-2">
          {caseItem.case_name || '제목 없음'}
        </h4>
      </div>
      {caseItem.case_number && (
        <p className="text-xs text-gray-500 font-mono mt-1">{caseItem.case_number}</p>
      )}
      <p className="text-xs text-gray-600 mt-2 line-clamp-3">{caseItem.summary}</p>

      {/* 판결 보기 토글 */}
      {hasDetails && (
        <>
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="mt-2 flex items-center gap-1 text-xs text-blue-600 hover:text-blue-800 font-medium transition"
          >
            <svg
              className={`w-3.5 h-3.5 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
            {isExpanded ? '판결 접기' : '판결 보기'}
          </button>

          {isExpanded && (
            <div className="mt-2 space-y-2 border-t border-gray-200 pt-2">
              {caseItem.ruling && (
                <div>
                  <p className="text-xs font-semibold text-gray-700 mb-0.5">주문 (판결)</p>
                  <p className="text-xs text-gray-600 line-clamp-4">{caseItem.ruling}</p>
                </div>
              )}
              {caseItem.reasoning && (
                <div>
                  <p className="text-xs font-semibold text-gray-700 mb-0.5">판결요지</p>
                  <p className="text-xs text-gray-600 line-clamp-6">{caseItem.reasoning}</p>
                </div>
              )}
            </div>
          )}
        </>
      )}

      <div className="flex items-center justify-end mt-2 pt-2 border-t border-gray-200">
        <a
          href={`/case-precedent?id=${caseItem.id}`}
          target="_blank"
          rel="noopener noreferrer"
          className="text-xs text-blue-600 hover:underline"
        >
          상세 보기
        </a>
      </div>
    </div>
  )
}
