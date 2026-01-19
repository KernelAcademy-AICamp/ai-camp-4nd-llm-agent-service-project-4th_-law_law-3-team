'use client'

import type { RelatedCaseItem } from '../types'

interface RelatedCasesProps {
  cases: RelatedCaseItem[]
  isLoading: boolean
  disputeType: string | null
}

export function RelatedCases({ cases, isLoading, disputeType }: RelatedCasesProps) {
  if (!disputeType) {
    return null
  }

  return (
    <div className="w-80 bg-white border-l border-gray-200 flex flex-col h-full">
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
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
            {cases.map((case_) => (
              <div
                key={case_.id}
                className="p-3 bg-gray-50 rounded-lg border border-gray-200 hover:border-gray-300 transition"
              >
                <h4 className="font-medium text-sm text-gray-900 line-clamp-2">
                  {case_.case_name || '제목 없음'}
                </h4>
                {case_.case_number && (
                  <p className="text-xs text-gray-500 font-mono mt-1">{case_.case_number}</p>
                )}
                <p className="text-xs text-gray-600 mt-2 line-clamp-3">{case_.summary}</p>
                <div className="flex items-center justify-between mt-2 pt-2 border-t border-gray-200">
                  <span className="text-xs text-gray-400">
                    유사도 {Math.round(case_.similarity * 100)}%
                  </span>
                  <a
                    href={`/case-precedent?id=${case_.id}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-xs text-blue-600 hover:underline"
                  >
                    상세 보기
                  </a>
                </div>
              </div>
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
