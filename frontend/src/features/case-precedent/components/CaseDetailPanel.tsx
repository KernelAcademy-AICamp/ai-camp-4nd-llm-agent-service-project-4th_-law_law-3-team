'use client'

import type { PrecedentDetail } from '../types'
import { PrecedentFullTextViewer } from './PrecedentFullTextViewer'

interface CaseDetailPanelProps {
  case_: PrecedentDetail | null
  isLoading: boolean
  error: string | null
}

export function CaseDetailPanel({
  case_,
  isLoading,
  error,
}: CaseDetailPanelProps) {
  const getDocTypeLabel = (docType: string) => {
    const labels: Record<string, string> = {
      precedent: '판례',
      constitutional: '헌재결정',
      administrative: '행정심판',
      interpretation: '법령해석',
    }
    return labels[docType] || docType
  }

  if (isLoading) {
    return (
      <div className="flex-1 flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-600 mx-auto mb-3" />
          <p className="text-gray-500">판례 내용을 불러오는 중...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex-1 flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <svg
            className="w-12 h-12 mx-auto mb-3 text-red-400"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
            />
          </svg>
          <p className="text-red-600">{error}</p>
        </div>
      </div>
    )
  }

  if (!case_) {
    return (
      <div className="flex-1 flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <svg
            className="w-16 h-16 mx-auto mb-4 text-gray-300"
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
          <p className="text-gray-500 text-lg">판례를 선택하세요</p>
          <p className="text-gray-400 text-sm mt-1">
            왼쪽에서 판례를 선택하면 상세 내용을 볼 수 있습니다
          </p>
        </div>
      </div>
    )
  }

  const formatDate = (dateStr?: string) => {
    if (!dateStr) return ''
    const cleaned = String(dateStr).replace(/-/g, '')
    if (cleaned.length === 8) {
      const year = cleaned.slice(0, 4)
      const month = parseInt(cleaned.slice(4, 6), 10)
      const day = parseInt(cleaned.slice(6, 8), 10)
      return `${year}. ${month}. ${day}.`
    }
    return dateStr
  }

  const court = case_.court_name || case_.court || '대법원'
  const date = formatDate(case_.decision_date || case_.date)
  const caseNumber = case_.case_number || ''
  const caseName = case_.case_name || ''

  return (
    <div className="flex-1 flex flex-col bg-white overflow-hidden">
      {/* 얇은 헤더바 */}
      <div className="px-4 py-2 border-b border-gray-200 bg-gray-50 text-sm text-gray-700 truncate">
        <span className="font-medium">
          {court} {date} 선고 {caseNumber} 판결
        </span>
        {caseName && (
          <span className="ml-1">[{caseName}]</span>
        )}
        <span className="text-gray-400 ml-3">| 대법원 종합법률정보</span>
      </div>

      {/* Content - 판례 원문 뷰어 */}
      <div className="flex-1 overflow-y-auto">
        <PrecedentFullTextViewer data={case_} mode="direct" />
      </div>
    </div>
  )
}
