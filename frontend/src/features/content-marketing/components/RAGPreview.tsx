'use client'

import { useState } from 'react'
import type { RelatedCase, RelatedLaw } from '../types'

interface RAGPreviewProps {
  relatedLaws: RelatedLaw[]
  relatedCases: RelatedCase[]
}

export function RAGPreview({ relatedLaws, relatedCases }: RAGPreviewProps) {
  const [isLawsOpen, setIsLawsOpen] = useState(false)
  const [isCasesOpen, setIsCasesOpen] = useState(false)

  if (relatedLaws.length === 0 && relatedCases.length === 0) return null

  return (
    <div className="bg-white rounded-xl border border-gray-200 divide-y divide-gray-100">
      {/* 관련 법령 */}
      {relatedLaws.length > 0 && (
        <div>
          <button
            onClick={() => setIsLawsOpen(!isLawsOpen)}
            className="w-full flex items-center justify-between px-5 py-3 text-sm font-medium text-gray-700 hover:bg-gray-50 transition-colors"
          >
            <span>관련 법령 ({relatedLaws.length}건)</span>
            <svg
              className={`w-4 h-4 text-gray-400 transition-transform ${isLawsOpen ? 'rotate-180' : ''}`}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
            </svg>
          </button>
          {isLawsOpen && (
            <div className="px-5 pb-4 space-y-1.5">
              {relatedLaws.map((law) => (
                <div
                  key={law.law_id}
                  className="flex items-center justify-between text-sm bg-blue-50 rounded-lg px-3 py-2"
                >
                  <span className="text-gray-700 truncate mr-2">{law.law_name}</span>
                  <span className="text-blue-600 font-medium shrink-0">
                    {(law.relevance_score * 100).toFixed(0)}%
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* 관련 판례 */}
      {relatedCases.length > 0 && (
        <div>
          <button
            onClick={() => setIsCasesOpen(!isCasesOpen)}
            className="w-full flex items-center justify-between px-5 py-3 text-sm font-medium text-gray-700 hover:bg-gray-50 transition-colors"
          >
            <span>관련 판례 ({relatedCases.length}건)</span>
            <svg
              className={`w-4 h-4 text-gray-400 transition-transform ${isCasesOpen ? 'rotate-180' : ''}`}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
            </svg>
          </button>
          {isCasesOpen && (
            <div className="px-5 pb-4 space-y-1.5">
              {relatedCases.map((caseItem) => (
                <div
                  key={caseItem.case_id}
                  className="flex items-center justify-between text-sm bg-purple-50 rounded-lg px-3 py-2"
                >
                  <span className="text-gray-700 truncate mr-2">
                    {caseItem.case_number} ({caseItem.case_name})
                  </span>
                  <span className="text-purple-600 font-medium shrink-0">
                    {(caseItem.relevance_score * 100).toFixed(0)}%
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
