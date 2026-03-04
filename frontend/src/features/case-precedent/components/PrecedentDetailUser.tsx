'use client'

import { useState } from 'react'
import type { ChatSource } from '../types'
import { PrecedentFullTextViewer } from './PrecedentFullTextViewer'

interface PrecedentDetailUserProps {
  source: ChatSource
}

export function PrecedentDetailUser({ source }: PrecedentDetailUserProps) {
  const [isProvisionsOpen, setIsProvisionsOpen] = useState(false)

  const provisions = source.reference_provisions
    ? source.reference_provisions
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean)
    : []

  return (
    <div className="space-y-4">
      {/* 판결요지 (핵심) */}
      {source.reasoning && (
        <div className="bg-blue-50 p-5 rounded-xl border border-blue-100">
          <h3 className="font-bold text-blue-800 mb-3 flex items-center gap-2">
            <span>📋</span> 법원 판단의 핵심 (판결요지)
          </h3>
          <p className="text-gray-700 leading-relaxed whitespace-pre-wrap">
            {source.reasoning}
          </p>
        </div>
      )}

      {/* 주문 (결과) */}
      {source.ruling && (
        <div className="bg-green-50 p-5 rounded-xl border border-green-100">
          <h3 className="font-bold text-green-800 mb-3 flex items-center gap-2">
            <span>⚖️</span> 처벌·판결 결과 (주문)
          </h3>
          <p className="text-gray-700 leading-relaxed whitespace-pre-wrap">
            {source.ruling}
          </p>
        </div>
      )}

      {/* 참조 조문 */}
      {provisions.length > 0 && (
        <div className="border border-gray-200 rounded-xl overflow-hidden">
          <button
            onClick={() => setIsProvisionsOpen(!isProvisionsOpen)}
            className="w-full flex items-center gap-2 px-4 py-3 hover:bg-gray-50 transition-colors"
          >
            <svg className="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25" />
            </svg>
            <span className="text-sm font-medium text-gray-700">참조 조문</span>
            <span className="text-xs text-gray-400">{provisions.length}</span>
            <svg
              className={`w-4 h-4 text-gray-400 ml-auto transition-transform ${isProvisionsOpen ? 'rotate-180' : ''}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
            </svg>
          </button>
          {isProvisionsOpen && (
            <ul className="px-4 pb-3 space-y-1 max-h-48 overflow-y-auto">
              {provisions.map((provision, idx) => (
                <li
                  key={`provision-${idx}`}
                  className="text-sm text-gray-700 py-1.5 px-3 rounded hover:bg-gray-50 cursor-default"
                >
                  {provision}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {/* 판결문 전체 보기 (아코디언) */}
      <PrecedentFullTextViewer
        data={source}
        mode="accordion"
        title="📄 판결문 전체 보기"
      />

      {/* 그래프 보강 정보 */}
      {(source.cited_statutes?.length || source.similar_cases?.length) ? (
        <div className="mt-6 pt-4 border-t border-gray-200">
          <h3 className="font-bold text-gray-700 mb-3">📊 관련 정보</h3>
          {source.cited_statutes && source.cited_statutes.length > 0 && (
            <p className="text-sm text-gray-600 mb-2">
              <span className="font-medium">인용 법령:</span>{' '}
              {source.cited_statutes.join(', ')}
            </p>
          )}
          {source.similar_cases && source.similar_cases.length > 0 && (
            <p className="text-sm text-gray-600">
              <span className="font-medium">유사 판례:</span>{' '}
              {source.similar_cases.join(', ')}
            </p>
          )}
        </div>
      ) : null}
    </div>
  )
}
