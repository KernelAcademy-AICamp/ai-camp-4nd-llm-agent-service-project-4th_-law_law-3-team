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
        <div className="bg-gray-50 p-5 rounded-xl border border-gray-200 border-l-4 border-l-blue-500">
          <h3 className="text-sm font-semibold text-gray-900 mb-3 flex items-center gap-2">
            <svg className="w-4 h-4 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h3.75M9 15h3.75M9 18h3.75m3 .75H18a2.25 2.25 0 002.25-2.25V6.108c0-1.135-.845-2.098-1.976-2.192a48.424 48.424 0 00-1.123-.08m-5.801 0c-.065.21-.1.433-.1.664 0 .414.336.75.75.75h4.5a.75.75 0 00.75-.75 2.25 2.25 0 00-.1-.664m-5.8 0A2.251 2.251 0 0113.5 2.25H15c1.012 0 1.867.668 2.15 1.586m-5.8 0c-.376.023-.75.05-1.124.08C9.095 4.01 8.25 4.973 8.25 6.108V8.25m0 0H4.875c-.621 0-1.125.504-1.125 1.125v11.25c0 .621.504 1.125 1.125 1.125h9.75c.621 0 1.125-.504 1.125-1.125V9.375c0-.621-.504-1.125-1.125-1.125H8.25z" />
            </svg>
            법원 판단의 핵심 (판결요지)
          </h3>
          <p className="text-gray-700 leading-relaxed whitespace-pre-wrap">
            {source.reasoning}
          </p>
        </div>
      )}

      {/* 주문 (결과) */}
      {source.ruling && (
        <div className="bg-gray-50 p-5 rounded-xl border border-gray-200 border-l-4 border-l-gray-400">
          <h3 className="text-sm font-semibold text-gray-900 mb-3 flex items-center gap-2">
            <svg className="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 3v17.25m0 0c-1.472 0-2.882.265-4.185.75M12 20.25c1.472 0 2.882.265 4.185.75M18.75 4.97A48.416 48.416 0 0012 4.5c-2.291 0-4.545.16-6.75.47m13.5 0c1.01.143 2.01.317 3 .52m-3-.52l2.62 10.726c.122.499-.106 1.028-.589 1.202a5.988 5.988 0 01-2.031.352 5.988 5.988 0 01-2.031-.352c-.483-.174-.711-.703-.59-1.202L18.75 4.971zm-16.5.52c.99-.203 1.99-.377 3-.52m0 0l2.62 10.726c.122.499-.106 1.028-.589 1.202a5.989 5.989 0 01-2.031.352 5.989 5.989 0 01-2.031-.352c-.483-.174-.711-.703-.59-1.202L5.25 4.971z" />
            </svg>
            처벌·판결 결과 (주문)
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
        title="판결문 전체 보기"
        highlightContent={source.content}
      />

      {/* 그래프 보강 정보 */}
      {(source.cited_statutes?.length || source.similar_cases?.length) ? (
        <div className="mt-6 pt-4 border-t border-gray-200">
          <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">관련 정보</h3>
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
