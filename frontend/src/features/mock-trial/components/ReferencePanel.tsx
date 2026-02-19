'use client'

import { useState } from 'react'
import type { ReferenceItem } from '../types'

interface ReferencePanelProps {
  references: ReferenceItem[]
}

type TabType = 'case' | 'law'

export function ReferencePanel({ references }: ReferencePanelProps) {
  const [activeTab, setActiveTab] = useState<TabType>('case')
  const [expandedId, setExpandedId] = useState<string | null>(null)

  const filteredReferences = references.filter((r) => r.type === activeTab)

  const caseCount = references.filter((r) => r.type === 'case').length
  const lawCount = references.filter((r) => r.type === 'law').length

  const toggleExpand = (id: string): void => {
    setExpandedId((prev) => (prev === id ? null : id))
  }

  return (
    <div className="flex flex-col h-full">
      {/* 헤더 */}
      <div className="px-4 py-3 border-b border-gray-200">
        <h2 className="text-sm font-bold text-gray-900 flex items-center gap-1.5">
          <span>참조 자료</span>
        </h2>
      </div>

      {/* 탭 */}
      <div className="flex border-b border-gray-200">
        <button
          onClick={() => setActiveTab('case')}
          className={`flex-1 px-3 py-2 text-xs font-medium transition-colors ${
            activeTab === 'case'
              ? 'text-blue-600 border-b-2 border-blue-600 bg-blue-50'
              : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'
          }`}
        >
          판결문 ({caseCount})
        </button>
        <button
          onClick={() => setActiveTab('law')}
          className={`flex-1 px-3 py-2 text-xs font-medium transition-colors ${
            activeTab === 'law'
              ? 'text-blue-600 border-b-2 border-blue-600 bg-blue-50'
              : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'
          }`}
        >
          법령 ({lawCount})
        </button>
      </div>

      {/* 리스트 */}
      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {filteredReferences.length === 0 ? (
          <div className="text-center py-8 text-gray-400 text-xs leading-relaxed">
            재판이 진행되면
            <br />
            언급된 {activeTab === 'case' ? '판결문' : '법령'}이
            <br />
            여기에 표시됩니다
          </div>
        ) : (
          filteredReferences.map((item) => {
            const isExpanded = expandedId === item.id
            return (
              <button
                key={item.id}
                onClick={() => toggleExpand(item.id)}
                className="w-full text-left border border-gray-200 rounded-lg p-3 hover:border-blue-300 hover:bg-blue-50/30 transition-colors cursor-pointer"
              >
                {/* 카드 헤더 */}
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1 min-w-0">
                    <p className="text-xs font-semibold text-gray-800 truncate">
                      {item.title}
                    </p>
                    <p className="text-[11px] text-gray-500 mt-0.5 truncate">
                      {item.matched_text}
                    </p>
                  </div>
                  <span
                    className={`text-[10px] transition-transform ${isExpanded ? 'rotate-180' : ''}`}
                  >
                    ▼
                  </span>
                </div>

                {/* 펼친 상태: 상세 내용 */}
                {isExpanded && (
                  <div className="mt-2 pt-2 border-t border-gray-100">
                    <p className="text-xs text-gray-600 leading-relaxed whitespace-pre-wrap">
                      {item.summary}
                    </p>
                    <div className="flex items-center gap-2 mt-2">
                      <span className="text-[10px] text-gray-400">
                        {item.source}
                      </span>
                      <span className="text-[10px] px-1.5 py-0.5 bg-blue-100 text-blue-600 rounded">
                        관련도 {Math.round(item.relevance_score * 100)}%
                      </span>
                    </div>
                  </div>
                )}
              </button>
            )
          })
        )}
      </div>
    </div>
  )
}
