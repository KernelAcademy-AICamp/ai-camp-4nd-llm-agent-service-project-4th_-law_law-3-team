'use client'

import { useState, memo } from 'react'
import { X, ChevronRight, ChevronDown, ChevronUp, Loader2, Sparkles, FileText } from 'lucide-react'
import type { StatuteHierarchyResponse, StatuteNode } from '../types'

interface StatuteDetailPanelProps {
  data: StatuteHierarchyResponse
  loading: boolean
  onClose: () => void
  onNodeClick: (node: StatuteNode) => void
}

const TYPE_COLORS: Record<string, string> = {
  헌법: 'bg-red-50 text-red-700',
  법률: 'bg-amber-50 text-amber-700',
  대통령령: 'bg-blue-50 text-blue-700',
  총리령: 'bg-cyan-50 text-cyan-700',
  부령: 'bg-teal-50 text-teal-700',
  행정규칙: 'bg-gray-100 text-gray-600',
}

function getTypeBadgeColor(type: string): string {
  return TYPE_COLORS[type] || 'bg-gray-100 text-gray-600'
}

const StatuteList = memo(function StatuteList({
  title,
  nodes,
  onNodeClick,
}: {
  title: string
  nodes: StatuteNode[]
  onNodeClick: (node: StatuteNode) => void
}) {
  return (
    <div>
      <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
        {title} ({nodes.length})
      </h4>
      {nodes.length === 0 ? (
        <p className="text-xs text-gray-400 px-3 py-2">없음</p>
      ) : (
        <ul className="space-y-1">
          {nodes.map((node) => (
            <li key={node.id}>
              <button
                onClick={() => onNodeClick(node)}
                className="w-full text-left px-3 py-2 rounded-lg hover:bg-gray-50
                           transition-colors group flex items-center justify-between"
              >
                <div className="flex-1 min-w-0">
                  <span className="text-sm text-gray-700 group-hover:text-gray-900 truncate block">
                    {node.name}
                  </span>
                  <span className="text-xs text-gray-500">
                    {node.type}
                    {node.abbreviation && ` · ${node.abbreviation}`}
                  </span>
                </div>
                <ChevronRight className="w-4 h-4 text-gray-400 group-hover:text-gray-600 shrink-0 ml-2" />
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
})

function StatuteContentSection({
  content,
  supplementary,
}: {
  content: string
  supplementary?: string | null
}) {
  const [isOpen, setIsOpen] = useState(false)

  return (
    <div>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-center gap-2 px-3 py-2 text-sm text-amber-600
                   border border-amber-300 rounded-lg hover:bg-amber-50 transition-colors"
      >
        <FileText className="w-3.5 h-3.5" />
        원문 보기
        {isOpen ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
      </button>
      {isOpen && (
        <div className="mt-3 bg-gray-50 border border-gray-200 rounded-lg p-3 space-y-3">
          <pre className="text-xs text-gray-700 leading-relaxed whitespace-pre-wrap break-words font-sans">
            {content}
          </pre>
          {supplementary && (
            <>
              <hr className="border-gray-200" />
              <div>
                <span className="text-xs font-semibold text-gray-500 block mb-1">부칙</span>
                <pre className="text-xs text-gray-500 leading-relaxed whitespace-pre-wrap break-words font-sans">
                  {supplementary}
                </pre>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  )
}

export const StatuteDetailPanel = memo(function StatuteDetailPanel({
  data,
  loading,
  onClose,
  onNodeClick,
}: StatuteDetailPanelProps) {
  const root = data.root

  return (
    <div className="w-80 h-full border-r border-gray-200 bg-white flex flex-col shrink-0 overflow-hidden relative z-10">
      {/* 헤더 */}
      <div className="p-4 border-b border-gray-200 flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          {root ? (
            <>
              <h3 className="text-sm font-semibold text-gray-900 truncate">{root.name}</h3>
              <span className={`inline-block mt-1 px-2 py-0.5 text-xs rounded ${getTypeBadgeColor(root.type)}`}>
                {root.type}
              </span>
            </>
          ) : (
            <h3 className="text-sm font-semibold text-gray-400">법령 정보 없음</h3>
          )}
        </div>
        <button
          onClick={onClose}
          className="p-1 hover:bg-gray-100 rounded transition-colors shrink-0"
          aria-label="닫기"
        >
          <X className="w-4 h-4 text-gray-400" />
        </button>
      </div>

      {/* 로딩 */}
      {loading && (
        <div className="flex items-center justify-center py-8">
          <Loader2 className="w-5 h-5 text-amber-400 animate-spin" />
        </div>
      )}

      {/* 본문 */}
      {!loading && root && (
        <div className="flex-1 overflow-y-auto p-4 space-y-5 statute-scrollbar">
          {/* 기본 정보 */}
          <div className="space-y-2">
            {root.abbreviation && (
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-500">약칭</span>
                <span className="text-gray-700">{root.abbreviation}</span>
              </div>
            )}
            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-500">인용 횟수</span>
              <span className="text-gray-700">{root.citation_count.toLocaleString()}회</span>
            </div>
          </div>

          {/* AI 요약 */}
          {root.ai_summary && (
            <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
              <div className="flex items-center gap-1.5 mb-2">
                <Sparkles className="w-3.5 h-3.5 text-amber-500" />
                <span className="text-xs font-semibold text-amber-600">AI 요약</span>
              </div>
              <p className="text-sm text-gray-700 leading-relaxed">{root.ai_summary}</p>
            </div>
          )}

          {/* 원문 보기 (펼치기/접기) */}
          {root.content && <StatuteContentSection content={root.content} supplementary={root.supplementary} />}

          <hr className="border-gray-200" />

          {/* 상위 법령 */}
          <StatuteList title="상위 법령" nodes={data.upper} onNodeClick={onNodeClick} />

          {/* 하위 법령 */}
          <StatuteList title="하위 법령" nodes={data.lower} onNodeClick={onNodeClick} />

          {/* 관련 법령 */}
          <StatuteList title="관련 법령" nodes={data.related} onNodeClick={onNodeClick} />
        </div>
      )}
    </div>
  )
})
