'use client'

import { memo } from 'react'
import ReactMarkdown from 'react-markdown'
import { X, ChevronRight, Loader2, Sparkles, FileText } from 'lucide-react'
import { normalizeSummaryMarkdown } from '../utils/normalizeSummaryMarkdown'
import type { StatuteHierarchyResponse, StatuteNode } from '../types'

interface StatuteDetailPanelProps {
  data: StatuteHierarchyResponse
  loading: boolean
  onClose: () => void
  onNodeClick: (node: StatuteNode) => void
}

const TYPE_COLORS: Record<string, string> = {
  헌법: 'bg-orange-50 text-orange-700 border border-orange-200/60',
  법률: 'bg-amber-50 text-amber-700 border border-amber-200/60',
  대통령령: 'bg-blue-50 text-blue-700 border border-blue-200/60',
  총리령: 'bg-cyan-50 text-cyan-700 border border-cyan-200/60',
  부령: 'bg-teal-50 text-teal-700 border border-teal-200/60',
  행정규칙: 'bg-gray-50 text-gray-600 border border-gray-200/60',
}

function getTypeBadgeColor(type: string): string {
  return TYPE_COLORS[type] || 'bg-gray-100 text-gray-600'
}

const SECTION_ICONS: Record<string, string> = {
  '상위 법령': '↑',
  '하위 법령': '↓',
  '관련 법령': '↔',
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
      <h4 className="flex items-center gap-1.5 text-xs font-semibold text-gray-500 tracking-wider mb-2">
        <span className="text-sm opacity-60">{SECTION_ICONS[title] || '·'}</span>
        {title}
        <span className="text-gray-400 font-normal">({nodes.length})</span>
      </h4>
      {nodes.length === 0 ? (
        <p className="text-xs text-gray-400 px-3 py-2 italic">없음</p>
      ) : (
        <ul className="space-y-0.5">
          {nodes.map((node) => (
            <li key={node.id}>
              <button
                onClick={() => onNodeClick(node)}
                className="w-full text-left px-3 py-2 rounded-lg hover:bg-amber-50/60
                           transition-all duration-150 group flex items-center justify-between"
              >
                <div className="flex-1 min-w-0">
                  <span className="text-sm text-gray-700 group-hover:text-amber-800 truncate block font-medium">
                    {node.name}
                  </span>
                  <span className="text-[11px] text-gray-400 group-hover:text-amber-600/60">
                    {node.type}
                    {node.abbreviation && ` · ${node.abbreviation}`}
                  </span>
                </div>
                <ChevronRight className="w-3.5 h-3.5 text-gray-300 group-hover:text-amber-500 group-hover:translate-x-0.5 shrink-0 ml-2 transition-all" />
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
})

export const StatuteDetailPanel = memo(function StatuteDetailPanel({
  data,
  loading,
  onClose,
  onNodeClick,
}: StatuteDetailPanelProps) {
  const root = data.root

  return (
    <div className="w-80 h-full border-r border-gray-200/60 bg-white flex flex-col shrink-0 overflow-hidden relative z-10 shadow-sm">
      {/* 헤더 */}
      <div className="p-4 border-b border-gray-100 bg-gradient-to-b from-gray-50/80 to-white flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          {root ? (
            <>
              <h3 className="text-sm font-bold text-gray-900 truncate leading-tight">{root.name}</h3>
              <span className={`inline-block mt-1.5 px-2 py-0.5 text-[11px] font-medium rounded-md ${getTypeBadgeColor(root.type)}`}>
                {root.type}
              </span>
            </>
          ) : (
            <h3 className="text-sm font-semibold text-gray-400">법령 정보 없음</h3>
          )}
        </div>
        <button
          onClick={onClose}
          className="p-1.5 hover:bg-gray-100 rounded-lg transition-colors shrink-0"
          aria-label="닫기"
        >
          <X className="w-3.5 h-3.5 text-gray-400" />
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
        <div className="flex-1 overflow-y-auto p-4 space-y-4 statute-scrollbar">
          {/* 기본 정보 */}
          <div className="bg-gray-50/80 rounded-lg p-3 space-y-2">
            {root.abbreviation && (
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-400 text-xs">약칭</span>
                <span className="text-gray-700 font-medium text-xs">{root.abbreviation}</span>
              </div>
            )}
            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-400 text-xs">인용 횟수</span>
              <span className="text-amber-600 font-semibold text-xs">{root.citation_count.toLocaleString()}회</span>
            </div>
          </div>

          {/* AI 요약 */}
          {root.ai_summary && (
            <div className="bg-gradient-to-br from-amber-50 to-orange-50/50 border border-amber-200/60 rounded-xl p-3">
              <div className="flex items-center gap-1.5 mb-2">
                <Sparkles className="w-3.5 h-3.5 text-amber-500" />
                <span className="text-xs font-bold text-amber-700">AI 요약</span>
              </div>
              <div className="prose prose-sm max-w-none text-gray-700
                            prose-headings:text-amber-800 prose-headings:text-sm prose-headings:mt-3 prose-headings:mb-1
                            prose-p:text-[13px] prose-p:leading-relaxed prose-p:my-1
                            prose-li:text-[13px] prose-li:my-0
                            prose-strong:text-gray-800">
                <ReactMarkdown>{normalizeSummaryMarkdown(root.ai_summary)}</ReactMarkdown>
              </div>
            </div>
          )}

          {/* 원문 보기 (법령 검색 페이지로 이동) */}
          {root.id && (
            <a
              href={`/law-search?id=${root.id}`}
              target="_blank"
              rel="noopener noreferrer"
              className="w-full flex items-center justify-center gap-2 px-3 py-2 text-sm font-medium text-amber-700
                         bg-amber-50/80 border border-amber-200/60 rounded-xl hover:bg-amber-100/80 transition-all shadow-sm"
            >
              <FileText className="w-3.5 h-3.5" />
              원문 보기
            </a>
          )}

          <hr className="border-gray-100" />

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
