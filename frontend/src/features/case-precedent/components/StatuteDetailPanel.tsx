'use client'

import { useState } from 'react'
import { X, ChevronRight, ChevronDown, ChevronUp, Loader2, Sparkles, FileText } from 'lucide-react'
import type { StatuteHierarchyResponse, StatuteNode } from '../types'

interface StatuteDetailPanelProps {
  data: StatuteHierarchyResponse
  loading: boolean
  onClose: () => void
  onNodeClick: (node: StatuteNode) => void
}

const TYPE_COLORS: Record<string, string> = {
  헌법: 'bg-red-500/20 text-red-400',
  법률: 'bg-amber-500/20 text-amber-400',
  대통령령: 'bg-blue-500/20 text-blue-400',
  총리령: 'bg-cyan-500/20 text-cyan-400',
  부령: 'bg-teal-500/20 text-teal-400',
  행정규칙: 'bg-slate-500/20 text-slate-400',
}

function getTypeBadgeColor(type: string): string {
  return TYPE_COLORS[type] || 'bg-slate-500/20 text-slate-400'
}

function StatuteList({
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
      <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
        {title} ({nodes.length})
      </h4>
      {nodes.length === 0 ? (
        <p className="text-xs text-slate-600 px-3 py-2">없음</p>
      ) : (
        <ul className="space-y-1">
          {nodes.map((node) => (
            <li key={node.id}>
              <button
                onClick={() => onNodeClick(node)}
                className="w-full text-left px-3 py-2 rounded-lg hover:bg-slate-700/50
                           transition-colors group flex items-center justify-between"
              >
                <div className="flex-1 min-w-0">
                  <span className="text-sm text-slate-200 group-hover:text-white truncate block">
                    {node.name}
                  </span>
                  <span className="text-xs text-slate-500">
                    {node.type}
                    {node.abbreviation && ` · ${node.abbreviation}`}
                  </span>
                </div>
                <ChevronRight className="w-4 h-4 text-slate-600 group-hover:text-slate-400 shrink-0 ml-2" />
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

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
        className="w-full flex items-center justify-center gap-2 px-3 py-2 text-sm text-amber-400
                   border border-amber-500/30 rounded-lg hover:bg-amber-500/10 transition-colors"
      >
        <FileText className="w-3.5 h-3.5" />
        원문 보기
        {isOpen ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
      </button>
      {isOpen && (
        <div className="mt-3 bg-slate-900/50 border border-slate-700 rounded-lg p-3 space-y-3">
          <pre className="text-xs text-slate-300 leading-relaxed whitespace-pre-wrap break-words font-sans">
            {content}
          </pre>
          {supplementary && (
            <>
              <hr className="border-slate-700" />
              <div>
                <span className="text-xs font-semibold text-slate-400 block mb-1">부칙</span>
                <pre className="text-xs text-slate-400 leading-relaxed whitespace-pre-wrap break-words font-sans">
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

export function StatuteDetailPanel({
  data,
  loading,
  onClose,
  onNodeClick,
}: StatuteDetailPanelProps) {
  const root = data.root

  return (
    <div className="w-80 h-full border-l border-slate-700 bg-slate-800 flex flex-col shrink-0 overflow-hidden relative z-10">
      {/* 헤더 */}
      <div className="p-4 border-b border-slate-700 flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          {root ? (
            <>
              <h3 className="text-sm font-semibold text-white truncate">{root.name}</h3>
              <span className={`inline-block mt-1 px-2 py-0.5 text-xs rounded ${getTypeBadgeColor(root.type)}`}>
                {root.type}
              </span>
            </>
          ) : (
            <h3 className="text-sm font-semibold text-slate-400">법령 정보 없음</h3>
          )}
        </div>
        <button
          onClick={onClose}
          className="p-1 hover:bg-slate-700 rounded transition-colors shrink-0"
          aria-label="닫기"
        >
          <X className="w-4 h-4 text-slate-400" />
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
                <span className="text-slate-400">약칭</span>
                <span className="text-slate-200">{root.abbreviation}</span>
              </div>
            )}
            <div className="flex items-center justify-between text-sm">
              <span className="text-slate-400">인용 횟수</span>
              <span className="text-slate-200">{root.citation_count.toLocaleString()}회</span>
            </div>
          </div>

          {/* AI 요약 */}
          {root.ai_summary && (
            <div className="bg-amber-500/10 border border-amber-500/20 rounded-lg p-3">
              <div className="flex items-center gap-1.5 mb-2">
                <Sparkles className="w-3.5 h-3.5 text-amber-400" />
                <span className="text-xs font-semibold text-amber-400">AI 요약</span>
              </div>
              <p className="text-sm text-slate-200 leading-relaxed">{root.ai_summary}</p>
            </div>
          )}

          {/* 원문 보기 (펼치기/접기) */}
          {root.content && <StatuteContentSection content={root.content} supplementary={root.supplementary} />}

          <hr className="border-slate-700" />

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
}
