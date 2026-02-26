'use client'

import { X, ChevronRight, Loader2 } from 'lucide-react'
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
  if (nodes.length === 0) return null

  return (
    <div>
      <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
        {title} ({nodes.length})
      </h4>
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
    <div className="w-80 border-l border-slate-700 bg-slate-800 flex flex-col shrink-0 overflow-hidden">
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
        <div className="flex-1 overflow-y-auto p-4 space-y-5">
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

          <hr className="border-slate-700" />

          {/* 상위 법령 */}
          <StatuteList title="상위 법령" nodes={data.upper} onNodeClick={onNodeClick} />

          {/* 하위 법령 */}
          <StatuteList title="하위 법령" nodes={data.lower} onNodeClick={onNodeClick} />

          {/* 관련 법령 */}
          <StatuteList title="관련 법령" nodes={data.related} onNodeClick={onNodeClick} />

          {/* 관계 없는 경우 */}
          {data.upper.length === 0 && data.lower.length === 0 && data.related.length === 0 && (
            <p className="text-sm text-slate-500 text-center py-4">
              연결된 법령이 없습니다.
            </p>
          )}
        </div>
      )}
    </div>
  )
}
