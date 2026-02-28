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

function getStatuteDocumentText(root: StatuteNode | null): { title: string; content: string } | null {
  if (!root) {
    return null
  }

  const contents = [root.content, root.supplementary]
    .map((value) => value?.trim())
    .filter((value): value is string => Boolean(value))

  if (contents.length === 0) {
    return null
  }

  const parts: string[] = []
  if (contents[0]) {
    parts.push(contents[0])
  }
  if (contents[1]) {
    parts.push(contents[1])
  }

  return {
    title: contents.length > 1 ? '법령 원문 + 부칙' : '법령 원문',
    content: parts.join('\n\n---\n\n'),
  }
}

function renderReadableText(content: string): JSX.Element[] {
  const lines = content
    .replace(/\r/g, '')
    .replace(/\n{3,}/g, '\n\n')
    .split('\n')

  return lines.map((line, index) => {
    const trimmed = line.trim()
    const isHeading =
      /^제\s*\d+\s*조\b/.test(trimmed) ||
      /^제\s*\d+\s*장\b/.test(trimmed) ||
      /^부칙/.test(trimmed)

    if (!trimmed) {
      return <div key={`blank-${index}`} className="h-2" />
    }

    if (trimmed === '---') {
      return <div key={`sep-${index}`} className="my-3 border-t border-slate-600/80" />
    }

    if (isHeading) {
      return (
        <h5
          key={`heading-${index}`}
          className="mt-3 mb-2 text-sm font-semibold text-white border-l-4 border-amber-400 pl-3"
        >
          {trimmed}
        </h5>
      )
    }

    return (
      <p
        key={`line-${index}`}
        className="text-sm text-slate-200 leading-6 tracking-[0.01em] whitespace-pre-wrap break-keep"
      >
        {trimmed}
      </p>
    )
  })
}

export function StatuteDetailPanel({
  data,
  loading,
  onClose,
  onNodeClick,
}: StatuteDetailPanelProps) {
  const root = data.root
  const statuteDocument = getStatuteDocumentText(root)

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

          {/* 원문 */}
          {statuteDocument && (
            <details className="border border-slate-700 rounded-lg p-3">
              <summary className="text-sm font-semibold text-slate-200 cursor-pointer">
                {statuteDocument.title}
              </summary>
              <div className="mt-3 border border-slate-700/70 rounded-md bg-slate-900/80 p-3 max-h-96 overflow-y-auto">
                <div className="space-y-1">{renderReadableText(statuteDocument.content)}</div>
              </div>
            </details>
          )}

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
