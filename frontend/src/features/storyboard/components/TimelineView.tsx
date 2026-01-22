'use client'

import { useMemo } from 'react'
import type { TimelineItem, EditMode } from '../types'
import { TimelineCard } from './TimelineCard'

interface TimelineViewProps {
  items: TimelineItem[]
  editMode: EditMode
  selectedItemId: string | null
  onItemSelect: (id: string) => void
  onItemEdit: (id: string) => void
  onItemDelete: (id: string) => void
  onItemGenerateImage?: (id: string) => void
  generatingImageIds?: Set<string>
}

export function TimelineView({
  items,
  editMode,
  selectedItemId,
  onItemSelect,
  onItemEdit,
  onItemDelete,
  onItemGenerateImage,
  generatingImageIds = new Set(),
}: TimelineViewProps) {
  // 날짜별 그룹화
  const groupedItems = useMemo(() => {
    const groups: { [key: string]: TimelineItem[] } = {}

    items.forEach((item) => {
      let dateKey = '기타'

      const ymdMatch = item.date.match(/(\d{4})[-.](\d{1,2})/)
      const korMatch = item.date.match(/(\d{4})년\s*(\d{1,2})월/)

      if (ymdMatch) dateKey = `${ymdMatch[1]}.${ymdMatch[2].padStart(2, '0')}`
      else if (korMatch) dateKey = `${korMatch[1]}.${korMatch[2].padStart(2, '0')}`
      else {
        const yearMatch = item.date.match(/(\d{4})/)
        if (yearMatch) dateKey = yearMatch[1]
      }

      if (!groups[dateKey]) groups[dateKey] = []
      groups[dateKey].push(item)
    })

    const orderedKeys: string[] = []
    items.forEach(item => {
      let dateKey = '기타'
      const ymdMatch = item.date.match(/(\d{4})[-.](\d{1,2})/)
      const korMatch = item.date.match(/(\d{4})년\s*(\d{1,2})월/)

      if (ymdMatch) dateKey = `${ymdMatch[1]}.${ymdMatch[2].padStart(2, '0')}`
      else if (korMatch) dateKey = `${korMatch[1]}.${korMatch[2].padStart(2, '0')}`
      else {
        const yearMatch = item.date.match(/(\d{4})/)
        if (yearMatch) dateKey = yearMatch[1]
      }

      if (!orderedKeys.includes(dateKey)) orderedKeys.push(dateKey)
    })

    return orderedKeys.map(key => ({
      dateLabel: key,
      items: groups[key]
    }))
  }, [items])

  if (items.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center text-slate-500 min-h-[400px]">
        <div className="text-center p-12 rounded-3xl border border-white/5 bg-white/5 backdrop-blur-sm">
          <div className="w-20 h-20 bg-gradient-to-br from-blue-500/20 to-indigo-500/20 rounded-full flex items-center justify-center mx-auto mb-6">
            <svg
              className="w-10 h-10 text-blue-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              aria-hidden="true"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
          </div>
          <h3 className="text-xl font-bold text-white mb-2">타임라인이 비어있습니다</h3>
          <p className="text-sm text-slate-400">
            사건 내용을 입력하면 AI가 자동으로<br />
            타임라인과 이미지를 생성합니다
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-y-auto p-4 sm:p-8 custom-scrollbar">
      <div className="max-w-4xl mx-auto space-y-16">
        {groupedItems.map((group) => (
          <div key={group.dateLabel} className="relative">
            {/* 날짜 헤더 (스티키) - 왼쪽 정렬 */}
            <div className="sticky top-0 z-20 mb-8">
              <span className="inline-flex items-center px-4 py-1.5 rounded-lg bg-blue-600/90 text-white text-sm font-bold shadow-lg shadow-blue-900/50 backdrop-blur-md border border-blue-400/30">
                {group.dateLabel}
              </span>
            </div>

            <div className="relative">
              {/* 왼쪽 수직선 */}
              <div className="absolute left-8 top-0 bottom-0 w-[2px] bg-gradient-to-b from-blue-500/50 via-slate-700/50 to-transparent" />

              <div className="space-y-10">
                {group.items.map((item) => (
                  <div key={item.id} className="relative pl-24 group">
                    {/* 날짜/시간 라벨 (선 왼쪽) */}
                    {/* 위치 조정: top-6, left-0, width-20 */}
                    <div className="absolute left-0 top-7 w-[28px] text-right flex justify-end">
                      <span className="text-[11px] font-bold text-slate-400 bg-slate-800/50 px-1.5 py-0.5 rounded leading-tight">
                        {item.date.includes(group.dateLabel)
                          ? item.date.replace(group.dateLabel, '').replace(/^[-.년월\s]+/, '')
                          : item.date.slice(-2)}
                      </span>
                    </div>

                    {/* 타임라인 노드 */}
                    <div
                      className={`
                        absolute left-[26px] top-8 w-4 h-4 rounded-full border-[3px] z-10 transition-all duration-300
                        ${selectedItemId === item.id
                          ? 'bg-blue-500 border-white shadow-[0_0_15px_rgba(59,130,246,0.6)] scale-125'
                          : 'bg-slate-900 border-slate-500 group-hover:border-blue-400 group-hover:bg-slate-800'}
                      `}
                    />

                    {/* 연결 선 (노드 -> 카드) */}
                    <div className="absolute left-[34px] top-[40px] w-14 h-[2px] bg-gradient-to-r from-slate-600/50 to-transparent" />

                    <TimelineCard
                      item={item}
                      isSelected={selectedItemId === item.id}
                      editMode={editMode}
                      onSelect={() => onItemSelect(item.id)}
                      onEdit={() => onItemEdit(item.id)}
                      onDelete={() => onItemDelete(item.id)}
                      onGenerateImage={onItemGenerateImage ? () => onItemGenerateImage(item.id) : undefined}
                      isGeneratingImage={generatingImageIds.has(item.id)}
                    />
                  </div>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
