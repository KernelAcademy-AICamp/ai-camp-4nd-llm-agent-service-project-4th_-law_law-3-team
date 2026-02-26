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
      <div className="flex-1 flex items-center justify-center text-[#86868B] min-h-[400px]">
        <div className="text-center p-12 rounded-3xl border border-black/[0.06] bg-white shadow-apple-sm">
          <div className="w-20 h-20 bg-[#007AFF]/10 rounded-full flex items-center justify-center mx-auto mb-6">
            <svg
              className="w-10 h-10 text-[#007AFF]"
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
          <h3 className="text-xl font-bold text-[#1D1D1F] mb-2">타임라인이 비어있습니다</h3>
          <p className="text-sm text-[#86868B]">
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
              <span className="inline-flex items-center px-4 py-1.5 rounded-lg bg-[#007AFF] text-white text-sm font-bold shadow-apple backdrop-blur-md">
                {group.dateLabel}
              </span>
            </div>

            <div className="relative">
              {/* 왼쪽 수직선 */}
              <div className="absolute left-8 top-0 bottom-0 w-[2px] bg-gradient-to-b from-[#007AFF]/50 via-gray-300 to-transparent" />

              <div className="space-y-10">
                {group.items.map((item) => (
                  <div key={item.id} className="relative pl-24 group">
                    {/* 날짜/시간 라벨 (선 왼쪽) */}
                    {/* 위치 조정: top-6, left-0, width-20 */}
                    <div className="absolute left-0 top-7 w-[28px] text-right flex justify-end">
                      <span className="text-[11px] font-bold text-[#86868B] bg-white px-1.5 py-0.5 rounded leading-tight shadow-apple-sm">
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
                          ? 'bg-[#007AFF] border-white shadow-[0_0_15px_rgba(0,122,255,0.4)] scale-125'
                          : 'bg-white border-gray-300 group-hover:border-[#007AFF] group-hover:bg-[#F5F5F7]'}
                      `}
                    />

                    {/* 연결 선 (노드 -> 카드) */}
                    <div className="absolute left-[34px] top-[40px] w-14 h-[2px] bg-gradient-to-r from-gray-300 to-transparent" />

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
