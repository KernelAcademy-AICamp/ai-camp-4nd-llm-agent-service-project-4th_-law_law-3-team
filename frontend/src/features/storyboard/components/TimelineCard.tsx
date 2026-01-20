'use client'

import type { TimelineItem, EditMode } from '../types'
import Image from 'next/image'

interface TimelineCardProps {
  item: TimelineItem
  isSelected: boolean
  editMode: EditMode
  onSelect: () => void
  onEdit: () => void
  onDelete: () => void
}

export function TimelineCard({
  item,
  isSelected,
  editMode,
  onSelect,
  onEdit,
  onDelete,
}: TimelineCardProps) {
  return (
    <div
      role="button"
      tabIndex={0}
      onClick={onSelect}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault()
          onSelect()
        }
      }}
      className={`
        relative group/card overflow-hidden rounded-2xl border transition-all duration-300 ease-in-out
        ${isSelected
          ? 'bg-slate-800/80 border-blue-500 shadow-[0_0_30px_-5px_rgba(59,130,246,0.3)] ring-1 ring-blue-500/50'
          : 'bg-slate-900/50 border-slate-700/50 hover:border-slate-600 hover:bg-slate-800/60 hover:shadow-xl hover:-translate-y-1'
        }
        backdrop-blur-xl
      `}
    >
      {/* 이미지가 있을 경우 표시 */}
      {item.imageUrl && (
        <div className="relative w-full h-48 overflow-hidden bg-slate-950">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={item.imageUrl}
            alt={item.title}
            className="w-full h-full object-cover transition-transform duration-500 group-hover/card:scale-105"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-slate-900 via-transparent to-transparent opacity-60" />
        </div>
      )}

      <div className="p-6 relative">
        {/* 헤더 영역 (제목 & 액션) */}
        <div className="flex justify-between items-start gap-4 mb-3">
          <h3 className={`font-bold text-xl leading-snug ${isSelected ? 'text-blue-400' : 'text-slate-100'} group-hover/card:text-blue-400 transition-colors`}>
            {item.title}
          </h3>

          {/* 편집 모드 액션 버튼 */}
          {editMode === 'edit' && (
            <div className="flex gap-1 shrink-0 opacity-100 transition-opacity">
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation()
                  onEdit()
                }}
                className="p-2 text-slate-400 hover:text-blue-400 hover:bg-blue-500/10 rounded-lg transition-colors"
                title="편집"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                </svg>
              </button>
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation()
                  onDelete()
                }}
                className="p-2 text-slate-400 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-colors"
                title="삭제"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
              </button>
            </div>
          )}
        </div>

        {/* 본문 */}
        <p className="text-slate-400 text-sm leading-relaxed whitespace-pre-line mb-4 font-light">
          {item.description}
        </p>

        {/* 하단 정보 (참여자) */}
        {item.participants.length > 0 && (
          <div className="flex flex-wrap gap-2 mt-auto pt-4 border-t border-slate-700/50">
            {item.participants.map((participant, idx) => (
              <span
                key={idx}
                className="inline-flex items-center px-3 py-1 rounded-full bg-slate-800/80 text-slate-300 text-xs font-medium border border-slate-700/50"
              >
                <svg className="w-3 h-3 mr-1.5 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                </svg>
                {participant}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* 선택되었을 때 좌측 강조 선 (그라데이션) */}
      {isSelected && (
        <div className="absolute left-0 top-0 bottom-0 w-1 bg-gradient-to-b from-blue-400 to-indigo-500" />
      )}
    </div>
  )
}
