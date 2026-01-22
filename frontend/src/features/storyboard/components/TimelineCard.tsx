'use client'

import { useState } from 'react'
import type { TimelineItem, EditMode, ImageStatus, Participant } from '../types'
import { PARTICIPANT_ROLE_CONFIG } from '../types'

interface TimelineCardProps {
  item: TimelineItem
  isSelected: boolean
  editMode: EditMode
  onSelect: () => void
  onEdit: () => void
  onDelete: () => void
  onGenerateImage?: () => void
  isGeneratingImage?: boolean
}

function ParticipantBadge({ participant }: { participant: Participant }) {
  const config = PARTICIPANT_ROLE_CONFIG[participant.role]

  return (
    <div className={`flex flex-col gap-1 p-2 rounded-lg ${config.bgColor} border border-slate-700/50`}>
      <div className="flex items-center gap-2">
        <span className={`text-xs font-semibold px-2 py-0.5 rounded ${config.bgColor} ${config.color}`}>
          {config.label}
        </span>
        <span className="text-sm font-medium text-slate-200">{participant.name}</span>
      </div>
      {participant.action && (
        <p className="text-xs text-slate-400 pl-1">→ {participant.action}</p>
      )}
    </div>
  )
}

export function TimelineCard({
  item,
  isSelected,
  editMode,
  onSelect,
  onEdit,
  onDelete,
  onGenerateImage,
  isGeneratingImage = false,
}: TimelineCardProps) {
  const [isDetailExpanded, setIsDetailExpanded] = useState(false)
  const imageStatus: ImageStatus | undefined = item.imageStatus

  const hasDetailedInfo = item.descriptionDetailed || item.participantsDetailed?.length

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
      {/* 헤더: 장면 번호 + 날짜/장소 */}
      <div className="flex items-center justify-between px-4 py-2 bg-slate-800/50 border-b border-slate-700/50">
        <div className="flex items-center gap-3">
          {item.sceneNumber && (
            <span className="flex items-center justify-center w-8 h-8 rounded-lg bg-blue-500/20 text-blue-400 text-sm font-bold">
              {item.sceneNumber}
            </span>
          )}
          <div className="flex flex-col">
            <span className="text-sm font-medium text-slate-300">
              {item.date && item.date !== '날짜 미상' ? item.date : '정보 없음'}
              {item.time && <span className="ml-2 text-blue-400">{item.time}</span>}
            </span>
            <div className="flex items-center gap-2 text-xs text-slate-500">
              {item.timeOfDay && <span>{item.timeOfDay}</span>}
              {item.location && (
                <>
                  {item.timeOfDay && <span>·</span>}
                  <span>{item.location}</span>
                </>
              )}
            </div>
          </div>
        </div>

      </div>

      {/* 이미지가 있을 경우 표시 */}
      {item.imageUrl && (
        <div className="relative w-full bg-slate-950">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={item.imageUrl}
            alt={item.title}
            className="w-full h-auto object-contain transition-transform duration-500 group-hover/card:scale-105"
          />
        </div>
      )}

      {/* 이미지 생성 중 표시 */}
      {isGeneratingImage && (
        <div className="relative w-full h-48 overflow-hidden bg-slate-950 flex items-center justify-center">
          <div className="text-center">
            <svg className="animate-spin h-10 w-10 text-blue-400 mx-auto mb-3" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
            <p className="text-slate-400 text-sm">이미지 생성 중...</p>
          </div>
        </div>
      )}

      <div className="p-5 relative space-y-4">
        {/* 제목 & 액션 버튼 */}
        <div className="flex justify-between items-start gap-4">
          <h3 className={`font-bold text-xl leading-snug ${isSelected ? 'text-blue-400' : 'text-slate-100'} group-hover/card:text-blue-400 transition-colors`}>
            {item.title}
          </h3>

          {/* 액션 버튼 */}
          <div className="flex gap-1 shrink-0">
            {onGenerateImage && !item.imageUrl && !isGeneratingImage && (
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation()
                  onGenerateImage()
                }}
                className="p-2 text-slate-400 hover:text-purple-400 hover:bg-purple-500/10 rounded-lg transition-colors"
                title="이미지 생성"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
              </button>
            )}

            {onGenerateImage && item.imageUrl && !isGeneratingImage && (
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation()
                  onGenerateImage()
                }}
                className="p-2 text-slate-400 hover:text-purple-400 hover:bg-purple-500/10 rounded-lg transition-colors"
                title="이미지 재생성"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
              </button>
            )}

            {editMode === 'edit' && (
              <>
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
              </>
            )}
          </div>
        </div>

        {/* 한 줄 요약 */}
        {item.descriptionShort && (
          <p className="text-slate-300 text-sm leading-relaxed">
            {item.descriptionShort}
          </p>
        )}

        {/* 상세 설명 (접기/펼치기) */}
        {item.descriptionDetailed && (
          <div className="space-y-2">
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation()
                setIsDetailExpanded(!isDetailExpanded)
              }}
              className="flex items-center gap-2 text-xs text-slate-500 hover:text-slate-300 transition-colors"
            >
              <svg
                className={`w-4 h-4 transition-transform ${isDetailExpanded ? 'rotate-90' : ''}`}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
              상세 설명 {isDetailExpanded ? '접기' : '펼치기'}
            </button>
            {isDetailExpanded && (
              <p className="text-slate-400 text-sm leading-relaxed whitespace-pre-line pl-6 border-l-2 border-slate-700">
                {item.descriptionDetailed}
              </p>
            )}
          </div>
        )}

        {/* 기존 description 표시 (새 필드가 없을 경우) */}
        {!item.descriptionShort && !item.descriptionDetailed && item.description && (
          <p className="text-slate-400 text-sm leading-relaxed whitespace-pre-line">
            {item.description}
          </p>
        )}

        {/* 참여자 (상세 정보가 있는 경우) */}
        {item.participantsDetailed && item.participantsDetailed.length > 0 && (
          <div className="space-y-2">
            <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider flex items-center gap-2">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
              </svg>
              등장인물
            </h4>
            <div className="grid gap-2">
              {item.participantsDetailed.map((participant, idx) => (
                <ParticipantBadge key={idx} participant={participant} />
              ))}
            </div>
          </div>
        )}

        {/* 기존 participants 표시 (상세 정보가 없는 경우) */}
        {(!item.participantsDetailed || item.participantsDetailed.length === 0) && item.participants.length > 0 && (
          <div className="flex flex-wrap gap-2 pt-2 border-t border-slate-700/50">
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

        {/* 증거물 */}
        {item.evidenceItems && item.evidenceItems.length > 0 && (
          <div className="flex flex-wrap gap-2 pt-2">
            <span className="text-xs text-slate-500 flex items-center gap-1">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
              </svg>
              증거:
            </span>
            {item.evidenceItems.map((evidence, idx) => (
              <span
                key={idx}
                className="text-xs px-2 py-1 rounded bg-slate-700/50 text-slate-300"
              >
                {evidence}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* 선택되었을 때 좌측 강조 선 */}
      {isSelected && (
        <div className="absolute left-0 top-0 bottom-0 w-1 bg-gradient-to-b from-blue-400 to-indigo-500" />
      )}
    </div>
  )
}
