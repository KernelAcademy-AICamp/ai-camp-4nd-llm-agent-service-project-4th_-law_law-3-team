'use client'

import { useCallback, useRef, useState } from 'react'
import type {
  LawyerPersona,
  PersonaTone,
  PersonaUpdateRequest,
  TargetAudience,
  TrendCategory,
} from '../types'

const CATEGORY_LABELS: Record<TrendCategory, string> = {
  all: '전체',
  criminal: '형사',
  civil: '민사',
  labor: '노동',
  family: '가사',
  administrative: '행정',
  corporate: '기업',
  ip: '지식재산',
}

const TONE_LABELS: Record<PersonaTone, string> = {
  professional: '전문가',
  casual: '친근한',
  storytelling: '스토리텔링',
  educational: '교육형',
}

const AUDIENCE_LABELS: Record<TargetAudience, string> = {
  general_public: '일반 대중',
  business: '기업/사업자',
  legal_student: '법학 학생',
  legal_professional: '법률 전문가',
}

const SPECIALTY_OPTIONS: TrendCategory[] = [
  'criminal', 'civil', 'labor', 'family', 'administrative', 'corporate', 'ip',
]

interface PersonaBannerProps {
  persona: LawyerPersona
  onEdit: () => void
  onQuickUpdate: (update: PersonaUpdateRequest) => void
}

export function PersonaBanner({ persona, onEdit, onQuickUpdate }: PersonaBannerProps) {
  const [showSpecialtyPopover, setShowSpecialtyPopover] = useState(false)
  const popoverRef = useRef<HTMLDivElement>(null)

  const specialties = persona.specialty_areas
    .map((a) => CATEGORY_LABELS[a] ?? a)
    .join(', ')

  const handleAudienceChange = useCallback(
    (value: string) => {
      onQuickUpdate({ target_audience: value as TargetAudience })
    },
    [onQuickUpdate],
  )

  const handleToneChange = useCallback(
    (value: string) => {
      onQuickUpdate({ preferred_tone: value as PersonaTone })
    },
    [onQuickUpdate],
  )

  const handleSpecialtyToggle = useCallback(
    (category: TrendCategory) => {
      const current = persona.specialty_areas
      const updated = current.includes(category)
        ? current.filter((c) => c !== category)
        : [...current, category]

      if (updated.length === 0) return
      onQuickUpdate({ specialty_areas: updated })
    },
    [persona.specialty_areas, onQuickUpdate],
  )

  return (
    <div className="bg-blue-50 border-b border-blue-100 px-6 py-2.5">
      <div className="max-w-6xl mx-auto flex items-center justify-between text-sm">
        <div className="flex items-center gap-3 text-blue-700 min-w-0">
          {/* 전문분야: 체크박스 팝오버 */}
          <div className="relative">
            <button
              onClick={() => setShowSpecialtyPopover(!showSpecialtyPopover)}
              className="flex items-center gap-1 px-2 py-0.5 rounded hover:bg-blue-100 transition-colors font-medium"
            >
              <span className="text-blue-500 text-xs">전문분야</span>
              <span>{specialties}</span>
              <svg className="w-3 h-3 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>
            {showSpecialtyPopover && (
              <>
                <div
                  className="fixed inset-0 z-10"
                  onClick={() => setShowSpecialtyPopover(false)}
                />
                <div
                  ref={popoverRef}
                  className="absolute top-full left-0 mt-1 bg-white border border-gray-200 rounded-lg shadow-lg p-2 z-20 min-w-[160px]"
                >
                  {SPECIALTY_OPTIONS.map((cat) => (
                    <label
                      key={cat}
                      className="flex items-center gap-2 px-2 py-1.5 rounded hover:bg-gray-50 cursor-pointer text-gray-700"
                    >
                      <input
                        type="checkbox"
                        checked={persona.specialty_areas.includes(cat)}
                        onChange={() => handleSpecialtyToggle(cat)}
                        className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                      />
                      <span className="text-sm">{CATEGORY_LABELS[cat]}</span>
                    </label>
                  ))}
                </div>
              </>
            )}
          </div>

          <span className="text-blue-300">|</span>

          {/* 시청자층: 단일 select */}
          <label className="flex items-center gap-1">
            <span className="text-blue-500 text-xs">시청자</span>
            <select
              value={persona.target_audience}
              onChange={(e) => handleAudienceChange(e.target.value)}
              className="bg-transparent border-none text-blue-700 text-sm font-medium focus:ring-0 focus:outline-none cursor-pointer py-0 pr-6 pl-0 appearance-auto"
            >
              {Object.entries(AUDIENCE_LABELS).map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
          </label>

          <span className="text-blue-300">|</span>

          {/* 수준: 단일 select */}
          <label className="flex items-center gap-1">
            <span className="text-blue-500 text-xs">톤</span>
            <select
              value={persona.preferred_tone}
              onChange={(e) => handleToneChange(e.target.value)}
              className="bg-transparent border-none text-blue-700 text-sm font-medium focus:ring-0 focus:outline-none cursor-pointer py-0 pr-6 pl-0 appearance-auto"
            >
              {Object.entries(TONE_LABELS).map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
          </label>
        </div>

        <button
          onClick={onEdit}
          className="ml-3 px-3 py-1 text-xs font-medium text-blue-600 border border-blue-300 rounded-md hover:bg-blue-100 transition-colors shrink-0"
        >
          전체 수정
        </button>
      </div>
    </div>
  )
}
