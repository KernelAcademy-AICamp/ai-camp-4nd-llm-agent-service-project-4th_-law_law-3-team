'use client'

import { useState } from 'react'
import type { Lawyer } from '../types'
import { SPECIALTY_CATEGORIES, SPECIALTY_TO_CATEGORY } from '../constants'

// 카테고리별 색상 매핑
const CATEGORY_COLORS: Record<string, string> = {
  'civil-family': 'bg-pink-50 text-pink-700 border-pink-200',
  'criminal': 'bg-red-50 text-red-700 border-red-200',
  'real-estate': 'bg-amber-50 text-amber-700 border-amber-200',
  'labor': 'bg-orange-50 text-orange-700 border-orange-200',
  'corporate': 'bg-blue-50 text-blue-700 border-blue-200',
  'finance': 'bg-emerald-50 text-emerald-700 border-emerald-200',
  'tax': 'bg-lime-50 text-lime-700 border-lime-200',
  'public': 'bg-purple-50 text-purple-700 border-purple-200',
  'ip': 'bg-cyan-50 text-cyan-700 border-cyan-200',
  'it-media': 'bg-indigo-50 text-indigo-700 border-indigo-200',
  'medical': 'bg-teal-50 text-teal-700 border-teal-200',
  'international': 'bg-violet-50 text-violet-700 border-violet-200',
}

interface LawyerCardProps {
  lawyer: Lawyer
  selected?: boolean
  onClick?: () => void
}

export function LawyerCard({ lawyer, selected, onClick }: LawyerCardProps) {
  const [imgError, setImgError] = useState(false)

  return (
    <div
      className={`p-4 border rounded-lg cursor-pointer transition-all ${
        selected
          ? 'border-blue-500 bg-blue-50 shadow-md'
          : 'border-gray-200 hover:border-gray-300 hover:shadow-sm'
      }`}
      onClick={onClick}
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') onClick?.() }}
      role="button"
      tabIndex={0}
    >
      <div className="flex gap-3">
        {/* 사진 */}
        <div className="flex-shrink-0">
          {lawyer.photo_url && !imgError ? (
            <img
              src={lawyer.photo_url}
              alt={lawyer.name}
              className="w-16 h-20 object-cover rounded"
              onError={() => setImgError(true)}
            />
          ) : (
            <div className="w-16 h-20 bg-gray-200 rounded flex items-center justify-center">
              <svg
                className="w-8 h-8 text-gray-400"
                fill="currentColor"
                viewBox="0 0 24 24"
                aria-hidden="true"
              >
                <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z" />
              </svg>
            </div>
          )}
        </div>

        {/* 정보 */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1 flex-wrap">
            <h3 className="font-semibold text-gray-900 truncate">{lawyer.name}</h3>
            
            {/* 전문분야 태그 (이름 옆에 표시) */}
            {lawyer.specialties && lawyer.specialties.length > 0 && (
              <div className="flex gap-1">
                {(() => {
                  // 1. 전문분야 -> 대분류 ID 매핑 및 중복 제거
                  const categoryIds = Array.from(new Set(
                    lawyer.specialties
                      .map(spec => SPECIALTY_TO_CATEGORY[spec])
                      .filter(Boolean)
                  ))

                  // 2. 대분류 객체 찾기
                  const categories = categoryIds
                    .map(id => SPECIALTY_CATEGORIES.find(c => c.id === id))
                    .filter((c): c is typeof SPECIALTY_CATEGORIES[0] => !!c)
                    .slice(0, 3) // 최대 3개 대분류 표시 (공간 제약)

                  // 3. 매핑된 대분류가 없으면 원본 표시 (Fallback)
                  if (categories.length === 0) {
                    return lawyer.specialties.slice(0, 2).map((spec) => (
                      <span
                        key={spec}
                        className="px-1.5 py-0.5 text-[10px] font-medium bg-gray-100 text-gray-600 rounded border border-gray-200"
                      >
                        {spec}
                      </span>
                    ))
                  }

                  return categories.map((cat) => (
                    <span
                      key={cat.id}
                      className={`px-1.5 py-0.5 text-[10px] font-medium rounded border flex items-center gap-1 ${CATEGORY_COLORS[cat.id] || 'bg-gray-50 text-gray-700 border-gray-200'}`}
                    >
                      {cat.name}
                    </span>
                  ))
                })()}
              </div>
            )}
          </div>

          {lawyer.office_name && (
            <p className="text-sm text-gray-600 truncate">{lawyer.office_name}</p>
          )}

          {lawyer.address && (
            <p className="text-xs text-gray-500 truncate mt-1">{lawyer.address}</p>
          )}

          <div className="flex items-center gap-3 mt-2 text-xs text-gray-500">
            {lawyer.distance !== undefined && (
              <span className="text-blue-600 font-medium">
                {lawyer.distance < 1
                  ? `${Math.round(lawyer.distance * 1000)}m`
                  : `${lawyer.distance.toFixed(1)}km`}
              </span>
            )}
            {lawyer.phone && <span>{lawyer.phone}</span>}
          </div>
        </div>
      </div>
    </div>
  )
}
