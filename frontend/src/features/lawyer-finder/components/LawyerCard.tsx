'use client'

import { useState } from 'react'
import type { Lawyer } from '../types'

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
          <div className="flex items-center gap-2 mb-1">
            <h3 className="font-semibold text-gray-900 truncate">{lawyer.name}</h3>
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
