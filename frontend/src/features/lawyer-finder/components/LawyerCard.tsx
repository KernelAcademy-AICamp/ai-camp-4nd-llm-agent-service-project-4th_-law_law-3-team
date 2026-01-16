'use client'

import type { Lawyer } from '../types'

interface LawyerCardProps {
  lawyer: Lawyer
  selected?: boolean
  onClick?: () => void
}

export function LawyerCard({ lawyer, selected, onClick }: LawyerCardProps) {
  return (
    <div
      className={`p-4 border rounded-lg cursor-pointer transition-all ${
        selected
          ? 'border-blue-500 bg-blue-50 shadow-md'
          : 'border-gray-200 hover:border-gray-300 hover:shadow-sm'
      }`}
      onClick={onClick}
    >
      <div className="flex gap-3">
        {/* 사진 */}
        <div className="flex-shrink-0">
          {lawyer.photo_url ? (
            <img
              src={lawyer.photo_url}
              alt={lawyer.name}
              className="w-16 h-20 object-cover rounded"
              onError={(e) => {
                e.currentTarget.src = '/placeholder-lawyer.png'
              }}
            />
          ) : (
            <div className="w-16 h-20 bg-gray-200 rounded flex items-center justify-center">
              <span className="text-gray-400 text-2xl">?</span>
            </div>
          )}
        </div>

        {/* 정보 */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <h3 className="font-semibold text-gray-900 truncate">{lawyer.name}</h3>
            <span
              className={`text-xs px-2 py-0.5 rounded ${
                lawyer.status === '개업'
                  ? 'bg-green-100 text-green-700'
                  : 'bg-gray-100 text-gray-600'
              }`}
            >
              {lawyer.status}
            </span>
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
