'use client'

import { useState } from 'react'
import type { Office, Lawyer } from '../types'

// 변호사 아이템 컴포넌트 (이미지 에러 상태 관리용)
function LawyerItem({
  lawyer,
  onClick,
}: {
  lawyer: Lawyer
  onClick: () => void
}) {
  const [imgError, setImgError] = useState(false)

  return (
    <div
      onClick={onClick}
      className="p-4 border rounded-lg cursor-pointer transition-all border-gray-200 hover:border-blue-300 hover:shadow-sm hover:bg-blue-50/30"
    >
      <div className="flex gap-3">
        {/* 사진 */}
        <div className="flex-shrink-0 w-14 h-[72px]">
          {lawyer.photo_url && !imgError ? (
            <img
              src={lawyer.photo_url}
              alt={lawyer.name}
              className="w-full h-full object-cover rounded"
              onError={() => setImgError(true)}
            />
          ) : (
            <div className="w-full h-full bg-gray-200 rounded flex items-center justify-center">
              <svg
                className="w-7 h-7 text-gray-400"
                fill="currentColor"
                viewBox="0 0 24 24"
              >
                <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z" />
              </svg>
            </div>
          )}
        </div>

        {/* 정보 */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <h3 className="font-semibold text-gray-900">{lawyer.name}</h3>
          </div>

          {lawyer.phone && (
            <p className="text-sm text-gray-500">{lawyer.phone}</p>
          )}

          {lawyer.birth_year && (
            <p className="text-xs text-gray-400 mt-1">
              {lawyer.birth_year}년생
            </p>
          )}
        </div>
      </div>
    </div>
  )
}

interface OfficeDetailPanelProps {
  office: Office
  onClose: () => void
  onLawyerSelect: (lawyer: Lawyer) => void
}

export function OfficeDetailPanel({
  office,
  onClose,
  onLawyerSelect,
}: OfficeDetailPanelProps) {
  return (
    <div className="w-96 bg-white border-r border-gray-200 flex flex-col h-full">
      {/* 헤더 */}
      <div className="p-4 border-b flex items-start justify-between">
        <div className="flex-1 min-w-0">
          <h2 className="text-lg font-bold text-gray-900 truncate">
            {office.name}
          </h2>
          {office.address && (
            <p className="text-sm text-gray-500 mt-1">{office.address}</p>
          )}
        </div>
        <button
          onClick={onClose}
          className="ml-2 p-1 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded transition-colors"
          title="닫기"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* 사무소 정보 */}
      <div className="p-4 bg-gray-50 border-b">
        <div className="flex items-center gap-4 text-sm">
          <div className="flex items-center gap-1.5 text-blue-600">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
            </svg>
            <span className="font-medium">{office.lawyers.length}명의 변호사</span>
          </div>
        </div>
      </div>

      {/* 변호사 리스트 */}
      <div className="flex-1 overflow-y-auto">
        <div className="p-3 space-y-2">
          {office.lawyers.map((lawyer) => (
            <LawyerItem
              key={lawyer.id}
              lawyer={lawyer}
              onClick={() => onLawyerSelect(lawyer)}
            />
          ))}
        </div>
      </div>

      {/* 푸터 */}
      <div className="p-3 border-t bg-gray-50">
        <button
          onClick={onClose}
          className="w-full py-2 text-sm text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-lg transition-colors"
        >
          ← 검색 결과로 돌아가기
        </button>
      </div>
    </div>
  )
}
