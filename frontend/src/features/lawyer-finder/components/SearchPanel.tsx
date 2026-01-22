'use client'

import { useState } from 'react'
import type { Lawyer } from '../types'
import { LawyerCard } from './LawyerCard'
import { SEOUL_DISTRICTS, SPECIALTY_CATEGORIES } from '../constants'

interface SearchPanelProps {
  lawyers: Lawyer[]
  loading: boolean
  selectedLawyer: Lawyer | null
  onLawyerSelect: (lawyer: Lawyer) => void
  onRadiusChange: (radius: number) => void
  onSearch: (query: string) => void
  onSearchReset: () => void
  radius: number
  totalCount: number
  sigungu: string
  onSigunguChange: (sigungu: string) => void
  searchQuery: string  // 부모에서 관리하는 검색어
  category: string  // 선택된 전문분야 카테고리 ID
  onCategoryChange: (category: string) => void
  specialty?: string  // 특정 전문분야 (예: "이혼")
}

const RADIUS_OPTIONS = [
  { value: 500, label: '500m' },
  { value: 1000, label: '1km' },
  { value: 3000, label: '3km' },
  { value: 5000, label: '5km' },
  { value: 10000, label: '10km' },
]

export function SearchPanel({
  lawyers,
  loading,
  selectedLawyer,
  onLawyerSelect,
  onRadiusChange,
  onSearch,
  onSearchReset,
  radius,
  totalCount,
  sigungu,
  onSigunguChange,
  searchQuery,
  category,
  onCategoryChange,
  specialty,
}: SearchPanelProps) {
  const [inputValue, setInputValue] = useState('')

  const handleSearch = () => {
    if (inputValue.trim()) {
      onSearch(inputValue.trim())
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch()
    }
  }

  const handleReset = () => {
    setInputValue('')
    onSearchReset()
  }

  return (
    <div className="w-96 bg-white border-r border-gray-200 flex flex-col h-full">
      {/* 검색 옵션 */}
      <div className="p-4 border-b space-y-3">
        {/* 지역 선택 */}
        <div className="flex items-center gap-2">
          <svg className="w-4 h-4 text-gray-500 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
          <select
            value={sigungu}
            onChange={(e) => onSigunguChange(e.target.value)}
            className="flex-1 px-3 py-2 border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="">서울 전체</option>
            {SEOUL_DISTRICTS.map((d) => (
              <option key={d} value={d}>{d}</option>
            ))}
          </select>
        </div>

        {/* 전문분야 선택 (12대분류) */}
        <div className="flex items-center gap-2">
          <svg className="w-4 h-4 text-gray-500 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
          </svg>
          <select
            value={category}
            onChange={(e) => onCategoryChange(e.target.value)}
            className="flex-1 px-3 py-2 border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="">전문분야 전체</option>
            {SPECIALTY_CATEGORIES.map((cat) => (
              <option key={cat.id} value={cat.id}>
                {cat.icon} {cat.name}
              </option>
            ))}
          </select>
        </div>

        {/* 검색 반경 */}
        <div className="flex items-center gap-2">
          <svg className="w-4 h-4 text-gray-500 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
            <circle cx="12" cy="12" r="10" strokeWidth={2} />
            <circle cx="12" cy="12" r="3" strokeWidth={2} />
          </svg>
          <div className="flex-1 flex flex-wrap gap-2">
            {RADIUS_OPTIONS.map((option) => (
              <button
                type="button"
                key={option.value}
                className={`flex-1 px-3 py-1.5 text-sm rounded-full transition ${
                  radius === option.value
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
                onClick={() => onRadiusChange(option.value)}
              >
                {option.label}
              </button>
            ))}
          </div>
        </div>

        {/* 검색창 */}
        <div className="flex items-center gap-2">
          <svg className="w-4 h-4 text-gray-500 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="변호사 또는 사무소 검색 (Enter)"
            className="flex-1 px-3 py-2 border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
      </div>

      {/* 결과 헤더 */}
      <div className="px-4 py-2 bg-gray-50 border-b text-sm h-10 flex items-center">
        {loading ? (
          <span className="text-gray-600">검색 중...</span>
        ) : (
          <div className="flex items-center justify-between w-full">
            <div className="flex items-center gap-2 text-gray-600">
              {searchQuery && <span className="text-blue-600 font-semibold">"{searchQuery}"</span>}
              <span>검색 결과</span>
              <span className="text-gray-400">|</span>
              <span className="text-gray-900 font-medium">{totalCount}명</span>
            </div>
            {searchQuery && (
              <button
                type="button"
                onClick={handleReset}
                className="flex items-center gap-1 px-2 py-0.5 text-xs text-red-600 bg-red-50 hover:bg-red-100 border border-red-200 rounded transition"
              >
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
                검색 해제
              </button>
            )}
          </div>
        )}
      </div>

      {/* 변호사 목록 */}
      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {loading ? (
          <div className="flex items-center justify-center h-32">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600" />
          </div>
        ) : lawyers.length > 0 ? (
          lawyers.map((lawyer) => (
            <LawyerCard
              key={lawyer.id}
              lawyer={lawyer}
              selected={selectedLawyer?.id === lawyer.id}
              onClick={() => onLawyerSelect(lawyer)}
              highlightCategory={category}
            />
          ))
        ) : (
          <div className="text-center text-gray-500 py-8">
            <svg className="w-12 h-12 mx-auto mb-3 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            {specialty ? (
              <>
                <p className="font-medium text-gray-700 mb-1">
                  주변 {radius >= 1000 ? `${radius / 1000}km` : `${radius}m`} 내에
                </p>
                <p className="text-blue-600 font-semibold mb-2">"{specialty}" 전문 변호사</p>
                <p className="text-gray-500">검색 결과가 없습니다</p>
                <p className="text-xs text-gray-400 mt-2">반경을 넓히거나 다른 조건으로 검색해보세요</p>
              </>
            ) : (
              <p>검색 결과가 없습니다</p>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
