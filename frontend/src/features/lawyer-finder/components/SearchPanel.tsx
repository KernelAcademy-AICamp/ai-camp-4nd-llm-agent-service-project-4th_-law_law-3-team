'use client'

import { useState } from 'react'
import type { Lawyer } from '../types'
import { LawyerCard } from './LawyerCard'

interface SearchPanelProps {
  lawyers: Lawyer[]
  loading: boolean
  selectedLawyer: Lawyer | null
  onLawyerSelect: (lawyer: Lawyer) => void
  onRadiusChange: (radius: number) => void
  onSearch: (query: { name?: string; office?: string; district?: string }) => void
  radius: number
  totalCount: number
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
  radius,
  totalCount,
}: SearchPanelProps) {
  const [searchMode, setSearchMode] = useState<'nearby' | 'search'>('nearby')
  const [searchName, setSearchName] = useState('')
  const [searchOffice, setSearchOffice] = useState('')
  const [searchDistrict, setSearchDistrict] = useState('')

  const handleSearch = () => {
    if (searchName || searchOffice || searchDistrict) {
      onSearch({
        name: searchName || undefined,
        office: searchOffice || undefined,
        district: searchDistrict || undefined,
      })
    }
  }

  return (
    <div className="w-96 bg-white border-r border-gray-200 flex flex-col h-full">
      {/* 모드 탭 */}
      <div className="flex border-b">
        <button
          className={`flex-1 py-3 text-sm font-medium ${
            searchMode === 'nearby'
              ? 'text-blue-600 border-b-2 border-blue-600'
              : 'text-gray-500 hover:text-gray-700'
          }`}
          onClick={() => setSearchMode('nearby')}
        >
          주변 검색
        </button>
        <button
          className={`flex-1 py-3 text-sm font-medium ${
            searchMode === 'search'
              ? 'text-blue-600 border-b-2 border-blue-600'
              : 'text-gray-500 hover:text-gray-700'
          }`}
          onClick={() => setSearchMode('search')}
        >
          이름/사무소 검색
        </button>
      </div>

      {/* 검색 옵션 */}
      <div className="p-4 border-b">
        {searchMode === 'nearby' ? (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              검색 반경
            </label>
            <div className="flex flex-wrap gap-2">
              {RADIUS_OPTIONS.map((option) => (
                <button
                  key={option.value}
                  className={`px-3 py-1.5 text-sm rounded-full transition ${
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
        ) : (
          <div className="space-y-3">
            <div>
              <label className="block text-xs text-gray-500 mb-1">변호사 이름</label>
              <input
                type="text"
                value={searchName}
                onChange={(e) => setSearchName(e.target.value)}
                placeholder="예: 홍길동"
                className="w-full px-3 py-2 border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-500 mb-1">사무소명</label>
              <input
                type="text"
                value={searchOffice}
                onChange={(e) => setSearchOffice(e.target.value)}
                placeholder="예: 법무법인 ○○"
                className="w-full px-3 py-2 border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-500 mb-1">지역 (구/군)</label>
              <input
                type="text"
                value={searchDistrict}
                onChange={(e) => setSearchDistrict(e.target.value)}
                placeholder="예: 강남구"
                className="w-full px-3 py-2 border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <button
              onClick={handleSearch}
              disabled={!searchName && !searchOffice && !searchDistrict}
              className="w-full py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed"
            >
              검색
            </button>
          </div>
        )}
      </div>

      {/* 결과 헤더 */}
      <div className="px-4 py-2 bg-gray-50 border-b text-sm text-gray-600">
        {loading ? (
          <span>검색 중...</span>
        ) : (
          <span>
            검색 결과 <strong className="text-gray-900">{totalCount}</strong>명
          </span>
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
            />
          ))
        ) : (
          <div className="text-center text-gray-500 py-8">
            검색 결과가 없습니다
          </div>
        )}
      </div>
    </div>
  )
}
