'use client'

import { useState, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import type { ViewMode } from '@/app/lawyer-stat/page'
import type { DensityStat, RegionStat, SpecialtyStat } from '../types'
import { fetchRegionSpecialties } from '../services'

interface RegionDetailListProps {
  regions: (RegionStat | DensityStat)[]
  viewMode: ViewMode
  selectedProvince: string | null
  onRegionClick?: (region: string | null) => void
  mapSelectedRegion?: string | null
}

function SpecialtyItem({ spec, maxCount }: { spec: SpecialtyStat; maxCount: number }) {
  const [expanded, setExpanded] = useState(false)
  const barWidth = (spec.count / maxCount) * 100

  return (
    <div className="border-b border-gray-100 last:border-0">
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        className="w-full py-2 text-left"
      >
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-gray-700 flex items-center gap-1">
            <svg
              className={`w-3 h-3 text-gray-400 transition-transform ${expanded ? 'rotate-90' : ''}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
            {spec.category_name}
          </span>
          <span className="text-sm font-semibold text-emerald-600">
            {spec.count.toLocaleString()}명
          </span>
        </div>
        <div className="mt-1 h-1.5 w-full rounded-full bg-gray-100">
          <div
            className="h-1.5 rounded-full bg-emerald-500 transition-all"
            style={{ width: `${barWidth}%` }}
          />
        </div>
      </button>
      {expanded && spec.specialties.length > 0 && (
        <div className="pl-4 pb-2 space-y-1">
          {spec.specialties.map((detail) => (
            <div key={detail.name} className="flex items-center justify-between text-xs">
              <span className="text-gray-500">{detail.name}</span>
              <span className="text-gray-600 font-medium">{detail.count}명</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export function RegionDetailList({ regions, viewMode, selectedProvince, onRegionClick, mapSelectedRegion }: RegionDetailListProps) {
  const [selectedRegion, setSelectedRegion] = useState<string | null>(null)

  // 지도에서 선택된 지역이 변경되면 세부 화면으로 전환
  useEffect(() => {
    if (mapSelectedRegion) {
      setSelectedRegion(mapSelectedRegion)
    }
  }, [mapSelectedRegion])

  // 시/도 필터가 변경되면 목록으로 돌아가기
  useEffect(() => {
    setSelectedRegion(null)
  }, [selectedProvince])

  const filteredRegions = selectedProvince
    ? regions.filter((r) => r.region.startsWith(selectedProvince))
    : regions

  // viewMode에 따라 정렬 기준 결정
  const sortedRegions = [...filteredRegions].sort((a, b) => {
    if (viewMode === 'density') {
      const aDensity = 'density' in a ? a.density : 0
      const bDensity = 'density' in b ? b.density : 0
      return bDensity - aDensity
    }
    return b.count - a.count
  })

  const displayRegions = selectedProvince ? sortedRegions : sortedRegions.slice(0, 15)

  // viewMode에 따라 최대값 결정
  const maxValue = viewMode === 'density'
    ? ('density' in displayRegions[0] ? displayRegions[0].density : 1)
    : displayRegions[0]?.count ?? 1

  // 선택된 지역의 전문분야 데이터 조회
  const specialtiesQuery = useQuery({
    queryKey: ['lawyer-stat', 'region-specialties', selectedRegion],
    queryFn: () => fetchRegionSpecialties(selectedRegion!),
    enabled: !!selectedRegion,
  })

  const maxSpecialtyCount = specialtiesQuery.data?.data[0]?.count ?? 1

  // 상세 뷰
  if (selectedRegion) {
    return (
      <div className="h-[500px] flex flex-col">
        <button
          type="button"
          onClick={() => {
            setSelectedRegion(null)
            onRegionClick?.(null)
          }}
          className="flex items-center gap-1 text-sm text-gray-500 hover:text-gray-700 transition-colors mb-2"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
          목록으로
        </button>
        <div className="mb-3">
          <div className="text-lg font-semibold text-gray-900">📍 {selectedRegion}</div>
          <div className="text-sm text-gray-500">전문분야별 변호사 현황 (클릭하여 세부분야 보기)</div>
        </div>
        <div className="overflow-y-auto flex-1">
          {specialtiesQuery.isLoading ? (
            <div className="flex items-center justify-center py-8">
              <div className="h-6 w-6 animate-spin rounded-full border-2 border-emerald-500 border-t-transparent" />
            </div>
          ) : specialtiesQuery.data?.data && specialtiesQuery.data.data.length > 0 ? (
            specialtiesQuery.data.data.map((spec) => (
              <SpecialtyItem key={spec.category_id} spec={spec} maxCount={maxSpecialtyCount} />
            ))
          ) : (
            <div className="text-sm text-gray-500 py-4 text-center">
              전문분야 데이터가 없습니다.
            </div>
          )}
        </div>
      </div>
    )
  }

  // 리스트 뷰
  const titleText = viewMode === 'density'
    ? (selectedProvince ? `${selectedProvince} 내 인구 대비 밀도 순위` : '전체 지역 인구 대비 밀도 순위(Top15)')
    : (selectedProvince ? `${selectedProvince} 내 변호사 수 순위` : '전체 지역 변호사 수 순위(Top15)')

  return (
    <div className="h-[500px] flex flex-col">
      <div className="mb-3 text-sm font-medium text-gray-500">
        {titleText}
        <span className="text-xs text-gray-400 ml-2">(클릭하여 상세보기)</span>
      </div>
      <div className="space-y-2 overflow-y-auto flex-1">
        {displayRegions.map((region, index) => {
          const value = viewMode === 'density' && 'density' in region ? region.density : region.count
          const barWidth = (value / maxValue) * 100
          const displayValue = viewMode === 'density' && 'density' in region
            ? `${region.density.toFixed(1)}명/10만`
            : `${region.count.toLocaleString()}명`
          const barColor = viewMode === 'density' ? 'bg-emerald-500' : 'bg-blue-500'
          const textColor = viewMode === 'density' ? 'text-emerald-600' : 'text-blue-600'

          return (
            <button
              type="button"
              key={region.region}
              onClick={() => {
                onRegionClick?.(region.region)
                setSelectedRegion(region.region)
              }}
              className="w-full flex items-center gap-3 hover:bg-gray-50 rounded-lg p-1 -m-1 transition-colors text-left"
            >
              <span className="w-6 text-right text-sm font-medium text-gray-400">
                {index + 1}
              </span>
              <div className="flex-1">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-700">
                    📍 {region.region}
                  </span>
                  <span className={`text-sm font-semibold ${textColor}`}>
                    {displayValue}
                  </span>
                </div>
                <div className="mt-1 h-1.5 w-full rounded-full bg-gray-100">
                  <div
                    className={`h-1.5 rounded-full ${barColor} transition-all`}
                    style={{ width: `${barWidth}%` }}
                  />
                </div>
              </div>
            </button>
          )
        })}
      </div>
    </div>
  )
}
