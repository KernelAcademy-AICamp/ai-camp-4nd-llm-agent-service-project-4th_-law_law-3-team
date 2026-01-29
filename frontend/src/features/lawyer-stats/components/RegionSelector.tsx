'use client'

import { useMemo, useState, useCallback } from 'react'
import type { RegionStat } from '../types'

const PROVINCES = [
  '서울', '경기', '인천', '부산', '대구', '광주', '대전', '울산', '세종',
  '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주',
]

interface RegionSelectorProps {
  regions: RegionStat[]
  selectedRegions: string[]
  onSelectionChange: (regions: string[]) => void
}

export function RegionSelector({ regions, selectedRegions, onSelectionChange }: RegionSelectorProps) {
  const [selectedProvinces, setSelectedProvinces] = useState<Set<string>>(new Set())

  // 시/도별 시군구 그룹화
  const regionsByProvince = useMemo(() => {
    const grouped = new Map<string, RegionStat[]>()

    for (const region of regions) {
      const [province] = region.region.split(' ')
      if (!grouped.has(province)) {
        grouped.set(province, [])
      }
      grouped.get(province)!.push(region)
    }

    // 각 시/도 내에서 변호사 수 내림차순 정렬
    grouped.forEach((districts) => {
      districts.sort((a, b) => b.count - a.count)
    })

    return grouped
  }, [regions])

  // 시/도에 해당하는 모든 지역 선택/해제
  const toggleProvinceSelection = useCallback((province: string) => {
    const provinceRegions = regionsByProvince.get(province) || []
    const provinceRegionNames = provinceRegions.map(r => r.region)

    const allSelected = provinceRegionNames.every(r => selectedRegions.includes(r))

    if (allSelected) {
      // 전체 해제
      onSelectionChange(selectedRegions.filter(r => !provinceRegionNames.includes(r)))
    } else {
      // 전체 선택
      const newSelection = new Set(selectedRegions)
      provinceRegionNames.forEach(r => newSelection.add(r))
      onSelectionChange(Array.from(newSelection))
    }
  }, [regionsByProvince, selectedRegions, onSelectionChange])

  // 개별 지역 선택/해제
  const toggleRegion = useCallback((region: string) => {
    if (selectedRegions.includes(region)) {
      onSelectionChange(selectedRegions.filter(r => r !== region))
    } else {
      onSelectionChange([...selectedRegions, region])
    }
  }, [selectedRegions, onSelectionChange])

  // 시/도 펼치기/접기
  const toggleProvinceExpand = useCallback((province: string) => {
    setSelectedProvinces(prev => {
      const next = new Set(prev)
      if (next.has(province)) {
        next.delete(province)
      } else {
        next.add(province)
      }
      return next
    })
  }, [])

  // 시/도의 선택 상태 계산
  const getProvinceState = useCallback((province: string): 'none' | 'some' | 'all' => {
    const provinceRegions = regionsByProvince.get(province) || []
    if (provinceRegions.length === 0) return 'none'

    const selectedCount = provinceRegions.filter(r => selectedRegions.includes(r.region)).length

    if (selectedCount === 0) return 'none'
    if (selectedCount === provinceRegions.length) return 'all'
    return 'some'
  }, [regionsByProvince, selectedRegions])

  // 전체 선택/해제
  const toggleAll = useCallback(() => {
    if (selectedRegions.length === regions.length) {
      onSelectionChange([])
    } else {
      onSelectionChange(regions.map(r => r.region))
    }
  }, [regions, selectedRegions, onSelectionChange])

  return (
    <div className="space-y-3">
      {/* 헤더 */}
      <div className="flex items-center justify-between">
        <div className="text-sm font-medium text-gray-700">
          지역 선택 <span className="text-blue-600">({selectedRegions.length}개 선택)</span>
        </div>
        <button
          type="button"
          onClick={toggleAll}
          className="text-xs text-blue-600 hover:text-blue-800"
        >
          {selectedRegions.length === regions.length ? '전체 해제' : '전체 선택'}
        </button>
      </div>

      {/* 1단계: 시/도 체크박스 */}
      <div className="rounded-lg border border-gray-200 bg-gray-50 p-3">
        <div className="mb-2 text-xs font-medium text-gray-500">1단계: 시/도 선택</div>
        <div className="flex flex-wrap gap-2">
          {PROVINCES.map((province) => {
            const state = getProvinceState(province)
            const isExpanded = selectedProvinces.has(province)
            const hasRegions = regionsByProvince.has(province)

            return (
              <button
                key={province}
                type="button"
                onClick={() => hasRegions && toggleProvinceExpand(province)}
                disabled={!hasRegions}
                className={`
                  flex items-center gap-1.5 rounded-full px-3 py-1.5 text-sm font-medium transition-colors
                  ${!hasRegions
                    ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                    : isExpanded
                      ? 'bg-blue-600 text-white'
                      : state === 'all'
                        ? 'bg-blue-100 text-blue-700 border border-blue-300'
                        : state === 'some'
                          ? 'bg-blue-50 text-blue-600 border border-blue-200'
                          : 'bg-white text-gray-600 border border-gray-200 hover:bg-gray-50'
                  }
                `}
              >
                {state === 'all' && <span className="text-xs">✓</span>}
                {state === 'some' && <span className="text-xs">−</span>}
                {province}
              </button>
            )
          })}
        </div>
      </div>

      {/* 2단계: 선택된 시/도의 시군구 체크박스 */}
      {selectedProvinces.size > 0 && (
        <div className="rounded-lg border border-gray-200 bg-white p-3">
          <div className="mb-2 text-xs font-medium text-gray-500">2단계: 시/군/구 선택</div>
          <div className="max-h-64 space-y-3 overflow-y-auto">
            {Array.from(selectedProvinces).map((province) => {
              const provinceRegions = regionsByProvince.get(province) || []
              const state = getProvinceState(province)

              return (
                <div key={province} className="space-y-2">
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => toggleProvinceSelection(province)}
                      className="flex items-center gap-1.5 text-sm font-semibold text-gray-700 hover:text-blue-600"
                    >
                      <span className={`
                        flex h-4 w-4 items-center justify-center rounded border text-xs
                        ${state === 'all'
                          ? 'bg-blue-600 border-blue-600 text-white'
                          : state === 'some'
                            ? 'bg-blue-100 border-blue-400 text-blue-600'
                            : 'border-gray-300'
                        }
                      `}>
                        {state === 'all' && '✓'}
                        {state === 'some' && '−'}
                      </span>
                      {province} 전체
                    </button>
                    <span className="text-xs text-gray-400">
                      ({provinceRegions.filter(r => selectedRegions.includes(r.region)).length}/{provinceRegions.length})
                    </span>
                  </div>
                  <div className="ml-6 flex flex-wrap gap-1.5">
                    {provinceRegions.map((region) => {
                      const isSelected = selectedRegions.includes(region.region)
                      const districtName = region.region.split(' ')[1] || region.region

                      return (
                        <button
                          key={region.region}
                          type="button"
                          onClick={() => toggleRegion(region.region)}
                          className={`
                            rounded px-2 py-1 text-xs transition-colors
                            ${isSelected
                              ? 'bg-blue-600 text-white'
                              : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                            }
                          `}
                        >
                          {districtName}
                          <span className={`ml-1 ${isSelected ? 'text-blue-200' : 'text-gray-400'}`}>
                            ({region.count})
                          </span>
                        </button>
                      )
                    })}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* 선택된 지역 요약 */}
      {selectedRegions.length > 0 && (
        <div className="rounded-lg border border-blue-200 bg-blue-50 p-3">
          <div className="mb-2 flex items-center justify-between">
            <span className="text-xs font-medium text-blue-700">선택된 지역</span>
            <button
              type="button"
              onClick={() => onSelectionChange([])}
              className="text-xs text-blue-600 hover:text-blue-800"
            >
              초기화
            </button>
          </div>
          <div className="flex flex-wrap gap-1">
            {selectedRegions.slice(0, 10).map((region) => (
              <span
                key={region}
                className="inline-flex items-center gap-1 rounded-full bg-blue-100 px-2 py-0.5 text-xs text-blue-700"
              >
                {region}
                <button
                  type="button"
                  onClick={() => toggleRegion(region)}
                  className="text-blue-500 hover:text-blue-700"
                >
                  ×
                </button>
              </span>
            ))}
            {selectedRegions.length > 10 && (
              <span className="text-xs text-blue-600">
                외 {selectedRegions.length - 10}개
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
