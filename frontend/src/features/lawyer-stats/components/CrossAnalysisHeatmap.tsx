'use client'

import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { fetchCrossAnalysisByRegions, fetchRegionStats } from '../services'
import { RegionSelector } from './RegionSelector'

function getColorIntensity(value: number, max: number): string {
  if (value === 0) return 'bg-gray-100'
  const ratio = value / max
  if (ratio > 0.7) return 'bg-blue-600 text-white'
  if (ratio > 0.5) return 'bg-blue-500 text-white'
  if (ratio > 0.3) return 'bg-blue-400 text-white'
  if (ratio > 0.1) return 'bg-blue-300'
  return 'bg-blue-200'
}

function LoadingSpinner() {
  return (
    <div className="flex h-40 items-center justify-center">
      <div className="h-6 w-6 animate-spin rounded-full border-2 border-blue-500 border-t-transparent" />
    </div>
  )
}

type SortOption = 'count' | 'name'

export function CrossAnalysisHeatmap() {
  const [selectedRegions, setSelectedRegions] = useState<string[]>([])
  const [hoveredCell, setHoveredCell] = useState<string | null>(null)
  const [sortBy, setSortBy] = useState<SortOption>('count')
  const [selectedCell, setSelectedCell] = useState<{ region: string; category: string } | null>(null)

  // 전체 지역 목록 조회 (선택 UI용)
  const regionStatsQuery = useQuery({
    queryKey: ['lawyer-stats', 'region'],
    queryFn: fetchRegionStats,
  })

  // 선택된 지역에 대한 교차 분석
  const crossAnalysisQuery = useQuery({
    queryKey: ['lawyer-stats', 'cross-analysis-regions', selectedRegions],
    queryFn: () => fetchCrossAnalysisByRegions(selectedRegions),
    enabled: selectedRegions.length > 0,
  })

  const { matrix, maxValue, rowTotals, colTotals, grandTotal } = useMemo(() => {
    if (!crossAnalysisQuery.data) {
      return {
        matrix: new Map<string, number>(),
        maxValue: 0,
        rowTotals: new Map<string, number>(),
        colTotals: new Map<string, number>(),
        grandTotal: 0,
      }
    }

    const map = new Map<string, number>()
    let max = 0

    for (const cell of crossAnalysisQuery.data.data) {
      const key = `${cell.region}-${cell.category_name}`
      map.set(key, cell.count)
      if (cell.count > max) max = cell.count
    }

    const { regions, categories } = crossAnalysisQuery.data
    const getVal = (r: string, c: string) => map.get(`${r}-${c}`) ?? 0

    // 행 합계: 각 지역별 전체 카테고리 합
    const rowTotalsMap = new Map<string, number>()
    for (const region of regions) {
      let sum = 0
      for (const category of categories) {
        sum += getVal(region, category)
      }
      rowTotalsMap.set(region, sum)
    }

    // 열 합계: 각 카테고리별 전체 지역 합
    const colTotalsMap = new Map<string, number>()
    for (const category of categories) {
      let sum = 0
      for (const region of regions) {
        sum += getVal(region, category)
      }
      colTotalsMap.set(category, sum)
    }

    // 총합계
    const total = Array.from(rowTotalsMap.values()).reduce((a, b) => a + b, 0)

    return {
      matrix: map,
      maxValue: max,
      rowTotals: rowTotalsMap,
      colTotals: colTotalsMap,
      grandTotal: total,
    }
  }, [crossAnalysisQuery.data])

  const getValue = (region: string, category: string) => {
    return matrix.get(`${region}-${category}`) ?? 0
  }

  const data = crossAnalysisQuery.data

  // 정렬된 지역 목록
  const sortedRegions = useMemo(() => {
    if (!data) return []

    const regions = [...data.regions]
    if (sortBy === 'name') {
      return regions.sort((a, b) => a.localeCompare(b, 'ko'))
    }
    // 'count' - 이미 API에서 총합 내림차순으로 정렬되어 옴
    return regions
  }, [data, sortBy])

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm">
      <div className="mb-4">
        <h3 className="text-lg font-semibold text-gray-900">지역별 전문분야 현황</h3>
        <p className="text-sm text-gray-500 mt-1">선택한 지역의 전문분야별 변호사 수를 비교할 수 있습니다.</p>
      </div>

      {/* 지역 선택 UI */}
      {regionStatsQuery.isLoading && <LoadingSpinner />}

      {regionStatsQuery.data && (
        <div className="mb-6">
          <RegionSelector
            regions={regionStatsQuery.data.data}
            selectedRegions={selectedRegions}
            onSelectionChange={setSelectedRegions}
          />
        </div>
      )}

      {/* 분석 결과 */}
      {selectedRegions.length === 0 && (
        <div className="rounded-lg border border-gray-200 bg-gray-50 py-12 text-center">
          <div className="text-gray-400 mb-2">
            <svg className="mx-auto h-12 w-12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
            </svg>
          </div>
          <p className="text-sm text-gray-500">
            위에서 분석할 지역을 선택해 주세요
          </p>
        </div>
      )}

      {selectedRegions.length > 0 && crossAnalysisQuery.isLoading && <LoadingSpinner />}

      {selectedRegions.length > 0 && crossAnalysisQuery.isError && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-red-700">
          데이터를 불러오는 중 오류가 발생했습니다.
        </div>
      )}

      {data && data.regions.length > 0 && (
        <>
          {/* 정렬 옵션 */}
          <div className="mb-3 flex items-center justify-end gap-2">
            <span className="text-xs text-gray-500">행 정렬:</span>
            <div className="flex rounded-lg border border-gray-200 overflow-hidden">
              <button
                type="button"
                onClick={() => setSortBy('count')}
                className={`px-3 py-1 text-xs font-medium transition-colors ${
                  sortBy === 'count'
                    ? 'bg-blue-600 text-white'
                    : 'bg-white text-gray-600 hover:bg-gray-50'
                }`}
              >
                총합 많은 순
              </button>
              <button
                type="button"
                onClick={() => setSortBy('name')}
                className={`px-3 py-1 text-xs font-medium transition-colors border-l border-gray-200 ${
                  sortBy === 'name'
                    ? 'bg-blue-600 text-white'
                    : 'bg-white text-gray-600 hover:bg-gray-50'
                }`}
              >
                이름순
              </button>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="min-w-full text-xs border-collapse border border-gray-200 table-fixed">
              <thead>
                <tr>
                  <th className="sticky left-0 z-10 bg-white px-2 py-1 text-left font-medium text-gray-500 border border-gray-200 w-24">
                    지역
                  </th>
                  {data.categories.map((category) => {
                    // 긴 이름(7자 이상)만 "·" 기준으로 줄바꿈
                    const parts = category.split('·')
                    const needsWrap = category.length >= 7 && parts.length > 1
                    const isHighlighted = selectedCell?.category === category
                    return (
                      <th
                        key={category}
                        className={`px-2 py-2 text-center font-medium text-xs leading-tight w-16 transition-colors border border-gray-200 ${
                          isHighlighted ? 'bg-blue-100 text-blue-700' : 'text-gray-500'
                        }`}
                      >
                        {needsWrap ? (
                          <>
                            {parts[0]}·<br />{parts[1]}
                          </>
                        ) : (
                          category
                        )}
                      </th>
                    )
                  })}
                  <th className="px-2 py-2 text-center font-semibold text-xs bg-gray-100 text-gray-700 w-16 border border-gray-200 border-l-2 border-l-gray-400">
                    합계
                  </th>
                </tr>
              </thead>
              <tbody>
                {sortedRegions.map((region) => {
                  const isRowHighlighted = selectedCell?.region === region
                  return (
                  <tr key={region}>
                    <td className={`sticky left-0 z-10 px-2 py-1 font-medium whitespace-nowrap transition-colors border border-gray-200 ${
                      isRowHighlighted ? 'bg-blue-100 text-blue-700' : 'bg-white text-gray-700'
                    }`}>
                      {region}
                    </td>
                    {data.categories.map((category) => {
                      const value = getValue(region, category)
                      const cellKey = `${region}-${category}`
                      const isHovered = hoveredCell === cellKey
                      const isSelected = selectedCell?.region === region && selectedCell?.category === category
                      const isInHighlightedRow = selectedCell?.region === region
                      const isInHighlightedCol = selectedCell?.category === category

                      return (
                        <td
                          key={cellKey}
                          className={`relative cursor-pointer px-2 py-1 text-center transition-all border border-gray-200 ${getColorIntensity(value, maxValue)} ${
                            isSelected ? 'ring-2 ring-blue-700' : ''
                          } ${(isInHighlightedRow || isInHighlightedCol) && !isSelected ? 'ring-1 ring-blue-300' : ''}`}
                          onMouseEnter={() => setHoveredCell(cellKey)}
                          onMouseLeave={() => setHoveredCell(null)}
                          onClick={() => setSelectedCell(
                            isSelected ? null : { region, category }
                          )}
                        >
                          {value > 0 ? value : '-'}
                          {isHovered && value > 0 && (
                            <div className="absolute bottom-full left-1/2 z-20 mb-1 -translate-x-1/2 whitespace-nowrap rounded bg-gray-900 px-2 py-1 text-xs text-white">
                              {region} / {category}: {value}명
                            </div>
                          )}
                        </td>
                      )
                    })}
                    <td className="px-2 py-1 text-center font-semibold bg-gray-50 text-gray-700 border border-gray-200 border-l-2 border-l-gray-400">
                      {rowTotals.get(region) ?? 0}
                    </td>
                  </tr>
                )})}
              </tbody>
              <tfoot>
                <tr>
                  <td className="sticky left-0 z-10 px-2 py-1 font-semibold bg-gray-100 text-gray-700 border border-gray-200 border-t-2 border-t-gray-400">
                    합계
                  </td>
                  {data.categories.map((category) => (
                    <td
                      key={`total-${category}`}
                      className="px-2 py-1 text-center font-semibold bg-gray-100 text-gray-700 border border-gray-200 border-t-2 border-t-gray-400"
                    >
                      {colTotals.get(category) ?? 0}
                    </td>
                  ))}
                  <td className="px-2 py-1 text-center font-bold bg-gray-200 text-gray-800 border border-gray-200 border-t-2 border-t-gray-400 border-l-2 border-l-gray-400">
                    {grandTotal}
                  </td>
                </tr>
              </tfoot>
            </table>
          </div>
          <div className="mt-4 flex items-center justify-between text-xs text-gray-500">
            <span>
              {data.regions.length}개 지역 분석 결과
            </span>
            <div className="flex items-center gap-2">
              <span>낮음</span>
              <div className="flex gap-0.5">
                <div className="h-4 w-4 bg-blue-200" />
                <div className="h-4 w-4 bg-blue-300" />
                <div className="h-4 w-4 bg-blue-400" />
                <div className="h-4 w-4 bg-blue-500" />
                <div className="h-4 w-4 bg-blue-600" />
              </div>
              <span>높음</span>
            </div>
          </div>
        </>
      )}

      {data && data.regions.length === 0 && selectedRegions.length > 0 && (
        <div className="py-8 text-center text-gray-500">
          선택한 지역에 데이터가 없습니다.
        </div>
      )}
    </div>
  )
}
