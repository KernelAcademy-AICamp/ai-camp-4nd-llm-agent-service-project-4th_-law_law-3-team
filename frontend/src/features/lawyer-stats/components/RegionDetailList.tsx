'use client'

import { useState, useEffect, useMemo, memo } from 'react'
import { useQuery } from '@tanstack/react-query'
import type { PredictionYear, ViewMode } from '@/app/lawyer-stats/page'
import type { DensityStat, RegionStat, SpecialtyStat } from '../types'
import { fetchRegionSpecialties } from '../services'

interface RegionDetailListProps {
  regions: (RegionStat | DensityStat)[]
  viewMode: ViewMode
  predictionYear?: PredictionYear
  selectedProvince: string | null
  onRegionClick?: (region: string | null) => void
  mapSelectedRegion?: string | null
}

/** viewMode별 색상 매핑 (렌더링 외부에서 정의) */
const VIEW_MODE_COLORS = {
  count: { bar: 'bg-blue-500', text: 'text-blue-600' },
  density: { bar: 'bg-emerald-500', text: 'text-emerald-600' },
  prediction: { bar: 'bg-violet-500', text: 'text-violet-600' },
} as const

const SpecialtyItem = memo(function SpecialtyItem({ spec, maxCount }: { spec: SpecialtyStat; maxCount: number }) {
  const [expanded, setExpanded] = useState(false)
  const barWidth = useMemo(() => (spec.count / maxCount) * 100, [spec.count, maxCount])

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
})

type SortOrder = 'desc' | 'asc'

export function RegionDetailList({ regions, viewMode, predictionYear, selectedProvince, onRegionClick, mapSelectedRegion }: RegionDetailListProps) {
  const [selectedRegion, setSelectedRegion] = useState<string | null>(null)
  const [sortOrder, setSortOrder] = useState<SortOrder>('desc')

  // 지도에서 선택된 지역이 변경되면 세부 화면으로 전환 (null이면 목록으로 복귀)
  useEffect(() => {
    setSelectedRegion(mapSelectedRegion ?? null)
  }, [mapSelectedRegion])

  // 시/도 필터가 변경되면 목록으로 돌아가기
  useEffect(() => {
    setSelectedRegion(null)
  }, [selectedProvince])

  const filteredRegions = useMemo(
    () => selectedProvince
      ? regions.filter((r) => r.region.startsWith(selectedProvince))
      : regions,
    [regions, selectedProvince]
  )

  // viewMode와 sortOrder에 따라 정렬
  const sortedRegions = useMemo(() => {
    return [...filteredRegions].sort((a, b) => {
      const multiplier = sortOrder === 'desc' ? 1 : -1
      if (viewMode === 'density' || viewMode === 'prediction') {
        const aDensity = 'density' in a ? a.density : 0
        const bDensity = 'density' in b ? b.density : 0
        return (bDensity - aDensity) * multiplier
      }
      return (b.count - a.count) * multiplier
    })
  }, [filteredRegions, sortOrder, viewMode])

  const displayRegions = useMemo(
    () => selectedProvince ? sortedRegions : sortedRegions.slice(0, 15),
    [selectedProvince, sortedRegions]
  )

  // viewMode에 따라 최대값 결정 (바 그래프용 - 정렬 순서 무관하게 최대값)
  const maxValue = useMemo(() => {
    return (viewMode === 'density' || viewMode === 'prediction')
      ? Math.max(...displayRegions.map(r => 'density' in r ? r.density : 0), 1)
      : Math.max(...displayRegions.map(r => r.count), 1)
  }, [displayRegions, viewMode])

  // 선택된 지역의 전문분야 데이터 조회
  const specialtiesQuery = useQuery({
    queryKey: ['lawyer-stats', 'region-specialties', selectedRegion],
    queryFn: () => fetchRegionSpecialties(selectedRegion!),
    enabled: !!selectedRegion,
  })

  const maxSpecialtyCount = specialtiesQuery.data?.data[0]?.count ?? 1

  // 선택된 지역의 데이터 찾기
  const selectedRegionData = regions.find(r => r.region === selectedRegion)

  // 상세 뷰
  if (selectedRegion) {
    // 예측 모드 상세 뷰
    if (viewMode === 'prediction' && selectedRegionData && 'density' in selectedRegionData) {
      const data = selectedRegionData as DensityStat
      const densityCurrent = data.density_current ?? data.density
      const changePercent = data.change_percent ?? 0
      const populationCurrent = densityCurrent > 0 ? Math.round(data.count / densityCurrent * 100000) : data.population
      const populationChange = populationCurrent > 0 ? ((data.population - populationCurrent) / populationCurrent * 100) : 0

      // 바 너비 계산
      const maxDensity = Math.max(densityCurrent, data.density)
      const densityCurrentWidth = (densityCurrent / maxDensity) * 100
      const densityFutureWidth = (data.density / maxDensity) * 100
      const maxPop = Math.max(populationCurrent, data.population)
      const popCurrentWidth = (populationCurrent / maxPop) * 100
      const popFutureWidth = (data.population / maxPop) * 100

      // 시장 전망 게이지 위치 (-30% ~ +30% → 0 ~ 100)
      const gaugePosition = Math.min(Math.max((changePercent + 30) / 60 * 100, 0), 100)

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

          <div className="mb-6 flex items-baseline gap-2">
            <div className="text-lg font-semibold text-gray-900">{selectedRegion}</div>
            <div className="text-sm text-violet-600 font-medium">{predictionYear}년 시장 전망</div>
          </div>

          <div className="space-y-5 overflow-y-auto flex-1 divide-y divide-gray-200">
            {/* 밀도 변화 */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <span className="text-base font-medium text-gray-700">밀도</span>
                  <span className={`text-sm font-medium ${changePercent >= 0 ? 'text-red-500' : 'text-blue-500'}`}>
                    ({changePercent >= 0 ? '+' : ''}{changePercent.toFixed(1)}%)
                  </span>
                </div>
                <span className="text-xs text-gray-400">(명 / 10만명)</span>
              </div>
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <span className="text-sm text-gray-400 w-12">현재</span>
                  <div className="flex-1 h-3 bg-gray-100 rounded overflow-hidden">
                    <div className="h-full bg-gray-400 rounded" style={{ width: `${densityCurrentWidth}%` }} />
                  </div>
                  <span className="text-sm text-gray-600 w-14 text-right">{densityCurrent.toFixed(1)}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-sm text-violet-500 w-12">{predictionYear}</span>
                  <div className="flex-1 h-3 bg-violet-100 rounded overflow-hidden">
                    <div className="h-full bg-violet-500 rounded" style={{ width: `${densityFutureWidth}%` }} />
                  </div>
                  <span className="text-sm text-violet-600 w-14 text-right">{data.density.toFixed(1)}</span>
                </div>
              </div>
            </div>

            {/* 인구 변화 */}
            <div className="pt-4">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <span className="text-base font-medium text-gray-700">인구</span>
                  <span className={`text-sm font-medium ${populationChange >= 0 ? 'text-red-500' : 'text-blue-500'}`}>
                    ({populationChange >= 0 ? '+' : ''}{populationChange.toFixed(1)}%)
                  </span>
                </div>
                <span className="text-xs text-gray-400">(만명)</span>
              </div>
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <span className="text-sm text-gray-400 w-12">현재</span>
                  <div className="flex-1 h-3 bg-gray-100 rounded overflow-hidden">
                    <div className="h-full bg-gray-400 rounded" style={{ width: `${popCurrentWidth}%` }} />
                  </div>
                  <span className="text-sm text-gray-600 w-14 text-right">{(populationCurrent / 10000).toFixed(1)}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className={`text-sm w-12 ${populationChange >= 0 ? 'text-red-500' : 'text-blue-500'}`}>{predictionYear}</span>
                  <div className={`flex-1 h-3 rounded overflow-hidden ${populationChange >= 0 ? 'bg-red-100' : 'bg-blue-100'}`}>
                    <div className={`h-full rounded ${populationChange >= 0 ? 'bg-red-400' : 'bg-blue-400'}`} style={{ width: `${popFutureWidth}%` }} />
                  </div>
                  <span className={`text-sm w-14 text-right ${populationChange >= 0 ? 'text-red-500' : 'text-blue-500'}`}>{(data.population / 10000).toFixed(1)}</span>
                </div>
              </div>
            </div>

            {/* 시장 전망 게이지 */}
            <div className="pt-4 pb-1">
              <div className="text-base font-medium text-gray-700 mb-3">시장 전망</div>
              <div className="relative h-3 mx-2">
                <div className="absolute inset-0 rounded-full bg-gradient-to-r from-blue-400 via-gray-200 to-red-400" />
                <div
                  className="absolute top-1/2 -translate-y-1/2 w-3.5 h-3.5 bg-white border-2 border-gray-700 rounded-full shadow"
                  style={{ left: `calc(${gaugePosition}% - 8px)` }}
                />
              </div>
              <div className="flex justify-between text-sm text-gray-400 mt-2">
                <span>경쟁 완화</span>
                <span>유지</span>
                <span>경쟁 심화</span>
              </div>
            </div>

            {/* 변호사 수 */}
            <div className="pt-4">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-base font-medium text-gray-700">변호사 수</span>
                <span className="text-xs text-gray-400">(현재 기준 고정 가정)</span>
              </div>
              <div className="text-lg font-bold text-gray-800 mb-1">{data.count.toLocaleString()}명</div>
              <div className="text-xs text-gray-400">※ 향후 변호사 수 증감은 반영하지 않은 시나리오입니다</div>
            </div>
          </div>
        </div>
      )
    }

    // 기본 상세 뷰 (전문분야)
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
          <div className="text-lg font-semibold text-gray-900">{selectedRegion}</div>
          <div className="text-sm text-gray-500">전문분야별 변호사 현황</div>
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
  const titleText = viewMode === 'prediction'
    ? (selectedProvince ? `${selectedProvince} 내 ${predictionYear}년 예측 밀도` : `전체 지역 ${predictionYear}년 예측 밀도(Top15)`)
    : viewMode === 'density'
      ? (selectedProvince ? `${selectedProvince} 내 인구 대비 밀도 순위` : '전체 지역 인구 대비 밀도 순위(Top15)')
      : (selectedProvince ? `${selectedProvince} 내 변호사 수 순위` : '전체 지역 변호사 수 순위(Top15)')

  return (
    <div className="h-[500px] flex flex-col">
      <div className="mb-3 flex items-center justify-between">
        <span className="text-sm font-medium text-gray-500">{titleText}</span>
        <button
          type="button"
          onClick={() => setSortOrder(sortOrder === 'desc' ? 'asc' : 'desc')}
          className="flex items-center gap-1 px-2 py-1 text-xs text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded transition-colors"
        >
          <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 16 16">
            {sortOrder === 'desc' ? (
              <>
                <rect x="2" y="2" width="12" height="2" rx="0.5" />
                <rect x="2" y="7" width="8" height="2" rx="0.5" />
                <rect x="2" y="12" width="4" height="2" rx="0.5" />
              </>
            ) : (
              <>
                <rect x="2" y="2" width="4" height="2" rx="0.5" />
                <rect x="2" y="7" width="8" height="2" rx="0.5" />
                <rect x="2" y="12" width="12" height="2" rx="0.5" />
              </>
            )}
          </svg>
          {sortOrder === 'desc' ? '높은순' : '낮은순'}
        </button>
      </div>
      <div className="space-y-2 overflow-y-auto flex-1">
        {displayRegions.map((region, index) => {
          const isDensityMode = viewMode === 'density' || viewMode === 'prediction'
          const value = isDensityMode && 'density' in region ? region.density : region.count
          const barWidth = (value / maxValue) * 100
          const displayValue = isDensityMode && 'density' in region
            ? `${region.density.toFixed(1)}명/10만`
            : `${region.count.toLocaleString()}명`
          const { bar: barColor, text: textColor } = VIEW_MODE_COLORS[viewMode]

          // 예측 모드에서 변화율 표시
          const changePercent = 'change_percent' in region ? region.change_percent : undefined

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
                  <div className="flex items-center gap-1.5">
                    <span className="text-sm font-medium text-gray-700">
                      📍 {region.region}
                    </span>
                    {/* 예측 모드일 때 지역명 옆에 변화율 표시 */}
                    {viewMode === 'prediction' && changePercent !== undefined && (
                      <span className={`text-xs font-medium ${
                        changePercent > 0 ? 'text-red-600' : changePercent < 0 ? 'text-blue-600' : 'text-gray-500'
                      }`}>
                        ({changePercent > 0 ? '▲' : changePercent < 0 ? '▼' : '−'}{Math.abs(changePercent)}%)
                      </span>
                    )}
                  </div>
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
