'use client'

import { useState, useEffect, useMemo, memo } from 'react'
import { useQuery } from '@tanstack/react-query'
import type { PredictionYear, ViewMode } from '@/app/lawyer-stats/page'
import type { CourtDemandMarker, DemandStat, DensityStat, RegionStat, SpecialtyStat } from '../types'
import { fetchRegionSpecialties } from '../services'

interface RegionDetailListProps {
  regions: (RegionStat | DensityStat | DemandStat)[]
  viewMode: ViewMode
  predictionYear?: PredictionYear
  selectedProvince: string | null
  onRegionClick?: (region: string | null) => void
  mapSelectedRegion?: string | null
  courtMarkers?: CourtDemandMarker[]
  selectedCourt?: string | null
  onCourtSelect?: (courtName: string | null) => void
  demandCategory?: string
  demandYear?: number
}

/** viewMode별 색상 매핑 (렌더링 외부에서 정의) */
const VIEW_MODE_COLORS = {
  count: { bar: 'bg-blue-500', text: 'text-blue-600' },
  density: { bar: 'bg-emerald-500', text: 'text-emerald-600' },
  prediction: { bar: 'bg-violet-500', text: 'text-violet-600' },
  case_count: { bar: 'bg-amber-500', text: 'text-amber-600' },
  burden_index: { bar: 'bg-rose-500', text: 'text-rose-600' },
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

export function RegionDetailList({ regions, viewMode, predictionYear, selectedProvince, onRegionClick, mapSelectedRegion, courtMarkers, selectedCourt, onCourtSelect, demandCategory, demandYear }: RegionDetailListProps) {
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

  // 법원 선택 시 지역 선택 해제
  useEffect(() => {
    if (selectedCourt) setSelectedRegion(null)
  }, [selectedCourt])

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
      if (viewMode === 'burden_index') {
        const aVal = 'burden_index' in a ? (a as DemandStat).burden_index : 0
        const bVal = 'burden_index' in b ? (b as DemandStat).burden_index : 0
        return (bVal - aVal) * multiplier
      }
      if (viewMode === 'case_count') {
        const aVal = 'case_count' in a ? (a as DemandStat).case_count : 0
        const bVal = 'case_count' in b ? (b as DemandStat).case_count : 0
        return (bVal - aVal) * multiplier
      }
      if (viewMode === 'density' || viewMode === 'prediction') {
        const aDensity = 'density' in a ? (a as DensityStat).density : 0
        const bDensity = 'density' in b ? (b as DensityStat).density : 0
        return (bDensity - aDensity) * multiplier
      }
      return ((b as RegionStat).count - (a as RegionStat).count) * multiplier
    })
  }, [filteredRegions, sortOrder, viewMode])

  const displayRegions = useMemo(
    () => selectedProvince ? sortedRegions : sortedRegions.slice(0, 15),
    [selectedProvince, sortedRegions]
  )

  // viewMode에 따라 최대값 결정 (바 그래프용 - 정렬 순서 무관하게 최대값)
  const maxValue = useMemo(() => {
    if (viewMode === 'burden_index') {
      return Math.max(...displayRegions.map(r => 'burden_index' in r ? (r as DemandStat).burden_index : 0), 1)
    }
    if (viewMode === 'case_count') {
      return Math.max(...displayRegions.map(r => 'case_count' in r ? (r as DemandStat).case_count : 0), 1)
    }
    if (viewMode === 'density' || viewMode === 'prediction') {
      return Math.max(...displayRegions.map(r => 'density' in r ? (r as DensityStat).density : 0), 1)
    }
    return Math.max(...displayRegions.map(r => (r as RegionStat).count), 1)
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

  const isDemandMode = viewMode === 'case_count' || viewMode === 'burden_index'

  // 수요 모드에서 법원 마커가 있으면 법원 기반 뷰 표시
  const hasCourtMarkers = isDemandMode && courtMarkers && courtMarkers.length > 0

  // 법원 목록 정렬
  const sortedCourts = useMemo(() => {
    if (!hasCourtMarkers || !courtMarkers) return []
    return [...courtMarkers].sort((a, b) => {
      const multiplier = sortOrder === 'desc' ? 1 : -1
      if (viewMode === 'burden_index') return (b.burden_index - a.burden_index) * multiplier
      return (b.case_count - a.case_count) * multiplier
    })
  }, [hasCourtMarkers, courtMarkers, sortOrder, viewMode])

  const maxCourtValue = useMemo(() => {
    if (!sortedCourts.length) return 1
    if (viewMode === 'burden_index') return Math.max(...sortedCourts.map(c => c.burden_index), 1)
    return Math.max(...sortedCourts.map(c => c.case_count), 1)
  }, [sortedCourts, viewMode])

  // 수요 모드 - 법원 상세 뷰
  if (hasCourtMarkers && selectedCourt) {
    const court = courtMarkers?.find(m => m.court_name === selectedCourt)
    if (court) {
      return (
        <div className="h-[500px] flex flex-col">
          <button
            type="button"
            onClick={() => onCourtSelect?.(null)}
            className="flex items-center gap-1 text-sm text-gray-500 hover:text-gray-700 transition-colors mb-2"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
            목록으로
          </button>

          <div className="mb-6 flex items-baseline gap-2">
            <div className="text-lg font-semibold text-gray-900">{court.court_name}</div>
            <div className="text-xs text-gray-400">{demandYear}년 · {demandCategory}</div>
          </div>

          <div className="space-y-5 overflow-y-auto flex-1 divide-y divide-gray-200">
            {/* 사건 접수 수 */}
            <div>
              <div className="flex items-center gap-2 mb-1">
                <span className="text-base font-medium text-gray-700">사건 접수 수</span>
                <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                  court.case_count >= 10000 ? 'bg-red-100 text-red-700'
                    : court.case_count >= 5000 ? 'bg-orange-100 text-orange-700'
                    : court.case_count >= 2000 ? 'bg-gray-100 text-gray-600'
                    : court.case_count >= 1000 ? 'bg-blue-100 text-blue-600'
                    : 'bg-slate-100 text-slate-500'
                }`}>
                  수요 {court.case_count >= 10000 ? '매우 많음'
                    : court.case_count >= 5000 ? '많음'
                    : court.case_count >= 2000 ? '보통'
                    : court.case_count >= 1000 ? '적음'
                    : '매우 적음'}
                </span>
              </div>
              <div className="text-lg font-bold text-gray-800">{court.case_count.toLocaleString()}건</div>
            </div>

            {/* 관할 변호사 수 */}
            <div className="pt-4">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-base font-medium text-gray-700">관할 변호사 수</span>
                <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                  court.lawyer_count >= 1000 ? 'bg-red-100 text-red-700'
                    : court.lawyer_count >= 500 ? 'bg-orange-100 text-orange-700'
                    : court.lawyer_count >= 100 ? 'bg-gray-100 text-gray-600'
                    : court.lawyer_count >= 50 ? 'bg-blue-100 text-blue-600'
                    : 'bg-slate-100 text-slate-500'
                }`}>
                  공급 {court.lawyer_count >= 1000 ? '매우 많음'
                    : court.lawyer_count >= 500 ? '많음'
                    : court.lawyer_count >= 100 ? '보통'
                    : court.lawyer_count >= 50 ? '적음'
                    : '매우 적음'}
                </span>
              </div>
              <div className="text-lg font-bold text-gray-800">{court.lawyer_count.toLocaleString()}명</div>
            </div>

            {/* 부담지수 */}
            <div className="pt-4">
              <div className="flex items-center gap-1.5 mb-3">
                <span className="text-base font-medium text-gray-700">부담지수</span>
                {(() => {
                  const s = [...(courtMarkers ?? [])].map(c => c.burden_index).sort((a, b) => a - b)
                  const m = Math.floor(s.length / 2)
                  const med = s.length % 2 === 0 ? (s[m - 1] + s[m]) / 2 : s[m]
                  const d = med > 0 ? Math.round((court.burden_index - med) / med * 100) : 0
                  const c = Math.abs(d) <= 10 ? 'text-gray-400' : d > 0 ? 'text-red-500' : 'text-blue-500'
                  return <span className={`text-sm ${c}`}>{court.burden_index.toFixed(1)}</span>
                })()}
                <span className="relative group">
                  <span className="inline-flex items-center justify-center w-4 h-4 rounded-full bg-gray-200 text-gray-500 text-[10px] font-bold cursor-help">?</span>
                  <span className="absolute bottom-full left-0 mb-1.5 w-56 px-2.5 py-2 rounded-md bg-amber-50 border border-amber-200 text-gray-700 text-xs leading-relaxed opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-10 whitespace-normal">
                    관할 법원의 사건 수를 변호사 수로 나눈 값입니다. 수치가 높을수록 변호사 1인당 처리 사건이 많음을 의미합니다.
                  </span>
                </span>
              </div>
              {(() => {
                const sorted = [...(courtMarkers ?? [])].map(c => c.burden_index).sort((a, b) => a - b)
                const mid = Math.floor(sorted.length / 2)
                const median = sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid]
                const diff = median > 0 ? Math.round((court.burden_index - median) / median * 100) : 0
                const medianText = `중앙값(${median.toFixed(1)})`
                const isHigh = diff > 0
                const boldColor = Math.abs(diff) <= 10 ? 'text-gray-700' : isHigh ? 'text-red-600' : 'text-blue-600'
                const compPart = Math.abs(diff) <= 10
                  ? `전국 ${medianText} 수준으로`
                  : `전국 ${medianText} 대비 ${Math.abs(diff)}% ${isHigh ? '높아,' : '낮아,'}`
                const descPart = Math.abs(diff) <= 10
                  ? '수요와 공급이 비교적 균형적인 지역입니다.'
                  : isHigh
                    ? '업무 부담이 큰 지역에 해당합니다.'
                    : '업무 부담이 상대적으로 낮은 지역에 해당합니다.'
                return (
                  <div className="rounded-lg border p-3 bg-gray-50 border-gray-200">
                    <span className="text-sm leading-relaxed text-gray-500">변호사 1인당 사건 수가 <span className={`font-semibold ${boldColor}`}>{compPart}</span> {descPart}</span>
                    <div className="text-[10px] text-gray-400 mt-1.5">※ 극단값의 영향을 줄이기 위해 중앙값을 기준으로 비교합니다.</div>
                  </div>
                )
              })()}
            </div>

            {/* 관할 지역 */}
            <div className="pt-4">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-base font-medium text-gray-700">관할 지역</span>
                <span className="text-xs text-gray-400">({court.regions.length}개)</span>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {court.regions.map(region => (
                  <span key={region} className="text-xs text-gray-600 py-1 px-2 bg-gray-50 rounded">
                    {region}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>
      )
    }
  }

  // 수요 모드 - 법원 랭킹 목록
  if (hasCourtMarkers && !selectedRegion) {
    const displayCourts = selectedProvince ? sortedCourts : sortedCourts.slice(0, 15)
    const titleText = viewMode === 'burden_index'
      ? (selectedProvince ? `${selectedProvince} 내 부담지수 순위` : '전체 법원 부담지수 순위(Top15)')
      : (selectedProvince ? `${selectedProvince} 내 사건 수 순위` : '전체 법원 사건 수 순위(Top15)')
    const { bar: barColor, text: textColor } = VIEW_MODE_COLORS[viewMode]

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
          {displayCourts.map((court, index) => {
            const value = viewMode === 'burden_index' ? court.burden_index : court.case_count
            const displayValue = viewMode === 'burden_index'
              ? court.burden_index.toFixed(1)
              : `${court.case_count.toLocaleString()}건`
            const barWidth = (value / maxCourtValue) * 100

            return (
              <button
                type="button"
                key={court.court_name}
                onClick={() => onCourtSelect?.(court.court_name)}
                className="w-full flex items-center gap-3 hover:bg-gray-50 rounded-lg p-1 -m-1 transition-colors text-left"
              >
                <span className="w-6 text-right text-sm font-medium text-gray-400">
                  {index + 1}
                </span>
                <div className="flex-1">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-1.5">
                      <span className="text-sm font-medium text-gray-700">
                        🏛 {court.court_name.replace(/지방법원/, '지법').replace(/가정법원/, '가법').replace(/행정법원/, '행법')}
                      </span>
                      <span className="text-xs text-gray-400">
                        {court.regions.length}개 지역
                      </span>
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

  // 상세 뷰
  if (selectedRegion) {
    // 수요 모드 상세 뷰
    if (isDemandMode && selectedRegionData && 'case_count' in selectedRegionData) {
      const data = selectedRegionData as DemandStat
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

          <div className="mb-4">
            <div className="text-lg font-semibold text-gray-900">{selectedRegion}</div>
            <div className="text-sm text-gray-500">사건 수요 상세</div>
          </div>

          <div className="space-y-4 overflow-y-auto flex-1">
            {/* 관할법원 */}
            <div className="p-3 bg-gray-50 rounded-lg">
              <div className="text-xs text-gray-400 mb-1">관할법원</div>
              <div className="text-base font-medium text-gray-800">{data.court_name}</div>
            </div>

            {/* 사건 수 */}
            <div className="p-3 bg-amber-50 rounded-lg">
              <div className="text-xs text-amber-600 mb-1">사건 접수 수</div>
              <div className="text-2xl font-bold text-amber-700">{data.case_count.toLocaleString()}<span className="text-base font-normal ml-1">건</span></div>
            </div>

            {/* 변호사 수 */}
            <div className="p-3 bg-blue-50 rounded-lg">
              <div className="text-xs text-blue-600 mb-1">관할 변호사 수</div>
              <div className="text-2xl font-bold text-blue-700">{data.lawyer_count.toLocaleString()}<span className="text-base font-normal ml-1">명</span></div>
            </div>

            {/* 부담지수 */}
            <div className="p-3 bg-rose-50 rounded-lg">
              <div className="text-xs text-rose-600 mb-1">부담지수 (사건 수 / 변호사 수)</div>
              <div className="text-2xl font-bold text-rose-700">{data.burden_index.toFixed(1)}</div>
              <div className="mt-2 text-xs text-gray-500">
                {data.burden_index >= 50
                  ? '변호사 대비 사건이 매우 많습니다'
                  : data.burden_index >= 20
                    ? '변호사 대비 사건이 다소 많습니다'
                    : data.burden_index >= 10
                      ? '적정 수준입니다'
                      : '변호사 대비 사건이 적습니다'}
              </div>
            </div>
          </div>
        </div>
      )
    }

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
  const titleText = viewMode === 'burden_index'
    ? (selectedProvince ? `${selectedProvince} 내 부담지수 순위` : '전체 지역 부담지수 순위(Top15)')
    : viewMode === 'case_count'
      ? (selectedProvince ? `${selectedProvince} 내 사건 수 순위` : '전체 지역 사건 수 순위(Top15)')
      : viewMode === 'prediction'
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
          let value: number
          let displayValue: string
          if (viewMode === 'burden_index' && 'burden_index' in region) {
            value = (region as DemandStat).burden_index
            displayValue = `${value.toFixed(1)}`
          } else if (viewMode === 'case_count' && 'case_count' in region) {
            value = (region as DemandStat).case_count
            displayValue = `${value.toLocaleString()}건`
          } else if (isDensityMode && 'density' in region) {
            value = (region as DensityStat).density
            displayValue = `${value.toFixed(1)}명/10만`
          } else {
            value = (region as RegionStat).count
            displayValue = `${value.toLocaleString()}명`
          }
          const barWidth = (value / maxValue) * 100
          const { bar: barColor, text: textColor } = VIEW_MODE_COLORS[viewMode]

          // 예측 모드에서 변화율 표시
          const changePercent = 'change_percent' in region ? (region as DensityStat).change_percent : undefined

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
                    {/* 수요 모드일 때 관할법원 표시 */}
                    {'court_name' in region && isDemandMode && (
                      <span className="text-xs text-gray-400">
                        {(region as DemandStat).court_name}
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
