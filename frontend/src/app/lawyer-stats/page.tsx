'use client'

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { keepPreviousData, useQuery } from '@tanstack/react-query'
import dynamic from 'next/dynamic'
import { BackButton } from '@/components/ui/BackButton'
import { useChat } from '@/context/ChatContext'
import { useUI } from '@/context/UIContext'
import { StickyTabNav, type TabType } from '@/features/lawyer-stats/components/StickyTabNav'

// Inline loading component for dynamic imports
const DynamicLoadingFallback = () => (
  <div className="flex h-40 items-center justify-center">
    <div className="h-8 w-8 animate-spin rounded-full border-4 border-blue-500 border-t-transparent" />
  </div>
)

const MapLoadingFallback = () => (
  <div className="flex h-[500px] items-center justify-center">
    <div className="h-8 w-8 animate-spin rounded-full border-4 border-blue-500 border-t-transparent" />
  </div>
)

// Dynamic imports for heavy chart components (reduces initial bundle size)
const RegionGeoMap = dynamic(
  () => import('@/features/lawyer-stats/components/RegionGeoMap').then(m => m.RegionGeoMap),
  { ssr: false, loading: MapLoadingFallback }
)

const RegionDetailList = dynamic(
  () => import('@/features/lawyer-stats/components/RegionDetailList').then(m => m.RegionDetailList),
  { loading: DynamicLoadingFallback }
)

const CrossAnalysisHeatmap = dynamic(
  () => import('@/features/lawyer-stats/components/CrossAnalysisHeatmap').then(m => m.CrossAnalysisHeatmap),
  { loading: DynamicLoadingFallback }
)

import {
  fetchDemandStats,
  fetchDensityStats,
  fetchOverview,
  fetchRegionStats,
} from '@/features/lawyer-stats/services'
import type { CourtDemandMarker, DemandStat, StatsFilter } from '@/features/lawyer-stats/types'

export type IndicatorGroup = 'supply' | 'demand'
export type ViewMode = 'count' | 'density' | 'prediction' | 'case_count' | 'burden_index'
export type PredictionYear = 2030 | 2035 | 2040
export type DemandCategory = '민사' | '형사' | '가사' | '행정' | '소년보호' | '가정보호'

const DEMAND_CATEGORIES: DemandCategory[] = ['민사', '형사', '가사', '행정', '소년보호', '가정보호']

function LoadingSpinner() {
  return (
    <div className="flex h-40 items-center justify-center">
      <div className="h-8 w-8 animate-spin rounded-full border-4 border-blue-500 border-t-transparent" />
    </div>
  )
}

function ErrorMessage({ message }: { message: string }) {
  return (
    <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-red-700">
      <div className="flex items-center gap-2">
        <svg className="h-5 w-5" fill="currentColor" viewBox="0 0 20 20">
          <path
            fillRule="evenodd"
            d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
            clipRule="evenodd"
          />
        </svg>
        <span>{message}</span>
      </div>
    </div>
  )
}

const PROVINCES = [
  '전체',
  '서울',
  '경기',
  '인천',
  '부산',
  '대구',
  '광주',
  '대전',
  '울산',
  '세종',
  '강원',
  '충북',
  '충남',
  '전북',
  '전남',
  '경북',
  '경남',
  '제주',
]

export default function LawyerStatPage() {
  const { isChatOpen, chatMode } = useUI()
  const { sessionData } = useChat()
  const [activeTab, setActiveTab] = useState<TabType>('region')
  const [indicatorGroup, setIndicatorGroup] = useState<IndicatorGroup>('supply')
  const [viewMode, setViewMode] = useState<ViewMode>('count')
  const [predictionYear, setPredictionYear] = useState<PredictionYear>(2030)
  const [demandCategory, setDemandCategory] = useState<DemandCategory>('민사')
  const [demandYear, setDemandYear] = useState<number>(2024)
  const [selectedProvince, setSelectedProvince] = useState<string | null>(null)
  const [highlightedRegion, setHighlightedRegion] = useState<string | null>(null)
  const [mapSelectedRegion, setMapSelectedRegion] = useState<string | null>(null)
  const [selectedCourt, setSelectedCourt] = useState<string | null>(null)
  const [courtCoords, setCourtCoords] = useState<Record<string, [number, number]>>({})

  const regionSectionRef = useRef<HTMLDivElement>(null)
  const crossSectionRef = useRef<HTMLDivElement>(null)

  // 법원 좌표 데이터 로드
  useEffect(() => {
    fetch('/data/court_coordinates.json')
      .then(r => r.json())
      .then(setCourtCoords)
      .catch(() => {})
  }, [])

  // 채팅 에이전트 sessionData.stats_filter → 대시보드 필터 자동 적용
  const appliedFilterRef = useRef<string>('')
  useEffect(() => {
    const filter = sessionData.stats_filter as StatsFilter | undefined
    if (!filter) return
    const filterKey = JSON.stringify(filter)
    if (filterKey === appliedFilterRef.current) return
    appliedFilterRef.current = filterKey

    // 지역 선택 (null이면 전체)
    if ('selectedProvince' in filter) {
      setSelectedProvince(filter.selectedProvince ?? null)
      setHighlightedRegion(null)
      setMapSelectedRegion(null)
      setSelectedCourt(null)
    }
    if (filter.indicatorGroup) setIndicatorGroup(filter.indicatorGroup)
    if (filter.viewMode) setViewMode(filter.viewMode)
    if (filter.activeTab) {
      setActiveTab(filter.activeTab)
      const targetRef = filter.activeTab === 'region' ? regionSectionRef : crossSectionRef
      setTimeout(() => targetRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100)
    }
    if (filter.predictionYear) setPredictionYear(filter.predictionYear)
    if (filter.demandCategory) setDemandCategory(filter.demandCategory as DemandCategory)
    if (filter.demandYear) setDemandYear(filter.demandYear)
  }, [sessionData.stats_filter]) // eslint-disable-line react-hooks/exhaustive-deps

  // 공급 그룹으로 전환 시 viewMode 복원
  const handleIndicatorGroupChange = useCallback((group: IndicatorGroup) => {
    setIndicatorGroup(group)
    setSelectedCourt(null)
    if (group === 'supply') {
      setViewMode('count')
    } else {
      setViewMode('case_count')
    }
  }, [])

  // === Supply queries ===
  const overviewQuery = useQuery({
    queryKey: ['lawyer-stats', 'overview'],
    queryFn: fetchOverview,
  })

  const regionQuery = useQuery({
    queryKey: ['lawyer-stats', 'region'],
    queryFn: fetchRegionStats,
  })

  const isPredictionMode = viewMode === 'prediction'

  const densityQuery = useQuery({
    queryKey: ['lawyer-stats', 'density', viewMode, isPredictionMode ? predictionYear : null],
    queryFn: () => fetchDensityStats(
      isPredictionMode ? predictionYear : 'current',
      isPredictionMode
    ),
    placeholderData: keepPreviousData,
  })

  // === Demand query ===
  const demandQuery = useQuery({
    queryKey: ['lawyer-stats', 'demand', demandCategory, demandYear],
    queryFn: () => fetchDemandStats(demandCategory, demandYear),
    enabled: indicatorGroup === 'demand',
    placeholderData: keepPreviousData,
  })

  // 수요 모드에서 사용 가능한 연도 목록
  const availableDemandYears = useMemo(() => {
    if (demandQuery.data?.available_years && demandQuery.data.available_years.length > 0) {
      return demandQuery.data.available_years
    }
    // 기본값: 2015~2024
    return Array.from({ length: 10 }, (_, i) => 2015 + i)
  }, [demandQuery.data?.available_years])

  // 법원 마커 클릭 → 해당 시/도로 필터 + 법원 선택
  const handleCourtClick = useCallback((courtName: string | null) => {
    if (!courtName) {
      setSelectedCourt(null)
      return
    }
    if (courtName === selectedCourt) {
      setSelectedCourt(null)
      return
    }
    setSelectedCourt(courtName)
    const stat = demandQuery.data?.data.find(d => d.court_name === courtName)
    if (stat) {
      const province = stat.region.split(' ')[0]
      setSelectedProvince(province)
    }
    setHighlightedRegion(null)
    setMapSelectedRegion(null)
  }, [selectedCourt, demandQuery.data])

  const isDemandMode = indicatorGroup === 'demand'

  const isLoading = isDemandMode
    ? demandQuery.isLoading
    : (overviewQuery.isLoading || regionQuery.isLoading || densityQuery.isLoading)

  const hasError = isDemandMode
    ? demandQuery.isError
    : (overviewQuery.isError || regionQuery.isError || densityQuery.isError)

  const filteredRegionData = useMemo(() => {
    if (isDemandMode && demandQuery.data) {
      const sourceData = demandQuery.data.data
      if (!selectedProvince) return sourceData
      return sourceData.filter(r => r.region.startsWith(selectedProvince))
    }
    const sourceData = viewMode === 'count'
      ? regionQuery.data?.data
      : densityQuery.data?.data
    if (!sourceData) return []
    if (!selectedProvince) return sourceData
    return sourceData.filter((r) => r.region.startsWith(selectedProvince))
  }, [isDemandMode, viewMode, demandQuery.data, regionQuery.data, densityQuery.data, selectedProvince])

  // DemandStat[] → CourtDemandMarker[] 그룹화
  const courtMarkers = useMemo((): CourtDemandMarker[] => {
    if (!isDemandMode || !demandQuery.data || !Object.keys(courtCoords).length) return []

    const grouped = new Map<string, { regions: string[]; stat: DemandStat }>()
    for (const stat of demandQuery.data.data) {
      const existing = grouped.get(stat.court_name)
      if (existing) {
        existing.regions.push(stat.region)
      } else {
        grouped.set(stat.court_name, { regions: [stat.region], stat })
      }
    }

    const markers: CourtDemandMarker[] = []
    grouped.forEach(({ regions, stat }, court) => {
      const coords = courtCoords[court]
      if (!coords) return
      markers.push({
        court_name: court,
        coordinates: coords,
        case_count: stat.case_count,
        lawyer_count: stat.lawyer_count,
        burden_index: stat.burden_index,
        regions,
      })
    })

    if (selectedProvince) {
      return markers.filter(m => m.regions.some(r => r.startsWith(selectedProvince)))
    }
    return markers
  }, [isDemandMode, demandQuery.data, courtCoords, selectedProvince])

  const scrollToSection = useCallback((tab: TabType) => {
    const refs: Record<TabType, React.RefObject<HTMLDivElement | null>> = {
      region: regionSectionRef,
      cross: crossSectionRef,
    }
    refs[tab].current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }, [])

  const handleTabChange = useCallback((tab: TabType) => {
    setActiveTab(tab)
    scrollToSection(tab)
  }, [scrollToSection])

  useEffect(() => {
    const options = {
      root: null,
      rootMargin: '-100px 0px -50% 0px',
      threshold: 0,
    }

    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          const id = entry.target.id
          if (id === 'region-section') setActiveTab('region')
          else if (id === 'cross-section') setActiveTab('cross')
        }
      })
    }, options)

    const sections = [
      regionSectionRef.current,
      crossSectionRef.current,
    ]

    sections.forEach((section) => {
      if (section) observer.observe(section)
    })

    return () => {
      sections.forEach((section) => {
        if (section) observer.unobserve(section)
      })
    }
  }, [])

  return (
    <div
      className={`min-h-screen bg-gray-50 transition-all duration-500 ease-in-out ${
        isChatOpen && chatMode === 'split' ? 'w-1/2 border-r border-gray-200' : 'w-full'
      }`}
    >
      <header className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="flex items-center gap-3">
          <BackButton />
          <span className="text-2xl">📊</span>
          <div>
            <h1 className="text-xl font-bold text-gray-900">변호사 시장 분석</h1>
            <p className="text-sm text-gray-500">지역·전문분야·인구 대비 변호사 분포를 분석합니다.</p>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
        {isLoading && (
          <div className="flex min-h-[400px] items-center justify-center">
            <LoadingSpinner />
          </div>
        )}

        {hasError && (
          <ErrorMessage message="데이터를 불러오는 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요." />
        )}

        {!isLoading && !hasError && (
          <div className="space-y-6">
            {/* Sticky Tab Navigation */}
            <StickyTabNav activeTab={activeTab} onTabChange={handleTabChange} />

            {/* Region Section */}
            <section
              id="region-section"
              ref={regionSectionRef}
              className="scroll-mt-16 rounded-xl border border-gray-200 bg-white p-6 shadow-sm"
            >
              {/* ===== 2단 지표 토글 ===== */}
              <div className="flex items-center gap-3 mb-4 flex-wrap">
                {/* [공급][수요] pill 토글 */}
                <div className="flex rounded-lg bg-gray-100 p-0.5">
                  <button
                    type="button"
                    onClick={() => handleIndicatorGroupChange('supply')}
                    className={`rounded-md px-3.5 py-1.5 text-sm font-semibold transition-colors ${
                      indicatorGroup === 'supply'
                        ? 'bg-gray-900 text-white shadow-sm'
                        : 'text-gray-500 hover:text-gray-700'
                    }`}
                  >
                    공급
                  </button>
                  <button
                    type="button"
                    onClick={() => handleIndicatorGroupChange('demand')}
                    className={`rounded-md px-3.5 py-1.5 text-sm font-semibold transition-colors ${
                      indicatorGroup === 'demand'
                        ? 'bg-gray-900 text-white shadow-sm'
                        : 'text-gray-500 hover:text-gray-700'
                    }`}
                  >
                    수요
                  </button>
                </div>

                {/* 구분선 */}
                <div className="h-5 w-px bg-gray-300" />

                {/* 하위 지표 라디오 버튼 */}
                {indicatorGroup === 'supply' ? (
                  <div className="flex items-center gap-4">
                    {([
                      { mode: 'count' as ViewMode, label: '변호사 수' },
                      { mode: 'density' as ViewMode, label: '인구 대비 밀도' },
                      { mode: 'prediction' as ViewMode, label: '향후 예측' },
                    ]).map(({ mode, label }) => (
                      <label key={mode} className="flex items-center gap-1.5 cursor-pointer text-sm">
                        <input
                          type="radio"
                          name="supply-indicator"
                          checked={viewMode === mode}
                          onChange={() => setViewMode(mode)}
                          className="h-3.5 w-3.5 text-gray-900 focus:ring-gray-500"
                        />
                        <span className={viewMode === mode ? 'font-medium text-gray-900' : 'text-gray-600'}>
                          {label}
                        </span>
                      </label>
                    ))}

                    {/* 예측 연도 선택 */}
                    {viewMode === 'prediction' && (
                      <>
                        <div className="h-5 w-px bg-gray-300" />
                        <div className="flex gap-1 rounded-lg bg-violet-100 p-0.5">
                          {([2030, 2035, 2040] as const).map((year) => (
                            <button
                              key={year}
                              type="button"
                              onClick={() => setPredictionYear(year)}
                              className={`rounded-md px-2.5 py-1 text-sm font-medium transition-colors ${
                                predictionYear === year
                                  ? 'bg-violet-600 text-white shadow-sm'
                                  : 'text-violet-700 hover:text-violet-900'
                              }`}
                            >
                              {year}년
                            </button>
                          ))}
                        </div>
                      </>
                    )}
                  </div>
                ) : (
                  <>
                    {/* 수요 지표 라디오 */}
                    <div className="flex items-center gap-4">
                      <label className="flex items-center gap-1.5 cursor-pointer text-sm">
                        <input
                          type="radio"
                          name="demand-indicator"
                          checked={viewMode === 'case_count'}
                          onChange={() => setViewMode('case_count')}
                          className="h-3.5 w-3.5 text-gray-900 focus:ring-gray-500"
                        />
                        <span className={viewMode === 'case_count' ? 'font-medium text-gray-900' : 'text-gray-600'}>
                          사건 수
                        </span>
                      </label>
                    </div>

                    {/* 구분선 */}
                    <div className="h-5 w-px bg-gray-300" />

                    {/* 분야 필터 pill 토글 */}
                    <div className="flex gap-1 rounded-lg bg-amber-50 p-0.5">
                      {DEMAND_CATEGORIES.map((cat) => (
                        <button
                          key={cat}
                          type="button"
                          onClick={() => setDemandCategory(cat)}
                          className={`rounded-md px-2.5 py-1 text-sm font-medium transition-colors ${
                            demandCategory === cat
                              ? 'bg-amber-500 text-white shadow-sm'
                              : 'text-amber-800 hover:text-amber-900 hover:bg-amber-100'
                          }`}
                        >
                          {cat}
                        </button>
                      ))}
                    </div>

                    {/* 연도 드롭다운 */}
                    <select
                      value={demandYear}
                      onChange={(e) => setDemandYear(Number(e.target.value))}
                      className="rounded-md border border-gray-300 bg-white px-2.5 py-1.5 text-sm font-medium text-gray-700 shadow-sm focus:border-amber-500 focus:outline-none focus:ring-1 focus:ring-amber-500"
                    >
                      {availableDemandYears.map((y) => (
                        <option key={y} value={y}>{y}년</option>
                      ))}
                    </select>
                  </>
                )}
              </div>

              {/* 지역 탭 버튼 */}
              <div className="mb-4 flex gap-1.5 overflow-x-auto pb-2">
                {PROVINCES.map((province) => (
                  <button
                    key={province}
                    type="button"
                    onClick={() => {
                      setSelectedProvince(province === '전체' ? null : province)
                      setHighlightedRegion(null)
                      setMapSelectedRegion(null)
                      setSelectedCourt(null)
                    }}
                    className={`shrink-0 rounded-full px-3 py-1.5 text-sm font-medium transition-colors ${
                      (province === '전체' && !selectedProvince) || province === selectedProvince
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                    }`}
                  >
                    {province}
                  </button>
                ))}
              </div>

              <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
                <div className="lg:col-span-8">
                  <RegionGeoMap
                    data={filteredRegionData}
                    viewMode={viewMode}
                    predictionYear={isPredictionMode ? predictionYear : undefined}
                    selectedProvince={selectedProvince}
                    highlightedRegion={highlightedRegion}
                    courtMarkers={courtMarkers}
                    selectedCourt={selectedCourt}
                    onCourtClick={handleCourtClick}
                    onRegionClick={(region) => {
                      if (!region) {
                        setHighlightedRegion(null)
                        setMapSelectedRegion(null)
                        return
                      }
                      if (mapSelectedRegion === region) {
                        setHighlightedRegion(null)
                        setMapSelectedRegion(null)
                        return
                      }
                      const province = region.split(' ')[0]
                      setSelectedProvince(province)
                      setHighlightedRegion(region)
                      setMapSelectedRegion(region)
                    }}
                  />
                </div>
                <div className="lg:col-span-4">
                  {filteredRegionData.length > 0 && (
                    <RegionDetailList
                      regions={filteredRegionData}
                      viewMode={viewMode}
                      predictionYear={isPredictionMode ? predictionYear : undefined}
                      selectedProvince={selectedProvince}
                      mapSelectedRegion={mapSelectedRegion}
                      courtMarkers={courtMarkers}
                      selectedCourt={selectedCourt}
                      onCourtSelect={handleCourtClick}
                      demandCategory={demandCategory}
                      demandYear={demandYear}
                      onRegionClick={(region) => {
                        if (region) {
                          const province = region.split(' ')[0]
                          setSelectedProvince(province)
                          setHighlightedRegion(region)
                          setMapSelectedRegion(null)
                        } else {
                          setHighlightedRegion(null)
                          setMapSelectedRegion(null)
                        }
                      }}
                    />
                  )}
                </div>
              </div>
            </section>

            {/* Cross Analysis Section */}
            <section
              id="cross-section"
              ref={crossSectionRef}
              className="scroll-mt-16"
            >
              <CrossAnalysisHeatmap />
            </section>

          </div>
        )}
      </main>
    </div>
  )
}
