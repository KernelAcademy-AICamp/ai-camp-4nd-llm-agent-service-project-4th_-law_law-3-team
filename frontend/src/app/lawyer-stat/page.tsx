'use client'

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import Link from 'next/link'
import { CrossAnalysisHeatmap } from '@/features/lawyer-stat/components/CrossAnalysisHeatmap'
import { RegionDetailList } from '@/features/lawyer-stat/components/RegionDetailList'
import { RegionGeoMap } from '@/features/lawyer-stat/components/RegionGeoMap'
import { SpecialtyBarChart } from '@/features/lawyer-stat/components/SpecialtyBarChart'
import { StickyTabNav, type TabType } from '@/features/lawyer-stat/components/StickyTabNav'
import {
  fetchDensityStats,
  fetchOverview,
  fetchRegionStats,
  fetchSpecialtyStats,
} from '@/features/lawyer-stat/services'

export type ViewMode = 'count' | 'density'

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
  const [activeTab, setActiveTab] = useState<TabType>('region')
  const [viewMode, setViewMode] = useState<ViewMode>('count')
  const [selectedProvince, setSelectedProvince] = useState<string | null>(null)
  const [highlightedRegion, setHighlightedRegion] = useState<string | null>(null)
  const [mapSelectedRegion, setMapSelectedRegion] = useState<string | null>(null)

  const regionSectionRef = useRef<HTMLDivElement>(null)
  const crossSectionRef = useRef<HTMLDivElement>(null)

  const overviewQuery = useQuery({
    queryKey: ['lawyer-stat', 'overview'],
    queryFn: fetchOverview,
  })

  const regionQuery = useQuery({
    queryKey: ['lawyer-stat', 'region'],
    queryFn: fetchRegionStats,
  })

  const densityQuery = useQuery({
    queryKey: ['lawyer-stat', 'density'],
    queryFn: fetchDensityStats,
  })

  const specialtyQuery = useQuery({
    queryKey: ['lawyer-stat', 'specialty'],
    queryFn: fetchSpecialtyStats,
  })

  const isLoading =
    overviewQuery.isLoading ||
    regionQuery.isLoading ||
    densityQuery.isLoading ||
    specialtyQuery.isLoading

  const hasError =
    overviewQuery.isError ||
    regionQuery.isError ||
    densityQuery.isError ||
    specialtyQuery.isError

  const topRegion = useMemo(() => {
    if (!regionQuery.data?.data || regionQuery.data.data.length === 0) return null
    const top = regionQuery.data.data[0]
    return { name: top.region, count: top.count }
  }, [regionQuery.data])

  const topSpecialty = useMemo(() => {
    if (!specialtyQuery.data?.data || specialtyQuery.data.data.length === 0) return null
    const top = specialtyQuery.data.data[0]
    return { name: top.category_name, count: top.count }
  }, [specialtyQuery.data])

  const filteredRegionData = useMemo(() => {
    const sourceData = viewMode === 'count' ? regionQuery.data?.data : densityQuery.data?.data
    if (!sourceData) return []
    if (!selectedProvince) return sourceData
    return sourceData.filter((r) => r.region.startsWith(selectedProvince))
  }, [regionQuery.data, densityQuery.data, selectedProvince, viewMode])

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
    <div className="min-h-screen bg-gray-50">
      <header className="border-b border-gray-200 bg-white shadow-sm">
        <div className="mx-auto max-w-7xl px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex items-center gap-4">
            <Link
              href="/"
              className="flex items-center gap-1 text-gray-500 transition-colors hover:text-gray-700"
            >
              <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
              홈으로
            </Link>
            <div className="h-6 w-px bg-gray-200" />
            <div>
              <h1 className="text-2xl font-bold text-gray-900">변호사 시장 분석</h1>
              <p className="text-sm text-gray-500 mt-1">지역·전문분야·인구 대비 변호사 분포를 분석합니다.</p>
            </div>
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
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-gray-900">지역별 변호사 현황</h2>
                {/* 변호사 수 / 인구 대비 밀도 토글 */}
                <div className="flex gap-1 rounded-lg bg-gray-100 p-1">
                  <button
                    type="button"
                    onClick={() => setViewMode('count')}
                    className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
                      viewMode === 'count'
                        ? 'bg-white text-gray-900 shadow-sm'
                        : 'text-gray-600 hover:text-gray-900'
                    }`}
                  >
                    변호사 수
                  </button>
                  <button
                    type="button"
                    onClick={() => setViewMode('density')}
                    className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
                      viewMode === 'density'
                        ? 'bg-white text-gray-900 shadow-sm'
                        : 'text-gray-600 hover:text-gray-900'
                    }`}
                  >
                    인구 대비 밀도
                  </button>
                </div>
              </div>

              {/* 지역 탭 버튼 */}
              <div className="mb-4 flex gap-1.5 overflow-x-auto pb-2">
                {PROVINCES.map((province) => (
                  <button
                    key={province}
                    type="button"
                    onClick={() => setSelectedProvince(province === '전체' ? null : province)}
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
                    selectedProvince={selectedProvince}
                    highlightedRegion={highlightedRegion}
                    onRegionClick={(region) => {
                      // 지도에서 클릭 시 해당 지역 세부 화면으로 전환
                      const province = region.split(' ')[0]
                      setSelectedProvince(province)
                      setHighlightedRegion(region)
                      setMapSelectedRegion(region)
                    }}
                  />
                </div>
                <div className="lg:col-span-4">
                  {(viewMode === 'count' ? regionQuery.data : densityQuery.data) && (
                    <RegionDetailList
                      regions={viewMode === 'count' ? regionQuery.data!.data : densityQuery.data!.data}
                      viewMode={viewMode}
                      selectedProvince={selectedProvince}
                      mapSelectedRegion={mapSelectedRegion}
                      onRegionClick={(region) => {
                        if (region) {
                          const province = region.split(' ')[0]
                          setSelectedProvince(province)
                          setHighlightedRegion(region)
                          setMapSelectedRegion(null)  // 리스트에서 선택 시 초기화
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

            {/* Specialty Bottom Card (탭 네비게이션과 무관) */}
            <section className="rounded-xl border border-gray-200 bg-gray-50 p-5 shadow-sm">
              <div className="flex items-center gap-2 mb-4">
                <span className="text-lg">📊</span>
                <h2 className="text-base font-semibold text-gray-800">전문분야별 변호사 분포</h2>
              </div>
              {specialtyQuery.data && <SpecialtyBarChart data={specialtyQuery.data.data} />}
            </section>
          </div>
        )}
      </main>
    </div>
  )
}
