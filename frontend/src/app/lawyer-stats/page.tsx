'use client'

import dynamic from 'next/dynamic'
import { BackButton } from '@/components/ui/BackButton'
import { useUI } from '@/context/UIContext'
import { StickyTabNav } from '@/features/lawyer-stats/components/StickyTabNav'
import { useStatsFilter } from '@/features/lawyer-stats/hooks/useStatsFilter'
import { DEMAND_CATEGORIES, PROVINCES } from '@/features/lawyer-stats/types'
import type { ViewMode } from '@/features/lawyer-stats/types'

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

export default function LawyerStatPage() {
  const { isChatOpen, chatMode } = useUI()
  const {
    activeTab,
    indicatorGroup,
    viewMode,
    predictionYear,
    demandCategory,
    demandYear,
    selectedProvince,
    highlightedRegion,
    mapSelectedRegion,
    selectedCourt,
    isPredictionMode,
    isDemandMode,
    availableDemandYears,
    isLoading,
    hasError,
    filteredRegionData,
    courtMarkers,
    regionSectionRef,
    crossSectionRef,
    setViewMode,
    setPredictionYear,
    setDemandCategory,
    setDemandYear,
    handleIndicatorGroupChange,
    handleCourtClick,
    handleTabChange,
    handleProvinceSelect,
    handleMapRegionClick,
    handleListRegionClick,
  } = useStatsFilter()

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
            <StickyTabNav activeTab={activeTab} onTabChange={handleTabChange} />

            {/* Region Section */}
            <section
              id="region-section"
              ref={regionSectionRef}
              className="scroll-mt-16 rounded-xl border border-gray-200 bg-white p-6 shadow-sm"
            >
              {/* 2단 지표 토글 */}
              <div className="flex items-center gap-3 mb-4 flex-wrap">
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

                <div className="h-5 w-px bg-gray-300" />

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

                    <div className="h-5 w-px bg-gray-300" />

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
                    onClick={() => handleProvinceSelect(province)}
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
                    onRegionClick={handleMapRegionClick}
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
                      onRegionClick={handleListRegionClick}
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
