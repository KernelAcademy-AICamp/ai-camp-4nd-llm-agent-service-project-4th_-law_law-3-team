'use client'

import { Suspense } from 'react'
import dynamic from 'next/dynamic'
import Script from 'next/script'
import { useUI } from '@/context/UIContext'
import { BackButton } from '@/components/ui/BackButton'
import { SearchPanel } from '@/features/lawyer-finder/components/SearchPanel'
import { OfficeDetailPanel } from '@/features/lawyer-finder/components/OfficeDetailPanel'
import { useLawyerFinder } from '@/features/lawyer-finder/hooks/useLawyerFinder'

const KakaoMap = dynamic(
  () => import('@/features/lawyer-finder/components/KakaoMap').then((m) => m.MemoizedKakaoMap),
  {
    ssr: false,
    loading: () => (
      <div className="w-full h-full flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <div className="w-8 h-8 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto mb-2" />
          <p className="text-sm text-gray-500">지도 로딩 중...</p>
        </div>
      </div>
    ),
  }
)

const KAKAO_MAP_API_KEY = process.env.NEXT_PUBLIC_KAKAO_MAP_API_KEY

export default function LawyerFinderPageWrapper() {
  return (
    <Suspense fallback={<LawyerFinderLoading />}>
      <LawyerFinderPage />
    </Suspense>
  )
}

function LawyerFinderLoading() {
  return (
    <div className="h-screen flex items-center justify-center bg-gray-50">
      <div className="text-center">
        <div className="w-8 h-8 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
        <p className="text-gray-600">변호사 찾기 페이지 로딩 중...</p>
      </div>
    </div>
  )
}

function LawyerFinderPage() {
  const { isChatOpen, chatMode } = useUI()
  const {
    lawyers,
    loading,
    totalCount,
    error,
    searchQuery,
    category,
    specialty,
    selectedLawyer,
    selectionTrigger,
    selectedOffice,
    radius,
    center,
    userLocation,
    province,
    regions,
    sigungu,
    initialZoom,
    useClusterMode,
    clusters,
    geoLoading,
    geoError,
    handleSearch,
    handleSearchInArea,
    handleSearchReset,
    handleLawyerSelect,
    handleRadiusChange,
    handleMapReady,
    handleMyLocation,
    handleCenterChange,
    handleZoomChange,
    handleBoundsChange,
    handleProvinceChange,
    handleSigunguChange,
    handleCategoryChange,
    handleOfficeClick,
    handleOfficeClose,
    handleLawyerFromOffice,
  } = useLawyerFinder()

  return (
    <>
      {KAKAO_MAP_API_KEY && (
        <Script
          src={`https://dapi.kakao.com/v2/maps/sdk.js?appkey=${KAKAO_MAP_API_KEY}&libraries=clusterer,services&autoload=false`}
          strategy="lazyOnload"
        />
      )}
      <div
        className={`h-screen flex flex-col transition-all duration-500 ease-in-out ${
          isChatOpen && chatMode === 'split' ? 'w-1/2 border-r border-gray-200' : 'w-full'
        }`}
      >
        {/* 헤더 */}
        <header className="bg-white border-b px-6 py-4 flex items-center justify-between">
          <div className="flex items-center">
            <BackButton />
            <div>
              <h1 className="text-xl font-bold text-gray-900">가까운 변호사를 빠르게 찾아보세요</h1>
            </div>
          </div>
          {geoLoading && (
            <span className="text-sm text-blue-600">위치 확인 중...</span>
          )}
          {geoError && (
            <span className="text-sm text-orange-600">{geoError}</span>
          )}
        </header>

        {/* 메인 컨텐츠 */}
        <div className="flex-1 flex overflow-hidden">
          {/* 사이드 패널 */}
          {selectedOffice ? (
            <OfficeDetailPanel
              office={selectedOffice}
              onClose={handleOfficeClose}
              onLawyerSelect={handleLawyerFromOffice}
            />
          ) : (
            <SearchPanel
              lawyers={lawyers}
              loading={loading}
              selectedLawyer={selectedLawyer}
              onLawyerSelect={handleLawyerSelect}
              onRadiusChange={handleRadiusChange}
              onSearch={handleSearch}
              onSearchReset={handleSearchReset}
              radius={radius}
              totalCount={totalCount}
              province={province}
              onProvinceChange={handleProvinceChange}
              provinces={regions}
              sigungu={sigungu}
              onSigunguChange={handleSigunguChange}
              searchQuery={searchQuery}
              category={category}
              onCategoryChange={handleCategoryChange}
              specialty={specialty}
            />
          )}

          {/* 지도 영역 */}
          <div className="flex-1 relative">
            {error && (
              <div className="absolute top-4 left-1/2 -translate-x-1/2 z-10 bg-red-100 text-red-700 px-4 py-2 rounded-lg shadow">
                {error}
              </div>
            )}

            {searchQuery && (
              <button
                type="button"
                onClick={handleSearchInArea}
                className="absolute top-4 left-1/2 -translate-x-1/2 z-10
                           bg-white px-4 py-2 rounded-full shadow-lg border border-gray-200
                           text-sm font-medium text-gray-700 hover:bg-gray-50 transition
                           flex items-center gap-2"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                이 지역에서 재검색
              </button>
            )}

            <KakaoMap
              center={center}
              userLocation={userLocation}
              lawyers={useClusterMode ? [] : lawyers}
              clusters={useClusterMode ? clusters : undefined}
              selectedLawyer={selectedLawyer}
              selectionTrigger={selectionTrigger}
              radius={radius}
              onMapReady={handleMapReady}
              onLawyerClick={handleLawyerSelect}
              onOfficeClick={handleOfficeClick}
              onMyLocationClick={handleMyLocation}
              onCenterChange={handleCenterChange}
              onZoomChange={handleZoomChange}
              onBoundsChange={handleBoundsChange}
              showRadius={true}
              initialLevel={initialZoom}
            />
          </div>
        </div>
      </div>
    </>
  )
}
