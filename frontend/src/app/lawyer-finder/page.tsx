'use client'

import { useState, useEffect, useCallback } from 'react'
import { KakaoMap } from '@/features/lawyer-finder/components/KakaoMap'
import { SearchPanel } from '@/features/lawyer-finder/components/SearchPanel'
import { useGeolocation } from '@/features/lawyer-finder/hooks/useGeolocation'
import { lawyerFinderService } from '@/features/lawyer-finder/services'
import type { Lawyer } from '@/features/lawyer-finder/types'

export default function LawyerFinderPage() {
  const [lawyers, setLawyers] = useState<Lawyer[]>([])
  const [selectedLawyer, setSelectedLawyer] = useState<Lawyer | null>(null)
  const [loading, setLoading] = useState(false)
  const [radius, setRadius] = useState(5000)
  const [totalCount, setTotalCount] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [mapReady, setMapReady] = useState(false)
  const [searchCenter, setSearchCenter] = useState<{ lat: number; lng: number } | null>(null)

  const {
    getCurrentPosition,
    getEffectiveLocation,
    hasLocation,
    loading: geoLoading,
    error: geoError,
  } = useGeolocation()

  // 페이지 로드 시 위치 권한 요청
  useEffect(() => {
    getCurrentPosition()
  }, [getCurrentPosition])

  // 검색에 사용할 위치 (지도 드래그 위치 또는 GPS 위치)
  const getSearchLocation = useCallback(() => {
    return searchCenter || getEffectiveLocation()
  }, [searchCenter, getEffectiveLocation])

  // 주변 변호사 검색
  const fetchNearbyLawyers = useCallback(async () => {
    if (!mapReady) return

    const location = getSearchLocation()
    setLoading(true)
    setError(null)

    try {
      const response = await lawyerFinderService.getNearbyLawyers(
        location.lat,
        location.lng,
        radius,
        100
      )
      setLawyers(response.lawyers)
      setTotalCount(response.total_count)
    } catch (err) {
      console.error('Failed to fetch lawyers:', err)
      setError('변호사 정보를 불러오는데 실패했습니다')
      setLawyers([])
      setTotalCount(0)
    } finally {
      setLoading(false)
    }
  }, [getSearchLocation, radius, mapReady])

  // 위치 또는 반경 변경 시 검색
  useEffect(() => {
    if (mapReady) {
      fetchNearbyLawyers()
    }
  }, [fetchNearbyLawyers, mapReady])

  // 이름/사무소/지역 검색
  const handleSearch = async (query: {
    name?: string
    office?: string
    district?: string
  }) => {
    setLoading(true)
    setError(null)

    try {
      const response = await lawyerFinderService.searchLawyers({
        ...query,
        limit: 100,
      })
      setLawyers(response.lawyers)
      setTotalCount(response.total_count)
    } catch (err) {
      console.error('Search failed:', err)
      setError('검색에 실패했습니다')
      setLawyers([])
      setTotalCount(0)
    } finally {
      setLoading(false)
    }
  }

  // 변호사 선택
  const handleLawyerSelect = (lawyer: Lawyer) => {
    setSelectedLawyer(lawyer)
  }

  // 반경 변경
  const handleRadiusChange = (newRadius: number) => {
    setRadius(newRadius)
  }

  // 지도 준비 완료
  const handleMapReady = () => {
    setMapReady(true)
  }

  // 내 위치로 이동
  const handleMyLocation = useCallback(() => {
    setSearchCenter(null)  // 드래그 위치 초기화
    getCurrentPosition()
  }, [getCurrentPosition])

  // 지도 드래그 완료 시 중심 변경
  const handleCenterChange = useCallback((newCenter: { lat: number; lng: number }) => {
    setSearchCenter(newCenter)
  }, [])

  const center = getSearchLocation()

  // 실제 GPS 위치 (파란 마커용) - 항상 GPS 위치에 표시
  const userLocation = hasLocation ? getEffectiveLocation() : null

  return (
    <div className="h-screen flex flex-col">
      {/* 헤더 */}
      <header className="bg-white border-b px-6 py-4 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-gray-900">주변 변호사 찾기</h1>
          <p className="text-sm text-gray-500 mt-0.5">
            {hasLocation
              ? '현재 위치 기준으로 검색합니다'
              : '서울 시청 기준으로 검색합니다 (위치 권한 허용 시 현재 위치로 변경)'}
          </p>
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
        {/* 검색 패널 */}
        <SearchPanel
          lawyers={lawyers}
          loading={loading}
          selectedLawyer={selectedLawyer}
          onLawyerSelect={handleLawyerSelect}
          onRadiusChange={handleRadiusChange}
          onSearch={handleSearch}
          radius={radius}
          totalCount={totalCount}
        />

        {/* 지도 영역 */}
        <div className="flex-1 relative">
          {error && (
            <div className="absolute top-4 left-1/2 -translate-x-1/2 z-10 bg-red-100 text-red-700 px-4 py-2 rounded-lg shadow">
              {error}
            </div>
          )}
          <KakaoMap
            center={center}
            userLocation={userLocation}
            lawyers={lawyers}
            selectedLawyer={selectedLawyer}
            radius={radius}
            onMapReady={handleMapReady}
            onLawyerClick={handleLawyerSelect}
            onMyLocationClick={handleMyLocation}
            onCenterChange={handleCenterChange}
            showRadius={true}
          />
        </div>
      </div>
    </div>
  )
}
