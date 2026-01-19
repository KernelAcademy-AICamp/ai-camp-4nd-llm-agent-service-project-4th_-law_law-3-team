'use client'

import { useState, useEffect, useCallback } from 'react'
import { KakaoMap } from '@/features/lawyer-finder/components/KakaoMap'
import { SearchPanel } from '@/features/lawyer-finder/components/SearchPanel'
import { OfficeDetailPanel } from '@/features/lawyer-finder/components/OfficeDetailPanel'
import { useGeolocation } from '@/features/lawyer-finder/hooks/useGeolocation'
import { lawyerFinderService } from '@/features/lawyer-finder/services'
import type { Lawyer, Office } from '@/features/lawyer-finder/types'

// 서울 구별 좌표 데이터
const DISTRICT_COORDS: Record<string, { lat: number; lng: number }> = {
  '강남구': { lat: 37.5172, lng: 127.0473 },
  '강동구': { lat: 37.5301, lng: 127.1238 },
  '강북구': { lat: 37.6396, lng: 127.0257 },
  '강서구': { lat: 37.5509, lng: 126.8495 },
  '관악구': { lat: 37.4784, lng: 126.9516 },
  '광진구': { lat: 37.5385, lng: 127.0823 },
  '구로구': { lat: 37.4954, lng: 126.8874 },
  '금천구': { lat: 37.4519, lng: 126.9020 },
  '노원구': { lat: 37.6542, lng: 127.0568 },
  '도봉구': { lat: 37.6688, lng: 127.0471 },
  '동대문구': { lat: 37.5744, lng: 127.0396 },
  '동작구': { lat: 37.5124, lng: 126.9393 },
  '마포구': { lat: 37.5663, lng: 126.9014 },
  '서대문구': { lat: 37.5791, lng: 126.9368 },
  '서초구': { lat: 37.4837, lng: 127.0324 },
  '성동구': { lat: 37.5633, lng: 127.0371 },
  '성북구': { lat: 37.5894, lng: 127.0167 },
  '송파구': { lat: 37.5145, lng: 127.1066 },
  '양천구': { lat: 37.5170, lng: 126.8667 },
  '영등포구': { lat: 37.5264, lng: 126.8962 },
  '용산구': { lat: 37.5324, lng: 126.9906 },
  '은평구': { lat: 37.6027, lng: 126.9291 },
  '종로구': { lat: 37.5735, lng: 126.9790 },
  '중구': { lat: 37.5636, lng: 126.9976 },
  '중랑구': { lat: 37.6063, lng: 127.0925 },
}

export default function LawyerFinderPage() {
  const [lawyers, setLawyers] = useState<Lawyer[]>([])
  const [selectedLawyer, setSelectedLawyer] = useState<Lawyer | null>(null)
  const [selectedOffice, setSelectedOffice] = useState<Office | null>(null)
  const [loading, setLoading] = useState(false)
  const [radius, setRadius] = useState(3000)
  const [totalCount, setTotalCount] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [mapReady, setMapReady] = useState(false)
  const [searchCenter, setSearchCenter] = useState<{ lat: number; lng: number } | null>(null)
  const [sigungu, setSigungu] = useState('')
  const [searchQuery, setSearchQuery] = useState('')  // 검색어 (빈 문자열 = 주변 탐색 모드)

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
    if (!mapReady || searchQuery) return  // 검색어가 있으면 자동 fetch 안 함

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
  }, [getSearchLocation, radius, mapReady, searchQuery])

  // 위치 또는 반경 변경 시 검색 (검색어 없을 때만)
  useEffect(() => {
    if (mapReady && !searchQuery) {
      fetchNearbyLawyers()
    }
  }, [fetchNearbyLawyers, mapReady, searchQuery])

  // 이름/사무소 검색
  const handleSearch = async (query: string) => {
    if (!query.trim()) return

    setSearchQuery(query.trim())
    setLoading(true)
    setError(null)

    try {
      const response = await lawyerFinderService.searchLawyers({
        name: query.trim(),
        office: query.trim(),
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

  // 이 지역에서 재검색 (검색어 유지 + 현재 위치 기준)
  const handleSearchInArea = async () => {
    if (!searchQuery) return

    setLoading(true)
    setError(null)

    try {
      // TODO: 백엔드에서 위치 + 검색어 조합 지원 시 개선
      const response = await lawyerFinderService.searchLawyers({
        name: searchQuery,
        office: searchQuery,
        limit: 100,
      })
      setLawyers(response.lawyers)
      setTotalCount(response.total_count)
    } catch (err) {
      console.error('Search failed:', err)
      setError('검색에 실패했습니다')
    } finally {
      setLoading(false)
    }
  }

  // 검색 초기화 (주변 탐색 모드로 복귀)
  const handleSearchReset = useCallback(() => {
    setSearchQuery('')
    // searchQuery가 빈 문자열이 되면 useEffect에서 fetchNearbyLawyers 호출됨
  }, [])

  // 변호사 선택 (지도 이동 트리거를 위해 새 객체 생성)
  const handleLawyerSelect = (lawyer: Lawyer) => {
    setSelectedLawyer({ ...lawyer })
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

  // 구 변경 (선택 시 자동으로 해당 지역으로 지도 이동)
  const handleSigunguChange = useCallback((newSigungu: string) => {
    setSigungu(newSigungu)
    if (newSigungu && DISTRICT_COORDS[newSigungu]) {
      setSearchCenter(DISTRICT_COORDS[newSigungu])
    }
  }, [])

  // 사무소 클릭 (지도 팝업에서)
  const handleOfficeClick = useCallback((office: Office) => {
    setSelectedOffice(office)
  }, [])

  // 사무소 패널 닫기
  const handleOfficeClose = useCallback(() => {
    setSelectedOffice(null)
  }, [])

  // 사무소 패널에서 변호사 선택 시 (나중에 상세페이지로 이동 예정)
  const handleLawyerFromOffice = useCallback((_lawyer: Lawyer) => {
    // TODO: 변호사 상세페이지로 이동
  }, [])

  const center = getSearchLocation()

  // 실제 GPS 위치 (파란 마커용) - 항상 GPS 위치에 표시
  const userLocation = hasLocation ? getEffectiveLocation() : null

  return (
    <div className="h-screen flex flex-col">
      {/* 헤더 */}
      <header className="bg-white border-b px-6 py-4 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-gray-900">가까운 변호사를 빠르게 찾아보세요</h1>
          <p className="text-sm text-gray-500 mt-0.5">
            위치와 조건으로 쉽게 검색할 수 있습니다
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
        {/* 사이드 패널 - 사무소 선택 시 사무소 패널, 아니면 검색 패널 */}
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
            sigungu={sigungu}
            onSigunguChange={handleSigunguChange}
            searchQuery={searchQuery}
          />
        )}

        {/* 지도 영역 */}
        <div className="flex-1 relative">
          {/* 에러 메시지 */}
          {error && (
            <div className="absolute top-4 left-1/2 -translate-x-1/2 z-10 bg-red-100 text-red-700 px-4 py-2 rounded-lg shadow">
              {error}
            </div>
          )}

          {/* 이 지역에서 재검색 버튼 (검색어가 있을 때만) */}
          {searchQuery && (
            <button
              onClick={handleSearchInArea}
              className="absolute top-4 left-1/2 -translate-x-1/2 z-10
                         bg-white px-4 py-2 rounded-full shadow-lg border border-gray-200
                         text-sm font-medium text-gray-700 hover:bg-gray-50 transition
                         flex items-center gap-2"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              이 지역에서 재검색
            </button>
          )}

          <KakaoMap
            center={center}
            userLocation={userLocation}
            lawyers={lawyers}
            selectedLawyer={selectedLawyer}
            radius={radius}
            onMapReady={handleMapReady}
            onLawyerClick={handleLawyerSelect}
            onOfficeClick={handleOfficeClick}
            onMyLocationClick={handleMyLocation}
            onCenterChange={handleCenterChange}
            showRadius={true}
          />
        </div>
      </div>
    </div>
  )
}
