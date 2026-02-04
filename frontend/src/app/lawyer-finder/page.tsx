'use client'

import { useState, useEffect, useCallback, useRef, Suspense } from 'react'
import dynamic from 'next/dynamic'
import Script from 'next/script'
import { useSearchParams } from 'next/navigation'
import { useUI } from '@/context/UIContext'
import { BackButton } from '@/components/ui/BackButton'
import { SearchPanel } from '@/features/lawyer-finder/components/SearchPanel'
import { OfficeDetailPanel } from '@/features/lawyer-finder/components/OfficeDetailPanel'
import { useGeolocation } from '@/features/lawyer-finder/hooks/useGeolocation'
import { lawyerFinderService } from '@/features/lawyer-finder/services'
import { DISTRICT_COORDS } from '@/features/lawyer-finder/constants'
import type { Lawyer, Office, ClusterData } from '@/features/lawyer-finder/types'

const KakaoMap = dynamic(
  () => import('@/features/lawyer-finder/components/KakaoMap').then((m) => m.MemoizedKakaoMap),
  {
    ssr: false,
    loading: () => (
      <div className="w-full h-full flex items-center justify-center bg-navy-50">
        <div className="text-center">
          <div className="w-8 h-8 border-4 border-navy-600 border-t-transparent rounded-full animate-spin mx-auto mb-2" />
          <p className="text-sm text-navy-500">지도 로딩 중...</p>
        </div>
      </div>
    ),
  }
)

const KAKAO_MAP_API_KEY = process.env.NEXT_PUBLIC_KAKAO_MAP_API_KEY

// 카카오 줌 레벨: 숫자 클수록 줌아웃. 6 이상이면 클러스터 모드
const CLUSTER_ZOOM_THRESHOLD = 6
// 지도 드래그 디바운스 (ms)
const DRAG_DEBOUNCE_MS = 400

// Suspense boundary를 위한 wrapper
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
  const [lawyers, setLawyers] = useState<Lawyer[]>([])
  const [selectedLawyer, setSelectedLawyer] = useState<Lawyer | null>(null)
  const [selectionTrigger, setSelectionTrigger] = useState<number>(0)  // 선택 시점 (timestamp)
  const [selectedOffice, setSelectedOffice] = useState<Office | null>(null)
  const [loading, setLoading] = useState(false)
  const [radius, setRadius] = useState(3000)
  const [totalCount, setTotalCount] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [mapReady, setMapReady] = useState(false)
  const [searchCenter, setSearchCenter] = useState<{ lat: number; lng: number } | null>(null)
  const [sigungu, setSigungu] = useState('')
  const [searchQuery, setSearchQuery] = useState('')  // 검색어 (빈 문자열 = 주변 탐색 모드)
  const [category, setCategory] = useState('')  // 선택된 전문분야 카테고리 ID
  const [specialty, setSpecialty] = useState('')  // 특정 전문분야 (예: "이혼") - category보다 우선

  // Phase 4: 클러스터 모드 상태
  const [zoomLevel, setZoomLevel] = useState(5)
  const [mapBounds, setMapBounds] = useState<{ min_lat: number; max_lat: number; min_lng: number; max_lng: number } | null>(null)
  const [clusters, setClusters] = useState<ClusterData[]>([])
  const useClusterMode = zoomLevel >= CLUSTER_ZOOM_THRESHOLD

  const { isChatOpen, chatMode } = useUI()
  const searchParams = useSearchParams()
  const initialSearchDone = useRef(false)
  const urlSearchInProgress = useRef(false)  // URL 파라미터 검색 진행 중
  const lastSearchParamsKey = useRef('')  // 마지막으로 처리한 URL 파라미터 키
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

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

  // URL 쿼리 파라미터로 검색 (챗봇에서 이동 시)
  useEffect(() => {
    if (!mapReady) return

    const lat = searchParams.get('lat')
    const lng = searchParams.get('lng')
    const categoryParam = searchParams.get('category')
    const specialtyParam = searchParams.get('specialty')
    const sigunguParam = searchParams.get('sigungu')
    const radiusParam = searchParams.get('radius')
    const searchAllParam = searchParams.get('searchAll')

    // 파라미터가 없으면 스킵
    if (!lat && !lng && !categoryParam && !specialtyParam && !sigunguParam && !searchAllParam) {
      return
    }

    // 파라미터 조합으로 고유 키 생성 (같은 파라미터면 중복 검색 방지)
    const paramsKey = `${lat}-${lng}-${categoryParam}-${specialtyParam}-${sigunguParam}-${radiusParam}-${searchAllParam}`

    if (lastSearchParamsKey.current === paramsKey) {
      return
    }
    lastSearchParamsKey.current = paramsKey
    initialSearchDone.current = true
    urlSearchInProgress.current = true

    // 검색에 사용할 값들 (파라미터에서 직접 파싱)
    const searchRadius = radiusParam ? parseInt(radiusParam, 10) : 3000
    const searchLat = lat ? parseFloat(lat) : null
    const searchLng = lng ? parseFloat(lng) : null

    // state 업데이트 (UI 표시용)
    if (radiusParam) {
      setRadius(searchRadius)
    }
    if (searchLat && searchLng) {
      setSearchCenter({ lat: searchLat, lng: searchLng })
    }
    if (specialtyParam) {
      setSpecialty(specialtyParam)
    } else if (categoryParam) {
      setCategory(categoryParam)
    }
    if (sigunguParam) {
      setSigungu(sigunguParam)
    }

    // 검색 실행 (state가 아닌 파라미터 값 직접 사용)
    setLoading(true)
    setError(null)

    const finishSearch = () => {
      setLoading(false)
      urlSearchInProgress.current = false
    }

    if (searchAllParam === 'true' && categoryParam) {
      // 전체 검색 모드 (위치 없이 카테고리로만 검색)
      lawyerFinderService.searchLawyers({
        category: categoryParam,
      }).then((response) => {
        setLawyers(response.lawyers)
        setTotalCount(response.total_count)
      }).catch(() => {
        setError('검색에 실패했습니다')
      }).finally(finishSearch)
    } else if (searchLat && searchLng) {
      // 위치 기반 검색 (파라미터 값 직접 사용)
      lawyerFinderService.getNearbyLawyers(
        searchLat,
        searchLng,
        searchRadius,
        categoryParam || undefined,
        specialtyParam || undefined
      ).then((response) => {
        setLawyers(response.lawyers)
        setTotalCount(response.total_count)
      }).catch(() => {
        setError('변호사 정보를 불러오는데 실패했습니다')
      }).finally(finishSearch)
    } else {
      finishSearch()
    }
  }, [searchParams, mapReady])

  // 검색에 사용할 위치 (지도 드래그 위치 또는 GPS 위치)
  const getSearchLocation = useCallback(() => {
    return searchCenter || getEffectiveLocation()
  }, [searchCenter, getEffectiveLocation])

  // 주변 변호사 검색
  const fetchNearbyLawyers = useCallback(async () => {
    if (!mapReady || searchQuery) return

    const location = getSearchLocation()

    setLoading(true)
    setError(null)

    try {
      const response = await lawyerFinderService.getNearbyLawyers(
        location.lat,
        location.lng,
        radius,
        category || undefined,
        specialty || undefined
      )
      setLawyers(response.lawyers)
      setTotalCount(response.total_count)
    } catch {
      setError('변호사 정보를 불러오는데 실패했습니다')
      setLawyers([])
      setTotalCount(0)
    } finally {
      setLoading(false)
    }
  }, [getSearchLocation, radius, mapReady, searchQuery, category, specialty])

  // 위치, 반경, 전문분야 변경 시 검색 (검색어 없을 때만)
  useEffect(() => {
    // URL 파라미터 검색 중이거나 완료 직후에는 스킵
    if (urlSearchInProgress.current) return
    // URL 파라미터로 이미 검색했으면 스킵 (첫 검색만)
    if (initialSearchDone.current) {
      initialSearchDone.current = false  // 다음 변경부터는 검색 가능
      return
    }
    if (mapReady && !searchQuery) {
      fetchNearbyLawyers()
    }
  }, [fetchNearbyLawyers, mapReady, searchQuery, category, specialty])

  // Phase 4: 줌아웃 시 클러스터 API 호출
  useEffect(() => {
    if (!useClusterMode || !mapBounds) {
      setClusters([])
      return
    }

    lawyerFinderService.getClusters(
      mapBounds,
      zoomLevel,
      category || undefined,
      specialty || undefined,
    ).then((res) => {
      setClusters(res.clusters)
    }).catch(() => {
      setClusters([])
    })
  }, [useClusterMode, mapBounds, zoomLevel, category, specialty])

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
        category: category || undefined,
        specialty: specialty || undefined,
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

  // 이 지역에서 재검색 (검색어 유지 + 현재 지도 위치 기준)
  const handleSearchInArea = async () => {
    if (!searchQuery) return

    const location = getSearchLocation()
    setLoading(true)
    setError(null)

    try {
      const response = await lawyerFinderService.searchLawyers({
        name: searchQuery,
        office: searchQuery,
        category: category || undefined,
        specialty: specialty || undefined,
        latitude: location.lat,
        longitude: location.lng,
        radius: radius,
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
    setSpecialty('')  // specialty도 초기화
    // searchQuery가 빈 문자열이 되면 useEffect에서 fetchNearbyLawyers 호출됨
  }, [])

  // 변호사 선택 (명시적 트리거로 지도 이동)
  const handleLawyerSelect = useCallback((lawyer: Lawyer) => {
    setSelectedLawyer(lawyer)
    setSelectionTrigger(Date.now())
  }, [])

  // 반경 변경
  const handleRadiusChange = useCallback((newRadius: number) => {
    setRadius(newRadius)
  }, [])

  // 지도 준비 완료
  const handleMapReady = useCallback(() => {
    setMapReady(true)
  }, [])

  // 내 위치로 이동
  const handleMyLocation = useCallback(() => {
    setSearchCenter(null)  // 드래그 위치 초기화
    getCurrentPosition()
  }, [getCurrentPosition])

  // 지도 드래그 완료 시 중심 변경 (400ms 디바운스)
  const handleCenterChange = useCallback((newCenter: { lat: number; lng: number }) => {
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => {
      setSearchCenter(newCenter)
    }, DRAG_DEBOUNCE_MS)
  }, [])

  // 줌 변경
  const handleZoomChange = useCallback((zoom: number) => {
    setZoomLevel(zoom)
  }, [])

  // 바운드 변경
  const handleBoundsChange = useCallback((bounds: { min_lat: number; max_lat: number; min_lng: number; max_lng: number }) => {
    setMapBounds(bounds)
  }, [])

  // 구 변경 (선택 시 자동으로 해당 지역으로 지도 이동)
  const handleSigunguChange = useCallback((newSigungu: string) => {
    setSigungu(newSigungu)
    if (newSigungu && DISTRICT_COORDS[newSigungu]) {
      setSearchCenter(DISTRICT_COORDS[newSigungu])
    }
  }, [])

  // 전문분야 카테고리 변경
  const handleCategoryChange = useCallback((newCategory: string) => {
    setCategory(newCategory)
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
            <p className="text-sm text-gray-500 mt-0.5">
              위치와 조건으로 쉽게 검색할 수 있습니다
            </p>
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
            category={category}
            onCategoryChange={handleCategoryChange}
            specialty={specialty}
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
          />
        </div>
      </div>
    </div>
    </>
  )
}
