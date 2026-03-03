'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
import dynamic from 'next/dynamic'
import Script from 'next/script'
import { useSearchParams } from 'next/navigation'
import { useUI } from '@/context/UIContext'
import { BackButton } from '@/components/ui/BackButton'
import { SearchPanel } from './SearchPanel'
import { OfficeDetailPanel } from './OfficeDetailPanel'
import { useGeolocation } from '../hooks/useGeolocation'
import { lawyerFinderService } from '../services'
import { DISTRICT_COORDS } from '../constants'
import type { Lawyer, Office, ClusterData } from '../types'

const KakaoMap = dynamic(
  () => import('./KakaoMap').then((m) => m.MemoizedKakaoMap),
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
const CLUSTER_ZOOM_THRESHOLD = 6
const DRAG_DEBOUNCE_MS = 400

interface LawyerFinderMainProps {
  isInline?: boolean
}

export function LawyerFinderMain({ isInline = false }: LawyerFinderMainProps) {
  const searchParams = useSearchParams()

  const [lawyers, setLawyers] = useState<Lawyer[]>([])
  const [selectedLawyer, setSelectedLawyer] = useState<Lawyer | null>(null)
  const [selectionTrigger, setSelectionTrigger] = useState<number>(0)
  const [selectedOffice, setSelectedOffice] = useState<Office | null>(null)
  const [loading, setLoading] = useState(false)
  const [radius, setRadius] = useState(() => {
    const r = searchParams.get('radius')
    if (r) {
      const parsed = parseInt(r, 10)
      if (!isNaN(parsed)) return parsed
    }
    return 3000
  })
  const [totalCount, setTotalCount] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [mapReady, setMapReady] = useState(false)
  const [searchCenter, setSearchCenter] = useState<{ lat: number; lng: number } | null>(() => {
    const lat = searchParams.get('lat')
    const lng = searchParams.get('lng')
    if (lat && lng) {
      const parsedLat = parseFloat(lat)
      const parsedLng = parseFloat(lng)
      if (!isNaN(parsedLat) && !isNaN(parsedLng)) {
        return { lat: parsedLat, lng: parsedLng }
      }
    }
    return null
  })
  const [sigungu, setSigungu] = useState('')
  const [searchQuery, setSearchQuery] = useState('')
  const [category, setCategory] = useState('')
  const [specialty, setSpecialty] = useState('')

  const [initialZoom, setInitialZoom] = useState<number | undefined>(() => {
    const zoom = searchParams.get('zoom')
    if (zoom) {
      const parsed = parseInt(zoom, 10)
      if (!isNaN(parsed)) return parsed
    }
    return undefined
  })
  const [zoomLevel, setZoomLevel] = useState(5)
  const [mapBounds, setMapBounds] = useState<{ min_lat: number; max_lat: number; min_lng: number; max_lng: number } | null>(null)
  const [clusters, setClusters] = useState<ClusterData[]>([])
  const useClusterMode = zoomLevel >= CLUSTER_ZOOM_THRESHOLD

  const { isChatOpen, chatMode } = useUI()
  const initialSearchDone = useRef(false)
  const urlSearchInProgress = useRef(false)
  const lastSearchParamsKey = useRef('')
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const {
    getCurrentPosition,
    getEffectiveLocation,
    hasLocation,
    loading: geoLoading,
    error: geoError,
  } = useGeolocation()

  useEffect(() => {
    getCurrentPosition()
  }, [getCurrentPosition])

  useEffect(() => {
    if (!mapReady) return

    const lat = searchParams.get('lat')
    const lng = searchParams.get('lng')
    const categoryParam = searchParams.get('category')
    const specialtyParam = searchParams.get('specialty')
    const sigunguParam = searchParams.get('sigungu')
    const radiusParam = searchParams.get('radius')
    const searchAllParam = searchParams.get('searchAll')
    const zoomParam = searchParams.get('zoom')

    if (!lat && !lng && !categoryParam && !specialtyParam && !sigunguParam && !searchAllParam) {
      return
    }

    const paramsKey = `${lat}-${lng}-${categoryParam}-${specialtyParam}-${sigunguParam}-${radiusParam}-${searchAllParam}-${zoomParam}`

    if (lastSearchParamsKey.current === paramsKey) {
      return
    }
    lastSearchParamsKey.current = paramsKey
    initialSearchDone.current = true
    urlSearchInProgress.current = true

    const searchRadius = radiusParam ? parseInt(radiusParam, 10) : 3000
    const searchLat = lat ? parseFloat(lat) : null
    const searchLng = lng ? parseFloat(lng) : null

    if (radiusParam) setRadius(searchRadius)
    if (searchLat && searchLng) setSearchCenter({ lat: searchLat, lng: searchLng })
    if (specialtyParam) setSpecialty(specialtyParam)
    else if (categoryParam) setCategory(categoryParam)
    if (sigunguParam) setSigungu(sigunguParam)
    if (zoomParam) {
      const parsedZoom = parseInt(zoomParam, 10)
      if (!isNaN(parsedZoom)) setInitialZoom(parsedZoom)
    }

    setLoading(true)
    setError(null)

    const finishSearch = () => {
      setLoading(false)
      urlSearchInProgress.current = false
    }

    if (searchAllParam === 'true' && categoryParam) {
      lawyerFinderService.searchLawyers({
        category: categoryParam,
      }).then((response) => {
        setLawyers(response.lawyers)
        setTotalCount(response.total_count)
      }).catch(() => {
        setError('검색에 실패했습니다')
      }).finally(finishSearch)
    } else if (searchLat && searchLng) {
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

  const getSearchLocation = useCallback(() => {
    return searchCenter || getEffectiveLocation()
  }, [searchCenter, getEffectiveLocation])

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

  useEffect(() => {
    if (urlSearchInProgress.current) return
    if (initialSearchDone.current) {
      initialSearchDone.current = false
      return
    }
    if (mapReady && !searchQuery) {
      fetchNearbyLawyers()
    }
  }, [fetchNearbyLawyers, mapReady, searchQuery, category, specialty])

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

  const handleSearchReset = useCallback(() => {
    setSearchQuery('')
    setSpecialty('')
  }, [])

  const handleLawyerSelect = useCallback((lawyer: Lawyer) => {
    setSelectedLawyer(lawyer)
    setSelectionTrigger(Date.now())
  }, [])

  const handleRadiusChange = useCallback((newRadius: number) => {
    setRadius(newRadius)
  }, [])

  const handleMapReady = useCallback(() => {
    setMapReady(true)
  }, [])

  const handleMyLocation = useCallback(() => {
    setSearchCenter(null)
    getCurrentPosition()
  }, [getCurrentPosition])

  const handleCenterChange = useCallback((newCenter: { lat: number; lng: number }) => {
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => {
      setSearchCenter(newCenter)
    }, DRAG_DEBOUNCE_MS)
  }, [])

  const handleZoomChange = useCallback((zoom: number) => {
    setZoomLevel(zoom)
  }, [])

  const handleBoundsChange = useCallback((bounds: { min_lat: number; max_lat: number; min_lng: number; max_lng: number }) => {
    setMapBounds(bounds)
  }, [])

  const handleSigunguChange = useCallback((newSigungu: string) => {
    setSigungu(newSigungu)
    if (newSigungu && DISTRICT_COORDS[newSigungu]) {
      setSearchCenter(DISTRICT_COORDS[newSigungu])
    }
  }, [])

  const handleCategoryChange = useCallback((newCategory: string) => {
    setCategory(newCategory)
  }, [])

  const handleOfficeClick = useCallback((office: Office) => {
    setSelectedOffice(office)
  }, [])

  const handleOfficeClose = useCallback(() => {
    setSelectedOffice(null)
  }, [])

  const handleLawyerFromOffice = useCallback((_lawyer: Lawyer) => {
    // TODO: 상세페이지 이동
  }, [])

  const center = getSearchLocation()
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
        className={`flex flex-col h-full bg-white transition-all duration-500 ease-in-out ${!isInline && isChatOpen && chatMode === 'split' ? 'w-1/2 border-r border-gray-200' : 'w-full'
          }`}
      >
        {!isInline && (
          <header className="bg-white border-b px-6 py-4 flex items-center justify-between">
            <div className="flex items-center">
              <BackButton />
              <div>
                <h1 className="text-xl font-bold text-gray-900">가까운 변호사를 빠르게 찾아보세요</h1>
                <p className="text-sm text-gray-500 mt-0.5">위치와 조건으로 쉽게 검색할 수 있습니다</p>
              </div>
            </div>
            {geoLoading && <span className="text-sm text-blue-600">위치 확인 중...</span>}
            {geoError && <span className="text-sm text-orange-600">{geoError}</span>}
          </header>
        )}

        <div className="flex-1 flex overflow-hidden">
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
                className="absolute top-4 left-1/2 -translate-x-1/2 z-10 bg-white px-4 py-2 rounded-full shadow-lg border border-gray-200 text-sm font-medium text-gray-700 hover:bg-gray-50 transition flex items-center gap-2"
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
