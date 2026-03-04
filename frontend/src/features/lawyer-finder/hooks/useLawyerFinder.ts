import { useState, useEffect, useCallback, useRef } from 'react'
import { useSearchParams } from 'next/navigation'
import { useGeolocation } from './useGeolocation'
import { lawyerFinderService } from '../services'
import { CLUSTER_ZOOM_THRESHOLD, DRAG_DEBOUNCE_MS, DISTRICT_COORDS, PROVINCE_CENTERS } from '../constants'
import type { Lawyer, Office, ClusterData, ProvinceData } from '../types'

export function useLawyerFinder() {
  const searchParams = useSearchParams()

  // 변호사 목록 + 검색 상태
  const [lawyers, setLawyers] = useState<Lawyer[]>([])
  const [loading, setLoading] = useState(false)
  const [totalCount, setTotalCount] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [category, setCategory] = useState('')
  const [specialty, setSpecialty] = useState('')

  // 선택 상태
  const [selectedLawyer, setSelectedLawyer] = useState<Lawyer | null>(null)
  const [selectionTrigger, setSelectionTrigger] = useState<number>(0)
  const [selectedOffice, setSelectedOffice] = useState<Office | null>(null)

  // 지도 상태
  const [mapReady, setMapReady] = useState(false)
  const [radius, setRadius] = useState(() => {
    const r = searchParams.get('radius')
    if (r) {
      const parsed = parseInt(r, 10)
      if (!isNaN(parsed)) return parsed
    }
    return 3000
  })
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
  const [province, setProvince] = useState(() => {
    return searchParams.get('province') || '서울'
  })
  const [regions, setRegions] = useState<ProvinceData[]>([])
  const [sigungu, setSigungu] = useState('')

  // 클러스터 모드 상태
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

  // Refs
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

  // 페이지 로드 시 위치 권한 요청
  useEffect(() => {
    getCurrentPosition()
  }, [getCurrentPosition])

  // 전국 지역 데이터 로드
  useEffect(() => {
    lawyerFinderService.getRegions()
      .then((data) => setRegions(data.provinces))
      .catch(() => {})
  }, [])

  // URL 쿼리 파라미터로 검색 (챗봇에서 이동 시)
  useEffect(() => {
    if (!mapReady) return

    const lat = searchParams.get('lat')
    const lng = searchParams.get('lng')
    const categoryParam = searchParams.get('category')
    const specialtyParam = searchParams.get('specialty')
    const provinceParam = searchParams.get('province')
    const sigunguParam = searchParams.get('sigungu')
    const radiusParam = searchParams.get('radius')
    const searchAllParam = searchParams.get('searchAll')
    const zoomParam = searchParams.get('zoom')

    if (!lat && !lng && !categoryParam && !specialtyParam && !sigunguParam && !searchAllParam && !provinceParam) {
      return
    }

    const paramsKey = `${lat}-${lng}-${categoryParam}-${specialtyParam}-${provinceParam}-${sigunguParam}-${radiusParam}-${searchAllParam}-${zoomParam}`

    if (lastSearchParamsKey.current === paramsKey) return
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
    if (provinceParam) setProvince(provinceParam)
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
      lawyerFinderService.searchLawyers({ category: categoryParam })
        .then((response) => {
          setLawyers(response.lawyers)
          setTotalCount(response.total_count)
        })
        .catch(() => setError('검색에 실패했습니다'))
        .finally(finishSearch)
    } else if (searchLat && searchLng) {
      lawyerFinderService.getNearbyLawyers(
        searchLat, searchLng, searchRadius,
        categoryParam || undefined, specialtyParam || undefined
      )
        .then((response) => {
          setLawyers(response.lawyers)
          setTotalCount(response.total_count)
        })
        .catch(() => setError('변호사 정보를 불러오는데 실패했습니다'))
        .finally(finishSearch)
    } else {
      finishSearch()
    }
  }, [searchParams, mapReady])

  // 검색에 사용할 위치
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
        location.lat, location.lng, radius,
        category || undefined, specialty || undefined
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

  // 위치, 반경, 전문분야 변경 시 검색
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

  // 줌아웃 시 클러스터 API 호출
  useEffect(() => {
    if (!useClusterMode || !mapBounds) {
      setClusters([])
      return
    }

    lawyerFinderService.getClusters(
      mapBounds, zoomLevel,
      category || undefined, specialty || undefined
    )
      .then((res) => setClusters(res.clusters))
      .catch(() => setClusters([]))
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

  // 이 지역에서 재검색
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

  // 검색 초기화
  const handleSearchReset = useCallback(() => {
    setSearchQuery('')
    setSpecialty('')
  }, [])

  // 변호사 선택
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
    setSearchCenter(null)
    getCurrentPosition()
  }, [getCurrentPosition])

  // 지도 드래그 완료 시 중심 변경 (디바운스)
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

  // 시/도 변경
  const handleProvinceChange = useCallback((newProvince: string) => {
    setProvince(newProvince)
    setSigungu('')

    const regionData = regions.find((r) => r.name === newProvince)
    if (regionData) {
      setSearchCenter({ lat: regionData.center_lat, lng: regionData.center_lng })
    } else if (PROVINCE_CENTERS[newProvince]) {
      setSearchCenter({ lat: PROVINCE_CENTERS[newProvince].lat, lng: PROVINCE_CENTERS[newProvince].lng })
    }

    const defaultRadius = PROVINCE_CENTERS[newProvince]?.defaultRadius ?? 15000
    setRadius(defaultRadius)
  }, [regions])

  // 시/군/구 변경
  const handleSigunguChange = useCallback((newSigungu: string) => {
    setSigungu(newSigungu)
    if (!newSigungu) return

    const regionData = regions.find((r) => r.name === province)
    const districtData = regionData?.districts.find((d) => d.name === newSigungu)
    if (districtData) {
      setSearchCenter({ lat: districtData.center_lat, lng: districtData.center_lng })
      setRadius(5000)
    } else if (DISTRICT_COORDS[newSigungu]) {
      setSearchCenter(DISTRICT_COORDS[newSigungu])
    }
  }, [regions, province])

  // 전문분야 카테고리 변경
  const handleCategoryChange = useCallback((newCategory: string) => {
    setCategory(newCategory)
  }, [])

  // 사무소 클릭/닫기
  const handleOfficeClick = useCallback((office: Office) => {
    setSelectedOffice(office)
  }, [])

  const handleOfficeClose = useCallback(() => {
    setSelectedOffice(null)
  }, [])

  const handleLawyerFromOffice = useCallback((_lawyer: Lawyer) => {
    // TODO: 변호사 상세페이지로 이동
  }, [])

  // 파생 상태
  const center = getSearchLocation()
  const userLocation = hasLocation ? getEffectiveLocation() : null

  return {
    // 검색 결과
    lawyers,
    loading,
    totalCount,
    error,
    searchQuery,
    category,
    specialty,

    // 선택 상태
    selectedLawyer,
    selectionTrigger,
    selectedOffice,

    // 지도 상태
    radius,
    center,
    userLocation,
    province,
    regions,
    sigungu,
    initialZoom,
    useClusterMode,
    clusters,

    // 위치
    geoLoading,
    geoError,

    // 핸들러
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
  }
}
