import { useState, useEffect, useCallback, useRef, useMemo } from 'react'
import { useGeolocation } from './useGeolocation'
import { useLawyerSearchParams } from './useLawyerSearchParams'
import { lawyerFinderService } from '../services'
import { CLUSTER_ZOOM_THRESHOLD, DRAG_DEBOUNCE_MS, DISTRICT_COORDS, PROVINCE_CENTERS } from '../constants'
import type { Lawyer, Office, ClusterData, ProvinceData } from '../types'

/**
 * 변호사 찾기 통합 훅
 *
 * 리팩토링 요약:
 * - useLawyerSearchParams로 URL 단일 파싱 (이중 읽기 제거)
 * - 3개 중복 방지 ref 제거 → 0개 ref (debounce ref만 유지)
 * - 2개 경쟁 이펙트 → 1개 통합 검색 이펙트
 * - 필터 변경 시 URL 동기화 (공유/북마크 지원)
 */
export function useLawyerFinder() {
  const { state: urlState, updateUrl } = useLawyerSearchParams()

  // ── 검색 결과 ──
  const [lawyers, setLawyers] = useState<Lawyer[]>([])
  const [loading, setLoading] = useState(false)
  const [totalCount, setTotalCount] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState('')

  // ── 필터 상태 (URL 초기값 → 로컬 관리 + URL 동기화) ──
  const [category, setCategory] = useState(urlState.category)
  const [specialty, setSpecialty] = useState(urlState.specialty)
  const [radius, setRadius] = useState(urlState.radius)
  const [province, setProvince] = useState(urlState.province)
  const [sigungu, setSigungu] = useState(urlState.sigungu)
  const [isSearchAll, setIsSearchAll] = useState(urlState.searchAll)

  // ── 선택 상태 ──
  const [selectedLawyer, setSelectedLawyer] = useState<Lawyer | null>(null)
  const [selectionTrigger, setSelectionTrigger] = useState(0)
  const [selectedOffice, setSelectedOffice] = useState<Office | null>(null)

  // ── 지도 상태 ──
  const [mapReady, setMapReady] = useState(false)
  const [searchCenter, setSearchCenter] = useState<{ lat: number; lng: number } | null>(() => {
    if (urlState.lat !== null && urlState.lng !== null) {
      return { lat: urlState.lat, lng: urlState.lng }
    }
    return null
  })
  const [regions, setRegions] = useState<ProvinceData[]>([])
  const [zoomLevel, setZoomLevel] = useState(5)
  const [mapBounds, setMapBounds] = useState<{
    min_lat: number; max_lat: number; min_lng: number; max_lng: number
  } | null>(null)
  const [clusters, setClusters] = useState<ClusterData[]>([])
  const useClusterMode = zoomLevel >= CLUSTER_ZOOM_THRESHOLD

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

  // 검색에 사용할 위치
  const getSearchLocation = useCallback(() => {
    return searchCenter || getEffectiveLocation()
  }, [searchCenter, getEffectiveLocation])

  // ── 통합 검색 이펙트 ──
  // 기존: URL 검색 이펙트 + 자동 검색 이펙트 + 3개 ref로 조율
  // 리팩토링: 단일 이펙트, primitive deps로 자동 중복 제거
  const location = useMemo(() => getSearchLocation(), [getSearchLocation])

  useEffect(() => {
    if (!mapReady || searchQuery) return

    let cancelled = false
    setLoading(true)
    setError(null)

    const execute = async () => {
      try {
        let response: { lawyers: Lawyer[]; total_count: number }

        if (isSearchAll && category) {
          response = await lawyerFinderService.searchLawyers({ category })
        } else {
          response = await lawyerFinderService.getNearbyLawyers(
            location.lat, location.lng, radius,
            category || undefined, specialty || undefined,
          )
        }

        if (!cancelled) {
          setLawyers(response.lawyers)
          setTotalCount(response.total_count)
        }
      } catch {
        if (!cancelled) {
          setError('변호사 정보를 불러오는데 실패했습니다')
          setLawyers([])
          setTotalCount(0)
        }
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    execute()
    return () => { cancelled = true }
  }, [mapReady, searchQuery, isSearchAll, category, specialty, radius, location.lat, location.lng])

  // 줌아웃 시 클러스터 API 호출
  useEffect(() => {
    if (!useClusterMode || !mapBounds) {
      setClusters([])
      return
    }

    lawyerFinderService.getClusters(
      mapBounds, zoomLevel,
      category || undefined, specialty || undefined,
    )
      .then((res) => setClusters(res.clusters))
      .catch(() => setClusters([]))
  }, [useClusterMode, mapBounds, zoomLevel, category, specialty])

  // ── 핸들러: 텍스트 검색 ──
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

    const loc = getSearchLocation()
    setLoading(true)
    setError(null)

    try {
      const response = await lawyerFinderService.searchLawyers({
        name: searchQuery,
        office: searchQuery,
        category: category || undefined,
        specialty: specialty || undefined,
        latitude: loc.lat,
        longitude: loc.lng,
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

  // ── 핸들러: 선택 ──
  const handleLawyerSelect = useCallback((lawyer: Lawyer) => {
    setSelectedLawyer(lawyer)
    setSelectionTrigger(Date.now())
  }, [])

  // ── 핸들러: 필터 변경 (로컬 상태 + URL 동기화) ──
  const handleRadiusChange = useCallback((newRadius: number) => {
    setRadius(newRadius)
    setIsSearchAll(false)
    updateUrl({ radius: newRadius, searchAll: null, lat: null, lng: null, zoom: null })
  }, [updateUrl])

  const handleProvinceChange = useCallback((newProvince: string) => {
    setProvince(newProvince)
    setSigungu('')
    setIsSearchAll(false)

    const regionData = regions.find((r) => r.name === newProvince)
    if (regionData) {
      setSearchCenter({ lat: regionData.center_lat, lng: regionData.center_lng })
    } else if (PROVINCE_CENTERS[newProvince]) {
      setSearchCenter({ lat: PROVINCE_CENTERS[newProvince].lat, lng: PROVINCE_CENTERS[newProvince].lng })
    }

    const defaultRadius = PROVINCE_CENTERS[newProvince]?.defaultRadius ?? 15000
    setRadius(defaultRadius)

    updateUrl({
      province: newProvince,
      sigungu: null,
      radius: defaultRadius,
      searchAll: null,
      lat: null,
      lng: null,
      zoom: null,
    })
  }, [regions, updateUrl])

  const handleSigunguChange = useCallback((newSigungu: string) => {
    setSigungu(newSigungu)
    setIsSearchAll(false)
    if (!newSigungu) return

    const regionData = regions.find((r) => r.name === province)
    const districtData = regionData?.districts.find((d) => d.name === newSigungu)
    if (districtData) {
      setSearchCenter({ lat: districtData.center_lat, lng: districtData.center_lng })
      setRadius(5000)
      updateUrl({ sigungu: newSigungu, radius: 5000, searchAll: null, lat: null, lng: null, zoom: null })
    } else if (DISTRICT_COORDS[newSigungu]) {
      setSearchCenter(DISTRICT_COORDS[newSigungu])
      updateUrl({ sigungu: newSigungu, searchAll: null, lat: null, lng: null, zoom: null })
    }
  }, [regions, province, updateUrl])

  const handleCategoryChange = useCallback((newCategory: string) => {
    setCategory(newCategory)
    setIsSearchAll(false)
    updateUrl({ category: newCategory || null, searchAll: null })
  }, [updateUrl])

  // ── 핸들러: 지도 ──
  const handleMapReady = useCallback(() => {
    setMapReady(true)
  }, [])

  const handleMyLocation = useCallback(() => {
    setSearchCenter(null)
    setIsSearchAll(false)
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

  const handleBoundsChange = useCallback((bounds: {
    min_lat: number; max_lat: number; min_lng: number; max_lng: number
  }) => {
    setMapBounds(bounds)
  }, [])

  // ── 핸들러: 사무소 ──
  const handleOfficeClick = useCallback((office: Office) => {
    setSelectedOffice(office)
  }, [])

  const handleOfficeClose = useCallback(() => {
    setSelectedOffice(null)
  }, [])

  const handleLawyerFromOffice = useCallback((_lawyer: Lawyer) => {
    // TODO: 변호사 상세페이지로 이동
  }, [])

  // ── 파생 상태 ──
  const center = location
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
    initialZoom: urlState.zoom,
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
