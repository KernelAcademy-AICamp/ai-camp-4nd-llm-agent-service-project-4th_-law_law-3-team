import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { keepPreviousData, useQuery } from '@tanstack/react-query'
import { useChat } from '@/context/ChatContext'
import type { TabType } from '@/features/lawyer-stats/components/StickyTabNav'
import {
  fetchDemandStats,
  fetchDensityStats,
  fetchOverview,
  fetchRegionStats,
} from '@/features/lawyer-stats/services'
import type {
  CourtDemandMarker,
  DemandCategory,
  DemandStat,
  IndicatorGroup,
  PredictionYear,
  StatsFilter,
  ViewMode,
} from '@/features/lawyer-stats/types'
import courtCoordsData from '../../../../public/data/court_coordinates.json'

const courtCoords = courtCoordsData as unknown as Record<string, [number, number]>

export function useStatsFilter() {
  const { sessionData } = useChat()

  // UI 상태
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
  const regionSectionRef = useRef<HTMLDivElement>(null)
  const crossSectionRef = useRef<HTMLDivElement>(null)

  // 채팅 에이전트 sessionData.stats_filter → 대시보드 필터 자동 적용
  const appliedFilterRef = useRef<string>('')
  useEffect(() => {
    const filter = sessionData.stats_filter as StatsFilter | undefined
    if (!filter) return
    const filterKey = JSON.stringify(filter)
    if (filterKey === appliedFilterRef.current) return
    appliedFilterRef.current = filterKey

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
    return Array.from({ length: 10 }, (_, i) => 2015 + i)
  }, [demandQuery.data?.available_years])

  // 법원 마커 클릭
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

  // 탭 스크롤
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

  // IntersectionObserver로 스크롤 위치에 따라 탭 자동 전환
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

    const sections = [regionSectionRef.current, crossSectionRef.current]
    sections.forEach((section) => {
      if (section) observer.observe(section)
    })

    return () => {
      sections.forEach((section) => {
        if (section) observer.unobserve(section)
      })
    }
  }, [])

  // 지역 선택 핸들러
  const handleProvinceSelect = useCallback((province: string) => {
    setSelectedProvince(province === '전체' ? null : province)
    setHighlightedRegion(null)
    setMapSelectedRegion(null)
    setSelectedCourt(null)
  }, [])

  const handleMapRegionClick = useCallback((region: string | null) => {
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
  }, [mapSelectedRegion])

  const handleListRegionClick = useCallback((region: string | null) => {
    if (region) {
      const province = region.split(' ')[0]
      setSelectedProvince(province)
      setHighlightedRegion(region)
      setMapSelectedRegion(null)
    } else {
      setHighlightedRegion(null)
      setMapSelectedRegion(null)
    }
  }, [])

  return {
    // UI 상태
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

    // 데이터
    isLoading,
    hasError,
    filteredRegionData,
    courtMarkers,

    // Refs
    regionSectionRef,
    crossSectionRef,

    // Setters
    setViewMode,
    setPredictionYear,
    setDemandCategory,
    setDemandYear,

    // 핸들러
    handleIndicatorGroupChange,
    handleCourtClick,
    handleTabChange,
    handleProvinceSelect,
    handleMapRegionClick,
    handleListRegionClick,
  }
}
