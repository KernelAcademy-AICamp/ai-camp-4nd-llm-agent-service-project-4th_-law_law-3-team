"use client"

import React, { useEffect, useMemo, useState } from "react"
import { ComposableMap, Geographies, Geography, Marker, ZoomableGroup } from "react-simple-maps"
import type { PredictionYear, ViewMode } from "@/app/lawyer-stats/page"
import type { CourtDemandMarker, DemandStat, DensityStat, RegionStat } from "../types"

// react-simple-maps 지리 객체 타입
interface GeoProperties {
  code: string
  name: string
}

interface GeoFeature {
  rsmKey: string
  properties: GeoProperties
}

// GeoJSON path (Nationwide)
const GEO_URL = "/data/korea_geo.json"

// Region Code Prefix Mapping (GeoJSON 2-digit code -> Province Name)
const PROVINCE_PREFIX_MAP: Record<string, string> = {
  "11": "서울",
  "21": "부산",
  "22": "대구",
  "23": "인천",
  "24": "광주",
  "25": "대전",
  "26": "울산",
  "29": "세종",
  "31": "경기",
  "32": "강원",
  "33": "충북",
  "34": "충남",
  "35": "전북",
  "36": "전남",
  "37": "경북",
  "38": "경남",
  "39": "제주",
}

interface Props {
  data: (RegionStat | DensityStat | DemandStat)[]
  viewMode: ViewMode
  predictionYear?: PredictionYear
  selectedProvince?: string | null
  highlightedRegion?: string | null
  onRegionClick?: (region: string | null) => void
  courtMarkers?: CourtDemandMarker[]
  selectedCourt?: string | null
  onCourtClick?: (courtName: string | null) => void
}

// 시/도 이름 -> GeoJSON 코드 prefix 매핑
const PROVINCE_TO_CODE: Record<string, string> = {
  서울: "11",
  부산: "21",
  대구: "22",
  인천: "23",
  광주: "24",
  대전: "25",
  울산: "26",
  세종: "29",
  경기: "31",
  강원: "32",
  충북: "33",
  충남: "34",
  전북: "35",
  전남: "36",
  경북: "37",
  경남: "38",
  제주: "39",
}

// 기본 뷰 설정 (전체)
const DEFAULT_VIEW_CONFIG = { center: [127.5, 36] as [number, number], zoom: 1 }

// 시/도별 중심 좌표 및 줌 레벨
const PROVINCE_VIEW_CONFIG: Record<string, { center: [number, number]; zoom: number }> = {
  서울: { center: [127.0, 37.56], zoom: 16 },
  경기: { center: [127.15, 37.6], zoom: 3.8 },
  인천: { center: [126.2, 37.45], zoom: 6 },
  부산: { center: [129.05, 35.18], zoom: 12 },
  대구: { center: [128.55, 35.82], zoom: 12 },
  광주: { center: [126.85, 35.15], zoom: 15 },
  대전: { center: [127.4, 36.35], zoom: 15 },
  울산: { center: [129.3, 35.52], zoom: 12 },
  세종: { center: [127.25, 36.6], zoom: 11 },
  강원: { center: [128.3, 37.9], zoom: 3 },
  충북: { center: [127.8, 36.7], zoom: 4 },
  충남: { center: [126.8, 36.5], zoom: 4 },
  전북: { center: [127.1, 35.7], zoom: 4 },
  전남: { center: [126.7, 34.7], zoom: 3 },
  경북: { center: [129.3, 36.55], zoom: 2.6 },
  경남: { center: [128.3, 35.2], zoom: 3.5 },
  제주: { center: [126.55, 33.4], zoom: 8 },
}

// 마커 색상 (사건 수 - 앰버)
const CASE_MARKER_COLORS = [
  { min: 10000, color: "#0F2A44" },
  { min: 5000, color: "#2A4F78" },
  { min: 2000, color: "#4E7AAD" },
  { min: 1000, color: "#7FA3C7" },
  { min: 0, color: "#C5D8EA" },
]

// 마커 색상 (부담지수 - 로즈)
const BURDEN_MARKER_COLORS = [
  { min: 100, color: "#9F1239" },
  { min: 50, color: "#E11D48" },
  { min: 20, color: "#FB7185" },
  { min: 10, color: "#FDA4AF" },
  { min: 0, color: "#FECDD3" },
]

export function RegionGeoMap({ data, viewMode, predictionYear, selectedProvince, highlightedRegion, onRegionClick, courtMarkers, selectedCourt, onCourtClick }: Props) {
  const isDemandMode = viewMode === 'case_count' || viewMode === 'burden_index'

  // 선택된 법원의 관할 지역 Set (하이라이트용)
  const selectedCourtRegions = useMemo(() => {
    if (!selectedCourt || !courtMarkers) return new Set<string>()
    const marker = courtMarkers.find(m => m.court_name === selectedCourt)
    return new Set(marker?.regions ?? [])
  }, [selectedCourt, courtMarkers])

  // 선택된 시/도의 코드 prefix
  const selectedCodePrefix = selectedProvince ? PROVINCE_TO_CODE[selectedProvince] : null

  // 뷰 설정 (선택된 시/도가 있으면 해당 지역으로, 없으면 전국)
  const viewConfig = selectedProvince && PROVINCE_VIEW_CONFIG[selectedProvince]
    ? PROVINCE_VIEW_CONFIG[selectedProvince]
    : DEFAULT_VIEW_CONFIG

  // 줌 및 중심 위치 상태 관리
  const [zoom, setZoom] = useState(viewConfig.zoom)
  const [center, setCenter] = useState<[number, number]>(viewConfig.center)
  const [tooltipContent, setTooltipContent] = useState<{
    region: string
    density?: number
    count: number
    changePercent?: number
    caseCount?: number
    lawyerCount?: number
    burdenIndex?: number
    courtName?: string
    isCourtMarker?: boolean
  } | null>(null)
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 })

  // 선택된 시/도가 변경되면 줌 레벨과 중심 위치 리셋
  useEffect(() => {
    const config = selectedProvince && PROVINCE_VIEW_CONFIG[selectedProvince]
      ? PROVINCE_VIEW_CONFIG[selectedProvince]
      : DEFAULT_VIEW_CONFIG
    setZoom(config.zoom)
    setCenter(config.center)
  }, [selectedProvince])



  // 1. Create a map of "Full Region Name" -> region data
  const regionDataMap = useMemo(() => {
    const map = new Map<string, {
      count: number; density?: number; changePercent?: number
      caseCount?: number; lawyerCount?: number; burdenIndex?: number; courtName?: string
    }>()
    data.forEach((d) => {
      map.set(d.region, {
        count: 'count' in d ? (d as RegionStat).count : 0,
        density: 'density' in d ? (d as DensityStat).density : undefined,
        changePercent: 'change_percent' in d ? (d as DensityStat).change_percent : undefined,
        caseCount: 'case_count' in d ? (d as DemandStat).case_count : undefined,
        lawyerCount: 'lawyer_count' in d ? (d as DemandStat).lawyer_count : undefined,
        burdenIndex: 'burden_index' in d ? (d as DemandStat).burden_index : undefined,
        courtName: 'court_name' in d ? (d as DemandStat).court_name : undefined,
      })
    })
    return map
  }, [data])

  // 2. Create a map for color scale values
  const dataMap = useMemo(() => {
    const map = new Map<string, number>()
    const useDensity = viewMode === 'density' || viewMode === 'prediction'
    data.forEach((d) => {
      if (viewMode === 'burden_index' && 'burden_index' in d) {
        map.set(d.region, (d as DemandStat).burden_index)
      } else if (viewMode === 'case_count' && 'case_count' in d) {
        map.set(d.region, (d as DemandStat).case_count)
      } else if (useDensity && 'density' in d) {
        map.set(d.region, (d as DensityStat).density)
      } else if ('count' in d) {
        map.set(d.region, (d as RegionStat).count)
      }
    })
    return map
  }, [data, viewMode])

  // 2. Color Scale - count mode (빨강 그라데이션)
  const COUNT_COLOR_RANGES = [
    { min: 500, max: Infinity, color: "#7F1D1D" },  // 500명 이상 - 가장 진한 빨강
    { min: 100, max: 500, color: "#DC2626" },       // 100~500명
    { min: 30, max: 100, color: "#EF4444" },        // 30~100명
    { min: 10, max: 30, color: "#F87171" },         // 10~30명
    { min: 1, max: 10, color: "#FCA5A5" },          // 1~10명 - 가장 연한 빨강
  ]

  // Color Scale - density mode (에메랄드 그라데이션 - 6단계)
  const DENSITY_COLOR_RANGES = [
    { min: 100, max: Infinity, color: "#022C22" },  // 100명 이상 - emerald-950
    { min: 10, max: 100, color: "#047857" },        // 10~100명 - emerald-700
    { min: 5, max: 10, color: "#059669" },          // 5~10명 - emerald-600
    { min: 2, max: 5, color: "#10B981" },           // 2~5명 - emerald-500
    { min: 1, max: 2, color: "#6EE7B7" },           // 1~2명 - emerald-300
    { min: 0, max: 1, color: "#A7F3D0" },           // 1명 미만 - emerald-200
  ]

  // Color Scale - prediction mode (보라색 그라데이션 - 6단계)
  const PREDICTION_COLOR_RANGES = [
    { min: 100, max: Infinity, color: "#3B0764" },  // 100명 이상 - purple-950
    { min: 10, max: 100, color: "#7C3AED" },        // 10~100명 - violet-600
    { min: 5, max: 10, color: "#8B5CF6" },          // 5~10명 - violet-500
    { min: 2, max: 5, color: "#A78BFA" },           // 2~5명 - violet-400
    { min: 1, max: 2, color: "#C4B5FD" },           // 1~2명 - violet-300
    { min: 0, max: 1, color: "#EDE9FE" },           // 1명 미만 - violet-100
  ]

  // Color Scale - case_count mode (앰버 그라데이션 - 6단계)
  const CASE_COUNT_COLOR_RANGES = [
    { min: 50000, max: Infinity, color: "#78350F" },  // 5만건 이상 - amber-900
    { min: 20000, max: 50000, color: "#92400E" },     // 2~5만건 - amber-800
    { min: 10000, max: 20000, color: "#B45309" },     // 1~2만건 - amber-700
    { min: 5000, max: 10000, color: "#D97706" },      // 5천~1만건 - amber-600
    { min: 1000, max: 5000, color: "#F59E0B" },       // 1~5천건 - amber-500
    { min: 0, max: 1000, color: "#FCD34D" },          // 1천건 미만 - amber-300
  ]

  // Color Scale - burden_index mode (로즈 그라데이션 - 6단계)
  const BURDEN_COLOR_RANGES = [
    { min: 100, max: Infinity, color: "#881337" },  // 100 이상 - rose-900
    { min: 50, max: 100, color: "#BE123C" },        // 50~100 - rose-700
    { min: 20, max: 50, color: "#E11D48" },         // 20~50 - rose-600
    { min: 10, max: 20, color: "#FB7185" },         // 10~20 - rose-400
    { min: 5, max: 10, color: "#FDA4AF" },          // 5~10 - rose-300
    { min: 0, max: 5, color: "#FFE4E6" },           // 5 미만 - rose-100
  ]

  const colorScale = (value: number) => {
    if (value === 0) return "#ffffff"
    const ranges = viewMode === 'case_count'
      ? CASE_COUNT_COLOR_RANGES
      : viewMode === 'burden_index'
        ? BURDEN_COLOR_RANGES
        : viewMode === 'prediction'
          ? PREDICTION_COLOR_RANGES
          : viewMode === 'density'
            ? DENSITY_COLOR_RANGES
            : COUNT_COLOR_RANGES
    for (const range of ranges) {
      if (value >= range.min && value < range.max) {
        return range.color
      }
    }
    return ranges[0].color
  }

  // 3. Helper to resolve full name from GeoJSON properties
  const getFullName = (geo: GeoFeature) => {
    const code = geo.properties.code // e.g. "11250"
    const name = geo.properties.name // e.g. "강동구" or "수원시장안구"

    const prefix = code.substring(0, 2)
    const province = PROVINCE_PREFIX_MAP[prefix]

    if (!province) return name // Fallback

    // 세종은 특별 처리
    if (prefix === "29") return "세종 세종시"

    // 광역시/특별시는 구 단위 그대로 사용 (서울, 부산, 대구, 인천, 광주, 대전, 울산)
    const metropolitanPrefixes = ["11", "21", "22", "23", "24", "25", "26"]
    if (metropolitanPrefixes.includes(prefix)) {
      return `${province} ${name}`
    }

    // 그 외 도 지역은 시/군 단위까지만 추출 (예: "수원시장안구" → "수원시")
    const match = name.match(/^(.+?시|.+?군)/)
    const district = match ? match[1] : name

    return `${province} ${district}`
  }

  const handleMouseEnter = (geo: GeoFeature, event: React.MouseEvent) => {
    const fullName = getFullName(geo)
    const regionData = regionDataMap.get(fullName)
    setTooltipContent({
      region: fullName,
      density: regionData?.density,
      count: regionData?.count ?? 0,
      changePercent: regionData?.changePercent,
      caseCount: regionData?.caseCount,
      lawyerCount: regionData?.lawyerCount,
      burdenIndex: regionData?.burdenIndex,
      courtName: regionData?.courtName,
    })
    const e = event.nativeEvent || event
    setTooltipPos({ x: e.clientX, y: e.clientY })
  }

  const handleMouseMove = (event: React.MouseEvent) => {
    const e = event.nativeEvent || event
    setTooltipPos({ x: e.clientX, y: e.clientY })
  }

  const handleMouseLeave = () => {
    setTooltipContent(null)
  }

  return (
    <div
      className="relative w-full h-[500px] bg-slate-50 rounded-xl border border-slate-200 overflow-hidden"
      onClick={() => onRegionClick?.(null)}
    >
      {/* Tooltip */}
      {tooltipContent && (
        <div
          className="fixed z-50 px-3 py-2 text-sm text-white bg-gray-900 rounded pointer-events-none"
          style={{ left: tooltipPos.x + 15, top: tooltipPos.y + 15 }}
        >
          <div className="font-medium">{tooltipContent.region}</div>
          {/* 수요 모드 지역 호버: 지역명 + 관할법원만 */}
          {isDemandMode && !tooltipContent.isCourtMarker && tooltipContent.courtName ? (
            <div className="text-gray-300">관할: {tooltipContent.courtName}</div>
          ) : /* 수요 모드 법원 마커 호버: 전체 정보 */
          viewMode === 'case_count' && tooltipContent.caseCount !== undefined && tooltipContent.isCourtMarker ? (
            <>
              <div>사건 수: {tooltipContent.caseCount.toLocaleString()}건</div>
              {tooltipContent.lawyerCount !== undefined && (
                <div className="text-gray-300">관할 변호사 수: {tooltipContent.lawyerCount.toLocaleString()}명</div>
              )}
            </>
          ) : viewMode === 'burden_index' && tooltipContent.burdenIndex !== undefined && tooltipContent.isCourtMarker ? (
            <>
              <div>부담지수: {tooltipContent.burdenIndex.toFixed(1)}</div>
              <div className="text-gray-300">
                사건 {tooltipContent.caseCount?.toLocaleString() ?? 0}건 / 관할 변호사 {tooltipContent.lawyerCount?.toLocaleString() ?? 0}명
              </div>
            </>
          ) : viewMode === 'prediction' && tooltipContent.density !== undefined ? (
            <div>
              {predictionYear} 예측: {tooltipContent.density.toFixed(1)}명 / 10만명
              {tooltipContent.changePercent !== undefined && (
                <span className={tooltipContent.changePercent >= 0 ? 'text-red-400' : 'text-blue-400'}>
                  {' '}({tooltipContent.changePercent >= 0 ? '+' : ''}{tooltipContent.changePercent}%)
                </span>
              )}
            </div>
          ) : viewMode === 'density' && tooltipContent.density !== undefined ? (
            <>
              <div>{tooltipContent.density.toFixed(1)}명 / 10만명</div>
              <div className="text-gray-300">변호사 {tooltipContent.count.toLocaleString()}명</div>
            </>
          ) : (
            <div>변호사 {tooltipContent.count.toLocaleString()}명</div>
          )}
        </div>
      )}

      <ComposableMap
        projection="geoMercator"
        projectionConfig={{
          scale: 4000,
          center: [127.5, 36], // Approx center of South Korea
        }}
        width={400} // Reduce internal width to make relative scale larger
        height={500}
        className="w-full h-full"
      >
        <ZoomableGroup
          center={center}
          zoom={zoom}
          minZoom={1}
          maxZoom={20}
          onMoveEnd={({ coordinates }) => setCenter(coordinates as [number, number])}
        >
          <Geographies geography={GEO_URL}>
            {({ geographies }: { geographies: GeoFeature[] }) =>
              geographies.map((geo: GeoFeature) => {
                const code = geo.properties.code
                const isSelected = !selectedCodePrefix || code.startsWith(selectedCodePrefix)
                const fullName = getFullName(geo)
                const isHighlighted = highlightedRegion === fullName

                // 선택된 지역만 데이터 매칭
                let count = 0
                if (isSelected) {
                  count = dataMap.get(fullName) ?? 0
                }

                // 줌 레벨에 반비례하여 테두리 두께 조절
                const baseStrokeWidth = 0.5 / zoom

                // 수요 모드: 지역을 회색 단색으로 표시, 선택된 법원 관할만 하이라이트
                // 공급 모드: 기존 색칠 방식
                const isCourtRegion = selectedCourtRegions.has(fullName)
                let fillColor: string
                if (isDemandMode && courtMarkers && courtMarkers.length > 0) {
                  if (isCourtRegion) {
                    fillColor = viewMode === 'case_count' ? "#DBEAFE" : "#DBEAFE"
                  } else {
                    fillColor = isSelected ? "#F1F5F9" : "#E5E7EB"
                  }
                } else {
                  fillColor = isSelected ? colorScale(count) : "#E5E7EB"
                }

                // 선택된 지역이 있을 때 나머지는 dim 처리
                const fillOpacity = isCourtRegion
                  ? 1
                  : isHighlighted
                    ? 1
                    : highlightedRegion && isSelected
                      ? 0.5
                      : isSelected
                        ? 1
                        : 0.4

                // Stroke 설정
                // Stroke 설정
                const isDemandZoomed = isDemandMode && courtMarkers && courtMarkers.length > 0 && selectedCodePrefix
                const strokeColor = isHighlighted
                  ? "#2563EB"  // 선택: blue-600
                  : isDemandZoomed && isSelected
                    ? "#C0C7CF"  // 수요 확대: 연한 경계 (바깥 #D6D6DA보다는 진하게)
                    : isSelected && selectedCodePrefix
                      ? "#6B7280"
                      : isSelected
                        ? "#9CA3AF"
                        : "#D6D6DA"

                const strokeWidth = isHighlighted
                  ? baseStrokeWidth * 5  // 선택: 굵게
                  : isSelected && selectedCodePrefix
                    ? baseStrokeWidth * 1.5
                    : baseStrokeWidth

                return (
                  <Geography
                    key={geo.rsmKey}
                    geography={geo}
                    fill={fillColor}
                    fillOpacity={fillOpacity}
                    stroke={strokeColor}
                    strokeWidth={strokeWidth}
                    onMouseEnter={isSelected ? (e: React.MouseEvent) => handleMouseEnter(geo, e) : undefined}
                    onMouseMove={isSelected ? handleMouseMove : undefined}
                    onMouseLeave={isSelected ? handleMouseLeave : undefined}
                    onClick={(e) => {
                      e.stopPropagation()
                      if (isDemandMode && isSelected) {
                        // 수요 모드: 지역의 관할법원으로 연결
                        const rd = regionDataMap.get(fullName)
                        if (rd?.courtName) onCourtClick?.(rd.courtName)
                      } else {
                        onRegionClick?.(isSelected ? fullName : null)
                      }
                    }}
                    tabIndex={-1}
                    style={{
                      default: { outline: "none" },
                      hover: isSelected
                        ? {
                            fill: isDemandZoomed ? "#E2E8F0" : (isHighlighted ? colorScale(count) : "#BFDBFE"),
                            stroke: isDemandZoomed ? "#CBD5E1" : "#93C5FD",
                            strokeWidth: baseStrokeWidth * 1.5,
                            outline: "none",
                            cursor: isDemandMode ? "pointer" : undefined,
                          }
                        : { outline: "none" },
                      pressed: { outline: "none" },
                    }}
                  />
                )
              })
            }
          </Geographies>

          {/* 수요 모드: 법원 마커 렌더링 */}
          {isDemandMode && courtMarkers?.map((court) => {
            const isMarkerSelected = selectedCourt === court.court_name
            const value = viewMode === 'burden_index' ? court.burden_index : court.case_count
            const size = 12

            const colors = viewMode === 'burden_index' ? BURDEN_MARKER_COLORS : CASE_MARKER_COLORS
            let markerColor = colors[colors.length - 1].color
            for (const c of colors) {
              if (value >= c.min) { markerColor = c.color; break }
            }

            {/* 아이콘 크기: 원 반지름의 ~65% */}
            const iconScale = (size * 0.65) / zoom / 12

            return (
              <Marker key={court.court_name} coordinates={court.coordinates}>
                <g
                  onClick={(e) => { e.stopPropagation(); onCourtClick?.(court.court_name) }}
                  onMouseEnter={(e) => {
                    const regionsPreview = court.regions.length > 3
                      ? court.regions.slice(0, 3).join(', ') + ` 외 ${court.regions.length - 3}개`
                      : court.regions.join(', ')
                    setTooltipContent({
                      region: court.court_name,
                      count: 0,
                      caseCount: court.case_count,
                      lawyerCount: court.lawyer_count,
                      burdenIndex: court.burden_index,
                      courtName: regionsPreview,
                      isCourtMarker: true,
                    })
                    setTooltipPos({ x: e.clientX, y: e.clientY })
                  }}
                  onMouseMove={handleMouseMove}
                  onMouseLeave={handleMouseLeave}
                  style={{ cursor: 'pointer' }}
                >
                  {/* 배경 원 */}
                  <circle
                    r={size / zoom}
                    fill={markerColor}
                    fillOpacity={0.85}
                    stroke={isMarkerSelected ? "#1E40AF" : "#FFFFFF"}
                    strokeWidth={isMarkerSelected ? 2 / zoom : 1 / zoom}
                  />
                  {/* 법원 건물 아이콘 (Landmark) */}
                  <g transform={`translate(${-12 * iconScale}, ${-12 * iconScale}) scale(${iconScale})`}>
                    <polygon points="12,3 21,8 3,8" fill="white" fillOpacity={0.95} />
                    <rect x="5" y="9.5" width="2" height="8" rx="0.5" fill="white" fillOpacity={0.95} />
                    <rect x="9" y="9.5" width="2" height="8" rx="0.5" fill="white" fillOpacity={0.95} />
                    <rect x="13" y="9.5" width="2" height="8" rx="0.5" fill="white" fillOpacity={0.95} />
                    <rect x="17" y="9.5" width="2" height="8" rx="0.5" fill="white" fillOpacity={0.95} />
                    <rect x="2" y="18" width="20" height="2.5" rx="0.5" fill="white" fillOpacity={0.95} />
                  </g>
                </g>
                {zoom >= 3 && (
                  <text
                    textAnchor="middle"
                    y={-size / zoom - 4 / zoom}
                    style={{ fontSize: `${10 / zoom}px`, fill: '#374151', fontWeight: 500, pointerEvents: 'none' }}
                  >
                    {court.court_name.replace(/지방법원|가정법원|행정법원/, '')}
                  </text>
                )}
              </Marker>
            )
          })}
        </ZoomableGroup>
      </ComposableMap>

      {/* Zoom Slider + Legend */}
      <div className="absolute bottom-4 right-4 flex flex-col items-end gap-2">
        {/* Zoom Buttons */}
        <div className="flex flex-col bg-white rounded-lg shadow-md border border-gray-200 overflow-hidden">
          <button
            type="button"
            onClick={() => setZoom(Math.min(selectedProvince ? viewConfig.zoom * 2 : 20, zoom * 1.3))}
            className="px-3 py-2 text-gray-600 hover:bg-gray-100 transition-colors border-b border-gray-200"
            aria-label="줌 인"
          >
            +
          </button>
          <button
            type="button"
            onClick={() => setZoom(Math.max(Math.max(1, viewConfig.zoom * 0.5), zoom / 1.3))}
            className="px-3 py-2 text-gray-600 hover:bg-gray-100 transition-colors"
            aria-label="줌 아웃"
          >
            −
          </button>
        </div>

        {/* Legend */}
        <div className="flex flex-col gap-1 text-xs text-gray-600 bg-white/95 p-2.5 rounded-lg shadow-sm border border-gray-200">
          {isDemandMode && courtMarkers && courtMarkers.length > 0 ? (
            viewMode === 'case_count' ? (
              <>
                <div className="font-medium text-slate-700 mb-1">지역별 사건 수<br /><span className="font-normal text-gray-400">(관할법원 기준)</span></div>
                {[
                  { min: 10000, label: '1만 건 이상' },
                  { min: 5000, label: '5천 ~ 1만 건' },
                  { min: 2000, label: '2천 ~ 5천 건' },
                  { min: 1000, label: '1천 ~ 2천 건' },
                  { min: 0, label: '1천 건 미만' },
                ].map(({ min, label }) => {
                  let c = CASE_MARKER_COLORS[CASE_MARKER_COLORS.length - 1].color
                  for (const mc of CASE_MARKER_COLORS) { if (min >= mc.min) { c = mc.color; break } }
                  return (
                    <div key={min} className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-full" style={{ backgroundColor: c, opacity: 0.85 }} />
                      <span>{label}</span>
                    </div>
                  )
                })}
              </>
            ) : (
              <>
                <div className="font-medium text-rose-700 mb-1">법원별 부담지수</div>
                {[
                  { min: 100, label: '100 이상' },
                  { min: 50, label: '50 ~ 100' },
                  { min: 20, label: '20 ~ 50' },
                  { min: 10, label: '10 ~ 20' },
                  { min: 0, label: '10 미만' },
                ].map(({ min, label }) => {
                  let c = BURDEN_MARKER_COLORS[BURDEN_MARKER_COLORS.length - 1].color
                  for (const mc of BURDEN_MARKER_COLORS) { if (min >= mc.min) { c = mc.color; break } }
                  return (
                    <div key={min} className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-full" style={{ backgroundColor: c, opacity: 0.85 }} />
                      <span>{label}</span>
                    </div>
                  )
                })}
              </>
            )
          ) : viewMode === 'count' ? (
            <>
              <div className="font-medium text-gray-700 mb-1">변호사 수</div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: colorScale(500) }}></div>
                <span>500명 이상</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: colorScale(200) }}></div>
                <span>100 ~ 500명</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: colorScale(50) }}></div>
                <span>30 ~ 100명</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: colorScale(15) }}></div>
                <span>10 ~ 30명</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: colorScale(5) }}></div>
                <span>1 ~ 10명</span>
              </div>
            </>
          ) : viewMode === 'prediction' ? (
            <>
              <div className="font-medium text-violet-700 mb-1">{predictionYear}년 예측 (10만명당)</div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: colorScale(100) }}></div>
                <span>100명 이상</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: colorScale(50) }}></div>
                <span>10 ~ 100명</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: colorScale(7) }}></div>
                <span>5 ~ 10명</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: colorScale(3) }}></div>
                <span>2 ~ 5명</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: colorScale(1.5) }}></div>
                <span>1 ~ 2명</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: colorScale(0.5) }}></div>
                <span>1명 미만</span>
              </div>
            </>
          ) : (
            <>
              <div className="font-medium text-gray-700 mb-1">인구 10만명당</div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: colorScale(100) }}></div>
                <span>100명 이상</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: colorScale(50) }}></div>
                <span>10 ~ 100명</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: colorScale(7) }}></div>
                <span>5 ~ 10명</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: colorScale(3) }}></div>
                <span>2 ~ 5명</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: colorScale(1.5) }}></div>
                <span>1 ~ 2명</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: colorScale(0.5) }}></div>
                <span>1명 미만</span>
              </div>
            </>
          )}
          {!(isDemandMode && courtMarkers && courtMarkers.length > 0) && (
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-sm bg-white border border-gray-300"></div>
              <span>0명</span>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
