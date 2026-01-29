"use client"

import React, { useEffect, useMemo, useState } from "react"
import { ComposableMap, Geographies, Geography, ZoomableGroup } from "react-simple-maps"
import type { PredictionYear, ViewMode } from "@/app/lawyer-stats/page"
import type { DensityStat, RegionStat } from "../types"

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
  data: (RegionStat | DensityStat)[]
  viewMode: ViewMode
  predictionYear?: PredictionYear
  selectedProvince?: string | null
  highlightedRegion?: string | null
  onRegionClick?: (region: string | null) => void
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

export function RegionGeoMap({ data, viewMode, predictionYear, selectedProvince, highlightedRegion, onRegionClick }: Props) {
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
    const map = new Map<string, { count: number; density?: number; changePercent?: number }>()
    data.forEach((d) => {
      map.set(d.region, {
        count: d.count,
        density: 'density' in d ? d.density : undefined,
        changePercent: 'change_percent' in d ? d.change_percent : undefined,
      })
    })
    return map
  }, [data])

  // 2. Create a map for color scale values
  const dataMap = useMemo(() => {
    const map = new Map<string, number>()
    const useDensity = viewMode === 'density' || viewMode === 'prediction'
    data.forEach((d) => {
      if (useDensity && 'density' in d) {
        map.set(d.region, d.density)
      } else {
        map.set(d.region, d.count)
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

  const colorScale = (value: number) => {
    if (value === 0) return "#ffffff"
    const ranges = viewMode === 'prediction'
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
  const getFullName = (geo: any) => {
    const code = geo.properties.code as string // e.g. "11250"
    const name = geo.properties.name as string // e.g. "강동구" or "수원시장안구"

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

  const handleMouseEnter = (geo: any, event: React.MouseEvent) => {
    const fullName = getFullName(geo)
    const regionData = regionDataMap.get(fullName)
    setTooltipContent({
      region: fullName,
      density: regionData?.density,
      count: regionData?.count ?? 0,
      changePercent: regionData?.changePercent,
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
          {viewMode === 'prediction' && tooltipContent.density !== undefined ? (
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
            {({ geographies }: { geographies: any[] }) =>
              geographies.map((geo: any) => {
                const code = geo.properties.code as string
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

                // 1️⃣ 선택된 지역: Fill 기존 색 유지 + 테두리 굵고 진하게
                // 2️⃣ 나머지 지역: dim 처리 (opacity 낮게)
                const fillColor = isSelected ? colorScale(count) : "#E5E7EB"

                // 선택된 지역이 있을 때 나머지는 dim 처리
                const fillOpacity = isHighlighted
                  ? 1
                  : highlightedRegion && isSelected
                    ? 0.5
                    : isSelected
                      ? 1
                      : 0.4

                // Stroke 설정
                const strokeColor = isHighlighted
                  ? "#2563EB"  // 선택: blue-600
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
                      onRegionClick?.(isSelected ? fullName : null)
                    }}
                    tabIndex={-1}
                    style={{
                      default: { outline: "none" },
                      hover: isSelected
                        ? {
                            fill: isHighlighted ? colorScale(count) : "#BFDBFE",
                            stroke: "#93C5FD",
                            strokeWidth: baseStrokeWidth * 1.5,
                            outline: "none"
                          }
                        : { outline: "none" },
                      pressed: { outline: "none" },
                    }}
                  />
                )
              })
            }
          </Geographies>
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
          {viewMode === 'count' ? (
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
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-sm bg-white border border-gray-300"></div>
            <span>0명</span>
          </div>
        </div>
      </div>
    </div>
  )
}
