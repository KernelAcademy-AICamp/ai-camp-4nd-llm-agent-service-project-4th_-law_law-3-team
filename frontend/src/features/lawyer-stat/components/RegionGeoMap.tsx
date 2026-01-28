"use client"

import React, { useMemo, useState } from "react"
import { ComposableMap, Geographies, Geography, ZoomableGroup } from "react-simple-maps"
import { scaleLinear } from "d3-scale"
import { RegionStat } from "../types"

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
  data: RegionStat[]
  selectedProvince?: string | null
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

export function RegionGeoMap({ data, selectedProvince }: Props) {
  // 선택된 시/도의 코드 prefix
  const selectedCodePrefix = selectedProvince ? PROVINCE_TO_CODE[selectedProvince] : null

  // 뷰 설정 (선택된 시/도가 있으면 해당 지역으로, 없으면 전국)
  const viewConfig = selectedProvince && PROVINCE_VIEW_CONFIG[selectedProvince]
    ? PROVINCE_VIEW_CONFIG[selectedProvince]
    : { center: [127.5, 36] as [number, number], zoom: 1 }
  const [tooltipContent, setTooltipContent] = useState<string | null>(null)
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 })

  // 1. Create a map of "Full Region Name" -> count
  //    e.g., "서울 서초구": 2849
  const dataMap = useMemo(() => {
    const map = new Map<string, number>()
    data.forEach((d) => {
      map.set(d.region, d.count)
    })
    return map
  }, [data])

  // 2. Color Scale
  //    Domain: 0 to Max count
  const maxCount = useMemo(() => Math.max(...data.map((d) => d.count), 0), [data])

  const colorScale = scaleLinear<string>()
    .domain([0, maxCount > 0 ? maxCount : 1]) // Avoid division by zero
    .range(["#ffffff", "#ef4444"]) // White to Red-500

  // 3. Helper to resolve full name from GeoJSON properties
  const getFullName = (geo: any) => {
    const code = geo.properties.code as string // e.g. "11250"
    const name = geo.properties.name as string // e.g. "강동구"

    const prefix = code.substring(0, 2)
    const province = PROVINCE_PREFIX_MAP[prefix]

    if (!province) return name // Fallback

    if (province === "세종특별자치시") return province // Sejong has no district usually in this level

    // Standard format: "Province District"
    // Handle specific cases if API returns different format (e.g. "강원도" vs "강원")
    // The map above uses short names "강원", "경기" which likely matches API behavior or is close enough.
    // Ideally we might need normalization if API returns "강원도".

    // Check if API uses full province names. The current data shows "서울 ...".
    // Let's assume the mapping is correct for now. 
    // Wait, typical API might return "경기도 수원시". 
    // Let's try to construct "Province Name"
    return `${province} ${name}`
  }

  const handleMouseEnter = (geo: any, event: React.MouseEvent) => {
    const fullName = getFullName(geo)
    const count = dataMap.get(fullName) || 0
    setTooltipContent(`${fullName}: ${count.toLocaleString()}명`)
    setTooltipPos({ x: event.clientX, y: event.clientY })
  }

  const handleMouseMove = (event: React.MouseEvent) => {
    setTooltipPos({ x: event.clientX, y: event.clientY })
  }

  const handleMouseLeave = () => {
    setTooltipContent(null)
  }

  return (
    <div className="relative w-full h-[500px] bg-slate-50 rounded-xl border border-slate-200 overflow-hidden">
      {/* Tooltip */}
      {tooltipContent && (
        <div
          className="fixed z-50 px-2 py-1 text-sm text-white bg-gray-900 rounded pointer-events-none transform -translate-x-1/2 -translate-y-full"
          style={{ left: tooltipPos.x, top: tooltipPos.y - 10 }}
        >
          {tooltipContent}
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
        <ZoomableGroup center={viewConfig.center} zoom={viewConfig.zoom} minZoom={0.5} maxZoom={8}>
          <Geographies geography={GEO_URL}>
            {({ geographies }: { geographies: any[] }) =>
              geographies.map((geo: any) => {
                const code = geo.properties.code as string
                const isSelected = !selectedCodePrefix || code.startsWith(selectedCodePrefix)
                const fullName = getFullName(geo)

                // 선택된 지역만 데이터 매칭
                let count = 0
                if (isSelected) {
                  count = dataMap.get(fullName) ?? 0
                }

                // 줌 레벨에 반비례하여 테두리 두께 조절
                const strokeWidth = 0.5 / viewConfig.zoom

                return (
                  <Geography
                    key={geo.rsmKey}
                    geography={geo}
                    fill={isSelected ? colorScale(count) : "#E5E7EB"}
                    fillOpacity={isSelected ? 1 : 0.4}
                    stroke={isSelected ? "#9CA3AF" : "#D6D6DA"}
                    strokeWidth={strokeWidth}
                    onMouseEnter={isSelected ? (e: React.MouseEvent) => handleMouseEnter(geo, e) : undefined}
                    onMouseMove={isSelected ? handleMouseMove : undefined}
                    onMouseLeave={isSelected ? handleMouseLeave : undefined}
                    style={{
                      default: { outline: "none" },
                      hover: isSelected ? { fill: "#F53", outline: "none" } : { outline: "none" },
                      pressed: { outline: "none" },
                    }}
                  />
                )
              })
            }
          </Geographies>
        </ZoomableGroup>
      </ComposableMap>

      {/* Legend-ish indicator */}
      <div className="absolute bottom-4 right-4 flex flex-col items-end gap-1 text-xs text-gray-500 bg-white/80 p-2 rounded">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-red-500"></div>
          <span>Many Lawyers</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-white border border-gray-200"></div>
          <span>Zero / No Data</span>
        </div>
      </div>
    </div>
  )
}
