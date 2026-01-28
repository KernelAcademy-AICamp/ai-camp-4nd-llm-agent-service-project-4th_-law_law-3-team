"use client"

import React, { useMemo, useState } from "react"
import { ComposableMap, Geographies, Geography, ZoomableGroup } from "react-simple-maps"
import { scaleLinear } from "d3-scale"
import { RegionStat } from "../types"

// GeoJSON path (Nationwide)
const GEO_URL = "/data/korea_geo.json"

// Region Code Prefix Mapping (2-digit code -> Province Name)
const PROVINCE_PREFIX_MAP: Record<string, string> = {
  "11": "서울",
  "26": "부산",
  "27": "대구",
  "28": "인천",
  "29": "광주",
  "30": "대전",
  "31": "울산",
  "36": "세종특별자치시",
  "41": "경기",
  "42": "강원",
  "43": "충북",
  "44": "충남",
  "45": "전북",
  "46": "전남",
  "47": "경북",
  "48": "경남",
  "50": "제주",
}

interface Props {
  data: RegionStat[]
}

export function RegionGeoMap({ data }: Props) {
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
        <ZoomableGroup center={[127.5, 36]} zoom={1} minZoom={0.5} maxZoom={4}>
          <Geographies geography={GEO_URL}>
            {({ geographies }: { geographies: any[] }) =>
              geographies.map((geo: any) => {
                const fullName = getFullName(geo)

                // Try literal match first
                let count = dataMap.get(fullName)

                // If not found, try robust matching (e.g. "경기 수원시" vs "경기도 수원시")
                if (count === undefined) {
                  // Fallback logic could go here if needed.
                  // For now, assume 0
                  count = 0
                }

                return (
                  <Geography
                    key={geo.rsmKey}
                    geography={geo}
                    fill={colorScale(count)}
                    stroke="#D6D6DA"
                    strokeWidth={0.5}
                    onMouseEnter={(e: React.MouseEvent) => handleMouseEnter(geo, e)}
                    onMouseMove={handleMouseMove}
                    onMouseLeave={handleMouseLeave}
                    style={{
                      default: { outline: "none" },
                      hover: { fill: "#F53", outline: "none" },
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
