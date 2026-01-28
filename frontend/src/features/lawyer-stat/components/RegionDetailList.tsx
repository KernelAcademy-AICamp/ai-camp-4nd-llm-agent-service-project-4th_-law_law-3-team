'use client'

import type { RegionStat } from '../types'

interface RegionDetailListProps {
  regions: RegionStat[]
  selectedProvince: string | null
}

export function RegionDetailList({ regions, selectedProvince }: RegionDetailListProps) {
  const filteredRegions = selectedProvince
    ? regions.filter((r) => r.region.startsWith(selectedProvince))
    : regions

  const displayRegions = filteredRegions.slice(0, 15)
  const maxCount = displayRegions[0]?.count ?? 1

  return (
    <div className="space-y-2">
      <div className="mb-3 text-sm font-medium text-gray-500">
        {selectedProvince ? `${selectedProvince} 지역 순위` : '전체 지역 순위'}
      </div>
      <div className="space-y-2 overflow-y-auto" style={{ maxHeight: '360px' }}>
        {displayRegions.map((region, index) => {
          const barWidth = (region.count / maxCount) * 100
          return (
            <div key={region.region} className="flex items-center gap-3">
              <span className="w-6 text-right text-sm font-medium text-gray-400">
                {index + 1}
              </span>
              <div className="flex-1">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-700">
                    📍 {region.region}
                  </span>
                  <span className="text-sm font-semibold text-blue-600">
                    {region.count.toLocaleString()}명
                  </span>
                </div>
                <div className="mt-1 h-1.5 w-full rounded-full bg-gray-100">
                  <div
                    className="h-1.5 rounded-full bg-blue-500 transition-all"
                    style={{ width: `${barWidth}%` }}
                  />
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
