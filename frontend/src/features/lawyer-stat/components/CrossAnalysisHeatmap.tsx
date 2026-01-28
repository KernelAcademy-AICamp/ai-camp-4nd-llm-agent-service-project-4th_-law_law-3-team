'use client'

import { useMemo, useState } from 'react'
import type { CrossAnalysisResponse } from '../types'

interface CrossAnalysisHeatmapProps {
  data: CrossAnalysisResponse
}

function getColorIntensity(value: number, max: number): string {
  if (value === 0) return 'bg-gray-100'
  const ratio = value / max
  if (ratio > 0.7) return 'bg-blue-600 text-white'
  if (ratio > 0.5) return 'bg-blue-500 text-white'
  if (ratio > 0.3) return 'bg-blue-400 text-white'
  if (ratio > 0.1) return 'bg-blue-300'
  return 'bg-blue-200'
}

export function CrossAnalysisHeatmap({ data }: CrossAnalysisHeatmapProps) {
  const [hoveredCell, setHoveredCell] = useState<string | null>(null)

  const { matrix, maxValue } = useMemo(() => {
    const map = new Map<string, number>()
    let max = 0

    for (const cell of data.data) {
      const key = `${cell.region}-${cell.category_name}`
      map.set(key, cell.count)
      if (cell.count > max) max = cell.count
    }

    return { matrix: map, maxValue: max }
  }, [data.data])

  const getValue = (region: string, category: string) => {
    return matrix.get(`${region}-${category}`) ?? 0
  }

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm">
      <h3 className="mb-4 text-lg font-semibold text-gray-900">
        지역 x 전문분야 교차 분석
      </h3>
      <div className="overflow-x-auto">
        <table className="min-w-full text-xs">
          <thead>
            <tr>
              <th className="sticky left-0 z-10 bg-white px-2 py-1 text-left font-medium text-gray-500">
                지역
              </th>
              {data.categories.map((category) => (
                <th
                  key={category}
                  className="px-2 py-1 text-center font-medium text-gray-500"
                  style={{ writingMode: 'vertical-rl', height: '100px' }}
                >
                  {category}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.regions.map((region) => (
              <tr key={region}>
                <td className="sticky left-0 z-10 bg-white px-2 py-1 font-medium text-gray-700">
                  {region}
                </td>
                {data.categories.map((category) => {
                  const value = getValue(region, category)
                  const cellKey = `${region}-${category}`
                  const isHovered = hoveredCell === cellKey

                  return (
                    <td
                      key={cellKey}
                      className={`relative cursor-pointer px-2 py-1 text-center transition-all ${getColorIntensity(value, maxValue)} ${isHovered ? 'ring-2 ring-blue-700' : ''}`}
                      onMouseEnter={() => setHoveredCell(cellKey)}
                      onMouseLeave={() => setHoveredCell(null)}
                    >
                      {value > 0 ? value : '-'}
                      {isHovered && value > 0 && (
                        <div className="absolute bottom-full left-1/2 z-20 mb-1 -translate-x-1/2 whitespace-nowrap rounded bg-gray-900 px-2 py-1 text-xs text-white">
                          {region} / {category}: {value}명
                        </div>
                      )}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-4 flex items-center justify-end gap-2 text-xs text-gray-500">
        <span>낮음</span>
        <div className="flex gap-0.5">
          <div className="h-4 w-4 bg-blue-200" />
          <div className="h-4 w-4 bg-blue-300" />
          <div className="h-4 w-4 bg-blue-400" />
          <div className="h-4 w-4 bg-blue-500" />
          <div className="h-4 w-4 bg-blue-600" />
        </div>
        <span>높음</span>
      </div>
    </div>
  )
}
