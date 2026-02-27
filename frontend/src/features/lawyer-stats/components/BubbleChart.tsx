'use client'

import { useMemo, useState } from 'react'
import type { ComponentType } from 'react'
import dynamic from 'next/dynamic'
import type { PlotData, PlotParams } from 'react-plotly.js'
import type { BubbleDataResponse } from '../types'

const Plot = dynamic(() => import('react-plotly.js'), {
  ssr: false,
}) as unknown as ComponentType<PlotParams>

// 12대분류 고정 색상 맵
const CATEGORY_COLORS: Record<string, string> = {
  '민사': '#3B82F6',
  '형사': '#EF4444',
  '가사': '#F59E0B',
  '행정': '#10B981',
  '헌법': '#8B5CF6',
  '특허/지적재산권': '#EC4899',
  '조세/관세': '#F97316',
  '노동': '#06B6D4',
  '국제거래/무역': '#6366F1',
  '의료/보건': '#14B8A6',
  '환경': '#84CC16',
  '기타': '#6B7280',
}

// 버블 크기 스케일링 (sqrt로 차이 완화)
const BUBBLE_MIN_SIZE = 6
const BUBBLE_MAX_SIZE = 60

function scaleBubbleSize(value: number, maxValue: number): number {
  if (maxValue === 0) return BUBBLE_MIN_SIZE
  const ratio = Math.sqrt(value) / Math.sqrt(maxValue)
  return BUBBLE_MIN_SIZE + ratio * (BUBBLE_MAX_SIZE - BUBBLE_MIN_SIZE)
}

interface BubbleChartProps {
  data: BubbleDataResponse
}

export function BubbleChart({ data }: BubbleChartProps) {
  const { cross, regions, categories, demand_by_year, available_years } = data

  const defaultYear = available_years.length > 0
    ? available_years[available_years.length - 1]
    : 2024

  const [selectedYear, setSelectedYear] = useState<number>(defaultYear)
  const [userRegion, setUserRegion] = useState<string>(regions[0] ?? '')
  const [userCategory, setUserCategory] = useState<string>(categories[0] ?? '')

  // 최대 변호사 수 계산
  const maxCount = useMemo(() => {
    return cross.reduce((max, cell) => Math.max(max, cell.count), 0)
  }, [cross])

  // 선택 연도의 지역별 총 사건 수
  const demandMap = useMemo(() => {
    const yearKey = String(selectedYear)
    const demandList = demand_by_year[yearKey] ?? []
    const map = new Map<string, number>()
    for (const item of demandList) {
      map.set(item.region, item.total_cases)
    }
    return map
  }, [demand_by_year, selectedYear])

  // 수요 최대값 (opacity 스케일용)
  const maxDemand = useMemo(() => {
    let max = 0
    demandMap.forEach((v) => { if (v > max) max = v })
    return max
  }, [demandMap])

  // Plotly 트레이스: 카테고리별로 분리하여 범례 생성
  const plotTraces = useMemo((): PlotData[] => {
    // 카테고리별 버블 데이터
    const categoryMap = new Map<string, {
      x: string[]
      y: string[]
      sizes: number[]
      counts: number[]
    }>()

    for (const cell of cross) {
      if (!categoryMap.has(cell.category_name)) {
        categoryMap.set(cell.category_name, { x: [], y: [], sizes: [], counts: [] })
      }
      const entry = categoryMap.get(cell.category_name)!
      entry.x.push(cell.region)
      entry.y.push(cell.category_name)
      entry.sizes.push(scaleBubbleSize(cell.count, maxCount))
      entry.counts.push(cell.count)
    }

    const traces: PlotData[] = []

    categoryMap.forEach((entry, categoryName) => {
      const color = CATEGORY_COLORS[categoryName] ?? '#6B7280'
      traces.push({
        type: 'scatter',
        mode: 'markers',
        name: categoryName,
        x: entry.x,
        y: entry.y,
        marker: {
          size: entry.sizes,
          color,
          opacity: 0.8,
          line: {
            color: 'white',
            width: 1,
          },
          sizemode: 'diameter',
        },
        text: entry.x.map((region, i) => {
          const demand = demandMap.get(region)
          const demandText = demand != null
            ? `<br>사건 수요(${selectedYear}): ${demand.toLocaleString()}건`
            : ''
          return `${region} ${categoryName}: ${entry.counts[i].toLocaleString()}명${demandText}`
        }),
        hovertemplate: '%{text}<extra></extra>',
        customdata: entry.counts,
      })
    })

    // 유저 마커 트레이스
    if (userRegion && userCategory) {
      const userCell = cross.find(
        (c) => c.region === userRegion && c.category_name === userCategory
      )
      const userCount = userCell?.count ?? 0
      traces.push({
        type: 'scatter',
        mode: 'markers+text',
        name: '내 위치',
        x: [userRegion],
        y: [userCategory],
        marker: {
          symbol: 'star',
          size: 28,
          color: '#FBBF24',
          line: {
            color: '#92400E',
            width: 2,
          },
        },
        text: ['★'],
        textposition: 'top center',
        hovertemplate: `${userRegion} ${userCategory}: ${userCount.toLocaleString()}명<br>사건 수요(${selectedYear}): ${(demandMap.get(userRegion) ?? 0).toLocaleString()}건<extra>내 위치</extra>`,
        showlegend: true,
      })
    }

    return traces
  }, [cross, maxCount, userRegion, userCategory, demandMap, selectedYear])

  // 배경 사각형 (수요 표시용 Plotly shapes)
  const demandShapes = useMemo(() => {
    if (maxDemand === 0) return []

    return regions.map((region) => {
      const demand = demandMap.get(region) ?? 0
      const opacity = maxDemand > 0 ? (demand / maxDemand) * 0.25 : 0

      return {
        type: 'rect' as const,
        xref: 'x' as const,
        yref: 'paper' as const,
        x0: region,
        x1: region,
        y0: 0,
        y1: 1,
        fillcolor: `rgba(251, 191, 36, ${opacity})`,
        line: { width: 0 },
        layer: 'below' as const,
      }
    })
  }, [regions, demandMap, maxDemand])

  // 모바일 감지 (SSR 안전)
  const isMobile = typeof window !== 'undefined' && window.innerWidth < 768

  // 레이아웃
  const layout = useMemo(() => ({
    autosize: true,
    margin: isMobile
      ? { l: 80, r: 10, t: 10, b: 80 }
      : { l: 150, r: 30, t: 20, b: 120 },
    xaxis: {
      title: isMobile ? undefined : { text: '지역', standoff: 12 },
      categoryorder: 'array' as const,
      categoryarray: regions,
      tickangle: -45,
      tickfont: { size: isMobile ? 9 : 11 },
      gridcolor: '#F3F4F6',
    },
    yaxis: {
      title: isMobile ? undefined : { text: '전문분야', standoff: 8 },
      categoryorder: 'array' as const,
      categoryarray: [...categories].reverse(),
      tickfont: { size: isMobile ? 9 : 11 },
      gridcolor: '#F3F4F6',
    },
    legend: isMobile
      ? { orientation: 'h' as const, x: 0, y: -0.25, font: { size: 9 } }
      : { orientation: 'v' as const, x: 1.02, y: 1, xanchor: 'left' as const, font: { size: 11 } },
    plot_bgcolor: '#FAFAFA',
    paper_bgcolor: '#FFFFFF',
    shapes: demandShapes,
    hovermode: 'closest' as const,
    transition: { duration: 300, easing: 'cubic-in-out' },
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }), [regions, categories, demandShapes, isMobile])

  const config = {
    responsive: true,
    displayModeBar: false,
  }

  return (
    <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
      {/* 헤더 */}
      <div className="mb-4">
        <h3 className="text-lg font-semibold text-gray-900">변호사 분포 시각화</h3>
        <p className="text-sm text-gray-500 mt-1">
          지역 × 전문분야별 변호사 수를 버블 크기로 표현합니다.
          배경 음영은 선택 연도의 지역별 사건 수요를 나타냅니다.
        </p>
      </div>

      {/* 컨트롤 영역 */}
      <div className="mb-4 flex flex-wrap items-center gap-4">
        {/* 연도 슬라이더 */}
        <div className="flex flex-col gap-1">
          <label className="text-xs font-medium text-gray-600">
            수요 연도: <span className="font-semibold text-amber-600">{selectedYear}년</span>
          </label>
          <input
            type="range"
            min={available_years[0] ?? 2015}
            max={available_years[available_years.length - 1] ?? 2024}
            step={1}
            value={selectedYear}
            onChange={(e) => setSelectedYear(Number(e.target.value))}
            className="w-40 accent-amber-500"
          />
          <div className="flex justify-between text-xs text-gray-400">
            <span>{available_years[0] ?? 2015}</span>
            <span>{available_years[available_years.length - 1] ?? 2024}</span>
          </div>
        </div>

        {/* 구분선 */}
        <div className="h-12 w-px bg-gray-200" />

        {/* 유저 마커 설정 */}
        <div className="flex flex-col gap-1">
          <label className="text-xs font-medium text-gray-600">
            내 위치 표시 (★)
          </label>
          <div className="flex gap-2">
            <select
              value={userRegion}
              onChange={(e) => setUserRegion(e.target.value)}
              className="rounded-md border border-gray-300 bg-white px-2 py-1 text-xs font-medium text-gray-700 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
            >
              {regions.map((region) => (
                <option key={region} value={region}>{region}</option>
              ))}
            </select>
            <select
              value={userCategory}
              onChange={(e) => setUserCategory(e.target.value)}
              className="rounded-md border border-gray-300 bg-white px-2 py-1 text-xs font-medium text-gray-700 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
            >
              {categories.map((category) => (
                <option key={category} value={category}>{category}</option>
              ))}
            </select>
          </div>
        </div>

        {/* 범례 설명 */}
        <div className="ml-auto flex items-center gap-3 text-xs text-gray-500">
          <div className="flex items-center gap-1.5">
            <div className="h-3 w-3 rounded-full bg-blue-400 opacity-80" />
            <span>버블 크기 = 변호사 수</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="h-3 w-6 rounded" style={{ background: 'rgba(251,191,36,0.25)' }} />
            <span>배경 = 사건 수요({selectedYear})</span>
          </div>
        </div>
      </div>

      {/* 차트 */}
      <div className="w-full">
        <Plot
          data={plotTraces}
          layout={layout}
          config={config}
          style={{ width: '100%', height: isMobile ? '400px' : '520px' }}
          useResizeHandler
        />
      </div>

      {/* 하단 안내 */}
      <p className="mt-2 text-center text-xs text-gray-400">
        호버 시 지역 · 전문분야 · 변호사 수 · 사건 수요 확인 가능
      </p>
    </div>
  )
}
