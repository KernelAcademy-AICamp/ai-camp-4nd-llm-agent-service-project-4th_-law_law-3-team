'use client'

import { useMemo } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  LabelList,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { DailyStatItem } from '../types'

interface ChartRow {
  date: string
  lawtimes: number
  naver: number
}

/** KST 기준 날짜 문자열 (백엔드 KST와 일치) */
function toKSTDateString(date: Date): string {
  const kst = new Date(date.getTime() + 9 * 60 * 60 * 1000)
  return kst.toISOString().slice(0, 10)
}

/** API 데이터를 차트 데이터로 변환 + 빈 날짜 0건 보간 */
function transformData(items: DailyStatItem[], periodDays: number): ChartRow[] {
  const map = new Map<string, ChartRow>()

  // API 데이터 집계 (lawtimes/naver만 분류, 미지 소스는 무시)
  for (const item of items) {
    const key = item.source === 'lawtimes' ? 'lawtimes' : item.source === 'naver' ? 'naver' : null
    if (!key) continue
    const existing = map.get(item.date)
    if (existing) {
      existing[key] += item.count
    } else {
      map.set(item.date, {
        date: item.date,
        lawtimes: key === 'lawtimes' ? item.count : 0,
        naver: key === 'naver' ? item.count : 0,
      })
    }
  }

  // 빈 날짜 0건 보간 (KST 기준)
  const today = new Date()
  for (let i = periodDays - 1; i >= 0; i--) {
    const d = new Date(today)
    d.setDate(d.getDate() - i)
    const dateStr = toKSTDateString(d)
    if (!map.has(dateStr)) {
      map.set(dateStr, { date: dateStr, lawtimes: 0, naver: 0 })
    }
  }

  return Array.from(map.entries())
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([, row]) => row)
}

function formatDate(dateStr: string): string {
  const parts = dateStr.split('-')
  if (parts.length === 3) return `${parts[1]}/${parts[2]}`
  return dateStr
}

interface NewsBarChartProps {
  items: DailyStatItem[]
  periodDays: number
  onPeriodChange: (days: number) => void
}

const PERIOD_OPTIONS = [
  { value: 7, label: '7일' },
  { value: 14, label: '14일' },
  { value: 30, label: '30일' },
]

export function NewsBarChart({ items, periodDays, onPeriodChange }: NewsBarChartProps) {
  const data = transformData(items, periodDays)

  const totals = useMemo(() => ({
    lawtimes: data.reduce((sum, row) => sum + row.lawtimes, 0),
    naver: data.reduce((sum, row) => sum + row.naver, 0),
  }), [data])

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-4 shadow-sm">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-gray-800">일별 수집 건수</h3>
        <div className="flex gap-1">
          {PERIOD_OPTIONS.map((opt) => (
            <button
              key={opt.value}
              onClick={() => onPeriodChange(opt.value)}
              className={`px-2.5 py-1 text-xs rounded-md transition-colors ${
                periodDays === opt.value
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {data.length === 0 ? (
        <div className="flex items-center justify-center h-48 text-sm text-gray-400">
          데이터 없음
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={data} margin={{ top: 15, right: 5, bottom: 5, left: -10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis
              dataKey="date"
              tickFormatter={formatDate}
              tick={{ fontSize: 11, fill: '#6b7280' }}
            />
            <YAxis
              allowDecimals={false}
              tick={{ fontSize: 11, fill: '#6b7280' }}
            />
            <Tooltip
              labelFormatter={(label) => formatDate(String(label))}
              formatter={(value, name) => {
                const label = name === 'lawtimes' ? '로타임즈' : name === 'naver' ? '네이버' : String(name)
                return [`${Number(value)}건`, label]
              }}
              contentStyle={{ fontSize: 12, borderRadius: 8 }}
            />
            <Legend
              formatter={(value: string) =>
                value === 'lawtimes' ? '로타임즈' : value === 'naver' ? '네이버' : value
              }
              wrapperStyle={{ fontSize: 12 }}
            />
            <Bar dataKey="lawtimes" fill="#3b82f6" radius={[4, 4, 0, 0]}>
              {periodDays <= 14 && (
                <LabelList
                  dataKey="lawtimes"
                  position="top"
                  style={{ fontSize: 10, fill: '#374151' }}
                  formatter={(value) => (Number(value) > 0 ? String(value) : '')}
                />
              )}
            </Bar>
            <Bar dataKey="naver" fill="#22c55e" radius={[4, 4, 0, 0]}>
              {periodDays <= 14 && (
                <LabelList
                  dataKey="naver"
                  position="top"
                  style={{ fontSize: 10, fill: '#374151' }}
                  formatter={(value) => (Number(value) > 0 ? String(value) : '')}
                />
              )}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      )}

      {/* 누계 건수 */}
      {data.length > 0 && (
        <div className="flex items-center justify-center gap-6 mt-2 pt-2 border-t border-gray-100">
          <div className="flex items-center gap-1.5 text-xs">
            <span className="w-2.5 h-2.5 rounded-full bg-blue-500" />
            <span className="text-gray-600">로타임즈</span>
            <span className="font-semibold text-gray-800">
              {totals.lawtimes.toLocaleString()}건
            </span>
          </div>
          <div className="flex items-center gap-1.5 text-xs">
            <span className="w-2.5 h-2.5 rounded-full bg-green-500" />
            <span className="text-gray-600">네이버</span>
            <span className="font-semibold text-gray-800">
              {totals.naver.toLocaleString()}건
            </span>
          </div>
        </div>
      )}
    </div>
  )
}
