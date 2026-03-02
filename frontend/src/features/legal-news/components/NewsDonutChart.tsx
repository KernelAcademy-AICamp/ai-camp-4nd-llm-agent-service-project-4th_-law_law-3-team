'use client'

import {
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
} from 'recharts'
import type { CategoryStatItem } from '../types'

const COLORS = [
  '#3b82f6', '#22c55e', '#f59e0b', '#ef4444',
  '#8b5cf6', '#ec4899', '#14b8a6', '#f97316',
  '#6366f1', '#84cc16', '#06b6d4', '#e11d48',
]

const PERIOD_OPTIONS = [
  { value: 7, label: '7일' },
  { value: 14, label: '14일' },
  { value: 30, label: '30일' },
]

interface NewsDonutChartProps {
  items: CategoryStatItem[]
  total: number
  periodDays: number
  onPeriodChange: (days: number) => void
  loading?: boolean
}

export function NewsDonutChart({
  items, total, periodDays, onPeriodChange, loading,
}: NewsDonutChartProps) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-4 shadow-sm">
      {/* 헤더: 제목 + 기간 선택 */}
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-gray-800">카테고리 분포</h3>
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500">총 {total.toLocaleString()}건</span>
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
      </div>

      {loading ? (
        <div className="flex items-center justify-center h-48">
          <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600" />
        </div>
      ) : items.length === 0 ? (
        <div className="flex items-center justify-center h-48 text-sm text-gray-400">
          데이터 없음
        </div>
      ) : (
        <>
          {/* 도넛 차트 (Legend 제거, 파이 라벨은 퍼센트만) */}
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie
                data={items}
                dataKey="count"
                nameKey="category"
                cx="50%"
                cy="50%"
                innerRadius={55}
                outerRadius={80}
                paddingAngle={2}
                label={({ percent }) => {
                  if (Number(percent) < 0.05) return ''
                  return `${(Number(percent) * 100).toFixed(0)}%`
                }}
                labelLine={{ stroke: '#9ca3af', strokeWidth: 1 }}
              >
                {items.map((_, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip
                formatter={(value) => [`${Number(value).toLocaleString()}건`]}
                contentStyle={{ fontSize: 12, borderRadius: 8 }}
              />
            </PieChart>
          </ResponsiveContainer>

          {/* 커스텀 Legend: 카테고리별 건수 포함 */}
          <ul className="grid grid-cols-2 gap-x-4 gap-y-1.5 mt-2 pt-2 border-t border-gray-100">
            {items.map((item, index) => (
              <li key={item.category} className="flex items-center gap-1.5 text-xs">
                <span
                  className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                  style={{ backgroundColor: COLORS[index % COLORS.length] }}
                />
                <span className="text-gray-700 truncate">{item.category}</span>
                <span className="text-gray-500 ml-auto font-medium">
                  {item.count.toLocaleString()}건
                </span>
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  )
}
