'use client'

import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { RegionStat } from '../types'

interface RegionBarChartProps {
  data: RegionStat[]
}

const DISPLAY_COUNT = 15

export function RegionBarChart({ data }: RegionBarChartProps) {
  const chartData = data.slice(0, DISPLAY_COUNT)

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm">
      <h3 className="mb-4 text-lg font-semibold text-gray-900">지역별 변호사 분포</h3>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} layout="vertical" margin={{ left: 80, right: 20 }}>
            <CartesianGrid strokeDasharray="3 3" horizontal vertical={false} />
            <XAxis type="number" />
            <YAxis
              type="category"
              dataKey="region"
              tick={{ fontSize: 12 }}
              width={75}
            />
            <Tooltip
              formatter={(value) => [`${Number(value).toLocaleString()}명`, '변호사 수']}
              labelStyle={{ color: '#374151' }}
              contentStyle={{
                backgroundColor: '#fff',
                border: '1px solid #e5e7eb',
                borderRadius: '0.5rem',
              }}
            />
            <Bar dataKey="count" fill="#3b82f6" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      {data.length > DISPLAY_COUNT && (
        <p className="mt-2 text-center text-sm text-gray-500">
          상위 {DISPLAY_COUNT}개 지역 표시 (전체 {data.length}개 지역)
        </p>
      )}
    </div>
  )
}
