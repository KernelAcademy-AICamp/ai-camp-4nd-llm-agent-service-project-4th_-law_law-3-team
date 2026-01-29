'use client'

import {
  Bar,
  BarChart,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { SpecialtyStat } from '../types'

interface SpecialtyBarChartProps {
  data: SpecialtyStat[]
}

const COLORS = [
  '#3b82f6', '#10b981', '#f59e0b', '#ef4444',
  '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16',
  '#f97316', '#6366f1', '#14b8a6', '#d946ef',
]

interface TooltipPayload {
  payload: SpecialtyStat
}

function CustomTooltip({ active, payload }: { active?: boolean; payload?: TooltipPayload[] }) {
  if (!active || !payload || payload.length === 0) {
    return null
  }

  const data = payload[0].payload

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-3 shadow-lg">
      <div className="mb-2 font-semibold text-gray-900">
        {data.category_name} ({data.count.toLocaleString()}명)
      </div>
      {data.specialties && data.specialties.length > 0 && (
        <>
          <div className="mb-1 border-t border-gray-100 pt-2 text-xs text-gray-500">
            세부 전문분야
          </div>
          <div className="max-h-40 space-y-1 overflow-y-auto">
            {data.specialties.map((spec) => (
              <div key={spec.name} className="flex justify-between text-sm">
                <span className="text-gray-600">{spec.name}</span>
                <span className="ml-4 font-medium text-gray-900">
                  {spec.count.toLocaleString()}명
                </span>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  )
}

export function SpecialtyBarChart({ data }: SpecialtyBarChartProps) {
  const chartData = data.map((item, index) => ({
    ...item,
    fill: COLORS[index % COLORS.length],
  }))

  return (
    <div className="h-[400px]">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={chartData}
          layout="vertical"
          margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
        >
          <XAxis type="number" tickFormatter={(value) => `${value}명`} />
          <YAxis
            type="category"
            dataKey="category_name"
            tick={{ fontSize: 12 }}
            width={95}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(0, 0, 0, 0.05)' }} />
          <Bar dataKey="count" radius={[0, 4, 4, 0]} barSize={24}>
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.fill} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
