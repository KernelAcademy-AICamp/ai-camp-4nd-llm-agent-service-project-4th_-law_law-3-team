'use client'

import type { OverviewResponse } from '../types'

interface OverviewCardsProps {
  data: OverviewResponse
}

interface CardProps {
  label: string
  value: string | number
  suffix?: string
  icon: string
}

function Card({ label, value, suffix, icon }: CardProps) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm">
      <div className="flex items-center gap-3">
        <span className="text-3xl">{icon}</span>
        <div>
          <p className="text-sm text-gray-500">{label}</p>
          <p className="text-2xl font-bold text-gray-900">
            {typeof value === 'number' ? value.toLocaleString() : value}
            {suffix && <span className="text-lg font-normal text-gray-500">{suffix}</span>}
          </p>
        </div>
      </div>
    </div>
  )
}

export function OverviewCards({ data }: OverviewCardsProps) {
  const activeCount = data.status_counts.find((s) => s.status === '개업')?.count ?? 0
  const activeRate = data.total_lawyers > 0
    ? ((activeCount / data.total_lawyers) * 100).toFixed(1)
    : '0'

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
      <Card
        icon="👨‍⚖️"
        label="전체 변호사"
        value={data.total_lawyers}
        suffix="명"
      />
      <Card
        icon="📍"
        label="위치 정보 보유"
        value={data.coord_rate}
        suffix="%"
      />
      <Card
        icon="🎯"
        label="전문분야 등록"
        value={data.specialty_rate}
        suffix="%"
      />
      <Card
        icon="✅"
        label="개업 변호사"
        value={activeRate}
        suffix="%"
      />
    </div>
  )
}
