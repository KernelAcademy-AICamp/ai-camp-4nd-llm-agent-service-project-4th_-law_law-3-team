'use client'

interface KPICardsProps {
  totalLawyers: number
  topRegion: { name: string; count: number } | null
  topSpecialty: { name: string; count: number } | null
}

interface KPICardProps {
  icon: string
  value: string
  label: string
  sublabel?: string
}

function KPICard({ icon, value, label, sublabel }: KPICardProps) {
  return (
    <div className="flex items-center gap-4 rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
      <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-blue-50 text-2xl">
        {icon}
      </div>
      <div>
        <div className="text-2xl font-bold text-gray-900">{value}</div>
        <div className="text-sm text-gray-500">{label}</div>
        {sublabel && <div className="text-xs text-gray-400">{sublabel}</div>}
      </div>
    </div>
  )
}

export function KPICards({ totalLawyers, topRegion, topSpecialty }: KPICardsProps) {
  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
      <KPICard
        icon="👨‍⚖️"
        value={`${totalLawyers.toLocaleString()}명`}
        label="전체 변호사"
      />
      <KPICard
        icon="📍"
        value={topRegion?.name ?? '-'}
        label="1위 지역"
        sublabel={topRegion ? `${topRegion.count.toLocaleString()}명` : undefined}
      />
      <KPICard
        icon="⚖️"
        value={topSpecialty?.name ?? '-'}
        label="1위 전문분야"
        sublabel={topSpecialty ? `${topSpecialty.count.toLocaleString()}명` : undefined}
      />
    </div>
  )
}
