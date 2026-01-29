'use client'

export type TabType = 'region' | 'cross'

interface StickyTabNavProps {
  activeTab: TabType
  onTabChange: (tab: TabType) => void
}

const TABS: { id: TabType; label: string }[] = [
  { id: 'region', label: '지역별' },
  { id: 'cross', label: '지역 × 전문분야' },
]

export function StickyTabNav({ activeTab, onTabChange }: StickyTabNavProps) {
  return (
    <div className="sticky top-0 z-10 border-b border-gray-200 bg-white/95 backdrop-blur-sm">
      <nav className="flex gap-1 px-4 py-2">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            className={`rounded-lg px-4 py-2 text-sm font-medium transition-colors ${
              activeTab === tab.id
                ? 'bg-blue-500 text-white'
                : 'text-gray-600 hover:bg-gray-100'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </nav>
    </div>
  )
}
