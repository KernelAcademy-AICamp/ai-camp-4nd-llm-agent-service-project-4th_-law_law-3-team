'use client'

import type { TrendCategory, TrendFilters as TrendFiltersType, TimeRange } from '../types'

const CATEGORIES: { value: TrendCategory; label: string }[] = [
  { value: 'all', label: '전체' },
  { value: 'criminal', label: '형사' },
  { value: 'civil', label: '민사' },
  { value: 'labor', label: '노동' },
  { value: 'family', label: '가사' },
  { value: 'administrative', label: '행정' },
  { value: 'corporate', label: '기업' },
  { value: 'ip', label: '지식재산' },
]

const TIME_RANGES: { value: TimeRange; label: string }[] = [
  { value: '48h', label: '48시간' },
  { value: '7d', label: '일주일' },
  { value: '30d', label: '한 달' },
]

interface TrendFiltersProps {
  filters: TrendFiltersType
  onFilterChange: (filters: TrendFiltersType) => void
  onRefresh: () => void
  loading: boolean
  cacheHit: boolean
}

export function TrendFilters({
  filters,
  onFilterChange,
  onRefresh,
  loading,
  cacheHit,
}: TrendFiltersProps) {
  return (
    <div className="flex items-center gap-3 flex-wrap">
      <div className="flex items-center gap-2">
        <label className="text-sm font-medium text-gray-600">카테고리:</label>
        <select
          value={filters.category}
          onChange={(e) =>
            onFilterChange({ ...filters, category: e.target.value as TrendCategory })
          }
          className="px-3 py-1.5 text-sm border border-gray-300 rounded-lg bg-white focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
        >
          {CATEGORIES.map((cat) => (
            <option key={cat.value} value={cat.value}>
              {cat.label}
            </option>
          ))}
        </select>
      </div>

      <div className="flex items-center gap-2">
        <label className="text-sm font-medium text-gray-600">기간:</label>
        <select
          value={filters.time_range}
          onChange={(e) =>
            onFilterChange({ ...filters, time_range: e.target.value as TimeRange })
          }
          className="px-3 py-1.5 text-sm border border-gray-300 rounded-lg bg-white focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
        >
          {TIME_RANGES.map((range) => (
            <option key={range.value} value={range.value}>
              {range.label}
            </option>
          ))}
        </select>
      </div>

      <button
        onClick={onRefresh}
        disabled={loading}
        className="px-3 py-1.5 text-sm font-medium text-blue-600 border border-blue-300 rounded-lg hover:bg-blue-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        {loading ? '조회 중...' : '새로고침'}
      </button>

      {cacheHit && (
        <span className="text-xs text-gray-400">캐시 응답</span>
      )}
    </div>
  )
}
