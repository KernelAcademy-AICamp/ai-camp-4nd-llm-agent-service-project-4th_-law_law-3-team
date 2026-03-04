'use client'

import type { DatePreset } from '../types'

interface FilterPanelProps {
  keyword: string
  onKeywordChange: (value: string) => void
  caseType: string
  onCaseTypeChange: (value: string) => void
  datePreset: DatePreset
  onDatePresetChange: (value: DatePreset) => void
  dateFrom: string
  onDateFromChange: (value: string) => void
  dateTo: string
  onDateToChange: (value: string) => void
  caseTypes: string[]
  onSearch: () => void
  isLoading: boolean
}

const DATE_PRESETS: { value: DatePreset; label: string }[] = [
  { value: 'all', label: '전체' },
  { value: '3y', label: '3년' },
  { value: '5y', label: '5년' },
  { value: '10y', label: '10년' },
  { value: 'custom', label: '직접입력' },
]


export function FilterPanel({
  keyword, onKeywordChange,
  caseType, onCaseTypeChange,
  datePreset, onDatePresetChange,
  dateFrom, onDateFromChange,
  dateTo, onDateToChange,
  caseTypes,
  onSearch,
  isLoading,
}: FilterPanelProps) {
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') onSearch()
  }

  return (
    <div className="border-b border-gray-200 px-4 py-3 space-y-3 bg-white">
      {/* 검색어 입력 */}
      <div>
        <input
          type="text"
          value={keyword}
          onChange={(e) => onKeywordChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="검색어를 입력하세요 (사건명, 판시사항, 사건번호, 판결요지)"
          className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        />
      </div>

      {/* 사건종류 */}
      <div>
        <label className="block text-xs font-medium text-gray-600 mb-1">사건종류</label>
        <select
          value={caseType}
          onChange={(e) => onCaseTypeChange(e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <option value="">전체</option>
          {caseTypes.map((type) => (
            <option key={type} value={type}>{type}</option>
          ))}
        </select>
      </div>

      {/* 기간 pill 버튼 */}
      <div>
        <label className="block text-xs font-medium text-gray-600 mb-1">기간</label>
        <div className="flex gap-1.5 flex-wrap">
          {DATE_PRESETS.map(({ value, label }) => (
            <button
              key={value}
              onClick={() => onDatePresetChange(value)}
              className={`px-3 py-1.5 text-xs rounded-full border transition-colors ${
                datePreset === value
                  ? 'bg-blue-600 text-white border-blue-600'
                  : 'bg-white text-gray-600 border-gray-300 hover:border-gray-400'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* 직접입력 연도 */}
      {datePreset === 'custom' && (
        <div className="flex gap-2 items-center">
          <input
            type="number"
            value={dateFrom}
            onChange={(e) => onDateFromChange(e.target.value)}
            placeholder="시작 연도"
            min={1947}
            max={new Date().getFullYear()}
            className="flex-1 px-2 py-1.5 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <span className="text-gray-400 text-sm">~</span>
          <input
            type="number"
            value={dateTo}
            onChange={(e) => onDateToChange(e.target.value)}
            placeholder="종료 연도"
            min={1947}
            max={new Date().getFullYear()}
            className="flex-1 px-2 py-1.5 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
      )}

      {/* 검색 버튼 */}
      <button
        onClick={onSearch}
        disabled={isLoading}
        className="w-full py-2 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        {isLoading ? '검색 중...' : '검색'}
      </button>
    </div>
  )
}
