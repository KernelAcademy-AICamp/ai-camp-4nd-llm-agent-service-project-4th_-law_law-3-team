'use client'

import type { DatePreset } from '../types'

interface LawFilterPanelProps {
  keyword: string
  onKeywordChange: (value: string) => void
  lawType: string
  onLawTypeChange: (value: string) => void
  ministry: string
  onMinistryChange: (value: string) => void
  promulgationPreset: DatePreset
  onPromulgationPresetChange: (value: DatePreset) => void
  promulgationFrom: string
  onPromulgationFromChange: (value: string) => void
  promulgationTo: string
  onPromulgationToChange: (value: string) => void
  enforcementPreset: DatePreset
  onEnforcementPresetChange: (value: DatePreset) => void
  enforcementFrom: string
  onEnforcementFromChange: (value: string) => void
  enforcementTo: string
  onEnforcementToChange: (value: string) => void
  lawTypes: string[]
  ministries: string[]
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

function DatePresetPills({
  preset,
  onPresetChange,
  dateFrom,
  onDateFromChange,
  dateTo,
  onDateToChange,
}: {
  preset: DatePreset
  onPresetChange: (value: DatePreset) => void
  dateFrom: string
  onDateFromChange: (value: string) => void
  dateTo: string
  onDateToChange: (value: string) => void
}) {
  return (
    <>
      <div className="flex gap-1.5 flex-wrap">
        {DATE_PRESETS.map(({ value, label }) => (
          <button
            key={value}
            onClick={() => onPresetChange(value)}
            className={`px-3 py-1.5 text-xs rounded-full border transition-colors ${
              preset === value
                ? 'bg-blue-600 text-white border-blue-600'
                : 'bg-white text-gray-600 border-gray-300 hover:border-gray-400'
            }`}
          >
            {label}
          </button>
        ))}
      </div>
      {preset === 'custom' && (
        <div className="flex gap-2 items-center mt-1.5">
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
    </>
  )
}

export function LawFilterPanel({
  keyword, onKeywordChange,
  lawType, onLawTypeChange,
  ministry, onMinistryChange,
  promulgationPreset, onPromulgationPresetChange,
  promulgationFrom, onPromulgationFromChange,
  promulgationTo, onPromulgationToChange,
  enforcementPreset, onEnforcementPresetChange,
  enforcementFrom, onEnforcementFromChange,
  enforcementTo, onEnforcementToChange,
  lawTypes, ministries,
  onSearch,
  isLoading,
}: LawFilterPanelProps) {
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
          placeholder="법령명 또는 키워드를 입력하세요"
          className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        />
      </div>

      {/* 법령유형 */}
      <div>
        <label className="block text-xs font-medium text-gray-600 mb-1">법령유형</label>
        <select
          value={lawType}
          onChange={(e) => onLawTypeChange(e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <option value="">전체</option>
          {lawTypes.map((type) => (
            <option key={type} value={type}>{type}</option>
          ))}
        </select>
      </div>

      {/* 소관부처 */}
      <div>
        <label className="block text-xs font-medium text-gray-600 mb-1">소관부처</label>
        <select
          value={ministry}
          onChange={(e) => onMinistryChange(e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <option value="">전체</option>
          {ministries.map((m) => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>
      </div>

      {/* 공포일자 */}
      <div>
        <label className="block text-xs font-medium text-gray-600 mb-1">공포일자</label>
        <DatePresetPills
          preset={promulgationPreset}
          onPresetChange={onPromulgationPresetChange}
          dateFrom={promulgationFrom}
          onDateFromChange={onPromulgationFromChange}
          dateTo={promulgationTo}
          onDateToChange={onPromulgationToChange}
        />
      </div>

      {/* 시행일자 */}
      <div>
        <label className="block text-xs font-medium text-gray-600 mb-1">시행일자</label>
        <DatePresetPills
          preset={enforcementPreset}
          onPresetChange={onEnforcementPresetChange}
          dateFrom={enforcementFrom}
          onDateFromChange={onEnforcementFromChange}
          dateTo={enforcementTo}
          onDateToChange={onEnforcementToChange}
        />
      </div>

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
