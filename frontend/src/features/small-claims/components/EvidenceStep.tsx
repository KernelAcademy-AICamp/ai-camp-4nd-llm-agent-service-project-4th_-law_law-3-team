'use client'

import type { EvidenceItem } from '../types'

interface EvidenceStepProps {
  items: EvidenceItem[]
  checkedItems: Set<string>
  isLoading: boolean
  onToggle: (id: string) => void
  onNext: () => void
  onPrevious: () => void
}

export function EvidenceStep({
  items,
  checkedItems,
  isLoading,
  onToggle,
  onNext,
  onPrevious,
}: EvidenceStepProps) {
  const requiredItems = items.filter((item) => item.required)
  const optionalItems = items.filter((item) => !item.required)

  const checkedRequiredCount = requiredItems.filter((item) => checkedItems.has(item.id)).length
  const allRequiredChecked = checkedRequiredCount === requiredItems.length

  if (isLoading) {
    return (
      <div className="max-w-2xl mx-auto">
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-600 mx-auto mb-3" />
            <p className="text-gray-500">증거 체크리스트를 불러오는 중...</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-2xl mx-auto">
      <div className="text-center mb-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">증거 체크리스트</h2>
        <p className="text-gray-600">소송에 필요한 증거 자료를 확인해주세요</p>
      </div>

      {/* Progress */}
      <div className="bg-blue-50 rounded-lg p-4 mb-6">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-blue-800">필수 증거 준비 현황</span>
          <span className="text-sm text-blue-600">
            {checkedRequiredCount} / {requiredItems.length}
          </span>
        </div>
        <div className="w-full bg-blue-200 rounded-full h-2">
          <div
            className="bg-blue-600 h-2 rounded-full transition-all duration-300"
            style={{
              width: `${requiredItems.length > 0 ? (checkedRequiredCount / requiredItems.length) * 100 : 0}%`,
            }}
          />
        </div>
      </div>

      {/* Required Evidence */}
      <div className="bg-white rounded-xl border border-gray-200 p-6 mb-4">
        <h3 className="font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <span className="text-red-500">*</span>
          필수 증거
        </h3>
        <div className="space-y-3">
          {requiredItems.map((item) => (
            <EvidenceCheckbox
              key={item.id}
              item={item}
              checked={checkedItems.has(item.id)}
              onToggle={() => onToggle(item.id)}
            />
          ))}
        </div>
      </div>

      {/* Optional Evidence */}
      {optionalItems.length > 0 && (
        <div className="bg-white rounded-xl border border-gray-200 p-6 mb-6">
          <h3 className="font-semibold text-gray-900 mb-4">권장 증거 (선택)</h3>
          <div className="space-y-3">
            {optionalItems.map((item) => (
              <EvidenceCheckbox
                key={item.id}
                item={item}
                checked={checkedItems.has(item.id)}
                onToggle={() => onToggle(item.id)}
              />
            ))}
          </div>
        </div>
      )}

      {/* Warning if not all required */}
      {!allRequiredChecked && (
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 mb-6">
          <div className="flex items-start gap-3">
            <svg
              className="w-5 h-5 text-amber-500 shrink-0 mt-0.5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
              />
            </svg>
            <div>
              <p className="text-sm font-medium text-amber-800">필수 증거가 부족합니다</p>
              <p className="text-sm text-amber-700 mt-1">
                필수 증거가 없어도 다음 단계로 진행할 수 있지만, 소송에서 불리할 수 있습니다.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Navigation */}
      <div className="flex justify-between">
        <button
          onClick={onPrevious}
          className="px-6 py-3 border border-gray-300 text-gray-700 rounded-lg font-medium hover:bg-gray-50 transition"
        >
          이전
        </button>
        <button
          onClick={onNext}
          className="px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition"
        >
          다음 단계
        </button>
      </div>
    </div>
  )
}

interface EvidenceCheckboxProps {
  item: EvidenceItem
  checked: boolean
  onToggle: () => void
}

function EvidenceCheckbox({ item, checked, onToggle }: EvidenceCheckboxProps) {
  return (
    <label
      className={`flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition ${
        checked ? 'border-blue-500 bg-blue-50' : 'border-gray-200 hover:border-gray-300'
      }`}
    >
      <input
        type="checkbox"
        checked={checked}
        onChange={onToggle}
        className="mt-1 h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
      />
      <div className="flex-1">
        <div className="flex items-center gap-2">
          <span className={`font-medium ${checked ? 'text-blue-800' : 'text-gray-900'}`}>
            {item.label}
          </span>
          {item.required && (
            <span className="px-1.5 py-0.5 text-xs bg-red-100 text-red-600 rounded">필수</span>
          )}
        </div>
        <p className="text-sm text-gray-500 mt-0.5">{item.description}</p>
      </div>
      {checked && (
        <svg className="w-5 h-5 text-blue-600 shrink-0" fill="currentColor" viewBox="0 0 20 20">
          <path
            fillRule="evenodd"
            d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
            clipRule="evenodd"
          />
        </svg>
      )}
    </label>
  )
}
