'use client'

import type { DisputeType } from '../types'
import { DISPUTE_TYPE_OPTIONS } from '../types'

interface DisputeTypeStepProps {
  selectedType: DisputeType | null
  onSelect: (type: DisputeType) => void
  onNext: () => void
}

export function DisputeTypeStep({ selectedType, onSelect, onNext }: DisputeTypeStepProps) {
  return (
    <div className="max-w-3xl mx-auto">
      <div className="text-center mb-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">분쟁 유형 선택</h2>
        <p className="text-gray-600">어떤 종류의 분쟁으로 소액소송을 진행하시나요?</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
        {DISPUTE_TYPE_OPTIONS.map((option) => (
          <button
            key={option.id}
            onClick={() => onSelect(option.id)}
            className={`p-6 rounded-xl border-2 text-left transition-all ${
              selectedType === option.id
                ? 'border-blue-500 bg-blue-50 shadow-md'
                : 'border-gray-200 bg-white hover:border-gray-300 hover:shadow-sm'
            }`}
          >
            <div className="text-3xl mb-3">{option.icon}</div>
            <h3 className="font-semibold text-gray-900 mb-1">{option.name}</h3>
            <p className="text-sm text-gray-500">{option.description}</p>
          </button>
        ))}
      </div>

      <div className="flex justify-end">
        <button
          onClick={onNext}
          disabled={!selectedType}
          className="px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition"
        >
          다음 단계
        </button>
      </div>
    </div>
  )
}
