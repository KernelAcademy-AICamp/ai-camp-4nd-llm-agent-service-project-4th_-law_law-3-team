'use client'

import type { StageInfo } from '../hooks/useScript'

interface StageProgressProps {
  stageInfo: StageInfo | null
  isGenerating: boolean
}

export function StageProgress({ stageInfo, isGenerating }: StageProgressProps) {
  if (!stageInfo || !isGenerating) return null

  const isCompleted = stageInfo.status === 'completed'

  return (
    <div
      className={`flex items-center gap-3 rounded-lg px-4 py-3 text-sm ${
        isCompleted
          ? 'bg-green-50 text-green-700'
          : 'bg-blue-50 text-blue-700'
      }`}
    >
      {isCompleted ? (
        <svg className="w-4 h-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
        </svg>
      ) : (
        <svg
          className="w-4 h-4 shrink-0 animate-spin"
          xmlns="http://www.w3.org/2000/svg"
          fill="none"
          viewBox="0 0 24 24"
        >
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path
            className="opacity-75"
            fill="currentColor"
            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
          />
        </svg>
      )}

      <div className="flex-1 min-w-0">
        <span className="font-medium">{stageInfo.stage}</span>
        {stageInfo.detail && (
          <span className="ml-2 text-xs opacity-75">{stageInfo.detail}</span>
        )}
      </div>

      {isCompleted && (
        <span className="text-xs shrink-0">완료</span>
      )}
    </div>
  )
}
