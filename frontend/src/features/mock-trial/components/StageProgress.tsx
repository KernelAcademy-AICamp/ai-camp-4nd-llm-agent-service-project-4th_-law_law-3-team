'use client'

import type { StageInfo } from '../types'

interface StageProgressProps {
  stages: StageInfo[]
  currentStageId: string
}

export function StageProgress({ stages, currentStageId }: StageProgressProps) {
  const currentIndex = stages.findIndex((stage) => stage.id === currentStageId)

  return (
    <div className="flex items-center gap-1 px-4 py-2 bg-white border-b border-gray-100">
      {stages.map((stage, index) => {
        const isCompleted = index < currentIndex
        const isCurrent = index === currentIndex

        return (
          <div key={stage.id} className="flex items-center">
            {index > 0 && (
              <div
                className={`w-4 h-0.5 ${
                  isCompleted ? 'bg-green-400' : 'bg-gray-200'
                }`}
              />
            )}
            <div className="flex flex-col items-center" title={stage.description}>
              <div
                className={`w-5 h-5 rounded-full flex items-center justify-center text-xs ${
                  isCompleted
                    ? 'bg-green-500 text-white'
                    : isCurrent
                      ? 'bg-blue-500 text-white ring-2 ring-blue-200'
                      : 'bg-gray-200 text-gray-400'
                }`}
              >
                {isCompleted ? '\u2713' : stage.order}
              </div>
              <span
                className={`text-[10px] mt-0.5 whitespace-nowrap ${
                  isCurrent
                    ? 'text-blue-600 font-semibold'
                    : isCompleted
                      ? 'text-green-600'
                      : 'text-gray-400'
                }`}
              >
                {stage.name}
              </span>
            </div>
          </div>
        )
      })}
    </div>
  )
}
