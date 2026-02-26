'use client'

import { useCallback } from 'react'
import { eventBus } from '@/features/mock-trial/game/EventBus'
import type { DialogueSpeed } from '@/features/mock-trial/types'

interface DialogueControlsProps {
  currentSpeed: DialogueSpeed
  onSpeedChange: (speed: DialogueSpeed) => void
}

const SPEED_OPTIONS: { speed: DialogueSpeed; label: string }[] = [
  { speed: 'normal', label: '▶ 1x' },
  { speed: 'fast', label: '▶▶ 2x' },
  { speed: 'faster', label: '▶▶▶ 4x' },
  { speed: 'instant', label: '⏩ 즉시' },
]

export function DialogueControls({ currentSpeed, onSpeedChange }: DialogueControlsProps) {
  const handleSpeedChange = useCallback(
    (speed: DialogueSpeed) => {
      onSpeedChange(speed)
      eventBus.emit('dialogue:set_speed', { speed })
    },
    [onSpeedChange]
  )

  const handleSkip = useCallback(() => {
    eventBus.emit('dialogue:skip', {} as Record<string, never>)
  }, [])

  return (
    <div className="flex items-center gap-1.5 px-3 py-1.5 bg-gray-50 border-t border-gray-200">
      {/* 속도 버튼 */}
      <div className="flex items-center gap-1">
        {SPEED_OPTIONS.map(({ speed, label }) => (
          <button
            key={speed}
            onClick={() => handleSpeedChange(speed)}
            className={`px-2 py-0.5 text-[11px] rounded transition-colors ${
              currentSpeed === speed
                ? 'bg-blue-500 text-white font-medium'
                : 'bg-gray-200 text-gray-600 hover:bg-gray-300'
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      <div className="flex-1" />

      {/* 스킵 버튼 */}
      <button
        onClick={handleSkip}
        className="px-2.5 py-0.5 text-[11px] bg-gray-200 text-gray-600 hover:bg-red-100 hover:text-red-600 rounded transition-colors"
      >
        스킵 ⏭
      </button>

      {/* 안내 */}
      <span className="text-[10px] text-gray-400 ml-1">
        Space: 다음
      </span>
    </div>
  )
}
