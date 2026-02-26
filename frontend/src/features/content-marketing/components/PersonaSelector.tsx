'use client'

import type { PersonaTone, PersonaType, ScriptDuration } from '../types'

interface PersonaSelectorProps {
  persona: PersonaType
  duration: ScriptDuration
  tone: PersonaTone
  onPersonaChange: (persona: PersonaType) => void
  onDurationChange: (duration: ScriptDuration) => void
  onToneChange: (tone: PersonaTone) => void
  disabled: boolean
}

const TONES: { value: PersonaTone; label: string; description: string }[] = [
  { value: 'professional', label: '전문가', description: '경어체, 신뢰감' },
  { value: 'casual', label: '친근한', description: '쉬운 비유, 편안함' },
  { value: 'storytelling', label: '스토리텔링', description: '극적 구성, 흥미' },
  { value: 'educational', label: '교육형', description: '단계별, 정확함' },
]

const DURATIONS: { value: ScriptDuration; label: string }[] = [
  { value: 5, label: '5분' },
  { value: 10, label: '10분' },
  { value: 15, label: '15분' },
]

export function PersonaSelector({
  persona,
  duration,
  tone,
  onPersonaChange,
  onDurationChange,
  onToneChange,
  disabled,
}: PersonaSelectorProps) {
  return (
    <div className="space-y-3">
      {/* 톤 선택 (v2.0: 4가지 PersonaTone) */}
      <div className="flex items-start gap-3">
        <span className="text-sm font-medium text-gray-600 mt-1 shrink-0">톤:</span>
        <div className="flex flex-wrap gap-2">
          {TONES.map((t) => (
            <label
              key={t.value}
              className={`flex flex-col px-3 py-1.5 rounded-lg border cursor-pointer transition-colors ${
                tone === t.value
                  ? 'border-blue-500 bg-blue-50 text-blue-700'
                  : 'border-gray-200 bg-white text-gray-700 hover:border-gray-300'
              } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              <input
                type="radio"
                name="tone"
                value={t.value}
                checked={tone === t.value}
                onChange={() => {
                  onToneChange(t.value)
                  // PersonaTone → PersonaType 동기화 (백엔드 호환)
                  if (t.value === 'professional' || t.value === 'educational') {
                    onPersonaChange('professional')
                  } else {
                    onPersonaChange('casual')
                  }
                }}
                disabled={disabled}
                className="sr-only"
              />
              <span className="text-sm font-medium">{t.label}</span>
              <span className="text-xs text-gray-500">{t.description}</span>
            </label>
          ))}
        </div>
      </div>

      {/* 길이 선택 */}
      <div className="flex items-center gap-3">
        <span className="text-sm font-medium text-gray-600">길이:</span>
        {DURATIONS.map((d) => (
          <label key={d.value} className="flex items-center gap-1.5 cursor-pointer">
            <input
              type="radio"
              name="duration"
              value={d.value}
              checked={duration === d.value}
              onChange={() => onDurationChange(d.value)}
              disabled={disabled}
              className="text-blue-600 focus:ring-blue-500"
            />
            <span className="text-sm text-gray-700">{d.label}</span>
          </label>
        ))}
      </div>
    </div>
  )
}
