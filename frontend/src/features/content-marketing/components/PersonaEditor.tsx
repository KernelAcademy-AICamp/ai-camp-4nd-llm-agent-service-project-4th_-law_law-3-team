'use client'

import { useCallback, useState } from 'react'
import type {
  ChannelStyle,
  LawyerPersona,
  PersonaTone,
  PersonaUpdateRequest,
  TargetAudience,
  TrendCategory,
} from '../types'

const CATEGORY_OPTIONS: { value: TrendCategory; label: string }[] = [
  { value: 'criminal', label: '형사' },
  { value: 'civil', label: '민사' },
  { value: 'labor', label: '노동' },
  { value: 'family', label: '가사' },
  { value: 'administrative', label: '행정' },
  { value: 'corporate', label: '기업' },
  { value: 'ip', label: '지식재산' },
]

const TONE_OPTIONS: { value: PersonaTone; label: string }[] = [
  { value: 'professional', label: '전문가' },
  { value: 'casual', label: '친근한' },
  { value: 'storytelling', label: '스토리텔링' },
  { value: 'educational', label: '교육형' },
]

const AUDIENCE_OPTIONS: { value: TargetAudience; label: string }[] = [
  { value: 'general_public', label: '일반 대중' },
  { value: 'business', label: '기업/사업자' },
  { value: 'legal_student', label: '법학 학생' },
  { value: 'legal_professional', label: '법률 전문가' },
]

const CHANNEL_OPTIONS: { value: ChannelStyle; label: string }[] = [
  { value: 'expert', label: '전문가 해설' },
  { value: 'casual_friendly', label: '친근한 유튜버' },
  { value: 'storytelling', label: '스토리텔링' },
  { value: 'lecture', label: '강의형' },
]

interface PersonaEditorProps {
  persona: LawyerPersona
  onSave: (request: PersonaUpdateRequest) => Promise<void>
  onClose: () => void
  isSaving: boolean
}

export function PersonaEditor({ persona, onSave, onClose, isSaving }: PersonaEditorProps) {
  const [specialtyAreas, setSpecialtyAreas] = useState<TrendCategory[]>(persona.specialty_areas)
  const [tone, setTone] = useState<PersonaTone>(persona.preferred_tone)
  const [audience, setAudience] = useState<TargetAudience>(persona.target_audience)
  const [channelStyle, setChannelStyle] = useState<ChannelStyle | null>(persona.channel_style)
  const [focusTopics, setFocusTopics] = useState<string[]>(persona.focus_topics)
  const [topicInput, setTopicInput] = useState('')

  const toggleSpecialty = useCallback((value: TrendCategory) => {
    setSpecialtyAreas((prev) => {
      if (prev.includes(value)) return prev.filter((v) => v !== value)
      if (prev.length >= 3) return prev
      return [...prev, value]
    })
  }, [])

  const addTopic = useCallback(() => {
    const trimmed = topicInput.trim()
    if (!trimmed || focusTopics.includes(trimmed)) return
    setFocusTopics((prev) => [...prev, trimmed])
    setTopicInput('')
  }, [topicInput, focusTopics])

  const removeTopic = useCallback((topic: string) => {
    setFocusTopics((prev) => prev.filter((t) => t !== topic))
  }, [])

  const handleSave = useCallback(async () => {
    await onSave({
      specialty_areas: specialtyAreas,
      preferred_tone: tone,
      target_audience: audience,
      channel_style: channelStyle,
      focus_topics: focusTopics,
    })
  }, [specialtyAreas, tone, audience, channelStyle, focusTopics, onSave])

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
      <div
        className="bg-white rounded-2xl shadow-xl w-full max-w-lg max-h-[85vh] flex flex-col m-4"
        onClick={(e) => e.stopPropagation()}
      >
        {/* 헤더 */}
        <div className="sticky top-0 bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between rounded-t-2xl">
          <h2 className="text-lg font-bold text-gray-900">페르소나 수정</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 text-xl leading-none"
          >
            &times;
          </button>
        </div>

        {/* 본문 */}
        <div className="flex-1 overflow-y-auto px-6 py-5 space-y-5">
          {/* 전문분야 */}
          <div>
            <p className="text-sm font-medium text-gray-700 mb-2">전문분야 (1~3개)</p>
            <div className="flex flex-wrap gap-2">
              {CATEGORY_OPTIONS.map((opt) => (
                <button
                  key={opt.value}
                  onClick={() => toggleSpecialty(opt.value)}
                  className={`px-3 py-1.5 text-sm rounded-lg border transition-colors ${
                    specialtyAreas.includes(opt.value)
                      ? 'border-blue-500 bg-blue-50 text-blue-700'
                      : 'border-gray-200 text-gray-600 hover:border-gray-300'
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>

          {/* 톤 */}
          <div>
            <p className="text-sm font-medium text-gray-700 mb-2">톤</p>
            <div className="flex flex-wrap gap-2">
              {TONE_OPTIONS.map((opt) => (
                <button
                  key={opt.value}
                  onClick={() => setTone(opt.value)}
                  className={`px-3 py-1.5 text-sm rounded-lg border transition-colors ${
                    tone === opt.value
                      ? 'border-blue-500 bg-blue-50 text-blue-700'
                      : 'border-gray-200 text-gray-600 hover:border-gray-300'
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>

          {/* 타겟 독자 */}
          <div>
            <p className="text-sm font-medium text-gray-700 mb-2">타겟 독자</p>
            <div className="flex flex-wrap gap-2">
              {AUDIENCE_OPTIONS.map((opt) => (
                <button
                  key={opt.value}
                  onClick={() => setAudience(opt.value)}
                  className={`px-3 py-1.5 text-sm rounded-lg border transition-colors ${
                    audience === opt.value
                      ? 'border-blue-500 bg-blue-50 text-blue-700'
                      : 'border-gray-200 text-gray-600 hover:border-gray-300'
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>

          {/* 채널 스타일 */}
          <div>
            <p className="text-sm font-medium text-gray-700 mb-2">채널 스타일 (선택)</p>
            <div className="flex flex-wrap gap-2">
              {CHANNEL_OPTIONS.map((opt) => (
                <button
                  key={opt.value}
                  onClick={() => setChannelStyle(channelStyle === opt.value ? null : opt.value)}
                  className={`px-3 py-1.5 text-sm rounded-lg border transition-colors ${
                    channelStyle === opt.value
                      ? 'border-blue-500 bg-blue-50 text-blue-700'
                      : 'border-gray-200 text-gray-600 hover:border-gray-300'
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>

          {/* 관심 주제 */}
          <div>
            <p className="text-sm font-medium text-gray-700 mb-2">관심 주제</p>
            <div className="flex gap-2 mb-2">
              <input
                value={topicInput}
                onChange={(e) => setTopicInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && addTopic()}
                placeholder="키워드 입력 후 Enter"
                className="flex-1 px-3 py-2 text-sm border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
              <button
                onClick={addTopic}
                disabled={!topicInput.trim()}
                className="px-3 py-2 text-sm font-medium text-blue-600 hover:text-blue-700 disabled:opacity-50"
              >
                추가
              </button>
            </div>
            {focusTopics.length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {focusTopics.map((topic) => (
                  <span
                    key={topic}
                    className="inline-flex items-center gap-1 px-2.5 py-0.5 text-sm bg-gray-100 text-gray-700 rounded-full"
                  >
                    {topic}
                    <button
                      onClick={() => removeTopic(topic)}
                      className="text-gray-400 hover:text-gray-600"
                    >
                      &times;
                    </button>
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* 푸터 */}
        <div className="sticky bottom-0 bg-white border-t border-gray-200 px-6 py-4 flex items-center justify-end gap-3 rounded-b-2xl">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm font-medium text-gray-600 hover:text-gray-800 transition-colors"
          >
            취소
          </button>
          <button
            onClick={handleSave}
            disabled={isSaving || specialtyAreas.length === 0}
            className="px-6 py-2.5 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors"
          >
            {isSaving ? '저장 중...' : '저장'}
          </button>
        </div>
      </div>
    </div>
  )
}
