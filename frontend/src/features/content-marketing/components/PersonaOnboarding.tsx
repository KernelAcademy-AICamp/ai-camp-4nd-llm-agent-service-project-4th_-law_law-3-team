'use client'

import { useCallback, useState } from 'react'
import type {
  ChannelStyle,
  OnboardingState,
  OnboardingStep,
  PersonaTone,
  TargetAudience,
  TrendCategory,
} from '../types'

const SPECIALTY_OPTIONS: { value: TrendCategory; label: string }[] = [
  { value: 'criminal', label: '형사' },
  { value: 'civil', label: '민사' },
  { value: 'labor', label: '노동' },
  { value: 'family', label: '가사' },
  { value: 'administrative', label: '행정' },
  { value: 'corporate', label: '기업' },
  { value: 'ip', label: '지식재산' },
]

const AUDIENCE_OPTIONS: { value: TargetAudience; label: string; description: string }[] = [
  { value: 'general_public', label: '일반 대중', description: '법률 지식이 적은 일반인' },
  { value: 'business', label: '기업/사업자', description: '경영·사업 관련 법률에 관심' },
  { value: 'legal_student', label: '법학 학생', description: '법률 공부 중인 학생' },
  { value: 'legal_professional', label: '법률 전문가', description: '변호사, 법무사 등 전문직' },
]

const TONE_OPTIONS: { value: PersonaTone; label: string; description: string }[] = [
  { value: 'professional', label: '전문가', description: '경어체, 신뢰감 있는 톤' },
  { value: 'casual', label: '친근한', description: '쉬운 비유, 편안한 톤' },
  { value: 'storytelling', label: '스토리텔링', description: '극적 구성, 흥미 위주' },
  { value: 'educational', label: '교육형', description: '단계별, 체계적 설명' },
]

const CHANNEL_OPTIONS: { value: ChannelStyle; label: string }[] = [
  { value: 'expert', label: '전문가 해설' },
  { value: 'casual_friendly', label: '친근한 유튜버' },
  { value: 'storytelling', label: '스토리텔링' },
  { value: 'lecture', label: '강의형' },
]

interface PersonaOnboardingProps {
  onComplete: (state: OnboardingState) => Promise<void>
  onBack: () => void
  isLoading: boolean
}

const INITIAL_STATE: OnboardingState = {
  step: 1,
  specialty_areas: [],
  target_audience: null,
  preferred_tone: null,
  channel_style: null,
  focus_topics: [],
}

export function PersonaOnboarding({ onComplete, onBack, isLoading }: PersonaOnboardingProps) {
  const [state, setState] = useState<OnboardingState>(INITIAL_STATE)
  const [topicInput, setTopicInput] = useState('')

  const setStep = useCallback((step: OnboardingStep) => {
    setState((prev) => ({ ...prev, step }))
  }, [])

  const canProceed = (): boolean => {
    switch (state.step) {
      case 1: return state.specialty_areas.length >= 1 && state.specialty_areas.length <= 3
      case 2: return state.target_audience !== null
      case 3: return state.preferred_tone !== null
      case 4: return true
    }
  }

  const handleNext = useCallback(() => {
    if (state.step < 4) {
      setStep((state.step + 1) as OnboardingStep)
    } else {
      onComplete(state)
    }
  }, [state, setStep, onComplete])

  const handlePrev = useCallback(() => {
    if (state.step > 1) {
      setStep((state.step - 1) as OnboardingStep)
    } else {
      onBack()
    }
  }, [state.step, setStep, onBack])

  const toggleSpecialty = useCallback((value: TrendCategory) => {
    setState((prev) => {
      const has = prev.specialty_areas.includes(value)
      if (has) {
        return { ...prev, specialty_areas: prev.specialty_areas.filter((v) => v !== value) }
      }
      if (prev.specialty_areas.length >= 3) return prev
      return { ...prev, specialty_areas: [...prev.specialty_areas, value] }
    })
  }, [])

  const addTopic = useCallback(() => {
    const trimmed = topicInput.trim()
    if (!trimmed || state.focus_topics.includes(trimmed)) return
    setState((prev) => ({ ...prev, focus_topics: [...prev.focus_topics, trimmed] }))
    setTopicInput('')
  }, [topicInput, state.focus_topics])

  const removeTopic = useCallback((topic: string) => {
    setState((prev) => ({
      ...prev,
      focus_topics: prev.focus_topics.filter((t) => t !== topic),
    }))
  }, [])

  return (
    <div className="max-w-lg mx-auto">
      {/* 스텝 인디케이터 */}
      <div className="flex items-center gap-2 mb-6">
        {[1, 2, 3, 4].map((s) => (
          <div
            key={s}
            className={`h-1.5 flex-1 rounded-full transition-colors ${
              s <= state.step ? 'bg-blue-500' : 'bg-gray-200'
            }`}
          />
        ))}
      </div>

      {/* Step 1: 전문분야 */}
      {state.step === 1 && (
        <div className="space-y-4">
          <div>
            <h2 className="text-lg font-bold text-gray-900">전문분야 선택</h2>
            <p className="text-sm text-gray-500 mt-1">관심 있는 법률 분야를 1~3개 선택하세요.</p>
          </div>
          <div className="grid grid-cols-2 gap-2">
            {SPECIALTY_OPTIONS.map((opt) => {
              const isSelected = state.specialty_areas.includes(opt.value)
              return (
                <button
                  key={opt.value}
                  onClick={() => toggleSpecialty(opt.value)}
                  className={`px-4 py-3 text-sm rounded-lg border transition-colors text-left ${
                    isSelected
                      ? 'border-blue-500 bg-blue-50 text-blue-700 font-medium'
                      : 'border-gray-200 text-gray-700 hover:border-gray-300'
                  }`}
                >
                  {opt.label}
                </button>
              )
            })}
          </div>
          <p className="text-xs text-gray-400">
            {state.specialty_areas.length}/3 선택됨
          </p>
        </div>
      )}

      {/* Step 2: 타겟 독자 */}
      {state.step === 2 && (
        <div className="space-y-4">
          <div>
            <h2 className="text-lg font-bold text-gray-900">타겟 독자</h2>
            <p className="text-sm text-gray-500 mt-1">주요 독자층을 선택하세요.</p>
          </div>
          <div className="space-y-2">
            {AUDIENCE_OPTIONS.map((opt) => (
              <button
                key={opt.value}
                onClick={() => setState((prev) => ({ ...prev, target_audience: opt.value }))}
                className={`w-full flex flex-col px-4 py-3 text-left rounded-lg border transition-colors ${
                  state.target_audience === opt.value
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
              >
                <span className={`text-sm font-medium ${
                  state.target_audience === opt.value ? 'text-blue-700' : 'text-gray-700'
                }`}>
                  {opt.label}
                </span>
                <span className="text-xs text-gray-500 mt-0.5">{opt.description}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Step 3: 톤 + 채널 스타일 */}
      {state.step === 3 && (
        <div className="space-y-5">
          <div>
            <h2 className="text-lg font-bold text-gray-900">톤 & 채널 스타일</h2>
            <p className="text-sm text-gray-500 mt-1">콘텐츠 톤과 채널 스타일을 선택하세요.</p>
          </div>
          <div>
            <p className="text-xs font-medium text-gray-500 mb-2">톤</p>
            <div className="grid grid-cols-2 gap-2">
              {TONE_OPTIONS.map((opt) => (
                <button
                  key={opt.value}
                  onClick={() => setState((prev) => ({ ...prev, preferred_tone: opt.value }))}
                  className={`flex flex-col px-3 py-2.5 text-left rounded-lg border transition-colors ${
                    state.preferred_tone === opt.value
                      ? 'border-blue-500 bg-blue-50'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                >
                  <span className={`text-sm font-medium ${
                    state.preferred_tone === opt.value ? 'text-blue-700' : 'text-gray-700'
                  }`}>
                    {opt.label}
                  </span>
                  <span className="text-xs text-gray-500">{opt.description}</span>
                </button>
              ))}
            </div>
          </div>
          <div>
            <p className="text-xs font-medium text-gray-500 mb-2">채널 스타일 (선택)</p>
            <div className="flex flex-wrap gap-2">
              {CHANNEL_OPTIONS.map((opt) => (
                <button
                  key={opt.value}
                  onClick={() =>
                    setState((prev) => ({
                      ...prev,
                      channel_style: prev.channel_style === opt.value ? null : opt.value,
                    }))
                  }
                  className={`px-3 py-1.5 text-sm rounded-lg border transition-colors ${
                    state.channel_style === opt.value
                      ? 'border-blue-500 bg-blue-50 text-blue-700'
                      : 'border-gray-200 text-gray-600 hover:border-gray-300'
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Step 4: 관심 주제 */}
      {state.step === 4 && (
        <div className="space-y-4">
          <div>
            <h2 className="text-lg font-bold text-gray-900">관심 주제</h2>
            <p className="text-sm text-gray-500 mt-1">
              자주 다루고 싶은 키워드를 추가하세요 (선택).
            </p>
          </div>
          <div className="flex gap-2">
            <input
              value={topicInput}
              onChange={(e) => setTopicInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && addTopic()}
              placeholder="예: 부동산 사기, 이혼 재산분할"
              className="flex-1 px-3 py-2 text-sm border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
            <button
              onClick={addTopic}
              disabled={!topicInput.trim()}
              className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors"
            >
              추가
            </button>
          </div>
          {state.focus_topics.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {state.focus_topics.map((topic) => (
                <span
                  key={topic}
                  className="inline-flex items-center gap-1 px-3 py-1 text-sm bg-gray-100 text-gray-700 rounded-full"
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
      )}

      {/* 네비게이션 버튼 */}
      <div className="flex items-center justify-between mt-8">
        <button
          onClick={handlePrev}
          className="px-4 py-2 text-sm font-medium text-gray-600 hover:text-gray-800 transition-colors"
        >
          {state.step === 1 ? '취소' : '이전'}
        </button>
        <button
          onClick={handleNext}
          disabled={!canProceed() || isLoading}
          className="px-6 py-2.5 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {isLoading ? '저장 중...' : state.step === 4 ? '완료' : '다음'}
        </button>
      </div>
    </div>
  )
}
