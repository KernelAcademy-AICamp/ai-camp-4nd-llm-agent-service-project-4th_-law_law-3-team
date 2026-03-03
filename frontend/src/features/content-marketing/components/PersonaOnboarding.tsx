'use client'

import { useCallback, useMemo, useState } from 'react'
import type {
  ChannelStyle,
  OnboardingState,
  OnboardingStep,
  PersonaTone,
  TargetAudience,
  TrendCategory,
} from '../types'
import { SPECIALTY_KEYWORDS, TONE_PREVIEW_TEXT } from '../types'

// ── 상수 ──

const SPECIALTY_CARDS: { value: TrendCategory; label: string; icon: string; keywords: string[] }[] = [
  { value: 'criminal', label: '형사', icon: '⚖️', keywords: ['음주운전', '사기/횡령', '폭행/상해'] },
  { value: 'civil', label: '민사', icon: '📋', keywords: ['손해배상', '계약분쟁', '부동산'] },
  { value: 'labor', label: '노동', icon: '👷', keywords: ['부당해고', '산업재해', '임금체불'] },
  { value: 'family', label: '가사', icon: '👨‍👩‍👧', keywords: ['이혼', '양육권', '상속'] },
  { value: 'administrative', label: '행정', icon: '🏛️', keywords: ['행정소송', '인허가', '징계'] },
  { value: 'corporate', label: '기업', icon: '🏢', keywords: ['M&A', '공정거래', '지배구조'] },
  { value: 'ip', label: '지식재산', icon: '💡', keywords: ['특허침해', '상표등록', '저작권'] },
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

// 타겟 독자 → 추천 톤 매핑
const AUDIENCE_TONE_SUGGESTION: Record<TargetAudience, PersonaTone> = {
  general_public: 'casual',
  business: 'professional',
  legal_student: 'educational',
  legal_professional: 'professional',
}

interface PersonaOnboardingProps {
  onComplete: (state: OnboardingState) => Promise<void>
  onBack: () => void
  isLoading: boolean
  error?: string | null
  initialState?: Partial<OnboardingState> | null
}

const DEFAULT_STATE: OnboardingState = {
  step: 1,
  specialty_areas: [],
  target_audience: null,
  preferred_tone: null,
  channel_style: null,
  focus_topics: [],
}

export function PersonaOnboarding({
  onComplete,
  onBack,
  isLoading,
  error,
  initialState,
}: PersonaOnboardingProps) {
  const [state, setState] = useState<OnboardingState>(() => ({
    ...DEFAULT_STATE,
    ...initialState,
    step: 1,
  }))
  const [topicInput, setTopicInput] = useState('')

  const setStep = useCallback((step: OnboardingStep) => {
    setState((prev) => ({ ...prev, step }))
  }, [])

  const canProceed = (): boolean => {
    switch (state.step) {
      case 1:
        return state.specialty_areas.length >= 1 && state.specialty_areas.length <= 3
      case 2:
        return state.target_audience !== null && state.preferred_tone !== null
      case 3:
        return true
    }
  }

  const handleNext = useCallback(() => {
    if (state.step < 3) {
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

  // 전문분야 기반 추천 키워드
  const suggestedTopics = useMemo(() => {
    return state.specialty_areas
      .flatMap((area) => SPECIALTY_KEYWORDS[area] ?? [])
      .filter((kw) => !state.focus_topics.includes(kw))
      .slice(0, 10)
  }, [state.specialty_areas, state.focus_topics])

  // 추천 톤
  const suggestedTone = state.target_audience
    ? AUDIENCE_TONE_SUGGESTION[state.target_audience]
    : null

  return (
    <div className="max-w-lg mx-auto">
      {/* 스텝 인디케이터 (3스텝) */}
      <div className="flex items-center gap-2 mb-6">
        {[1, 2, 3].map((s) => (
          <div
            key={s}
            className={`h-1.5 flex-1 rounded-full transition-colors ${
              s <= state.step ? 'bg-blue-500' : 'bg-gray-200'
            }`}
          />
        ))}
      </div>

      {/* Step 1: 전문분야 (아이콘 카드 + 대표 키워드) */}
      {state.step === 1 && (
        <div className="space-y-4">
          <div>
            <h2 className="text-lg font-bold text-gray-900">전문분야 선택</h2>
            <p className="text-sm text-gray-500 mt-1">관심 있는 법률 분야를 1~3개 선택하세요.</p>
          </div>
          <div className="grid grid-cols-3 gap-2">
            {SPECIALTY_CARDS.map((card) => {
              const isSelected = state.specialty_areas.includes(card.value)
              return (
                <button
                  key={card.value}
                  onClick={() => toggleSpecialty(card.value)}
                  className={`relative flex flex-col items-center p-3 rounded-xl border-2 transition-all text-center ${
                    isSelected
                      ? 'border-blue-500 bg-blue-50 ring-2 ring-blue-500/20'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                >
                  {isSelected && (
                    <span className="absolute top-1.5 right-1.5">
                      <svg className="w-4 h-4 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                      </svg>
                    </span>
                  )}
                  <span className="text-2xl mb-1">{card.icon}</span>
                  <span className={`text-sm font-medium mb-1 ${isSelected ? 'text-blue-700' : 'text-gray-700'}`}>
                    {card.label}
                  </span>
                  <div className="space-y-0.5">
                    {card.keywords.map((kw) => (
                      <p key={kw} className="text-xs text-gray-400 leading-tight">{kw}</p>
                    ))}
                  </div>
                </button>
              )
            })}
          </div>
          <p className="text-xs text-gray-400">
            {state.specialty_areas.length}/3 선택됨
          </p>
        </div>
      )}

      {/* Step 2: 톤 + 타겟 통합 */}
      {state.step === 2 && (
        <div className="space-y-5">
          <div>
            <h2 className="text-lg font-bold text-gray-900">타겟 독자 & 톤</h2>
            <p className="text-sm text-gray-500 mt-1">독자층과 콘텐츠 톤을 선택하세요.</p>
          </div>

          {/* 타겟 독자 */}
          <div>
            <p className="text-xs font-medium text-gray-500 mb-2">타겟 독자</p>
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

          {/* 콘텐츠 톤 */}
          <div>
            <p className="text-xs font-medium text-gray-500 mb-2">콘텐츠 톤</p>
            <div className="grid grid-cols-2 gap-2">
              {TONE_OPTIONS.map((opt) => {
                const isSuggested = suggestedTone === opt.value && state.preferred_tone !== opt.value
                return (
                  <button
                    key={opt.value}
                    onClick={() => setState((prev) => ({ ...prev, preferred_tone: opt.value }))}
                    className={`relative flex flex-col px-3 py-2.5 text-left rounded-lg border transition-colors ${
                      state.preferred_tone === opt.value
                        ? 'border-blue-500 bg-blue-50'
                        : isSuggested
                          ? 'border-blue-200 bg-blue-50/50'
                          : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    {isSuggested && (
                      <span className="absolute -top-2 right-2 text-xs px-1.5 py-0.5 bg-blue-100 text-blue-600 rounded font-medium">
                        추천
                      </span>
                    )}
                    <span className={`text-sm font-medium ${
                      state.preferred_tone === opt.value ? 'text-blue-700' : 'text-gray-700'
                    }`}>
                      {opt.label}
                    </span>
                    <span className="text-xs text-gray-500">{opt.description}</span>
                  </button>
                )
              })}
            </div>
            {/* 톤 미리보기 */}
            {state.preferred_tone && (
              <div className="mt-2 bg-gray-50 rounded-lg px-3 py-2">
                <p className="text-xs text-gray-400 mb-1">미리보기</p>
                <p className="text-xs text-gray-600 italic">
                  &ldquo;{TONE_PREVIEW_TEXT[state.preferred_tone]}&rdquo;
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Step 3: 채널 스타일 + 관심 주제 통합 */}
      {state.step === 3 && (
        <div className="space-y-5">
          <div>
            <h2 className="text-lg font-bold text-gray-900">채널 & 관심 주제</h2>
            <p className="text-sm text-gray-500 mt-1">채널 스타일과 관심 주제를 설정하세요.</p>
          </div>

          {/* 채널 스타일 */}
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

          {/* 관심 주제 */}
          <div>
            <p className="text-xs font-medium text-gray-500 mb-2">관심 주제 (선택)</p>
            <div className="flex gap-2 mb-3">
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

            {/* 추천 키워드 칩 */}
            {suggestedTopics.length > 0 && (
              <div className="mb-3">
                <p className="text-xs text-gray-400 mb-1.5">추천 키워드</p>
                <div className="flex flex-wrap gap-1.5">
                  {suggestedTopics.map((kw) => (
                    <button
                      key={kw}
                      onClick={() =>
                        setState((prev) => ({
                          ...prev,
                          focus_topics: [...prev.focus_topics, kw],
                        }))
                      }
                      className="px-2 py-0.5 text-xs text-blue-600 bg-blue-50 rounded-full hover:bg-blue-100 transition-colors"
                    >
                      + {kw}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* 선택된 주제 태그 */}
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
        </div>
      )}

      {/* 에러 메시지 */}
      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 text-sm rounded-lg px-4 py-3 mt-4">
          {error}
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
          {isLoading ? '저장 중...' : state.step === 3 ? '완료' : '다음'}
        </button>
      </div>
    </div>
  )
}
