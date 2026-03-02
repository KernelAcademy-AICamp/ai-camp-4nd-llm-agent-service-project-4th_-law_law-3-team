'use client'

import { useCallback, useState } from 'react'
import type {
  AnalysisInsights,
  LawyerPersona,
  PersonaTone,
  PersonaUpdateRequest,
  TargetAudience,
  TrendCategory,
} from '../types'
import { SPECIALTY_KEYWORDS, TONE_PREVIEW_TEXT } from '../types'

const CATEGORY_LABELS: Record<string, string> = {
  criminal: '형사',
  civil: '민사',
  labor: '노동',
  family: '가사',
  administrative: '행정',
  corporate: '기업',
  ip: '지식재산',
}

const AUDIENCE_LABELS: Record<TargetAudience, string> = {
  general_public: '일반 대중',
  business: '기업/사업자',
  legal_student: '법학 학생',
  legal_professional: '법률 전문가',
}

const TONE_LABELS: Record<PersonaTone, string> = {
  professional: '전문가',
  casual: '친근한',
  storytelling: '스토리텔링',
  educational: '교육형',
}

interface PersonaHybridReviewProps {
  persona: LawyerPersona
  insights: AnalysisInsights
  onConfirm: (modified: PersonaUpdateRequest) => void
  onSwitchToManual: () => void
  isSaving: boolean
}

export function PersonaHybridReview({
  persona,
  insights,
  onConfirm,
  onSwitchToManual,
  isSaving,
}: PersonaHybridReviewProps) {
  const [localEdits, setLocalEdits] = useState<PersonaUpdateRequest>({})
  const [topicInput, setTopicInput] = useState('')

  // 현재 값: 로컬 편집 > AI 원본
  const currentAreas = localEdits.specialty_areas ?? persona.specialty_areas.filter((a) => a !== 'all')
  const currentAudience = localEdits.target_audience ?? persona.target_audience
  const currentTone = localEdits.preferred_tone ?? persona.preferred_tone
  const currentTopics = localEdits.focus_topics ?? persona.focus_topics

  const isModified = (field: keyof PersonaUpdateRequest): boolean => field in localEdits

  // 전문분야 토글
  const toggleArea = useCallback(
    (area: TrendCategory) => {
      const current = localEdits.specialty_areas ?? persona.specialty_areas.filter((a) => a !== 'all')
      const updated = current.includes(area)
        ? current.filter((a) => a !== area)
        : current.length < 3
          ? [...current, area]
          : current
      setLocalEdits((prev) => ({ ...prev, specialty_areas: updated }))
    },
    [localEdits.specialty_areas, persona.specialty_areas],
  )

  // 타겟 독자 선택
  const selectAudience = useCallback((value: TargetAudience) => {
    setLocalEdits((prev) => ({ ...prev, target_audience: value }))
  }, [])

  // 톤 선택
  const selectTone = useCallback((value: PersonaTone) => {
    setLocalEdits((prev) => ({ ...prev, preferred_tone: value }))
  }, [])

  // 관심 주제 추가/삭제
  const addTopic = useCallback(() => {
    const trimmed = topicInput.trim()
    if (!trimmed || currentTopics.includes(trimmed)) return
    setLocalEdits((prev) => ({
      ...prev,
      focus_topics: [...(prev.focus_topics ?? persona.focus_topics), trimmed],
    }))
    setTopicInput('')
  }, [topicInput, currentTopics, persona.focus_topics])

  const removeTopic = useCallback(
    (topic: string) => {
      setLocalEdits((prev) => ({
        ...prev,
        focus_topics: (prev.focus_topics ?? persona.focus_topics).filter((t) => t !== topic),
      }))
    },
    [persona.focus_topics],
  )

  // 신뢰도 색상
  const confidencePercent = Math.round(persona.confidence * 100)
  const confidenceColor =
    confidencePercent >= 70
      ? 'bg-green-500'
      : confidencePercent >= 40
        ? 'bg-yellow-500'
        : 'bg-red-500'

  const handleConfirm = () => {
    onConfirm(localEdits)
  }

  return (
    <div className="max-w-2xl mx-auto py-6 px-4">
      {/* 헤더: 분석 결과 + 신뢰도 */}
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-lg font-bold text-slate-800">AI 분석 결과</h2>
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500">신뢰도</span>
          <div className="w-20 h-2 bg-gray-200 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full ${confidenceColor}`}
              style={{ width: `${confidencePercent}%` }}
            />
          </div>
          <span className="text-xs font-medium text-gray-600">{confidencePercent}%</span>
        </div>
      </div>

      {/* 전문분야 분석 */}
      <section className="mb-6">
        <div className="flex items-center gap-2 mb-3">
          <h3 className="text-sm font-semibold text-gray-700">전문분야 분석</h3>
          {isModified('specialty_areas') && (
            <span className="text-xs px-1.5 py-0.5 bg-amber-100 text-amber-700 rounded">수정됨</span>
          )}
        </div>

        {/* 분야별 스코어 바 */}
        {Object.entries(insights.area_scores)
          .sort(([, a], [, b]) => b - a)
          .slice(0, 5)
          .map(([area, score]) => (
            <div key={area} className="flex items-center gap-3 mb-1.5">
              <span className="text-xs text-gray-600 w-12 text-right">
                {CATEGORY_LABELS[area] ?? area}
              </span>
              <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue-500 rounded-full"
                  style={{ width: `${Math.round(score * 100)}%` }}
                />
              </div>
              <span className="text-xs text-gray-400 w-10">{Math.round(score * 100)}%</span>
            </div>
          ))}

        {/* 분야 칩 선택 */}
        <div className="flex flex-wrap gap-2 mt-3">
          {Object.entries(CATEGORY_LABELS).map(([key, label]) => {
            const isSelected = currentAreas.includes(key as TrendCategory)
            return (
              <button
                key={key}
                onClick={() => toggleArea(key as TrendCategory)}
                className={`px-3 py-1.5 text-xs rounded-full border transition-colors ${
                  isSelected
                    ? 'border-blue-500 bg-blue-50 text-blue-700 font-medium'
                    : 'border-gray-200 text-gray-500 hover:border-gray-300'
                }`}
              >
                {label} {isSelected && '✓'}
              </button>
            )
          })}
        </div>

        {/* 분석 근거 */}
        {insights.summary && (
          <p className="text-xs text-gray-500 mt-3 bg-gray-50 rounded-lg px-3 py-2">
            {insights.summary}
            <span className="text-gray-400 ml-1">
              ({insights.total_conversations_analyzed}건 분석)
            </span>
          </p>
        )}
      </section>

      {/* 타겟 독자 */}
      <section className="mb-6">
        <div className="flex items-center gap-2 mb-3">
          <h3 className="text-sm font-semibold text-gray-700">타겟 독자</h3>
          {isModified('target_audience') && (
            <span className="text-xs px-1.5 py-0.5 bg-amber-100 text-amber-700 rounded">수정됨</span>
          )}
        </div>
        <div className="flex flex-wrap gap-2">
          {(Object.entries(AUDIENCE_LABELS) as [TargetAudience, string][]).map(([value, label]) => (
            <button
              key={value}
              onClick={() => selectAudience(value)}
              className={`px-3 py-1.5 text-xs rounded-full border transition-colors ${
                currentAudience === value
                  ? 'border-blue-500 bg-blue-50 text-blue-700 font-medium'
                  : 'border-gray-200 text-gray-500 hover:border-gray-300'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </section>

      {/* 콘텐츠 톤 */}
      <section className="mb-6">
        <div className="flex items-center gap-2 mb-3">
          <h3 className="text-sm font-semibold text-gray-700">콘텐츠 톤</h3>
          {isModified('preferred_tone') && (
            <span className="text-xs px-1.5 py-0.5 bg-amber-100 text-amber-700 rounded">수정됨</span>
          )}
        </div>
        <div className="flex flex-wrap gap-2">
          {(Object.entries(TONE_LABELS) as [PersonaTone, string][]).map(([value, label]) => (
            <button
              key={value}
              onClick={() => selectTone(value)}
              className={`px-3 py-1.5 text-xs rounded-full border transition-colors ${
                currentTone === value
                  ? 'border-blue-500 bg-blue-50 text-blue-700 font-medium'
                  : 'border-gray-200 text-gray-500 hover:border-gray-300'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
        {/* 톤 미리보기 */}
        <div className="mt-2 bg-gray-50 rounded-lg px-3 py-2">
          <p className="text-xs text-gray-400 mb-1">미리보기</p>
          <p className="text-xs text-gray-600 italic">
            &ldquo;{TONE_PREVIEW_TEXT[currentTone]}&rdquo;
          </p>
        </div>
      </section>

      {/* 관심 주제 */}
      <section className="mb-8">
        <div className="flex items-center gap-2 mb-3">
          <h3 className="text-sm font-semibold text-gray-700">관심 주제</h3>
          {isModified('focus_topics') && (
            <span className="text-xs px-1.5 py-0.5 bg-amber-100 text-amber-700 rounded">수정됨</span>
          )}
        </div>

        {/* 현재 주제 태그 */}
        <div className="flex flex-wrap gap-2 mb-3">
          {currentTopics.map((topic) => (
            <span
              key={topic}
              className="inline-flex items-center gap-1 px-2.5 py-1 text-xs bg-gray-100 text-gray-700 rounded-full"
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

        {/* 추천 칩 (전문분야 기반) */}
        {currentAreas.length > 0 && (
          <div className="mb-3">
            <p className="text-xs text-gray-400 mb-1.5">추천 키워드</p>
            <div className="flex flex-wrap gap-1.5">
              {currentAreas
                .flatMap((area) => SPECIALTY_KEYWORDS[area] ?? [])
                .filter((kw) => !currentTopics.includes(kw))
                .slice(0, 8)
                .map((kw) => (
                  <button
                    key={kw}
                    onClick={() => {
                      setLocalEdits((prev) => ({
                        ...prev,
                        focus_topics: [...(prev.focus_topics ?? persona.focus_topics), kw],
                      }))
                    }}
                    className="px-2 py-0.5 text-xs text-blue-600 bg-blue-50 rounded-full hover:bg-blue-100 transition-colors"
                  >
                    + {kw}
                  </button>
                ))}
            </div>
          </div>
        )}

        {/* 자유 입력 */}
        <div className="flex gap-2">
          <input
            value={topicInput}
            onChange={(e) => setTopicInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && addTopic()}
            placeholder="주제 추가 (예: 음주운전)"
            className="flex-1 px-3 py-1.5 text-xs border border-gray-200 rounded-lg focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
          />
          <button
            onClick={addTopic}
            disabled={!topicInput.trim()}
            className="px-3 py-1.5 text-xs text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors"
          >
            추가
          </button>
        </div>
      </section>

      {/* 액션 버튼 */}
      <div className="flex items-center justify-between border-t border-gray-100 pt-4">
        <button
          onClick={onSwitchToManual}
          className="text-sm text-gray-500 hover:text-gray-700 transition-colors"
        >
          처음부터 직접 설정
        </button>
        <button
          onClick={handleConfirm}
          disabled={isSaving}
          className="px-6 py-2.5 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors"
        >
          {isSaving ? '저장 중...' : '이 설정으로 시작하기'}
        </button>
      </div>
    </div>
  )
}
