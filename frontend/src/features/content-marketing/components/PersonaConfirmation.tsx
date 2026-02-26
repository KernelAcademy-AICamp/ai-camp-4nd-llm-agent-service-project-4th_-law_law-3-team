'use client'

import type { LawyerPersona, TrendCategory } from '../types'

const CATEGORY_LABELS: Record<TrendCategory, string> = {
  all: '전체',
  criminal: '형사',
  civil: '민사',
  labor: '노동',
  family: '가사',
  administrative: '행정',
  corporate: '기업',
  ip: '지식재산',
}

const TONE_LABELS: Record<string, string> = {
  professional: '전문가',
  casual: '친근한',
  storytelling: '스토리텔링',
  educational: '교육형',
}

const AUDIENCE_LABELS: Record<string, string> = {
  general_public: '일반 대중',
  business: '기업/사업자',
  legal_student: '법학 학생',
  legal_professional: '법률 전문가',
}

interface PersonaConfirmationProps {
  persona: LawyerPersona
  onApprove: () => void
  onEdit: () => void
  onSwitchTrack: () => void
}

export function PersonaConfirmation({
  persona,
  onApprove,
  onEdit,
  onSwitchTrack,
}: PersonaConfirmationProps) {
  const confidencePercent = Math.round(persona.confidence * 100)
  const confidenceColor =
    confidencePercent >= 70
      ? 'text-green-600 bg-green-50'
      : confidencePercent >= 40
        ? 'text-yellow-600 bg-yellow-50'
        : 'text-red-600 bg-red-50'

  return (
    <div className="max-w-lg mx-auto space-y-6">
      <div className="text-center">
        <h2 className="text-lg font-bold text-gray-900">페르소나 분석 완료</h2>
        <p className="text-sm text-gray-500 mt-1">
          AI가 분석한 결과를 확인하고 승인해주세요.
        </p>
      </div>

      {/* 신뢰도 */}
      <div className="flex items-center justify-center gap-2">
        <span className="text-sm text-gray-600">분석 신뢰도:</span>
        <span className={`px-3 py-1 text-sm font-bold rounded-full ${confidenceColor}`}>
          {confidencePercent}%
        </span>
      </div>

      {/* 요약 카드 */}
      <div className="bg-white rounded-xl border border-gray-200 divide-y divide-gray-100">
        <div className="px-5 py-3">
          <p className="text-xs font-medium text-gray-500 mb-1.5">전문분야</p>
          <div className="flex flex-wrap gap-1.5">
            {persona.specialty_areas.map((area) => (
              <span
                key={area}
                className="px-2.5 py-0.5 text-sm bg-blue-50 text-blue-700 rounded-full"
              >
                {CATEGORY_LABELS[area] ?? area}
              </span>
            ))}
          </div>
        </div>

        <div className="px-5 py-3">
          <p className="text-xs font-medium text-gray-500 mb-1">타겟 독자</p>
          <p className="text-sm text-gray-700">
            {AUDIENCE_LABELS[persona.target_audience] ?? persona.target_audience}
          </p>
        </div>

        <div className="px-5 py-3">
          <p className="text-xs font-medium text-gray-500 mb-1">톤</p>
          <p className="text-sm text-gray-700">
            {TONE_LABELS[persona.preferred_tone] ?? persona.preferred_tone}
          </p>
        </div>

        {persona.focus_topics.length > 0 && (
          <div className="px-5 py-3">
            <p className="text-xs font-medium text-gray-500 mb-1.5">관심 주제</p>
            <div className="flex flex-wrap gap-1.5">
              {persona.focus_topics.map((topic) => (
                <span
                  key={topic}
                  className="px-2.5 py-0.5 text-sm bg-gray-100 text-gray-700 rounded-full"
                >
                  {topic}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* 액션 버튼 */}
      <div className="space-y-2">
        <button
          onClick={onApprove}
          className="w-full py-3 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 transition-colors"
        >
          이 페르소나로 시작하기
        </button>
        <div className="flex gap-2">
          <button
            onClick={onEdit}
            className="flex-1 py-2.5 text-sm font-medium text-gray-700 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
          >
            일부 수정
          </button>
          <button
            onClick={onSwitchTrack}
            className="flex-1 py-2.5 text-sm font-medium text-gray-700 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
          >
            직접 설정하기
          </button>
        </div>
      </div>
    </div>
  )
}
