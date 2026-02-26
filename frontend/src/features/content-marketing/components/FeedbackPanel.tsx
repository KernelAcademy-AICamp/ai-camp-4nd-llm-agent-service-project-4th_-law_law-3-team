'use client'

import { useCallback, useState } from 'react'
import type { PersonaFeedbackRequest } from '../types'

type FeedbackType = 'tone_mismatch' | 'specialty_mismatch' | 'audience_mismatch' | 'other'

const FEEDBACK_TYPES: { value: FeedbackType; label: string }[] = [
  { value: 'tone_mismatch', label: '톤이 맞지 않음' },
  { value: 'specialty_mismatch', label: '전문분야가 다름' },
  { value: 'audience_mismatch', label: '타겟 독자 불일치' },
  { value: 'other', label: '기타' },
]

interface FeedbackPanelProps {
  personaId: string
  scriptId: string | null
  onSubmit: (request: PersonaFeedbackRequest) => Promise<void>
}

export function FeedbackPanel({ personaId, scriptId, onSubmit }: FeedbackPanelProps) {
  const [rating, setRating] = useState(0)
  const [hoverRating, setHoverRating] = useState(0)
  const [feedbackType, setFeedbackType] = useState<FeedbackType | null>(null)
  const [feedbackText, setFeedbackText] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [isSubmitted, setIsSubmitted] = useState(false)

  const handleSubmit = useCallback(async () => {
    if (rating === 0) return
    setIsSubmitting(true)
    try {
      await onSubmit({
        persona_id: personaId,
        script_id: scriptId,
        rating,
        feedback_type: feedbackType,
        feedback_text: feedbackText.trim() || null,
      })
      setIsSubmitted(true)
    } finally {
      setIsSubmitting(false)
    }
  }, [personaId, scriptId, rating, feedbackType, feedbackText, onSubmit])

  if (isSubmitted) {
    return (
      <div className="bg-green-50 border border-green-200 rounded-xl px-5 py-4 text-center">
        <p className="text-sm font-medium text-green-700">
          피드백을 보내주셔서 감사합니다!
        </p>
        <p className="text-xs text-green-600 mt-1">
          다음 대본 생성에 반영됩니다.
        </p>
      </div>
    )
  }

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-5 space-y-4">
      <h3 className="text-sm font-bold text-gray-800">대본 피드백</h3>

      {/* 별점 */}
      <div>
        <p className="text-xs font-medium text-gray-500 mb-2">만족도</p>
        <div className="flex gap-1">
          {[1, 2, 3, 4, 5].map((star) => (
            <button
              key={star}
              onClick={() => setRating(star)}
              onMouseEnter={() => setHoverRating(star)}
              onMouseLeave={() => setHoverRating(0)}
              className="text-2xl transition-colors"
              aria-label={`${star}점`}
            >
              <span className={
                star <= (hoverRating || rating)
                  ? 'text-yellow-400'
                  : 'text-gray-300'
              }>
                &#9733;
              </span>
            </button>
          ))}
          {rating > 0 && (
            <span className="ml-2 text-sm text-gray-500 self-center">{rating}점</span>
          )}
        </div>
      </div>

      {/* 피드백 타입 */}
      {rating > 0 && rating < 4 && (
        <div>
          <p className="text-xs font-medium text-gray-500 mb-2">개선 항목 (선택)</p>
          <div className="flex flex-wrap gap-2">
            {FEEDBACK_TYPES.map((ft) => (
              <button
                key={ft.value}
                onClick={() => setFeedbackType(feedbackType === ft.value ? null : ft.value)}
                className={`px-3 py-1.5 text-xs rounded-lg border transition-colors ${
                  feedbackType === ft.value
                    ? 'border-blue-500 bg-blue-50 text-blue-700'
                    : 'border-gray-200 text-gray-600 hover:border-gray-300'
                }`}
              >
                {ft.label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* 자유 텍스트 */}
      {rating > 0 && (
        <div>
          <textarea
            value={feedbackText}
            onChange={(e) => setFeedbackText(e.target.value)}
            placeholder="추가 의견이 있으면 작성해주세요 (선택)"
            rows={2}
            className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg resize-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
        </div>
      )}

      {/* 제출 */}
      {rating > 0 && (
        <button
          onClick={handleSubmit}
          disabled={isSubmitting}
          className="px-5 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors"
        >
          {isSubmitting ? '전송 중...' : '피드백 보내기'}
        </button>
      )}
    </div>
  )
}
