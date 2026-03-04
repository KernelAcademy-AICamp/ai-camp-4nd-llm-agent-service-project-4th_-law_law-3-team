'use client'

import { FileText, Send, Loader2, PenLine } from 'lucide-react'
import { CATEGORY_MAP } from '../types'
import { ExamPaperView } from './ExamPaperView'

const CATEGORY_BADGE_COLORS: Record<string, string> = {
  CIVIL: 'bg-blue-100 text-blue-800 border-blue-300',
  CRIMINAL: 'bg-red-100 text-red-800 border-red-300',
  PUBLIC: 'bg-green-100 text-green-800 border-green-300',
}

interface ExamViewerProps {
  content: string | undefined
  isLoading: boolean
  title: string
  answerText: string
  onAnswerChange: (text: string) => void
  onRequestFeedback: () => void
  isFeedbackLoading: boolean
  category?: string
  session?: number
  year?: number
}

export function ExamViewer({
  content,
  isLoading,
  title,
  answerText,
  onAnswerChange,
  onRequestFeedback,
  isFeedbackLoading,
  category,
  session,
  year,
}: ExamViewerProps) {
  if (!content && !isLoading) {
    return (
      <div className="flex flex-col items-center justify-center h-full bg-stone-100">
        <div className="bg-white rounded-xl shadow-md p-12 text-center">
          <FileText size={48} className="mx-auto mb-4 text-gray-300" />
          <p className="text-lg font-medium text-gray-500">
            좌측에서 시험 문제를 선택하세요
          </p>
          <p className="text-sm text-gray-400 mt-1">
            과목과 회차를 선택하면 문제가 표시됩니다
          </p>
        </div>
      </div>
    )
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full bg-stone-100">
        <div className="bg-white rounded-xl shadow-md p-8 flex items-center gap-3">
          <Loader2 size={24} className="animate-spin text-blue-500" />
          <span className="text-gray-600">시험 문제를 불러오는 중...</span>
        </div>
      </div>
    )
  }

  const catInfo = category ? CATEGORY_MAP[category] : null
  const badgeColor = category ? CATEGORY_BADGE_COLORS[category] : ''

  return (
    <div className="flex flex-col h-full bg-stone-100">
      {/* 시험지 용지 */}
      <div className="flex-1 flex flex-col m-3 bg-white rounded-lg shadow-lg border border-gray-200 overflow-hidden">
        {/* 시험 헤더 */}
        <div className="shrink-0 px-6 py-4 border-b-2 border-gray-800">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-lg font-bold text-gray-900 tracking-tight">
                {session ? `제${session}회` : ''} 변호사시험 기록형
              </h2>
              {year && (
                <p className="text-xs text-gray-500 mt-0.5">{year}년도</p>
              )}
            </div>
            {catInfo && (
              <span
                className={`px-3 py-1 rounded-full text-xs font-bold border ${badgeColor}`}
              >
                {catInfo.label}
              </span>
            )}
          </div>
        </div>

        {/* 문제 영역 */}
        <div className="flex-1 overflow-y-auto px-8 py-6">
          <ExamPaperView markdown={content ?? ''} />
        </div>

        {/* 답안 작성 영역 */}
        <div className="shrink-0 border-t-2 border-gray-300">
          {/* 답안 헤더 */}
          <div className="px-6 py-2.5 bg-gray-50 flex items-center justify-between border-b border-gray-200">
            <div className="flex items-center gap-2">
              <PenLine size={14} className="text-gray-500" />
              <span className="text-sm font-semibold text-gray-700">
                답안 작성
              </span>
            </div>
            <span className="text-xs font-mono text-gray-500 bg-white px-2 py-0.5 rounded border border-gray-200">
              {answerText.length.toLocaleString()}자
            </span>
          </div>
          {/* 답안 입력 */}
          <div className="p-4">
            <textarea
              value={answerText}
              onChange={(e) => onAnswerChange(e.target.value)}
              placeholder="답안을 작성하세요..."
              className="w-full min-h-[200px] p-4 border border-gray-300 rounded-lg resize-y text-sm leading-7 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white"
              style={{
                backgroundImage:
                  'repeating-linear-gradient(transparent, transparent 27px, #f0f0f0 27px, #f0f0f0 28px)',
                backgroundPositionY: '15px',
              }}
            />
            <div className="flex justify-end mt-3">
              <button
                onClick={onRequestFeedback}
                disabled={!answerText.trim() || isFeedbackLoading}
                className="flex items-center gap-2 px-5 py-2.5 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-sm"
              >
                {isFeedbackLoading ? (
                  <>
                    <Loader2 size={16} className="animate-spin" />
                    피드백 생성 중...
                  </>
                ) : (
                  <>
                    <Send size={16} />
                    AI 피드백 받기
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
