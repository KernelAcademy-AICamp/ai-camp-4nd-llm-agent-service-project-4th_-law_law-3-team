'use client'

import ReactMarkdown from 'react-markdown'
import { FileText, Send, Loader2 } from 'lucide-react'

interface ExamViewerProps {
  content: string | undefined
  isLoading: boolean
  title: string
  answerText: string
  onAnswerChange: (text: string) => void
  onRequestFeedback: () => void
  isFeedbackLoading: boolean
}

export function ExamViewer({
  content,
  isLoading,
  title,
  answerText,
  onAnswerChange,
  onRequestFeedback,
  isFeedbackLoading,
}: ExamViewerProps) {
  if (!content && !isLoading) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-gray-400">
        <FileText size={48} className="mb-4" />
        <p className="text-lg">좌측에서 문제를 선택하세요</p>
      </div>
    )
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full text-gray-400">
        <Loader2 size={24} className="animate-spin mr-2" />
        문제를 불러오는 중...
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full">
      {/* 문제 표시 영역 */}
      <div className="flex-1 overflow-y-auto p-4 border-b border-gray-200">
        <h2 className="text-lg font-bold mb-4 text-gray-900">{title}</h2>
        <div className="prose prose-sm max-w-none text-gray-800">
          <ReactMarkdown>{content ?? ''}</ReactMarkdown>
        </div>
      </div>

      {/* 답안 작성 영역 */}
      <div className="shrink-0 p-4 bg-gray-50">
        <div className="flex items-center justify-between mb-2">
          <label className="text-sm font-medium text-gray-700">답안 작성</label>
          <span className="text-xs text-gray-400">
            {answerText.length.toLocaleString()}자
          </span>
        </div>
        <textarea
          value={answerText}
          onChange={(e) => onAnswerChange(e.target.value)}
          placeholder="답안을 작성하세요..."
          className="w-full min-h-48 p-3 border border-gray-300 rounded-lg resize-y text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        />
        <button
          onClick={onRequestFeedback}
          disabled={!answerText.trim() || isFeedbackLoading}
          className="mt-2 flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
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
  )
}
