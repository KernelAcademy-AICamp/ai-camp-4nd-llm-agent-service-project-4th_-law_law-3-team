'use client'

import { useState, KeyboardEvent } from 'react'
import type { AIQuestionResponse } from '../types'

interface AIChatSectionProps {
  response: AIQuestionResponse | null
  isLoading: boolean
  error: string | null
  onAsk: (question: string) => void
  onClear: () => void
}

export function AIChatSection({
  response,
  isLoading,
  error,
  onAsk,
  onClear,
}: AIChatSectionProps) {
  const [question, setQuestion] = useState('')

  const handleSubmit = () => {
    if (question.trim()) {
      onAsk(question.trim())
      setQuestion('')
    }
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  return (
    <div className="border-t border-gray-200 p-4 bg-gray-50">
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-medium text-gray-900 flex items-center gap-2">
          <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"
            />
          </svg>
          AI에게 질문하기
        </h3>
        {response && (
          <button
            onClick={onClear}
            className="text-xs text-gray-500 hover:text-gray-700"
          >
            대화 초기화
          </button>
        )}
      </div>

      {/* Question Input */}
      <div className="mb-3">
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="이 판례에 대해 궁금한 점을 물어보세요..."
          rows={2}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        />
        <div className="flex justify-between items-center mt-2">
          <span className="text-xs text-gray-400">Enter로 전송, Shift+Enter로 줄바꿈</span>
          <button
            onClick={handleSubmit}
            disabled={isLoading || !question.trim()}
            className="px-4 py-1.5 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition"
          >
            {isLoading ? '답변 생성 중...' : '질문하기'}
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-600 mb-3">
          {error}
        </div>
      )}

      {/* Loading */}
      {isLoading && (
        <div className="p-4 bg-white border border-gray-200 rounded-lg">
          <div className="flex items-center gap-3">
            <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600" />
            <span className="text-sm text-gray-600">AI가 판례를 분석하고 있습니다...</span>
          </div>
        </div>
      )}

      {/* Response */}
      {response && !isLoading && (
        <div className="p-4 bg-white border border-gray-200 rounded-lg">
          <div className="flex items-start gap-3">
            <div className="shrink-0 w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
              <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                />
              </svg>
            </div>
            <div className="flex-1">
              <p className="text-sm text-gray-800 whitespace-pre-wrap leading-relaxed">
                {response.answer}
              </p>
              {response.sources.length > 0 && (
                <div className="mt-3 pt-3 border-t border-gray-100">
                  <p className="text-xs text-gray-500 mb-1">참조 판례:</p>
                  <div className="flex flex-wrap gap-1">
                    {response.sources.map((source, idx) => (
                      <span
                        key={idx}
                        className="px-2 py-0.5 bg-gray-100 text-gray-600 text-xs rounded"
                      >
                        {source.case_number || source.case_name}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
