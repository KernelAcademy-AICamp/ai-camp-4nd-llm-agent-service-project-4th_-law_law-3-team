'use client'

import ReactMarkdown from 'react-markdown'
import { X, Loader2, MessageSquare } from 'lucide-react'

interface FeedbackPanelProps {
  feedback: string | undefined
  isLoading: boolean
  isError: boolean
  errorMessage?: string
  onClose: () => void
}

export function FeedbackPanel({
  feedback,
  isLoading,
  isError,
  errorMessage,
  onClose,
}: FeedbackPanelProps) {
  return (
    <div className="flex flex-col h-full">
      {/* 헤더 */}
      <div className="flex items-center justify-between p-3 border-b border-gray-200">
        <div className="flex items-center gap-2">
          <MessageSquare size={16} className="text-blue-600" />
          <h3 className="text-sm font-semibold text-gray-700">AI 피드백</h3>
        </div>
        <button
          onClick={onClose}
          className="p-1 text-gray-400 hover:text-gray-600 rounded transition-colors"
          aria-label="피드백 패널 닫기"
        >
          <X size={16} />
        </button>
      </div>

      {/* 본문 */}
      <div className="flex-1 overflow-y-auto p-4">
        {isLoading && (
          <div className="flex flex-col items-center justify-center py-12 text-gray-400">
            <Loader2 size={32} className="animate-spin mb-3" />
            <p className="text-sm">피드백을 생성하고 있습니다...</p>
            <p className="text-xs mt-1">잠시만 기다려주세요</p>
          </div>
        )}

        {isError && (
          <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
            <p className="font-medium">피드백 생성에 실패했습니다</p>
            {errorMessage && <p className="mt-1 text-xs">{errorMessage}</p>}
          </div>
        )}

        {feedback && !isLoading && (
          <div className="prose prose-sm max-w-none text-gray-800">
            <ReactMarkdown>{feedback}</ReactMarkdown>
          </div>
        )}
      </div>
    </div>
  )
}
