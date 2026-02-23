'use client'

import { useState } from 'react'
import { Send, Maximize2 } from 'lucide-react'
import type { CourtEvent } from '../types'
import { CHARACTER_NAMES } from '../game/config'

interface ChatBottomBarProps {
  messages: CourtEvent[]
  isWaiting: boolean
  placeholder: string
  onSend: (text: string) => void
  onExpand: () => void
  /** 데모 모드: 자동 입력할 텍스트 */
  demoInput?: string | null
  /** 데모 모드: 자동 입력 버튼 클릭 시 호출 */
  onDemoInput?: () => void
}

export function ChatBottomBar({
  messages,
  isWaiting,
  placeholder,
  onSend,
  onExpand,
  demoInput,
  onDemoInput,
}: ChatBottomBarProps) {
  const [input, setInput] = useState('')

  const handleSend = (): void => {
    const trimmed = input.trim()
    if (!trimmed || isWaiting) return
    onSend(trimmed)
    setInput('')
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>): void => {
    if (e.key === 'Enter') {
      e.preventDefault()
      handleSend()
    }
  }

  const lastAiMessage = [...messages].reverse().find((m) => m.speaker !== 'user')
  const lastSpeakerName = lastAiMessage
    ? (CHARACTER_NAMES[lastAiMessage.speaker] ?? lastAiMessage.speaker)
    : null

  return (
    <div className="border-t border-gray-200 bg-white px-3 py-2 space-y-1">
      {/* 최근 AI 발언 1줄 */}
      {lastAiMessage && (
        <div className="text-xs text-gray-500 truncate">
          <span className="font-semibold">[{lastSpeakerName}]</span>{' '}
          {lastAiMessage.content}
        </div>
      )}
      {isWaiting && (
        <div className="text-xs text-gray-400 animate-pulse">
          AI가 응답을 생성 중입니다...
        </div>
      )}

      {/* 입력 행 */}
      <div className="flex items-center gap-2">
        {/* 데모 자동 입력 버튼 */}
        {demoInput && onDemoInput && !isWaiting && (
          <button
            onClick={onDemoInput}
            className="shrink-0 px-2 py-1.5 text-xs font-medium rounded border-2 border-amber-400 text-amber-700 bg-amber-50 hover:bg-amber-100 transition-colors"
          >
            자동 입력
          </button>
        )}

        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={isWaiting}
          className="flex-1 min-w-0 px-3 py-1.5 border border-gray-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-400 disabled:bg-gray-50"
        />
        <button
          onClick={handleSend}
          disabled={!input.trim() || isWaiting}
          className="shrink-0 p-1.5 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
          aria-label="전송"
        >
          <Send className="w-4 h-4" />
        </button>
        <button
          onClick={onExpand}
          className="shrink-0 p-1.5 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
          aria-label="채팅 패널 열기"
        >
          <Maximize2 className="w-4 h-4" />
        </button>
      </div>
    </div>
  )
}
