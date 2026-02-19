'use client'

import { useState, useRef, useEffect } from 'react'
import type { CourtEvent } from '../types'
import { CHARACTER_NAMES } from '../game/config'

interface ChatPanelProps {
  messages: CourtEvent[]
  isWaiting: boolean
  placeholder: string
  onSend: (text: string) => void
  actions?: { label: string; action: string }[]
  onAction?: (action: string) => void
  /** 데모 모드: 자동 입력할 텍스트 (없으면 자동 입력 버튼 숨김) */
  demoInput?: string | null
  /** 데모 모드: 자동 입력 버튼 클릭 시 호출 */
  onDemoInput?: () => void
}

export function ChatPanel({
  messages,
  isWaiting,
  placeholder,
  onSend,
  actions,
  onAction,
  demoInput,
  onDemoInput,
}: ChatPanelProps) {
  const [input, setInput] = useState('')
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages])

  const handleSend = (): void => {
    const trimmed = input.trim()
    if (!trimmed || isWaiting) return
    onSend(trimmed)
    setInput('')
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>): void => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="flex flex-col h-full border-t border-gray-200 bg-white">
      {/* 메시지 목록 */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto p-3 space-y-2">
        {messages.map((message, index) => {
          const speakerName =
            CHARACTER_NAMES[message.speaker] ?? message.speaker
          return (
            <div key={index} className="text-sm">
              <span className="font-semibold text-gray-700">
                [{speakerName}]
              </span>{' '}
              <span className="text-gray-600">{message.content}</span>
            </div>
          )
        })}
        {isWaiting && (
          <div className="text-sm text-gray-400 animate-pulse">
            AI가 응답을 생성 중입니다...
          </div>
        )}
      </div>

      {/* 액션 버튼 */}
      {actions && actions.length > 0 && (
        <div className="px-3 py-2 flex gap-2 border-t border-gray-100">
          {actions.map((actionItem) => (
            <button
              key={actionItem.action}
              onClick={() => onAction?.(actionItem.action)}
              className="px-3 py-1 text-xs rounded-full border border-blue-300 text-blue-600 hover:bg-blue-50 transition-colors"
            >
              {actionItem.label}
            </button>
          ))}
        </div>
      )}

      {/* 데모 자동 입력 버튼 */}
      {demoInput && onDemoInput && !isWaiting && (
        <div className="px-3 py-2 border-t border-amber-100 bg-amber-50">
          <button
            onClick={onDemoInput}
            className="w-full px-3 py-2 text-sm rounded-lg border-2 border-amber-400 text-amber-700 bg-white hover:bg-amber-100 font-medium transition-colors"
          >
            자동 입력 (데모)
          </button>
          <p className="text-xs text-amber-500 mt-1 truncate">
            {demoInput.slice(0, 60)}...
          </p>
        </div>
      )}

      {/* 입력 영역 */}
      <div className="flex items-end gap-2 p-3 border-t border-gray-100">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={isWaiting}
          rows={1}
          className="flex-1 resize-none p-2 border border-gray-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-400 disabled:bg-gray-50"
        />
        <button
          onClick={handleSend}
          disabled={!input.trim() || isWaiting}
          className="px-4 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
        >
          전송
        </button>
      </div>
    </div>
  )
}
