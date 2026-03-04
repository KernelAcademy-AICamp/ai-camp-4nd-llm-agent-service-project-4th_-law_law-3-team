'use client'

import { useState, useRef, useEffect } from 'react'
import { Minimize2 } from 'lucide-react'
import type { CourtEvent } from '../types'
import { EMOTION_EMOJI, DEFAULT_ROLE_EMOTION } from '../types'
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
  /** 패널 접기 핸들러 (전달 시 상단에 접기 버튼 표시) */
  onCollapse?: () => void
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
  onCollapse,
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
    <div className="flex flex-col h-full bg-white">
      {/* 접기 헤더 */}
      {onCollapse && (
        <div className="flex items-center justify-between px-3 py-2 border-b border-gray-200 bg-gray-50">
          <span className="text-sm font-semibold text-gray-700">재판 채팅</span>
          <button
            onClick={onCollapse}
            className="p-1 text-gray-500 hover:text-gray-700 hover:bg-gray-200 rounded transition-colors"
            aria-label="채팅 패널 접기"
          >
            <Minimize2 className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* 메시지 목록 */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto p-3 space-y-2">
        {messages.map((message, index) => {
          const speakerName =
            CHARACTER_NAMES[message.speaker] ?? message.speaker
          const emotionKey =
            message.emotion ?? DEFAULT_ROLE_EMOTION[message.speaker] ?? 'neutral'
          const emoji = EMOTION_EMOJI[emotionKey] ?? '😐'
          return (
            <div key={`msg-${index}`} className="text-sm">
              <span className="font-semibold text-gray-700">
                [{speakerName} {emoji}]
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
