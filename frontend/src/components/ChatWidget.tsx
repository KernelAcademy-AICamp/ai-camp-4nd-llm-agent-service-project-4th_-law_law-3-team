'use client'

import { useState, useRef, useCallback, useEffect } from 'react'
import { useMarkdownComponents } from './chat/MessageBubble'
import { useChatMessages } from './chat/useChatMessages'
import { ChatHeader } from './chat/ChatHeader'
import { ChatMessageList } from './chat/ChatMessageList'
import { ChatInput } from './chat/ChatInput'

const BUTTON_SIZE = 56
const DRAG_THRESHOLD = 5

export default function ChatWidget() {
  const {
    // 상태
    messages,
    input,
    setInput,
    isLoading,
    isStreaming,
    streamingMessageId,
    loadingStatus,
    scrollRef,

    // UI 상태
    isChatOpen,
    toggleChat,
    setChatOpen,
    chatMode,
    supportsFloatingMode,
    isChatHiddenPage,

    // 사용자/세션
    userRole,
    setUserRole,
    sessionData,
    setHighlightedCaseNumber,

    // 핸들러
    handleSend,
    handleAction,
    handleResetChat,
    handleRequestLocation,
    toggleViewMode,
  } = useChatMessages()

  const isLightTheme = true
  const markdownComponents = useMarkdownComponents(setHighlightedCaseNumber, isLightTheme)

  const isDisabled = isLoading || isStreaming

  // 드래그 상태
  const [position, setPosition] = useState<{ x: number; y: number } | null>(null)
  const dragRef = useRef({ isDragging: false, startX: 0, startY: 0, startPosX: 0, startPosY: 0, hasMoved: false })

  // 초기 위치 설정 (우하단)
  useEffect(() => {
    if (position === null && typeof window !== 'undefined') {
      setPosition({
        x: window.innerWidth - BUTTON_SIZE - 24,
        y: window.innerHeight - BUTTON_SIZE - 24,
      })
    }
  }, [position])

  // 화면 리사이즈 시 경계 보정
  useEffect(() => {
    const handleResize = () => {
      setPosition(prev => {
        if (!prev) return prev
        return {
          x: Math.min(prev.x, window.innerWidth - BUTTON_SIZE),
          y: Math.min(prev.y, window.innerHeight - BUTTON_SIZE),
        }
      })
    }
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  const handlePointerDown = useCallback((e: React.PointerEvent) => {
    e.preventDefault()
    const currentPos = position ?? { x: window.innerWidth - BUTTON_SIZE - 24, y: window.innerHeight - BUTTON_SIZE - 24 }
    dragRef.current = {
      isDragging: true,
      startX: e.clientX,
      startY: e.clientY,
      startPosX: currentPos.x,
      startPosY: currentPos.y,
      hasMoved: false,
    }
    ;(e.target as HTMLElement).setPointerCapture(e.pointerId)
  }, [position])

  const handlePointerMove = useCallback((e: React.PointerEvent) => {
    const drag = dragRef.current
    if (!drag.isDragging) return

    const deltaX = e.clientX - drag.startX
    const deltaY = e.clientY - drag.startY

    if (!drag.hasMoved && Math.abs(deltaX) < DRAG_THRESHOLD && Math.abs(deltaY) < DRAG_THRESHOLD) {
      return
    }
    drag.hasMoved = true

    const newX = Math.max(0, Math.min(window.innerWidth - BUTTON_SIZE, drag.startPosX + deltaX))
    const newY = Math.max(0, Math.min(window.innerHeight - BUTTON_SIZE, drag.startPosY + deltaY))
    setPosition({ x: newX, y: newY })
  }, [])

  const handlePointerUp = useCallback(() => {
    const drag = dragRef.current
    if (!drag.isDragging) return
    drag.isDragging = false

    if (!drag.hasMoved) {
      toggleChat()
    }
  }, [toggleChat])

  const layoutClasses =
    chatMode === 'split'
      ? 'fixed top-0 right-0 w-1/2 h-screen z-50 flex flex-col animate-in slide-in-from-right duration-500'
      : 'fixed w-[380px] h-[600px] z-50 rounded-2xl flex flex-col animate-in slide-in-from-bottom zoom-in duration-300'

  // 채팅 창 위치 (버튼 기준 위쪽으로 팝업)
  const chatWindowStyle = chatMode === 'floating' && position
    ? {
        left: Math.max(0, Math.min(position.x - 380 + BUTTON_SIZE, window.innerWidth - 380)),
        top: Math.max(0, position.y - 600 - 8),
      }
    : undefined

  // Floating Button (Collapsed)
  if (!isChatOpen) {
    if (isChatHiddenPage) return null
    if (!position) return null
    return (
      <div
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        style={{ left: position.x, top: position.y, width: BUTTON_SIZE, height: BUTTON_SIZE }}
        className="fixed z-50 w-14 h-14 bg-[#007AFF] hover:bg-[#0056CC] text-white rounded-full shadow-apple-hover flex items-center justify-center transition-colors duration-200 hover:scale-105 active:scale-95 cursor-pointer select-none touch-none"
      >
        <svg className="w-8 h-8 pointer-events-none" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
          />
        </svg>
      </div>
    )
  }

  // Expanded Chat Interface
  return (
    <>
      {/* 드래그 가능 버튼 (열린 상태에서도 표시) */}
      {chatMode === 'floating' && position && (
        <div
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          style={{ left: position.x, top: position.y, width: BUTTON_SIZE, height: BUTTON_SIZE }}
          className="fixed z-[51] w-14 h-14 bg-[#007AFF] hover:bg-[#0056CC] text-white rounded-full shadow-apple-hover flex items-center justify-center transition-colors duration-200 cursor-pointer select-none touch-none"
        >
          <svg className="w-8 h-8 pointer-events-none" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
            />
          </svg>
        </div>
      )}

      <div
        className={`${layoutClasses} bg-white/95 backdrop-blur-xl border-l border-blue-500/20 shadow-[-20px_0_80px_-20px_rgba(0,0,0,0.15),-4px_0_20px_rgba(59,130,246,0.03)] text-[#1D1D1F]`}
        style={chatWindowStyle}
      >
        <ChatHeader
          activeAgent={sessionData.active_agent as string | undefined}
          userRole={userRole}
          onSetUserRole={setUserRole}
          supportsFloatingMode={supportsFloatingMode}
          chatMode={chatMode}
          onToggleViewMode={toggleViewMode}
          onReset={handleResetChat}
          onClose={() => setChatOpen(false)}
        />

        <ChatMessageList
          messages={messages}
          streamingMessageId={streamingMessageId}
          isLoading={isLoading}
          loadingStatus={loadingStatus}
          markdownComponents={markdownComponents}
          onAction={handleAction}
          onRequestLocation={handleRequestLocation}
          scrollRef={scrollRef}
        />

        <ChatInput
          input={input}
          onInputChange={setInput}
          isDisabled={isDisabled}
          chatMode={chatMode}
          onSend={() => handleSend()}
        />
      </div>
    </>
  )
}
