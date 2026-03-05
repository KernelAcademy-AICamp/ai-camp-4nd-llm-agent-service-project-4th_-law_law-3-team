'use client'

import { useMarkdownComponents } from './chat/MessageBubble'
import { useChatMessages } from './chat/useChatMessages'
import { ChatHeader } from './chat/ChatHeader'
import { ChatMessageList } from './chat/ChatMessageList'
import { ChatInput } from './chat/ChatInput'

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

  const layoutClasses =
    chatMode === 'split'
      ? 'fixed top-0 right-0 w-1/2 h-screen z-50 flex flex-col animate-in slide-in-from-right duration-500'
      : 'fixed bottom-6 right-6 w-[380px] h-[600px] z-50 rounded-2xl flex flex-col animate-in slide-in-from-bottom zoom-in duration-300'

  // Floating Button (Collapsed)
  if (!isChatOpen) {
    if (isChatHiddenPage) return null
    return (
      <button
        onClick={toggleChat}
        className="fixed bottom-6 right-6 w-14 h-14 bg-[#007AFF] hover:bg-[#0056CC] text-white rounded-full shadow-apple-hover flex items-center justify-center z-50 transition-all duration-200 hover:scale-105 active:scale-95 cursor-pointer"
      >
        <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
          />
        </svg>
      </button>
    )
  }

  // Expanded Chat Interface
  return (
    <div
      className={`${layoutClasses} bg-white/95 backdrop-blur-xl border-l border-blue-500/20 shadow-[-20px_0_80px_-20px_rgba(0,0,0,0.15),-4px_0_20px_rgba(59,130,246,0.03)] text-[#1D1D1F]`}
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
  )
}
