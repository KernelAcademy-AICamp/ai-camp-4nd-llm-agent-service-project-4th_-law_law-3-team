import type { RefObject } from 'react'
import type { Message } from './constants'
import MessageBubble, { ShimmerText } from './MessageBubble'
import type { MarkdownComponentsType } from './MessageBubble'

interface ChatMessageListProps {
  messages: Message[]
  streamingMessageId: string | null
  isLoading: boolean
  loadingStatus: { title: string }
  markdownComponents: MarkdownComponentsType
  onAction: (action: string) => void
  onRequestLocation: () => void
  scrollRef: RefObject<HTMLDivElement | null>
}

export function ChatMessageList({
  messages,
  streamingMessageId,
  isLoading,
  loadingStatus,
  markdownComponents,
  onAction,
  onRequestLocation,
  scrollRef,
}: ChatMessageListProps) {
  return (
    <div
      ref={scrollRef as RefObject<HTMLDivElement>}
      className="flex-1 overflow-y-auto p-6 space-y-6 scrollbar-thin scrollbar-thumb-gray-300/20 scrollbar-track-transparent"
    >
      {messages.map((msg) => (
        <MessageBubble
          key={msg.id}
          msg={msg}
          isStreamingMessage={msg.id === streamingMessageId}
          messageUserClass="bg-[#007AFF] text-white shadow-sm"
          messageBotClass="bg-[#F5F5F7] text-[#1D1D1F] border border-black/[0.04]"
          isLightTheme={true}
          markdownComponents={markdownComponents}
          loadingStatus={loadingStatus}
          onAction={onAction}
          onRequestLocation={onRequestLocation}
        />
      ))}
      {/* Loading indicator (스트리밍 중이 아닐 때만 표시) */}
      {isLoading && !streamingMessageId && (
        <div className="flex justify-start">
          <div className="max-w-[85%] p-4 rounded-2xl rounded-tl-none bg-[#F5F5F7] text-[#1D1D1F] border border-black/[0.04]">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
              <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
              <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
              <span className="ml-2 text-sm text-blue-600"><ShimmerText text={loadingStatus.title} /></span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
