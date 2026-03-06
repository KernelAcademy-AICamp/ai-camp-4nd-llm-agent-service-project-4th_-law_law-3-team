'use client'

import { memo, useMemo } from 'react'
import ReactMarkdown from 'react-markdown'
import { motion, type Variants } from 'framer-motion'
import ChatActions from '../ChatActions'
import type { ChatAction } from '../ChatActions'
import type { Message } from './constants'

// 글자 순차 반짝이 애니메이션 (Framer Motion)
function ShimmerText({ text, className }: { text: string; className?: string }) {
  const chars = text.split('')
  const totalDuration = chars.length * 0.06 + 0.8
  return (
    <span className={className} aria-label={text}>
      {chars.map((char, i) => (
        <motion.span
          key={i}
          className="inline-block"
          animate={{ opacity: [0.4, 1, 0.4] }}
          transition={{
            duration: 0.8,
            ease: 'easeInOut',
            delay: i * 0.06,
            repeat: Infinity,
            repeatDelay: totalDuration - 0.8,
          }}
        >
          {char === ' ' ? '\u00A0' : char}
        </motion.span>
      ))}
    </span>
  )
}

export { ShimmerText }

// 판례번호 패턴: 2023다12345, 88도820, 99가합1234 등
const CASE_NUMBER_PATTERN = /(\d{2,4}[가-힣]{1,3}\d{1,6})/g

// 텍스트에서 판례번호를 클릭 가능한 버튼으로 변환하는 컴포넌트
function CaseNumberLink({
  text,
  onCaseClick,
  isLightTheme
}: {
  text: string
  onCaseClick: (caseNumber: string) => void
  isLightTheme: boolean
}) {
  const parts = text.split(CASE_NUMBER_PATTERN)

  return (
    <>
      {parts.map((part, index) => {
        if (CASE_NUMBER_PATTERN.test(part)) {
          // Reset the regex lastIndex
          CASE_NUMBER_PATTERN.lastIndex = 0
          return (
            <button
              key={`case-${index}`}
              onClick={(e) => {
                e.preventDefault()
                e.stopPropagation()
                onCaseClick(part)
              }}
              className={`inline px-1 py-0.5 mx-0.5 rounded text-sm font-mono font-bold transition-all hover:scale-105 ${isLightTheme
                ? 'bg-blue-100 text-blue-700 hover:bg-blue-200'
                : 'bg-blue-500/30 text-blue-300 hover:bg-blue-500/50'
                }`}
              title={`${part} 판례 보기`}
            >
              {part}
            </button>
          )
        }
        return <span key={`part-${index}`}>{part}</span>
      })}
    </>
  )
}

// Memoized markdown components factory to prevent recreation on every render
export function useMarkdownComponents(
  onCaseClick: (caseNumber: string) => void,
  isLightTheme: boolean
) {
  return useMemo(() => ({
    p: ({ children }: { children?: React.ReactNode }) => (
      <p>
        {typeof children === 'string' ? (
          <CaseNumberLink
            text={children}
            onCaseClick={onCaseClick}
            isLightTheme={isLightTheme}
          />
        ) : Array.isArray(children) ? (
          children.map((child, i) =>
            typeof child === 'string' ? (
              <CaseNumberLink
                key={`md-${i}`}
                text={child}
                onCaseClick={onCaseClick}
                isLightTheme={isLightTheme}
              />
            ) : (
              <span key={`md-${i}`}>{child}</span>
            )
          )
        ) : (
          children
        )}
      </p>
    ),
    li: ({ children }: { children?: React.ReactNode }) => (
      <li>
        {typeof children === 'string' ? (
          <CaseNumberLink
            text={children}
            onCaseClick={onCaseClick}
            isLightTheme={isLightTheme}
          />
        ) : Array.isArray(children) ? (
          children.map((child, i) =>
            typeof child === 'string' ? (
              <CaseNumberLink
                key={`md-${i}`}
                text={child}
                onCaseClick={onCaseClick}
                isLightTheme={isLightTheme}
              />
            ) : (
              <span key={`md-${i}`}>{child}</span>
            )
          )
        ) : (
          children
        )}
      </li>
    ),
  }), [onCaseClick, isLightTheme])
}

export type MarkdownComponentsType = ReturnType<typeof useMarkdownComponents>

export interface MessageBubbleProps {
  msg: Message
  isStreamingMessage: boolean
  messageUserClass: string
  messageBotClass: string
  isLightTheme: boolean
  markdownComponents: MarkdownComponentsType
  loadingStatus: { title: string }
  onAction: (action: string) => void
  onRequestLocation: () => void
}

const MessageBubble = memo(function MessageBubble({
  msg,
  isStreamingMessage,
  messageUserClass,
  messageBotClass,
  isLightTheme,
  markdownComponents,
  loadingStatus,
  onAction,
  onRequestLocation,
}: MessageBubbleProps) {
  return (
    <div className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[85%] p-4 rounded-2xl text-base leading-relaxed ${msg.role === 'user'
          ? `${messageUserClass} rounded-tr-none`
          : `${messageBotClass} rounded-tl-none`
          }`}
      >
        {msg.role === 'assistant' ? (
          isStreamingMessage && !msg.content.trim() ? (
            <div className="space-y-3 min-w-[260px]">
              <div className="flex items-start gap-3">
                <div className="mt-1.5 flex h-3 w-3">
                  <span className="relative inline-flex h-3 w-3">
                    <span className="absolute inline-flex h-full w-full rounded-full bg-blue-500 opacity-75 animate-ping" />
                    <span className="relative inline-flex h-3 w-3 rounded-full bg-blue-500" />
                  </span>
                </div>
                <div className="min-w-0">
                  <p className="text-sm text-blue-600"><ShimmerText text={loadingStatus.title} /></p>
                </div>
              </div>
            </div>
          ) : (
            <div className={`prose prose-sm max-w-none prose-p:my-2 prose-ul:my-2 prose-ol:my-2 prose-li:my-0 prose-headings:my-2 prose-strong:text-inherit ${!isLightTheme ? 'prose-invert' : ''}`}>
              <ReactMarkdown components={markdownComponents}>
                {msg.content}
              </ReactMarkdown>
              {isStreamingMessage && (
                <span className="inline-block w-2 h-4 bg-current animate-pulse ml-0.5" />
              )}
              {msg.actions && msg.actions.length > 0 && (
                <ChatActions
                  actions={msg.actions as ChatAction[]}
                  onAction={onAction}
                  onRequestLocation={onRequestLocation}
                  isLightTheme={isLightTheme}
                />
              )}
            </div>
          )
        ) : (
          <span className="whitespace-pre-wrap">{msg.content}</span>
        )}
      </div>
    </div>
  )
}, (prevProps, nextProps) => {
  // 완료된 메시지는 스트리밍 중 재렌더 방지 (ReactMarkdown 파싱 비용 절감)
  if (prevProps.msg !== nextProps.msg) return false
  if (prevProps.isStreamingMessage !== nextProps.isStreamingMessage) return false
  if (prevProps.isLightTheme !== nextProps.isLightTheme) return false
  if (prevProps.isStreamingMessage && prevProps.loadingStatus !== nextProps.loadingStatus) return false
  return true
})

export default MessageBubble
