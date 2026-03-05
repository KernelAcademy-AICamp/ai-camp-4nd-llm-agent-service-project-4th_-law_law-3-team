/**
 * 스트리밍 채팅 훅
 *
 * SSE(Server-Sent Events)를 통해 LLM 응답을 실시간으로 처리합니다.
 */

import { useState, useCallback, useRef } from 'react'
import type { ChatAction } from '@/components/ChatActions'
import type { ChatSource } from '@/features/case-precedent/types'

export interface ChatMetadata {
  agent_used: string
  actions: ChatAction[]
  session_data: Record<string, unknown>
  // mock_trial 확장
  speaking_agent?: string
  emotion?: string
  stage?: string
  step?: string
  evidence?: { cases: unknown[]; articles: unknown[] }
  user_hints?: unknown[]
  scenario?: {
    title: string
    background: string
    characters: Array<{ role: string; name: string; description: string }>
    issues: string[]
    evidence_hints: Array<{ type: string; title: string; description: string; favorable_to: string }>
    objectives: string[]
  }
  references?: Array<{
    id: string
    type: string
    title: string
    summary: string
    relevance_score: number
    source: string
  }>
}

interface StreamingChatOptions {
  onToken?: (content: string) => void
  onSources?: (sources: ChatSource[]) => void
  onRouting?: (data: { selected_agent: string }) => void
  onMetadata?: (metadata: ChatMetadata) => void
  onDone?: (data?: Record<string, unknown>) => void
  onError?: (error: string) => void
}

interface StreamingChatRequest {
  message: string
  user_role?: string
  history?: Array<{ role: string; content: string }>
  session_data?: Record<string, unknown>
  user_location?: { latitude: number; longitude: number } | null
  agent?: string
  conversation_id?: string
  case_id?: string
}

interface UseStreamingChatReturn {
  sendStreamingMessage: (
    request: StreamingChatRequest,
    options: StreamingChatOptions
  ) => Promise<void>
  isStreaming: boolean
  abortStream: () => void
}

export function useStreamingChat(): UseStreamingChatReturn {
  const [isStreaming, setIsStreaming] = useState(false)
  const abortControllerRef = useRef<AbortController | null>(null)

  const abortStream = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      abortControllerRef.current = null
      setIsStreaming(false)
    }
  }, [])

  const sendStreamingMessage = useCallback(
    async (request: StreamingChatRequest, options: StreamingChatOptions) => {
      // 기존 스트림이 있으면 중단
      abortStream()

      const controller = new AbortController()
      abortControllerRef.current = controller
      setIsStreaming(true)

      try {
        const response = await fetch('/api/chat/stream', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(request),
          signal: controller.signal,
          credentials: 'include',
        })

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`)
        }

        const reader = response.body?.getReader()
        if (!reader) {
          throw new Error('Response body is null')
        }

        const decoder = new TextDecoder()
        let buffer = ''

        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          const chunk = decoder.decode(value, { stream: true })
          buffer += chunk

          // SSE 이벤트는 빈 줄로 구분됨 (\n\n 또는 \r\n\r\n)
          // 먼저 \r\n을 \n으로 정규화
          buffer = buffer.replace(/\r\n/g, '\n')
          const events = buffer.split('\n\n')
          buffer = events.pop() || ''

          for (const eventBlock of events) {
            if (!eventBlock.trim()) continue

            const lines = eventBlock.split('\n')
            let eventType = 'message'
            let eventData = ''

            for (const line of lines) {
              if (line.startsWith('event:')) {
                eventType = line.slice(6).trim()
              } else if (line.startsWith('data:')) {
                eventData = line.slice(5).trim()
              }
            }

            if (!eventData) continue

            try {
              const data = JSON.parse(eventData)

              switch (eventType) {
                case 'token':
                  options.onToken?.(data.content || '')
                  break
                case 'sources':
                  options.onSources?.(data.sources || [])
                  break
                case 'routing':
                  options.onRouting?.(data as { selected_agent: string })
                  break
                case 'metadata':
                  options.onMetadata?.(data as ChatMetadata)
                  break
                case 'done':
                  options.onDone?.(data as Record<string, unknown>)
                  break
                case 'error':
                  options.onError?.(data.message || 'Unknown error')
                  break
              }
            } catch (e) {
              console.error('[SSE] JSON parse error:', e, 'Data:', eventData)
            }
          }
        }
      } catch (error) {
        if (error instanceof Error && error.name === 'AbortError') {
          // 스트림이 의도적으로 중단됨
          return
        }
        options.onError?.(error instanceof Error ? error.message : 'Unknown error')
      } finally {
        setIsStreaming(false)
        abortControllerRef.current = null
      }
    },
    [abortStream]
  )

  return {
    sendStreamingMessage,
    isStreaming,
    abortStream,
  }
}
