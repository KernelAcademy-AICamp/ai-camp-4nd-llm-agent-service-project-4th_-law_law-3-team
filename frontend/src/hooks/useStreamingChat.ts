/**
 * 스트리밍 채팅 훅
 *
 * SSE(Server-Sent Events)를 통해 LLM 응답을 실시간으로 처리합니다.
 */

import { useState, useCallback, useRef } from 'react'

interface ChatSource {
  case_name?: string
  case_number?: string
  doc_type: string
  similarity: number
  summary?: string
  content?: string
  law_name?: string
  law_type?: string
  cited_statutes?: string[]
  similar_cases?: string[]
}

interface ChatAction {
  type: string
  label: string
  action?: string
  url?: string
  params?: Record<string, unknown>
}

interface ChatMetadata {
  agent_used: string
  actions: ChatAction[]
  session_data: Record<string, unknown>
}

interface StreamingChatOptions {
  onToken?: (content: string) => void
  onSources?: (sources: ChatSource[]) => void
  onMetadata?: (metadata: ChatMetadata) => void
  onDone?: () => void
  onError?: (error: string) => void
}

interface StreamingChatRequest {
  message: string
  user_role?: string
  history?: Array<{ role: string; content: string }>
  session_data?: Record<string, unknown>
  user_location?: { latitude: number; longitude: number } | null
  agent?: string
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
                case 'metadata':
                  options.onMetadata?.(data as ChatMetadata)
                  break
                case 'done':
                  options.onDone?.()
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
