/**
 * 대본 생성 SSE 스트리밍 프록시 API Route
 *
 * Next.js rewrites가 SSE를 버퍼링하는 문제를 우회하기 위해
 * API Route에서 직접 스트리밍 프록시합니다.
 *
 * v2.3: 유휴 타이머 전환 — 데이터 수신 시마다 리셋 (하드 타이머 제거)
 */

import { NextRequest } from 'next/server'

const BACKEND_URL = process.env.BACKEND_URL || 'http://127.0.0.1:8000'

// 유휴 타임아웃: 마지막 청크 수신 후 90초 무응답 시 중단
// (하드 타이머 대신 — 백엔드에서 데이터가 흐르는 한 연결 유지)
const IDLE_TIMEOUT_MS = 90_000

// 초기 연결 타임아웃: 백엔드 응답 시작까지 최대 대기
const CONNECT_TIMEOUT_MS = 30_000

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

export async function POST(request: NextRequest) {
  const abortController = new AbortController()

  // 클라이언트 이탈 시 업스트림 요청도 취소
  request.signal.addEventListener('abort', () => {
    abortController.abort()
  })

  try {
    const body = await request.json()

    // 초기 연결 타이머 (백엔드 응답 시작까지)
    let timeoutId: ReturnType<typeof setTimeout> = setTimeout(
      () => abortController.abort(),
      CONNECT_TIMEOUT_MS,
    )

    const backendResponse = await fetch(
      `${BACKEND_URL}/api/content-marketing/script/generate`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: abortController.signal,
      },
    )

    // 연결 성공 — 유휴 타이머로 전환
    clearTimeout(timeoutId)
    timeoutId = setTimeout(() => abortController.abort(), IDLE_TIMEOUT_MS)

    if (!backendResponse.ok) {
      clearTimeout(timeoutId)
      const errorText = await backendResponse.text()
      return new Response(
        JSON.stringify({ error: 'Backend request failed', detail: errorText }),
        {
          status: backendResponse.status,
          headers: { 'Content-Type': 'application/json' },
        },
      )
    }

    if (!backendResponse.body) {
      clearTimeout(timeoutId)
      return new Response(
        JSON.stringify({ error: 'No response body' }),
        { status: 500, headers: { 'Content-Type': 'application/json' } },
      )
    }

    const reader = backendResponse.body.getReader()

    const stream = new ReadableStream({
      async start(controller) {
        try {
          while (true) {
            const { done, value } = await reader.read()
            if (done) {
              controller.close()
              break
            }

            // 데이터 수신마다 유휴 타이머 리셋
            clearTimeout(timeoutId)
            timeoutId = setTimeout(() => abortController.abort(), IDLE_TIMEOUT_MS)

            controller.enqueue(value)
          }
        } catch (error) {
          console.error('[Script SSE Proxy] Stream error:', error)
          controller.error(error)
        } finally {
          clearTimeout(timeoutId)
          reader.releaseLock()
        }
      },
      cancel() {
        clearTimeout(timeoutId)
        reader.cancel()
        abortController.abort()
      },
    })

    return new Response(stream, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache, no-transform',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no',
      },
    })
  } catch (error) {
    const isTimeout =
      error instanceof DOMException && error.name === 'AbortError'
    console.error(
      '[Script SSE Proxy] Error:',
      isTimeout ? 'Timeout/Abort' : error,
    )

    if (isTimeout) {
      return new Response(
        JSON.stringify({
          error: 'Proxy timeout',
          detail: '대본 생성 시간이 초과되었습니다. 다시 시도해주세요.',
        }),
        { status: 504, headers: { 'Content-Type': 'application/json' } },
      )
    }

    return new Response(
      JSON.stringify({ error: 'Proxy error', detail: String(error) }),
      { status: 500, headers: { 'Content-Type': 'application/json' } },
    )
  }
}
