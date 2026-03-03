/**
 * 웹툰 스토리보드 SSE 스트리밍 프록시 API Route
 *
 * Next.js rewrites가 SSE를 버퍼링하는 문제를 우회하기 위해
 * API Route에서 직접 스트리밍 프록시합니다.
 *
 * 기존 패턴: /api/content-marketing/script/generate/route.ts (POST)
 * 차이점: GET 메서드, 600초 타임아웃 (이미지 생성 10패널 고려)
 */

import { NextRequest } from 'next/server'

const BACKEND_URL = process.env.BACKEND_URL || 'http://127.0.0.1:8000'
const PROXY_TIMEOUT_MS = 600_000 // 10분 (이미지 생성 파이프라인 여유)

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ jobId: string }> },
) {
  const { jobId } = await params
  const abortController = new AbortController()

  // 클라이언트 이탈 시 업스트림 요청도 취소
  request.signal.addEventListener('abort', () => {
    abortController.abort()
  })

  try {
    const timeoutId = setTimeout(() => abortController.abort(), PROXY_TIMEOUT_MS)

    const apiKey = process.env.API_KEY || ''
    const backendResponse = await fetch(
      `${BACKEND_URL}/api/content-marketing/script/webtoon/${jobId}/stream`,
      {
        method: 'GET',
        headers: {
          Accept: 'text/event-stream',
          ...(apiKey ? { 'X-API-Key': apiKey } : {}),
        },
        signal: abortController.signal,
      },
    )

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
            controller.enqueue(value)
          }
        } catch (error) {
          console.error('[Webtoon SSE Proxy] Stream error:', error)
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
      '[Webtoon SSE Proxy] Error:',
      isTimeout ? 'Timeout/Abort' : error,
    )

    if (isTimeout) {
      return new Response(
        JSON.stringify({
          error: 'Proxy timeout',
          detail: '스토리보드 생성 시간이 초과되었습니다. 다시 시도해주세요.',
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
