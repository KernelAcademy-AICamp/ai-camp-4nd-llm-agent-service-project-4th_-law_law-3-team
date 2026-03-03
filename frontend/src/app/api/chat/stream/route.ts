/**
 * SSE 스트리밍 프록시 API Route
 *
 * Next.js rewrites가 SSE를 버퍼링하는 문제를 우회하기 위해
 * API Route에서 직접 스트리밍 프록시합니다.
 */

import { NextRequest } from 'next/server'

const BACKEND_URL = process.env.BACKEND_URL || 'http://127.0.0.1:8000'
const MAX_RETRIES = 3
const INITIAL_DELAY_MS = 2000

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

function isConnectionError(error: unknown): boolean {
  if (error instanceof TypeError && error.cause) {
    const cause = error.cause as { code?: string }
    return cause.code === 'ETIMEDOUT' || cause.code === 'ECONNREFUSED'
  }
  return false
}

async function fetchWithRetry(
  url: string,
  options: RequestInit,
  retries: number = MAX_RETRIES
): Promise<Response> {
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      return await fetch(url, options)
    } catch (error) {
      if (attempt < retries && isConnectionError(error)) {
        const delay = INITIAL_DELAY_MS * Math.pow(2, attempt)
        // 재시도 로깅은 서버사이드에서만 노출되지만 프로덕션에서는 불필요
        await new Promise(resolve => setTimeout(resolve, delay))
        continue
      }
      throw error
    }
  }
  throw new Error('Unreachable')
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()

    // 클라이언트 쿠키를 백엔드로 전달 (세션 토큰)
    const cookieHeader = request.headers.get('cookie') || ''

    const backendResponse = await fetchWithRetry(`${BACKEND_URL}/api/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(cookieHeader ? { Cookie: cookieHeader } : {}),
      },
      body: JSON.stringify(body),
    })

    if (!backendResponse.ok) {
      const errorText = await backendResponse.text()
      console.error('[SSE Proxy] Backend error:', errorText)
      return new Response(
        JSON.stringify({ error: 'Backend request failed', detail: errorText }),
        { status: backendResponse.status, headers: { 'Content-Type': 'application/json' } }
      )
    }

    if (!backendResponse.body) {
      return new Response(
        JSON.stringify({ error: 'No response body' }),
        { status: 500, headers: { 'Content-Type': 'application/json' } }
      )
    }

    // ReadableStream을 사용하여 청크 단위로 전달
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
            // 청크를 그대로 전달
            controller.enqueue(value)
          }
        } catch (error) {
          console.error('[SSE Proxy] Stream error:', error)
          controller.error(error)
        }
      },
      cancel() {
        reader.cancel()
      },
    })

    // 백엔드 Set-Cookie 헤더를 클라이언트에 전달 (세션 토큰)
    const responseHeaders: Record<string, string> = {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache, no-transform',
      'Connection': 'keep-alive',
      'X-Accel-Buffering': 'no',
    }
    const setCookie = backendResponse.headers.get('set-cookie')
    if (setCookie) {
      responseHeaders['Set-Cookie'] = setCookie
    }

    return new Response(stream, { headers: responseHeaders })
  } catch (error) {
    console.error('[SSE Proxy] Error:', error)
    const isConnErr = isConnectionError(error)
    return new Response(
      JSON.stringify({
        error: isConnErr ? 'Backend not ready' : 'Proxy error',
        detail: String(error),
      }),
      {
        status: isConnErr ? 503 : 500,
        headers: { 'Content-Type': 'application/json' },
      }
    )
  }
}
