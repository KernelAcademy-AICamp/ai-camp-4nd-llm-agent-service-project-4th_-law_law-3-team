/**
 * SSE 스트리밍 프록시 API Route
 *
 * Next.js rewrites가 SSE를 버퍼링하는 문제를 우회하기 위해
 * API Route에서 직접 스트리밍 프록시합니다.
 */

import { NextRequest } from 'next/server'
import {
  BACKEND_URL,
  SSE_HEADERS,
  backendErrorResponse,
  noBodyResponse,
  pipeBackendStream,
  apiKeyHeader,
} from '@/lib/sse-proxy'

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
        ...apiKeyHeader(),
      },
      body: JSON.stringify(body),
    })

    if (!backendResponse.ok) return backendErrorResponse(backendResponse)
    if (!backendResponse.body) return noBodyResponse()

    const reader = backendResponse.body.getReader()
    const stream = pipeBackendStream(reader, { label: 'SSE Proxy' })

    // 백엔드 Set-Cookie 헤더를 클라이언트에 전달 (세션 토큰)
    const responseHeaders: Record<string, string> = { ...SSE_HEADERS }
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
