/**
 * 대본 생성 SSE 스트리밍 프록시 API Route
 *
 * v2.3: 유휴 타이머 전환 — 데이터 수신 시마다 리셋 (하드 타이머 제거)
 */

import { NextRequest } from 'next/server'
import {
  BACKEND_URL,
  SSE_HEADERS,
  backendErrorResponse,
  noBodyResponse,
  proxyErrorResponse,
  pipeBackendStream,
  apiKeyHeader,
} from '@/lib/sse-proxy'

// 유휴 타임아웃: 마지막 청크 수신 후 90초 무응답 시 중단
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
        headers: {
          'Content-Type': 'application/json',
          ...apiKeyHeader(),
        },
        body: JSON.stringify(body),
        signal: abortController.signal,
      },
    )

    // 연결 성공 — 유휴 타이머로 전환
    clearTimeout(timeoutId)
    timeoutId = setTimeout(() => abortController.abort(), IDLE_TIMEOUT_MS)

    if (!backendResponse.ok) {
      clearTimeout(timeoutId)
      return backendErrorResponse(backendResponse)
    }

    if (!backendResponse.body) {
      clearTimeout(timeoutId)
      return noBodyResponse()
    }

    const reader = backendResponse.body.getReader()
    const stream = pipeBackendStream(reader, {
      label: 'Script SSE Proxy',
      onChunk: () => {
        // 데이터 수신마다 유휴 타이머 리셋
        clearTimeout(timeoutId)
        timeoutId = setTimeout(() => abortController.abort(), IDLE_TIMEOUT_MS)
      },
      onCleanup: () => {
        clearTimeout(timeoutId)
        reader.releaseLock()
      },
    })

    return new Response(stream, { headers: SSE_HEADERS })
  } catch (error) {
    return proxyErrorResponse(error, 'Script SSE Proxy', {
      checkTimeout: true,
      timeoutMessage: '대본 생성 시간이 초과되었습니다. 다시 시도해주세요.',
    })
  }
}
