/**
 * 웹툰 스토리보드 SSE 스트리밍 프록시 API Route
 *
 * GET 메서드, 600초 타임아웃 (이미지 생성 10패널 고려)
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

    const backendResponse = await fetch(
      `${BACKEND_URL}/api/content-marketing/script/webtoon/${jobId}/stream`,
      {
        method: 'GET',
        headers: {
          Accept: 'text/event-stream',
          ...apiKeyHeader(),
        },
        signal: abortController.signal,
      },
    )

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
      label: 'Webtoon SSE Proxy',
      onCleanup: () => {
        clearTimeout(timeoutId)
        reader.releaseLock()
      },
    })

    return new Response(stream, { headers: SSE_HEADERS })
  } catch (error) {
    return proxyErrorResponse(error, 'Webtoon SSE Proxy', {
      checkTimeout: true,
      timeoutMessage: '스토리보드 생성 시간이 초과되었습니다. 다시 시도해주세요.',
    })
  }
}
