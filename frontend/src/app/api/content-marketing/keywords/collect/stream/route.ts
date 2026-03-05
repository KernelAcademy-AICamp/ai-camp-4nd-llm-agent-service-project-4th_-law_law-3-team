/**
 * 키워드 수집 SSE 스트리밍 프록시 API Route
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

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const maxKeywords = searchParams.get('max_keywords') ?? '10'
    const timeRange = searchParams.get('time_range') ?? '48h'
    const forceRefresh = searchParams.get('force_refresh') ?? ''
    const category = searchParams.get('category') ?? 'all'
    const personaId = searchParams.get('persona_id') ?? ''

    let backendUrl =
      `${BACKEND_URL}/api/content-marketing/keywords/collect/stream` +
      `?max_keywords=${encodeURIComponent(maxKeywords)}` +
      `&time_range=${encodeURIComponent(timeRange)}` +
      `&category=${encodeURIComponent(category)}`
    if (forceRefresh === 'true') {
      backendUrl += '&force_refresh=true'
    }
    if (personaId) {
      backendUrl += `&persona_id=${encodeURIComponent(personaId)}`
    }

    // 120초 타임아웃 (R-1: 키워드 수집 최대 소요 시간 기준)
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 120_000)

    let backendResponse: Response
    try {
      backendResponse = await fetch(backendUrl, {
        method: 'GET',
        headers: {
          Accept: 'text/event-stream',
          'Cache-Control': 'no-cache',
          ...apiKeyHeader(),
        },
        signal: controller.signal,
      })
    } catch (error) {
      clearTimeout(timeoutId)
      if (error instanceof Error && error.name === 'AbortError') {
        return new Response(
          JSON.stringify({ error: 'Backend request timeout (120s)' }),
          { status: 504, headers: { 'Content-Type': 'application/json' } },
        )
      }
      throw error
    }

    clearTimeout(timeoutId)

    if (!backendResponse.ok) return backendErrorResponse(backendResponse)
    if (!backendResponse.body) return noBodyResponse()

    const reader = backendResponse.body.getReader()
    const stream = pipeBackendStream(reader, { label: 'Keyword SSE Proxy' })

    return new Response(stream, { headers: SSE_HEADERS })
  } catch (error) {
    return proxyErrorResponse(error, 'Keyword SSE Proxy')
  }
}
