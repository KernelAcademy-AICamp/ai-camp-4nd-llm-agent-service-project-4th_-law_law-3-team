/**
 * 키워드 수집 SSE 스트리밍 프록시 API Route
 *
 * Next.js rewrites가 SSE를 버퍼링하는 문제를 우회하기 위해
 * API Route에서 직접 스트리밍 프록시합니다.
 *
 * 참고: /api/chat/stream, /api/content-marketing/script/generate 와 동일한 패턴
 */

import { NextRequest } from 'next/server'

const BACKEND_URL = process.env.BACKEND_URL || 'http://127.0.0.1:8000'

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

    // 연결 성공 시 타임아웃 해제 (스트리밍 중에는 불필요)
    clearTimeout(timeoutId)

    if (!backendResponse.ok) {
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
      return new Response(
        JSON.stringify({ error: 'No response body' }),
        { status: 500, headers: { 'Content-Type': 'application/json' } },
      )
    }

    const reader = backendResponse.body.getReader()

    const stream = new ReadableStream({
      async start(streamController) {
        try {
          while (true) {
            const { done, value } = await reader.read()
            if (done) {
              streamController.close()
              break
            }
            streamController.enqueue(value)
          }
        } catch (error) {
          console.error('[Keyword SSE Proxy] Stream error:', error)
          streamController.error(error)
        }
      },
      cancel() {
        reader.cancel()
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
    console.error('[Keyword SSE Proxy] Error:', error)
    return new Response(
      JSON.stringify({ error: 'Proxy error', detail: String(error) }),
      { status: 500, headers: { 'Content-Type': 'application/json' } },
    )
  }
}
