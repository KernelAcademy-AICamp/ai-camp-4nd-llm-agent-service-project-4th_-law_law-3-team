/**
 * 대본 생성 SSE 스트리밍 프록시 API Route
 *
 * Next.js rewrites가 SSE를 버퍼링하는 문제를 우회하기 위해
 * API Route에서 직접 스트리밍 프록시합니다.
 */

import { NextRequest } from 'next/server'

const BACKEND_URL = process.env.BACKEND_URL || 'http://127.0.0.1:8000'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()

    const backendResponse = await fetch(
      `${BACKEND_URL}/api/content-marketing/script/generate`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      },
    )

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
          console.error('[Script SSE Proxy] Stream error:', error)
          controller.error(error)
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
    console.error('[Script SSE Proxy] Error:', error)
    return new Response(
      JSON.stringify({ error: 'Proxy error', detail: String(error) }),
      { status: 500, headers: { 'Content-Type': 'application/json' } },
    )
  }
}
