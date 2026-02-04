/**
 * SSE 스트리밍 프록시 API Route
 *
 * Next.js rewrites가 SSE를 버퍼링하는 문제를 우회하기 위해
 * API Route에서 직접 스트리밍 프록시합니다.
 */

import { NextRequest } from 'next/server'

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()

    console.log('[SSE Proxy] Forwarding request to backend:', `${BACKEND_URL}/api/chat/stream`)

    const backendResponse = await fetch(`${BACKEND_URL}/api/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    })

    console.log('[SSE Proxy] Backend response status:', backendResponse.status)

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
    const encoder = new TextEncoder()
    const decoder = new TextDecoder()

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

    return new Response(stream, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache, no-transform',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no',
      },
    })
  } catch (error) {
    console.error('[SSE Proxy] Error:', error)
    return new Response(
      JSON.stringify({ error: 'Proxy error', detail: String(error) }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    )
  }
}
