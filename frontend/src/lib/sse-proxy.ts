/**
 * SSE 스트리밍 프록시 공통 유틸리티
 *
 * Next.js rewrites가 SSE를 버퍼링하는 문제를 우회하기 위해
 * API Route에서 직접 스트리밍 프록시할 때 사용하는 공통 함수들.
 */

export const BACKEND_URL = process.env.BACKEND_URL || 'http://127.0.0.1:8000'

export const SSE_HEADERS: Record<string, string> = {
  'Content-Type': 'text/event-stream',
  'Cache-Control': 'no-cache, no-transform',
  'Connection': 'keep-alive',
  'X-Accel-Buffering': 'no',
}

/** 백엔드 응답 실패 시 JSON 에러 응답 생성 */
export async function backendErrorResponse(backendResponse: Response): Promise<Response> {
  const errorText = await backendResponse.text()
  return new Response(
    JSON.stringify({ error: 'Backend request failed', detail: errorText }),
    { status: backendResponse.status, headers: { 'Content-Type': 'application/json' } },
  )
}

/** 응답 body가 없을 때 에러 응답 */
export function noBodyResponse(): Response {
  return new Response(
    JSON.stringify({ error: 'No response body' }),
    { status: 500, headers: { 'Content-Type': 'application/json' } },
  )
}

/** catch 블록에서 프록시 에러 응답 생성 */
export function proxyErrorResponse(
  error: unknown,
  label: string,
  options?: { checkTimeout?: boolean; timeoutMessage?: string },
): Response {
  const isTimeout = options?.checkTimeout &&
    error instanceof DOMException && error.name === 'AbortError'

  console.error(`[${label}] Error:`, isTimeout ? 'Timeout/Abort' : error)

  if (isTimeout) {
    return new Response(
      JSON.stringify({
        error: 'Proxy timeout',
        detail: options?.timeoutMessage ?? '요청 시간이 초과되었습니다.',
      }),
      { status: 504, headers: { 'Content-Type': 'application/json' } },
    )
  }

  return new Response(
    JSON.stringify({ error: 'Proxy error', detail: String(error) }),
    { status: 500, headers: { 'Content-Type': 'application/json' } },
  )
}

export interface PipeStreamOptions {
  label: string
  /** 청크 수신 시 콜백 (유휴 타이머 리셋 등) */
  onChunk?: () => void
  /** 스트림 종료/취소 시 정리 콜백 */
  onCleanup?: () => void
}

/** 백엔드 ReadableStream을 클라이언트로 파이핑하는 ReadableStream 생성 */
export function pipeBackendStream(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  options: PipeStreamOptions,
): ReadableStream {
  return new ReadableStream({
    async start(controller) {
      try {
        while (true) {
          const { done, value } = await reader.read()
          if (done) {
            controller.close()
            break
          }
          options.onChunk?.()
          controller.enqueue(value)
        }
      } catch (error) {
        console.error(`[${options.label}] Stream error:`, error)
        controller.error(error)
      } finally {
        options.onCleanup?.()
      }
    },
    cancel() {
      options.onCleanup?.()
      reader.cancel()
    },
  })
}

/** API Key 헤더 생성 (있을 때만 포함) */
export function apiKeyHeader(): Record<string, string> {
  const apiKey = process.env.API_KEY || ''
  return apiKey ? { 'X-API-Key': apiKey } : {}
}
