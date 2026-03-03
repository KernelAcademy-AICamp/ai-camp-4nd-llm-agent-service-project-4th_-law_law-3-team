# Red Team 검증 보고서: 스토리보드 SSE 타임아웃 수정

> **검증 대상**: `docs/01-plan/features/storyboard-sse-timeout-fix.plan.md`
> **검증 도구**: Gemini CLI (Red Team)
> **검증일**: 2026-03-01

---

## 1. 취약점 (Critical/High/Medium/Low)

- **[High] 인증 정보 누락 (Auth Passthrough Failure)**:
  - 신규 `route.ts`의 `fetch` 코드에서 클라이언트의 `Cookie` 또는 `Authorization` 헤더를 백엔드로 전달하지 않음
  - 백엔드가 인증이 필요한 API일 경우, 모든 요청이 'Unauthorized'로 실패하거나 익명 사용자로 처리됨

- **[Medium] 서버 자원 고갈 DoS (Resource Exhaustion)**:
  - `PROXY_TIMEOUT_MS`가 5분으로 설정되어 있어, 악의적 사용자가 다수의 스트리밍 연결을 유지할 경우 Node.js 워커 쓰레드나 커넥션 풀 점유로 전체 서비스 마비 가능

- **[Medium] 에러 핸들링 부재 (Unhandled Exception)**:
  - `fetch` 실패(DNS 에러, 백엔드 다운) 시 `try-catch` 블록이 없어 서버 프로세스 수준의 예외 발생 가능
  - `backendResponse.ok` 확인 없이 `body.getReader()` 호출 시 `null` 참조 오류 가능

- **[Low] 응답 타입 오염 (MIME Sniffing)**:
  - 백엔드가 에러(JSON)를 보냈음에도 프록시가 `text/event-stream` 헤더를 강제하여 클라이언트 측 파싱 에러 유도 가능

## 2. 아키텍처 개선 제안

- **스트림 직접 전달**: `backendResponse.body`를 직접 `Response`에 전달 (메모리 효율성, 지연 시간 개선)
- **헤더 포워딩**: 클라이언트 요청 헤더(`Authorization`, `Cookie`)를 백엔드 fetch 시 복사하여 인증 상태 유지

## 3. 고급 기능 추가 제안

- **Proxy Heartbeat**: 이미지 생성 중 백엔드 침묵 시 중간 프록시(Nginx, Cloudflare)가 연결을 끊을 수 있으므로, 프록시 레이어에서 주기적 `: ping` 주석 이벤트 전송 권장
- **Circuit Breaker**: 백엔드 타임아웃이 빈번할 경우 요청을 즉시 차단하여 전체 시스템 보호

## 4. 성능 최적화 제안

- **Edge Runtime 고려**: `export const runtime = 'edge'` — 롱 폴링/스트리밍에서 메모리 오버헤드 적고 더 많은 동시 연결 처리 가능 (호환성 확인 필요)
- **메모리 누수 방지**: 클라이언트 연결 종료 시 `AbortController`로 백엔드 요청 즉시 중단

## 5. 종합 평가

**[보류: 수정 권고]** — 올바른 방향성(API Route 사용)이나, 인증 처리와 예외 처리 누락으로 실제 배포 시 서비스 장애 가능성 있음. 수정된 권장 구현안 적용 강력 권고.

## 권장 구현안 (수정된 route.ts)

```typescript
import { NextRequest } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

const BACKEND_URL = process.env.BACKEND_URL || 'http://127.0.0.1:8000'
const PROXY_TIMEOUT_MS = 400_000

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ jobId: string }> }
) {
  const { jobId } = await params
  const abortController = new AbortController()

  try {
    const backendResponse = await fetch(
      `${BACKEND_URL}/api/content-marketing/script/webtoon/${jobId}/stream`,
      {
        method: 'GET',
        headers: {
          'Accept': 'text/event-stream',
          'Authorization': request.headers.get('Authorization') || '',
          'Cookie': request.headers.get('Cookie') || '',
        },
        signal: abortController.signal,
      }
    )

    if (!backendResponse.ok) {
      const errorMsg = await backendResponse.text()
      return new Response(`Backend Error: ${errorMsg}`, { status: backendResponse.status })
    }

    const stream = new ReadableStream({
      async start(controller) {
        const reader = backendResponse.body?.getReader()
        if (!reader) return controller.close()

        try {
          while (true) {
            const { done, value } = await reader.read()
            if (done) break
            controller.enqueue(value)
          }
        } finally {
          reader.releaseLock()
          controller.close()
        }
      },
      cancel() {
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
    console.error('SSE Proxy Error:', error)
    return new Response('Internal Stream Error', { status: 500 })
  }
}
```
