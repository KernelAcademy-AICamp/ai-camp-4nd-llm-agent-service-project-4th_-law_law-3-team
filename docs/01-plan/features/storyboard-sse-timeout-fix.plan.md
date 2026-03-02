# 스토리보드 SSE 타임아웃 수정 계획

## 문제

콘텐츠 마케팅 → 스토리보드 생성 시 "Request timed out" 에러가 지속 발생.

## 근본 원인

1. **SSE 전용 API Route 누락**: 웹툰 스토리보드 SSE 엔드포인트(`/api/content-marketing/script/webtoon/{jobId}/stream`)에 전용 API Route가 없어 Next.js rewrites를 통해 프록시됨. rewrites는 SSE 데이터를 버퍼링하므로 heartbeat가 클라이언트에 전달되지 않음.
2. **proxyTimeout 초과**: Next.js proxyTimeout(120초) < 웹툰 파이프라인 소요 시간(3~5분+)
3. **기존 패턴 미적용**: 동일 모듈의 `keywords/collect/stream`과 `script/generate`에는 이미 SSE 전용 API Route가 존재하지만, 웹툰 추가 시 누락됨

## 기존 패턴 (참조)

| SSE 엔드포인트 | API Route | 타임아웃 |
|---------------|-----------|---------|
| `keywords/collect/stream` | `frontend/src/app/api/content-marketing/keywords/collect/stream/route.ts` | 120초 |
| `script/generate` | `frontend/src/app/api/content-marketing/script/generate/route.ts` | 180초 |
| `script/webtoon/{jobId}/stream` | **없음 (누락)** | - |

## 수정 계획

### 변경 파일 (3개)

| # | 파일 | 변경 내용 |
|---|------|----------|
| 1 | `frontend/src/app/api/content-marketing/script/webtoon/[jobId]/stream/route.ts` | **신규 생성** — SSE 프록시 API Route |
| 2 | `frontend/src/features/content-marketing/services/index.ts` | SSE URL 경로 수정 (불필요할 수 있음 — API Route가 rewrites보다 우선) |
| 3 | `frontend/next.config.js` | 코멘트 업데이트 (웹툰 SSE도 API Route 처리 명시) |

### 파일 1: SSE 프록시 API Route (신규)

`frontend/src/app/api/content-marketing/script/webtoon/[jobId]/stream/route.ts`

기존 `script/generate/route.ts` 패턴을 따르되:
- **GET 메서드** (POST가 아님 — 백엔드가 GET)
- **타임아웃 300초** (이미지 생성 10패널 × 30초 = 5분)
- SSE 버퍼링 방지 헤더 (`X-Accel-Buffering: no`)
- `ReadableStream`으로 백엔드 SSE를 클라이언트에 패스스루

```typescript
export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

const BACKEND_URL = process.env.BACKEND_URL || 'http://127.0.0.1:8000'
const PROXY_TIMEOUT_MS = 300_000 // 5분 (이미지 생성 고려)

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ jobId: string }> }
) {
  const { jobId } = await params

  const backendResponse = await fetch(
    `${BACKEND_URL}/api/content-marketing/script/webtoon/${jobId}/stream`,
    {
      method: 'GET',
      headers: { Accept: 'text/event-stream' },
      signal: AbortSignal.timeout(PROXY_TIMEOUT_MS),
    },
  )

  // ReadableStream 패스스루 (버퍼링 없이)
  const reader = backendResponse.body.getReader()
  const stream = new ReadableStream({
    async start(controller) {
      while (true) {
        const { done, value } = await reader.read()
        if (done) { controller.close(); break }
        controller.enqueue(value)
      }
    },
    cancel() { reader.cancel() },
  })

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache, no-transform',
      'Connection': 'keep-alive',
      'X-Accel-Buffering': 'no',
    },
  })
}
```

### 파일 2: 프론트엔드 SSE URL 확인

`frontend/src/features/content-marketing/services/index.ts` 319행:
```typescript
const url = `/api${BASE}/script/webtoon/${jobId}/stream`
// → /api/content-marketing/script/webtoon/{jobId}/stream
```

이 URL은 이미 `/api/content-marketing/...` 형식이므로, 새 API Route(`/api/content-marketing/script/webtoon/[jobId]/stream/route.ts`)에 자동 매칭됨. **변경 불필요**.

### 파일 3: next.config.js 코멘트 업데이트

```javascript
// /api/content-marketing/keywords/collect/stream 은 Next.js API Route에서 SSE 프록시 처리
// /api/content-marketing/script/generate 도 API Route에서 SSE 프록시 처리
// /api/content-marketing/script/webtoon/{jobId}/stream 도 API Route에서 SSE 프록시 처리 ← 추가
// (rewrites는 SSE 스트리밍을 버퍼링하므로 API Route 사용)
```

## 검증 계획

1. `npm run build` — 타입 에러 확인
2. 브라우저에서 스토리보드 생성 → 타임아웃 없이 SSE 이벤트 수신 확인
3. Network 탭에서 SSE 응답이 chunked로 오는지 확인 (버퍼링 없음)

## 위험도

- **Fast-track 대상**: 기존 패턴 내 버그 수정, 고위험 영역 아님
- **영향 범위**: 프론트엔드 SSE 프록시만 변경, 백엔드 무변경
