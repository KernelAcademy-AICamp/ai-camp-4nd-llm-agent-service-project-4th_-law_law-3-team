# Content Marketing 키워드 탐색 4가지 이슈 수정 기획서

> **Summary**: 콘텐츠 마케팅 모듈의 키워드 탐색 기능에서 발견된 4가지 이슈(SSE 진행바 미작동, 캐싱 클리어 부재, 데이터 소스 미작동, 뉴스 검색 소스 실패)를 분석하고 해결 방안을 수립한다.
>
> **Project**: law-3 (Legal President)
> **Author**: PM (Product Manager)
> **Date**: 2026-03-01
> **Status**: Draft (v1.2)
> **Related Plan**: `content-marketing-source-fix.plan.md` (v2.0, 이슈 3/4의 일부 중복)

---

## 1. 개요

### 1.1 목적

콘텐츠 마케팅 키워드 탐색 기능에서 사용자 경험을 저해하는 4가지 이슈를 수정하여, 키워드 수집 및 뉴스 검색의 신뢰성과 투명성을 개선한다.

### 1.2 배경

키워드 탐색 기능은 3단계 파이프라인(커뮤니티 수집 -> LLM 키워드 추출 -> 뉴스 검색)으로 작동한다. 이 과정에서 다음 4가지 이슈가 발견되었다:

1. **SSE 진행바 미작동**: 진행바가 0%에서 멈추고, 완료 시 한번에 100%로 점프
2. **캐싱 클리어 부재**: 3개의 인메모리 캐시에 대한 수동 클리어 수단 없음
3. **데이터 소스 미작동**: 7개 중 3개만 작동 (Tavily/Naver/Perplexity)
4. **뉴스 검색 소스 실패**: 뉴스 검색에서도 일부 소스가 무음 실패

### 1.3 관련 문서

- 기존 소스 수정 기획서: `docs/01-plan/features/content-marketing-source-fix.plan.md` (v2.0)
- 분석 보고서: `docs/03-analysis/content-marketing-source-fix.analysis.md`
- Red Team: `docs/03-analysis/content-marketing-source-fix.redteam.md`
- QA 전략: `docs/03-analysis/content-marketing-source-fix.qa-strategy.md`

### 1.4 기존 기획과의 관계

`content-marketing-source-fix.plan.md` (v2.0)는 이슈 3/4의 백엔드 로직(소스 확장, safe_fetch_with_status, 보안 필터링)을 상세히 다룬다. 본 기획서는 해당 기획서에서 다루지 않은 **이슈 1(SSE)**, **이슈 2(캐시 클리어)**를 중심으로 하고, 이슈 3/4에 대해서는 기존 기획서의 진행 상황을 정리하며 누락된 관점을 보완한다.

| 이슈 | 기존 기획 커버리지 | 본 기획 역할 |
|------|-------------------|-------------|
| 이슈 1: SSE 진행바 | 미다룸 | **신규 기획** |
| 이슈 2: 캐시 클리어 | 미다룸 | **신규 기획** |
| 이슈 3: 소스 미작동 | v2.0에서 상세 다룸 (RC-1, FR-01) | 진행 상황 확인 + 보완 |
| 이슈 4: 뉴스 검색 실패 | v2.0에서 상세 다룸 (RC-2, RC-4) | 진행 상황 확인 + 보완 |

---

## 2. 이슈별 상세 분석

### 2.1 이슈 1: SSE 진행바가 0%에서 움직이지 않음

#### 현상

키워드 수집 시 SSE 스트리밍을 통해 진행 상황(10% -> 40% -> 50% -> 80% -> 90% -> 100%)을 전달하지만, 프론트엔드에서 진행바가 0%에 멈춰 있다가 수집 완료 시 한번에 100%로 점프한다.

#### 근본 원인

**`next.config.js`의 rewrites 프록시가 SSE 스트리밍을 버퍼링한다.**

```javascript
// next.config.js line 42-44
{
  source: '/api/content-marketing/:path*',
  destination: 'http://127.0.0.1:8000/api/content-marketing/:path*',
},
```

Next.js의 rewrites 프록시는 응답을 버퍼링하여 한번에 전달하므로, SSE 이벤트가 실시간으로 클라이언트에 도달하지 않는다. 이 문제는 코드 내에 이미 주석으로 명시되어 있다:

```javascript
// next.config.js line 49-50
// /api/chat/stream은 Next.js API Route에서 SSE 프록시 처리
// (rewrites는 SSE 스트리밍을 버퍼링하므로 API Route 사용)
```

동일한 문제를 `chat/stream`과 `script/generate`에서는 이미 API Route로 해결한 상태이다:

| SSE 엔드포인트 | 프록시 방식 | SSE 작동 |
|----------------|-----------|---------|
| `/api/chat/stream` | API Route (`frontend/src/app/api/chat/stream/route.ts`) | 정상 |
| `/api/content-marketing/script/generate` | API Route (`frontend/src/app/api/content-marketing/script/generate/route.ts`) | 정상 |
| `/api/content-marketing/keywords/collect/stream` | **rewrites** (next.config.js) | **버퍼링됨** |

#### 영향

- 사용자가 키워드 수집 중 진행 상황을 확인할 수 없어 "멈춘 것인지 작동 중인지" 판단 불가
- 15초 이상 소요될 수 있는 키워드 수집에서 UX 저하 (무반응 화면)

#### 코드 증거

프론트엔드 서비스(`frontend/src/features/content-marketing/services/index.ts` line 196)에서 SSE 요청 URL:

```typescript
const url = `/api${BASE}/keywords/collect/stream?max_keywords=${maxKeywords}&time_range=${timeRange}`
```

이 URL은 `/api/content-marketing/keywords/collect/stream`으로, rewrites 규칙(`/api/content-marketing/:path*`)에 의해 백엔드로 프록시되며 버퍼링이 발생한다.

---

### 2.2 이슈 2: 캐싱 클리어 버튼 없음

#### 현상

키워드 수집 결과가 캐시되어 동일한 요청을 반복하면 캐시된 결과가 반환된다. 그러나 사용자가 "최신 결과로 갱신"하고 싶을 때 캐시를 수동으로 클리어할 방법이 없다.

#### 캐시 현황

3개의 인메모리 캐시가 존재한다:

| 캐시 | 위치 | TTL | 키 형식 | 용도 |
|------|------|-----|---------|------|
| `_keyword_cache` | `content_marketing_service.py` | 1시간 (`KEYWORD_COLLECT_CACHE_TTL`) | `{user_id}:{time_range}:keywords` | 키워드 수집 결과 |
| `_trend_detail_cache` | `content_marketing_service.py` | 무제한 (LRU 200건) | `{trend_id}` | 트렌드 상세 조회 |
| `TrendCollector._cache` | `collector.py` | 24시간 (`CONTENT_MARKETING_CACHE_TTL`) | 요청 기반 | 트렌드 수집 결과 |

#### 영향

- 사용자가 새로운 트렌드/이슈가 발생했을 때 즉시 갱신된 결과를 확인할 수 없음
- 캐시 TTL(1시간)이 경과하기 전까지 동일 결과만 반복 표시
- 디버깅/테스트 시 캐시 영향을 배제하기 어려움

#### 설계 의도와 현실

캐시 TTL이 config에서 관리되므로(`settings.KEYWORD_COLLECT_CACHE_TTL`) 서버 설정 변경으로 TTL을 줄일 수 있지만, 이는 운영자 영역이며 사용자가 접근할 수 없다. "새로고침" 의미의 캐시 클리어 기능이 UI에 필요하다.

---

### 2.3 이슈 3: 일부 데이터 소스 미작동

#### 현상

7개 검색 소스 중 Tavily, Naver, Perplexity만 키워드 수집에 참여하고, YouTube, Google CSE, NewsData, NewsAPI는 참여하지 않는다.

#### 근본 원인

**기존 기획서(`content-marketing-source-fix.plan.md` v2.0)의 RC-1과 동일.**

`collector.py:collect_community_keywords()` (line 272-285)에서 Tavily + Naver만 호출하고 나머지 4개 소스는 호출하지 않는 설계 제한이 원인이다.

#### 현재 구현 상태 분석

코드 분석 결과, 기존 기획서에서 계획한 수정 중 일부가 이미 반영되어 있다:

| 항목 | 기획서 v2.0 계획 | 현재 코드 상태 |
|------|-----------------|---------------|
| `safe_fetch_with_status()` | 추가 예정 | **이미 구현됨** (sources/__init__.py) |
| `SourceFetchResult` | 추가 예정 | **이미 구현됨** (sources/__init__.py) |
| `SourceFailInfo` 스키마 | 추가 예정 | **이미 구현됨** (schema/__init__.py) |
| `sources_used` 하드코딩 제거 | 수정 예정 | **이미 수정됨** (캐시에 소스 정보 포함) |
| SSE 스트리밍에서 Secondary 소스 호출 | 수정 예정 | **이미 구현됨** (`collect_keywords_stream` line 454-468) |
| 비-SSE `collect_community_keywords` 확장 | 수정 예정 | **미반영** (여전히 Tavily+Naver만 호출) |
| `collect_community_keywords_with_status` | 미계획 | **이미 구현됨** (서비스에서 호출) |

**핵심 발견**: SSE 스트리밍 경로(`collect_keywords_stream`)는 이미 Secondary 소스를 호출하도록 구현되었지만, 비-SSE 경로(`collect_community_keywords`)는 여전히 Tavily+Naver만 호출한다. 다만, 비-SSE 경로는 `collect_community_keywords_with_status`가 별도로 구현되어 서비스 레이어에서 대체 사용 중이다.

#### 추가 확인 필요 사항

4개 소스(YouTube, Google CSE, NewsData, NewsAPI)의 API 키가 설정되지 않아 `is_available=False`인 경우, 소스가 참여하지 않는 것은 정상 동작이다. 실제 API 키 설정 여부를 확인하여 진짜 미작동인지 API 키 미설정인지를 구분해야 한다.

| 소스 | 필요 환경변수 | 미설정 시 동작 |
|------|-------------|---------------|
| YouTube | `YOUTUBE_API_KEY` | `is_available=False`, 자동 스킵 |
| Google CSE | `GOOGLE_CSE_API_KEY` + `GOOGLE_CSE_ID` | `is_available=False`, 자동 스킵 |
| NewsData | `NEWSDATA_API_KEY` | `is_available=False`, 자동 스킵 |
| NewsAPI | `NEWSAPI_API_KEY` | `is_available=False`, 자동 스킵 |

**사용자 관점 이슈**: API 키가 미설정되어 소스가 스킵되는 것이 정상 동작이라 하더라도, 사용자에게 "왜 3개 소스만 작동하는지" 안내가 없어 혼란을 초래한다.

---

### 2.4 이슈 4: 뉴스 검색 시 일부 소스 실패

#### 현상

키워드 선택 후 뉴스 검색(`search_news_for_keyword`)에서도 YouTube, Perplexity, NewsData, NewsAPI 소스가 결과를 반환하지 않는 경우가 있다.

#### 근본 원인

**기존 기획서의 RC-2, RC-4와 관련.**

1. **`safe_fetch()` 사용으로 에러 정보 누락**: `search_news_for_keyword()`의 `_fetch_with_timeout()` 내부에서 `safe_fetch()`를 사용하여, 소스별 실패 원인(인증 오류, 파싱 오류 등)이 무음 처리된다. 타임아웃만 추적되고 나머지 에러 유형은 빈 리스트로 반환되어 "결과 없음"과 "에러"를 구분할 수 없다.

2. **소스별 적합성 이슈**:
   - YouTube: 뉴스 검색보다 동영상 검색에 적합하여 관련 결과가 부족할 수 있음
   - NewsData: `timeframe=48` 제한과 검색 기본 `7d`의 불일치
   - Perplexity: 검색 엔진 특성상 실시간 뉴스보다 요약/분석에 강함

3. **에러 추적 불완전**: `fail_reasons` 딕셔너리가 타임아웃 에러만 추적하고, `safe_fetch()` 내부에서 발생하는 인증/파싱/네트워크 에러는 "unknown"으로 분류된다.

#### 현재 코드의 에러 추적 한계

```python
# collector.py line 343-356
fail_reasons: dict[str, str] = {}

async def _fetch_with_timeout(source: BaseTrendSource) -> list[RawTrendItem]:
    timeout = source_timeouts.get(source.name.value, 15)
    try:
        return await asyncio.wait_for(
            source.safe_fetch(sanitized, config),  # safe_fetch는 에러 시 [] 반환
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        fail_reasons[source.name.value] = "timeout"
        return []
```

`safe_fetch()`가 내부 에러를 삼키고 빈 리스트를 반환하므로, `_fetch_with_timeout`은 "빈 결과"와 "에러"를 구분할 수 없다. 결과적으로 line 383-391에서 빈 결과를 반환한 소스가 "실패"로 분류되지만 에러 유형은 항상 "unknown"이다.

---

## 3. 해결 방안

### 3.1 이슈 1 해결: SSE 프록시 API Route 추가

#### 방안

기존 `chat/stream`과 `script/generate`에서 사용한 패턴을 동일하게 적용하여, 키워드 수집 SSE 스트리밍용 API Route를 추가한다.

#### 구현

**신규 파일**: `frontend/src/app/api/content-marketing/keywords/collect/stream/route.ts`

```typescript
/**
 * 키워드 수집 SSE 스트리밍 프록시 API Route
 *
 * Next.js rewrites가 SSE를 버퍼링하는 문제를 우회하기 위해
 * API Route에서 직접 스트리밍 프록시한다.
 */
import { NextRequest } from 'next/server'

const BACKEND_URL = process.env.BACKEND_URL || 'http://127.0.0.1:8000'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams
  const maxKeywords = searchParams.get('max_keywords') || '10'
  const timeRange = searchParams.get('time_range') || '48h'

  const backendUrl = `${BACKEND_URL}/api/content-marketing/keywords/collect/stream?max_keywords=${maxKeywords}&time_range=${timeRange}`

  const backendResponse = await fetch(backendUrl, {
    method: 'GET',
    headers: { Accept: 'text/event-stream' },
  })

  if (!backendResponse.ok || !backendResponse.body) {
    return new Response(
      JSON.stringify({ error: 'Backend request failed' }),
      { status: backendResponse.status, headers: { 'Content-Type': 'application/json' } },
    )
  }

  const reader = backendResponse.body.getReader()
  const stream = new ReadableStream({
    async start(controller) {
      try {
        while (true) {
          const { done, value } = await reader.read()
          if (done) { controller.close(); break }
          controller.enqueue(value)
        }
      } catch (error) {
        controller.error(error)
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

**프론트엔드 서비스 변경**: `services/index.ts`의 `streamKeywordCollect()`에서 URL 경로 변경 불필요. API Route는 기존 URL 패턴(`/api/content-marketing/keywords/collect/stream`)과 동일하므로, Next.js가 API Route를 우선 매칭하여 rewrites를 건너뛴다.

단, 확실한 분리를 위해 rewrites에서 SSE 경로를 제외하는 것을 권장한다:

```javascript
// next.config.js - content-marketing rewrites를 세분화
{
  source: '/api/content-marketing/keywords/collect/stream',
  // API Route가 처리하므로 rewrites 불필요 (삭제 또는 주석 처리)
  // Next.js는 API Route > rewrites 우선이므로 자동으로 API Route가 매칭됨
},
```

#### 설계 원칙

- 기존 `chat/stream` API Route(`route.ts`)와 동일한 패턴 사용 (검증된 패턴 재사용)
- `ReadableStream`으로 청크 단위 전달, 버퍼링 없음
- `X-Accel-Buffering: no` 헤더로 리버스 프록시 버퍼링도 방지
- GET 메서드 사용 (기존 백엔드 엔드포인트가 GET)

---

### 3.2 이슈 2 해결: 캐시 클리어 기능 추가

#### 3.2.1 백엔드: 캐시 클리어 API

**엔드포인트**: `DELETE /api/content-marketing/keywords/cache`

```python
# router/__init__.py 추가
@router.delete("/keywords/cache", status_code=200)
async def clear_keyword_cache_endpoint() -> dict[str, str]:
    """키워드 캐시 클리어 (현재 사용자의 캐시만)"""
    user_id = TEMP_USER_ID  # TODO: Auth
    cleared_count = clear_keyword_cache(user_id)
    return {"status": "cleared", "cleared_count": str(cleared_count)}
```

```python
# content_marketing_service.py 추가
def clear_keyword_cache(user_id: str) -> int:
    """특정 사용자의 키워드 캐시를 클리어한다.

    Returns:
        삭제된 캐시 항목 수
    """
    keys_to_remove = [k for k in _keyword_cache if k.startswith(f"{user_id}:")]
    for key in keys_to_remove:
        del _keyword_cache[key]
    logger.info("키워드 캐시 클리어: user=%s, count=%d", user_id, len(keys_to_remove))
    return len(keys_to_remove)
```

#### 3.2.2 프론트엔드: 캐시 클리어 버튼

키워드 수집 결과 표시 영역에 "새로 수집" 버튼을 추가한다. 캐시 히트 상태일 때 "캐시된 결과입니다. 새로 수집하려면 클릭하세요" 안내와 함께 표시한다.

```typescript
// services/index.ts 추가
export async function clearKeywordCache(): Promise<void> {
  await api.delete(`${BASE}/keywords/cache`)
}
```

**UI 동작 흐름**:
1. 키워드 수집 결과에 `cache_hit=true` 표시
2. "새로 수집" 버튼 클릭
3. `clearKeywordCache()` API 호출
4. 자동으로 `handleCollect()` 재실행 (새 데이터 수집)

#### 설계 원칙

- 사용자별 캐시만 클리어 (다른 사용자 캐시에 영향 없음)
- 전체 캐시 클리어가 아닌 키워드 캐시만 대상 (트렌드 상세 캐시는 별도)
- DELETE 메서드 사용 (RESTful 의미론)
- 캐시 클리어 후 자동 재수집으로 사용자 경험 최소화

---

### 3.3 이슈 3 해결: 소스 가용 상태 표시 + 구현 검증

#### 3.3.1 기존 기획 진행 상황 확인

코드 분석 결과, `content-marketing-source-fix.plan.md` v2.0의 핵심 수정 사항 중 상당수가 이미 구현되었다:

- `safe_fetch_with_status()`, `SourceFetchResult`: 구현 완료
- `SourceFailInfo` 스키마: 구현 완료
- `sources_used` 하드코딩 제거: 구현 완료 (캐시에 소스 정보 포함)
- SSE 스트리밍에서 Secondary 소스 호출: 구현 완료

#### 3.3.2 추가 필요 작업

1. **소스 가용 상태 안내 UI**: 사용자에게 어떤 소스가 활성/비활성/실패인지 표시하는 UI 컴포넌트. 이는 기존 기획서 FR-05(Should)와 동일하며 아직 미구현 상태.

2. **비-SSE 경로 정합성 확인**: `collect_community_keywords()`(비-SSE)와 `collect_keywords_stream()`(SSE)의 소스 호출 범위가 다른 점을 확인하고, 비-SSE 경로가 실제로 사용되는 곳이 있는지 검증. 현재 서비스 레이어에서는 `collect_community_keywords_with_status()`를 사용하므로 직접적인 문제는 없지만, 일관성 확보가 필요.

3. **API 키 미설정 안내 메시지**: API 키가 미설정된 소스에 대해 "API 키가 설정되지 않아 이 소스를 사용할 수 없습니다"라는 명확한 안내.

---

### 3.4 이슈 4 해결: 뉴스 검색 에러 추적 개선

#### 3.4.1 `safe_fetch_with_status()` 활용

`search_news_for_keyword()`의 `_fetch_with_timeout()`에서 `safe_fetch()` 대신 `safe_fetch_with_status()`를 사용하도록 변경한다.

```python
# AS-IS (collector.py line 346-356)
async def _fetch_with_timeout(source: BaseTrendSource) -> list[RawTrendItem]:
    timeout = source_timeouts.get(source.name.value, 15)
    try:
        return await asyncio.wait_for(
            source.safe_fetch(sanitized, config),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        fail_reasons[source.name.value] = "timeout"
        return []

# TO-BE
async def _fetch_with_timeout(source: BaseTrendSource) -> SourceFetchResult:
    timeout = source_timeouts.get(source.name.value, 15)
    try:
        return await asyncio.wait_for(
            source.safe_fetch_with_status(sanitized, config),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return SourceFetchResult(
            items=[], source_name=source.name.value,
            is_success=False, error_type="timeout",
            error_message="응답 시간 초과",
        )
```

이로써 `fail_reasons` 딕셔너리를 별도로 관리할 필요 없이, `SourceFetchResult`에서 에러 정보를 직접 추출할 수 있다.

#### 3.4.2 소스별 에러 분류 개선

`SourceFetchResult.error_type`에 의해 다음 에러 유형이 구분된다:

| error_type | 의미 | 사용자 표시 메시지 |
|-----------|------|-------------------|
| `timeout` | 응답 시간 초과 | "응답 시간 초과" |
| `auth` | 인증 오류 (401/403) | "API 키 확인 필요" |
| `rate_limit` | 할당량 초과 (429) | "API 할당량 초과" |
| `network` | 네트워크 연결 실패 | "소스 연결 오류" |
| `parse` | 응답 파싱 오류 | "응답 형식 오류" |
| `not_available` | API 키 미설정 | "API 키 미설정" |
| `unknown` | 기타 오류 | "알 수 없는 오류" |

---

## 4. 구현 우선순위 및 의존 관계

### 4.1 MoSCoW 분류

| 우선순위 | 이슈 | 항목 | 사유 |
|---------|------|------|------|
| **Must** | 이슈 1 | SSE 프록시 API Route 추가 | UX 핵심. 진행바가 작동하지 않으면 사용자가 기능 고장으로 인식 |
| **Must** | 이슈 4 | 뉴스 검색 `safe_fetch_with_status()` 전환 | 에러 정보가 없으면 소스 실패 원인 추적 불가 |
| **Should** | 이슈 2 | 캐시 클리어 API + UI 버튼 | 편의 기능이지만 캐시 TTL 1시간은 너무 길 수 있음 |
| **Should** | 이슈 3 | 소스 가용 상태 표시 UI | 사용자 투명성. 기존 기획 FR-05와 동일 |
| **Could** | 이슈 2 | 트렌드 상세 캐시 + TrendCollector 캐시도 클리어 | 키워드 캐시만으로 충분할 수 있음 |

### 4.2 의존 관계

```
이슈 1 (SSE) ─────────── 독립 (프론트엔드만)
이슈 2 (캐시) ────────── 독립 (백엔드 API + 프론트엔드 버튼)
이슈 3 (소스 표시) ───── 이슈 4에 의존 (sources_failed 데이터 필요)
이슈 4 (뉴스 에러) ───── 독립 (백엔드 collector.py 수정)
```

### 4.3 구현 순서

| 단계 | 이슈 | 작업 | 예상 소요 | 누적 |
|------|------|------|----------|------|
| **1단계** | 이슈 1 | SSE 프록시 API Route 추가 | 0.5일 | 0.5일 |
| **2단계** | 이슈 4 | 뉴스 검색 `safe_fetch_with_status()` 전환 | 0.5일 | 1일 |
| **3단계** | 이슈 2 | 캐시 클리어 API + 프론트엔드 버튼 | 0.5일 | 1.5일 |
| **4단계** | 이슈 3 | 소스 가용 상태 표시 UI | 1일 | 2.5일 |
| **총 예상 소요** | | | **2.5일** | |

---

## 5. 영향 범위 (수정 대상 파일 목록)

### 5.1 이슈 1: SSE 프록시 (Must)

| 파일 | 변경 유형 | 설명 |
|------|----------|------|
| `frontend/src/app/api/content-marketing/keywords/collect/stream/route.ts` | **신규** | SSE 프록시 API Route |

### 5.2 이슈 2: 캐시 클리어 (Should)

| 파일 | 변경 유형 | 설명 |
|------|----------|------|
| `backend/app/services/service_function/content_marketing_service.py` | 수정 | `clear_keyword_cache()` 함수 추가 |
| `backend/app/modules/content_marketing/router/__init__.py` | 수정 | `DELETE /keywords/cache` 엔드포인트 추가 |
| `frontend/src/features/content-marketing/services/index.ts` | 수정 | `clearKeywordCache()` API 함수 추가 |
| `frontend/src/features/content-marketing/hooks/useKeywordFlow.ts` | 수정 | 캐시 클리어 + 재수집 핸들러 추가 |
| `frontend/src/features/content-marketing/components/KeywordCollector.tsx` | 수정 | "새로 수집" 버튼 UI 추가 |

### 5.3 이슈 3: 소스 상태 표시 (Should)

| 파일 | 변경 유형 | 설명 |
|------|----------|------|
| `frontend/src/features/content-marketing/components/KeywordCollector.tsx` | 수정 | 소스 상태 배지 컴포넌트 추가 |
| `frontend/src/features/content-marketing/components/KeywordNewsList.tsx` | 수정 | 뉴스 소스 상태 표시 |

### 5.4 이슈 4: 뉴스 검색 에러 추적 (Must)

| 파일 | 변경 유형 | 설명 |
|------|----------|------|
| `backend/app/tools/trend/collector.py` | 수정 | `search_news_for_keyword()`에서 `safe_fetch_with_status()` 사용 |

### 5.5 참조 파일 (변경 없음)

| 파일 | 참조 사유 |
|------|----------|
| `frontend/src/app/api/chat/stream/route.ts` | SSE 프록시 패턴 참조 |
| `frontend/src/app/api/content-marketing/script/generate/route.ts` | SSE 프록시 패턴 참조 |
| `frontend/next.config.js` | rewrites 규칙 확인 |
| `backend/app/tools/trend/sources/__init__.py` | `safe_fetch_with_status()` 인터페이스 확인 |
| `backend/app/core/config.py` | 캐시 TTL 설정 확인 |

---

## 6. 리스크 및 대안

### 6.1 리스크 평가

| 리스크 | 영향 | 가능성 | 완화 방안 |
|--------|------|--------|----------|
| API Route 추가 시 기존 rewrites와 충돌 | Low | Low | Next.js는 API Route를 rewrites보다 우선 매칭하므로 충돌 없음. `chat/stream`에서 검증된 패턴 |
| 캐시 클리어 API가 남용될 가능성 (Rate Limit 없음) | Low | Medium | 기존 `_collect_limiter`를 캐시 클리어에도 적용하거나, 별도 limiter 추가 |
| `safe_fetch_with_status()` 전환 시 반환 타입 변경으로 기존 로직 영향 | Medium | Low | `search_news_for_keyword()` 내부만 변경. 외부 인터페이스(반환 타입)는 동일 유지 |
| SSE 프록시 추가로 메모리/연결 리소스 증가 | Low | Low | 기존 `chat/stream` 프록시가 이미 운영 중이므로 동일 수준 |
| `collect_community_keywords()`와 `collect_keywords_stream()`의 소스 범위 불일치 | Low | High | 비-SSE 경로가 직접 사용되지 않으므로 당장 영향 없음. 장기적으로 통합 필요 |

### 6.2 하위호환성 보장

| 항목 | 보장 방법 |
|------|----------|
| SSE 프록시 | 기존 URL 패턴 유지. API Route가 우선 매칭되므로 프론트엔드 코드 변경 불필요 |
| 캐시 클리어 API | 신규 엔드포인트 추가만. 기존 API 변경 없음 |
| `search_news_for_keyword()` | 반환 타입 `tuple[list[NewsArticle], list[str], list[SourceFailInfo]]` 유지 |
| 스키마 | 기존 필드 변경/삭제 없음 |

### 6.3 대안 검토

#### 이슈 1 대안: EventSource Polyfill

SSE 프록시 대신 프론트엔드에서 백엔드 URL을 직접 호출하는 방안. CORS 설정이 필요하고 프로덕션 환경에서 백엔드 URL 노출 문제가 있어 **거부**. API Route 패턴이 이미 검증되어 있으므로 동일 패턴을 적용하는 것이 합리적.

#### 이슈 2 대안: force_refresh 쿼리 파라미터

캐시 클리어 API를 별도로 만들지 않고, 기존 키워드 수집 API에 `force_refresh=true` 파라미터를 추가하는 방안. 더 간결하지만, RESTful 의미론에서 부작용(캐시 삭제)을 GET/POST에 포함하는 것은 부적절. 별도 DELETE 엔드포인트가 명확하므로 **원안 유지**.

---

## 7. 일정 계획

### 7.1 전체 일정

> **총 일정: 약 8일** (3트랙 병렬 실행 기준, 최장 트랙 A 기준)

#### 트랙 A: 백엔드 (7.9일, 크리티컬 패스)

| 단계 | 기간 | 산출물 | 검증 |
|------|------|--------|------|
| A-1: SSE 프록시 | 0.5일 | API Route 파일 | 브라우저에서 진행바 실시간 업데이트 확인 |
| A-2: 뉴스 에러 추적 | 0.5일 | collector.py 수정 | API 응답의 `sources_failed` 에러 유형 확인 |
| A-3: 캐시 클리어 | 0.5일 | API + UI | "새로 수집" 버튼으로 캐시 클리어 + 재수집 확인 |
| A-4: 소스 상태 UI | 1일 | 컴포넌트 수정 | 소스별 상태 배지 표시 확인 |
| A-5: 채택 항목 반영 | 0.4일 | Rate Limiter, 타임아웃, 확인 모달 등 | 기존 채택 5건 반영 |
| A-6: Boot 헬스체크 (E-6) | 0.5일 | `health.py` 신규 | lifespan 시 소스별 키 검증 확인 |
| A-7: Circuit Breaker (C-5) + Bulkhead+Backoff (E-11) | 1일 | `circuit_breaker.py` 신규 | CircuitBreakerRegistry + asyncio.Semaphore + 지수 백오프 |
| A-8: 동적 가중치 (C-6) | 0.5일 | `source_stats.py` 신규 | 성공률+지연+품질 기반 가중치 자동 조정 |
| A-9: 파이프라인 분리 (E-3) | 1일 | `pipeline/` 디렉토리 | normalizer, validator, scorer 분리 |
| A-10: Redis 캐시 (R-2+E-10) | 1일 | `cache.py` 신규 | L1+L2 TwoLevelCache, Redis 미설정 시 L1만 |
| A-11: Prometheus 관측성 (E-4 BE) | 1일 | `metrics.py` 신규 | prometheus_client 미설치 시 no-op |
| 검증 | 포함 | 린트/빌드 | `ruff check`, `mypy` |
| **소계** | **7.9일** | | |

#### 트랙 B: 프론트엔드 (5.5일, 트랙 A와 병렬)

| 단계 | 기간 | 산출물 | 검증 |
|------|------|--------|------|
| B-1: 단계형 진행 UX (E-1) | 2.5일 | `KeywordCollectProgress.tsx` 신규 | SSE 이벤트에 phase/source_status/eta_seconds 반영 확인 |
| B-2: 관측성 대시보드 FE (E-4 FE) | 3일 | `SourceObservabilityDashboard.tsx` 신규 | `NEXT_PUBLIC_SHOW_DEV_TOOLS`로 숨김 처리, GET /sources/stats 연동 |
| 검증 | 포함 | 빌드 | `npm run build` |
| **소계** | **5.5일** | | |

#### 트랙 C: QA (6일, 트랙 A/B와 병렬)

| 단계 | 기간 | 산출물 | 검증 |
|------|------|--------|------|
| C-1: 계약+회귀 테스트 (E-9) | 6일 | respx 기반 httpx mock, 7개 소스 계약/회귀 테스트 | CI 통합 (GitHub Actions) |
| **소계** | **6일** | | |

#### 병렬화 타임라인

```
Day 1    2    3    4    5    6    7    8
├────────────────────────────────────────┤
│ 트랙 A (백엔드): 기존 이슈(2.9일) → 보류 항목(5일)    │ ← 크리티컬 패스
│ 트랙 B (프론트엔드): E-1(2.5일) → E-4 FE(3일)       │
│ 트랙 C (QA): E-9 계약+회귀 테스트(6일)               │
└────────────────────────────────────────┘
총 일정: 최장 트랙(A) 기준 약 8일
```

### 7.2 검증 기준

| 단계 | 검증 항목 | 방법 |
|------|----------|------|
| 1단계 | SSE 이벤트가 실시간으로 프론트엔드에 전달되는가 | 브라우저 Network 탭에서 SSE 이벤트 타이밍 확인 |
| 1단계 | 진행바가 10% -> 40% -> 50% -> 80% -> 90% -> 100% 순서로 업데이트되는가 | 시각적 확인 |
| 2단계 | `sources_failed`에 구체적 에러 유형(timeout/auth/rate_limit)이 표시되는가 | curl로 API 응답 확인 |
| 3단계 | 캐시 클리어 후 다음 수집이 캐시 미스로 새 데이터를 가져오는가 | `cache_hit=false` 확인 |
| 4단계 | 소스별 상태(성공/실패/미설정)가 UI에 표시되는가 | 시각적 확인 |

---

## 8. 성공 기준

### 8.1 정량적 기준

| 지표 | 목표 | 측정 방법 |
|------|------|----------|
| SSE 진행바 실시간 업데이트 | 각 단계별 이벤트가 수신 후 500ms 이내에 UI 반영 | 브라우저 Performance 탭 |
| 캐시 클리어 응답 시간 | 200ms 이내 | API 응답 시간 |
| 뉴스 검색 에러 분류 정확도 | 에러 소스의 100%가 구체적 error_type을 가짐 ("unknown" 최소화) | API 응답 검증 |
| 정적 검증 통과 | `ruff check`, `mypy`, `npm run build` 모두 통과 | CI 파이프라인 |

### 8.2 정성적 기준

- 사용자가 키워드 수집 중 진행 상황을 실시간으로 확인할 수 있다
- 사용자가 캐시된 결과를 인지하고, 필요 시 새로 수집할 수 있다
- 사용자가 어떤 소스가 작동/실패/미설정인지 한눈에 파악할 수 있다
- 운영자가 뉴스 검색 소스의 실패 원인을 API 응답으로 즉시 진단할 수 있다

---

## 9. 백엔드 소스별 상세 분석 (코드 리뷰 기반)

### 9.1 소스별 발견된 버그/이슈

| 소스 | 심각도 | 이슈 | 상세 |
|------|--------|------|------|
| **NewsAPISource** | Critical | `language=ko` 미지원 | NewsAPI.org Everything 엔드포인트 공식 지원 언어에 `ko`가 없음 → 항상 0건 반환. 실질적 소스 불능 |
| **NewsDataSource** | High | `status` 필드 미검증 | API가 HTTP 200을 반환하며 `status: "error"`를 포함할 수 있음. `data.get("results", [])`가 dict를 순회하여 TypeError 발생 |
| **YouTubeSource** | Medium | naive datetime 반환 | `datetime.strptime()`이 timezone-naive datetime 반환. 다른 소스의 timezone-aware datetime과 비교 시 TypeError 가능 |
| **GoogleSource** | Medium | `published_at` 항상 `now()` | 응답의 메타 태그에서 날짜를 추출하지 않고 수집 시각으로 고정. 날짜 기반 정렬/필터 무력화 |
| **PerplexitySource** | Medium | `sonar` 모델명 하드코딩 | 모델 deprecated 시 HTTP 400/404 반환. config로 분리 필요 |
| **PerplexitySource** | Medium | 외부 타임아웃(10s) < 내부 httpx 타임아웃(30s) | 외부 `asyncio.wait_for(10s)`가 먼저 취소되지만 내부 httpx 연결이 정리되지 않을 수 있음 |
| **collector.py** | Medium | `search_news_for_keyword()`의 `safe_fetch()` | 에러 유형 전량 소실, "unknown"만 반환 |

### 9.2 소스별 수정 방안

#### NewsAPISource (Critical)
- `language` 파라미터 제거하고, 검색 쿼리에 한국어 키워드만으로 검색
- 또는 `language` 파라미터를 config로 분리하여 비활성화 가능하게 설정
- 대안: `domains` 파라미터로 한국 뉴스 도메인 지정

#### NewsDataSource (High)
- 응답에서 `status` 필드 검증 로직 추가
- `"error"` 상태일 때 `TrendSourceError` 발생

#### YouTubeSource (Medium)
- `_parse_youtube_date()` 반환값에 `timezone.utc` 추가

#### GoogleSource (Medium)
- 응답의 `pagemap.metatags[0].article:published_time` 파싱 시도, 실패 시 `now()` 폴백

#### PerplexitySource (Medium)
- 모델명을 `settings.PERPLEXITY_MODEL`로 config 분리
- 타임아웃 불일치 해소: 내부 httpx 타임아웃을 외부 `source_timeouts`와 동기화

## 10. 에이전트 팀 검토 요청

| 파트 | 검토 항목 |
|------|----------|
| **프론트엔드 개발자** | SSE 프록시 API Route 구현, 캐시 클리어 UI, 소스 상태 배지 컴포넌트, `next.config.js` rewrites와 API Route 우선순위 동작 확인 |
| **백엔드 개발자** | `search_news_for_keyword()` `safe_fetch_with_status()` 전환, 캐시 클리어 서비스 함수, `collect_community_keywords()`와 SSE 경로의 소스 범위 일관성 |
| **QA 엔지니어** | SSE 스트리밍 테스트 계획 (버퍼링 vs 실시간 차이 검증), 캐시 클리어 후 재수집 시나리오, 소스별 실패 시나리오 |
| **UI/UX 디자이너** | 소스 상태 배지 디자인, 캐시 히트/클리어 UI 흐름, 진행바 UX |

---

## 10. 아키텍처 고려사항

### 10.1 프로젝트 수준

- **Enterprise 수준**: 기존 모듈형 아키텍처(FastAPI + Next.js)에 SSE 프록시 패턴이 이미 확립되어 있어, 동일 패턴 재사용

### 10.2 핵심 아키텍처 결정

| 결정 | 선택 | 근거 |
|------|------|------|
| SSE 프록시 방식 | API Route (기존 패턴) | `chat/stream`, `script/generate`에서 이미 검증됨 |
| 캐시 클리어 방식 | 별도 DELETE 엔드포인트 | RESTful 의미론 준수, `force_refresh` 파라미터보다 명확 |
| 에러 추적 방식 | `SourceFetchResult` 사용 | 기존 `safe_fetch_with_status()` 인프라 재사용 |
| 소스 상태 데이터 흐름 | 백엔드 API 응답의 `sources_failed` 필드 | 별도 health check API 없이도 수집 결과에서 실패 정보 획득 |

---

## 11. 외부 검증 피드백 통합

### 11.1 검증 개요

| 검증 주체 | 보고서 경로 | 핵심 관점 |
|----------|-----------|----------|
| Red Team (Gemini CLI) | `docs/03-analysis/content-marketing-keyword-fix.redteam.md` | 보안 취약점, 아키텍처 약점, 성능 최적화 |
| External Consultant (Codex CLI) | `docs/03-analysis/content-marketing-keyword-fix.consulting.md` | 업계 격차 분석, 기술 고도화, UX 개선, 데이터 파이프라인 |

---

### 11.2 공통 권고 사항 (양측 합의 영역)

두 검증 주체가 동일하게 지적한 사항으로, 신뢰도가 높은 권고이다.

| # | 공통 권고 | Red Team 표현 | Consultant 표현 | 기획서 반영 상태 |
|---|----------|--------------|-----------------|---------------|
| C-1 | **에러 소실 문제 (`safe_fetch()`)** | [Critical] 에러 소실, 데이터 무결성 이슈 | 에러 삼킴 → 구조화 에러 반환 필요 | **이미 반영** (Section 3.4, 이슈 4 해결) |
| C-2 | **NewsData `status` 필드 미검증** | [Critical] TypeError 위험 | 스키마 검증 미흡 | **이미 반영** (Section 9.1, NewsDataSource High) |
| C-3 | **캐시 클리어 API Rate Limiting 필요** | [Medium] DoS/Wallet-Drain 공격 가능 | Admin only + Rate Limiting + 감사로그 | **부분 반영** (Section 6.1 리스크로 인식, 구체적 구현 미포함) |
| C-4 | **YouTube naive datetime** | [Low] 타임존 불일치 | timezone-aware datetime 강제 | **이미 반영** (Section 9.2, YouTubeSource 수정 방안) |
| C-5 | **Circuit Breaker 패턴 도입** | 고급 기능 제안 #2 | 최우선 안정성 아키텍처 | **미반영** (기획서에 없음) |
| C-6 | **동적 소스 가중치 시스템** | 고급 기능 제안 #1 | 공급자 추상화/동적 라우팅 | **미반영** (기획서에 없음) |
| C-7 | **Perplexity 모델명 하드코딩 제거** | 아키텍처 약점 | 설정 레지스트리 + Feature Flag | **이미 반영** (Section 9.2, config 분리 계획) |
| C-8 | **NewsAPI `language=ko` 미지원** | Dead Code 상태 | 신뢰도 저하 원인 | **이미 반영** (Section 9.1, Critical) |
| C-9 | **GoogleSource `published_at=now()` 고정** | 심각한 설계 오류 | 날짜 기반 필터 무력화 | **이미 반영** (Section 9.2, 메타 태그 파싱 계획) |

---

### 11.3 Red Team 고유 권고 사항

Red Team만 제안한 항목으로, 보안/인프라 관점이 강하다.

| # | 권고 | 심각도 | 현재 기획 반영 |
|---|------|--------|-------------|
| R-1 | SSE 프록시 추가 시 서버 메모리 부하/커넥션 타임아웃 재설계 | Medium | 부분 반영 (리스크 6.1에 Low 영향으로 평가) |
| R-2 | 인메모리 캐시 → 분산 캐시(Redis) 전환 | High | 미반영 |
| R-3 | 비동기 병렬 호출 시 소스별 차등 타임아웃 ('빠른 실패' 유도) | Medium | 부분 반영 (Perplexity 타임아웃 불일치만 언급) |
| R-4 | SSE 데이터 스트림 압축 (불필요 필드 제거) | Low | 미반영 |
| R-5 | Cross-Language 검색 브릿지 (키워드 영→한 번역) | Low | 미반영 |

---

### 11.4 External Consultant 고유 권고 사항

Consultant만 제안한 항목으로, 제품 경쟁력/UX/거버넌스 관점이 강하다.

| # | 권고 | 우선순위 | 현재 기획 반영 |
|---|------|---------|-------------|
| E-1 | 단계형 진행 상태 UX (소스 수집→정규화→랭킹→완료) + ETA | High | 미반영 (현재 단순 % 진행바만 계획) |
| E-2 | 부분 성공 표시 ("3/7 소스 완료") | High | 미반영 |
| E-3 | 수집-정규화-검증-품질점수 파이프라인 분리 | Medium | 미반영 (단일 호출 구조) |
| E-4 | OpenTelemetry + Prometheus 관측성 대시보드 | Medium | 미반영 |
| E-5 | 캐시 클리어 시 확인 모달 + 영향 범위 표시 | Medium | 미반영 (자동 재수집만 계획) |
| E-6 | 키/권한/엔드포인트 boot-time 헬스체크 | Medium | 미반영 |
| E-7 | 뉴스 원본 발행시각과 수집시각 분리 표기 | Low | 미반영 |
| E-8 | 중복 기사 병합 + 출처 다양성 점수 | Low | 미반영 |
| E-9 | 계약 테스트(소스 응답 스키마) + 회귀 테스트 | Medium | 미반영 |
| E-10 | 다층 캐시(L1 메모리 + L2 Redis) + 태그 기반 무효화 | Medium | 미반영 |
| E-11 | Bulkhead + Exponential Backoff | Medium | 미반영 |

---

### 11.5 채택/보류/거부 결정

#### 판단 기준

- **채택**: 현재 스프린트에 추가 가능하고, 4가지 이슈 해결에 직접 기여
- **채택(축소)**: 아이디어는 유효하지만, 현재 스프린트에서는 축소 버전만 포함
- **채택(v1.2 전환)**: v1.1에서 보류되었으나, 파트별 분석으로 구현 가능성이 확인되어 채택
- **거부**: 현재 프로젝트 단계에 부적합하거나, 비용 대비 효과 불명확

#### 채택 항목

| # | 권고 | 출처 | 결정 | 반영 방법 | 추가 공수 |
|---|------|------|------|----------|----------|
| C-3 | 캐시 클리어 API Rate Limiting | 공통 | **채택** | Section 3.2 캐시 클리어 엔드포인트에 기존 `_collect_limiter`와 동일한 Rate Limiter 적용. 분당 5회 제한. | +0.1일 |
| R-1 | SSE 프록시 커넥션 타임아웃 설정 | Red Team | **채택** | Section 3.1 API Route에 `AbortController` + 120초 타임아웃 추가. 백엔드 키워드 수집 최대 소요 시간 기준. | +0.1일 |
| R-3 | 소스별 차등 타임아웃 확대 적용 | Red Team | **채택(축소)** | Section 3.4 `_fetch_with_timeout()`에서 이미 `source_timeouts` dict 사용 중. 현재 기본값 15초를 소스 특성별로 조정 (Tavily: 10s, Perplexity: 20s, YouTube: 8s, NewsData: 10s). 전면 Fail Fast 아키텍처가 아닌, 타임아웃 값 튜닝 수준. | +0.05일 |
| E-2 | 부분 성공 표시 ("N/M 소스 완료") | Consultant | **채택(축소)** | Section 3.3 소스 상태 UI에 "성공 N개 / 전체 M개" 텍스트 표시 추가. 진행바 단계 분리가 아닌, 결과 요약 레벨에서 표시. | +0.1일 |
| E-5 | 캐시 클리어 확인 모달 | Consultant | **채택(축소)** | Section 3.2 "새로 수집" 버튼 클릭 시 `window.confirm()` 수준의 간단한 확인. 영향 범위 표시/롤백 옵션은 보류. | +0.05일 |

**채택 항목 추가 공수 합계: +0.4일** (기존 2.5일 → 2.9일, 허용 범위 내)

#### 채택 항목 (보류 → 채택, 파트별 분석 기반)

> v1.2에서 보류 11건 전체를 채택으로 전환. 백엔드/프론트엔드/QA 파트별 분석을 통해 구현 방법과 공수가 구체화되었으며, 3트랙 병렬 실행으로 총 8일 내 완료 가능함이 확인되었다.

##### 백엔드 채택 항목 (7건, 5일)

권장 구현 순서: E-6 → C-5+E-11 → C-6 → E-3 → R-2+E-10 → E-4

| # | 권고 | 출처 | 결정 | 구현 방법 | 공수 |
|---|------|------|------|----------|------|
| E-6 | boot-time 헬스체크 | Consultant | **채택** | `health.py` 신규 생성. FastAPI `lifespan` 이벤트에 1줄 추가하여 소스별 API 키 검증 + 연결 테스트 수행. 독립 구현으로 다른 항목에 의존 없음. | 0.5일 |
| C-5 | Circuit Breaker 패턴 | 공통 | **채택** | `circuit_breaker.py` 신규 생성. `CircuitBreakerRegistry` 싱글턴 패턴으로 상태 관리(open/half-open/closed). E-11과 통합 구현하여 장애 격리 패턴을 일관되게 적용. | 0.5일 |
| E-11 | Bulkhead + Exponential Backoff | Consultant | **채택** | C-5와 통합 구현. `asyncio.Semaphore`로 소스별 동시 요청 수 제한(Bulkhead) + 지수 백오프로 재시도 로직 구현. Circuit Breaker와 함께 소스 안정성 3중 방어 구성. | 0.5일 |
| C-6 | 동적 소스 가중치 시스템 | 공통 | **채택** | `source_stats.py` 신규 생성. 성공률 + 응답 지연 + 품질 점수 3가지 지표 기반으로 소스별 가중치를 자동 조정. 관측 데이터 축적과 함께 정밀도 향상. | 0.5일 |
| E-3 | 수집-정규화-검증-품질점수 파이프라인 분리 | Consultant | **채택** | `pipeline/` 디렉토리 신규 생성. `normalizer.py`, `validator.py`, `scorer.py`로 기존 `collector.py` 단일 구조를 멀티 레이어로 분리. 단계별 독립 테스트 및 교체 가능한 구조. | 1일 |
| R-2 + E-10 | Redis 분산 캐시 + 다층 캐시 | Red Team + Consultant | **채택** | `cache.py` 신규 생성. L1(인메모리) + L2(Redis) `TwoLevelCache` 구현. **Redis 미설정 시 L1만 자동 사용**하여 인프라 미준비 환경에서도 동작. 태그 기반 무효화 지원. | 1일 |
| E-4 (BE) | Prometheus 관측성 (백엔드) | Consultant | **채택** | `metrics.py` 신규 생성. `prometheus_client` 미설치 시 no-op(빈 동작)으로 폴백하여 의존성 없이 동작. 소스별 호출 횟수/지연/에러율 메트릭 수집. 프론트엔드 대시보드(E-4 FE)에 데이터 제공을 위한 `GET /sources/stats` 엔드포인트 포함. | 1일 |

##### 프론트엔드 채택 항목 (2건, 5.5일)

| # | 권고 | 출처 | 결정 | 구현 방법 | 공수 |
|---|------|------|------|----------|------|
| E-1 | 단계형 진행 UX + ETA | Consultant | **채택** | `KeywordCollectProgress.tsx` 신규 생성. 백엔드 SSE 이벤트 스키마 확장 필요 (`phase`, `source_status`, `eta_seconds` 필드 추가). 소스 수집 → 정규화 → 랭킹 → 완료 4단계 진행 표시. E-3(파이프라인 분리)과 연동하여 단계별 실시간 피드백 제공. | 2.5일 |
| E-4 (FE) | 관측성 대시보드 (프론트엔드) | Consultant | **채택** | `SourceObservabilityDashboard.tsx` 신규 생성. `NEXT_PUBLIC_SHOW_DEV_TOOLS` 환경변수로 개발/운영 환경에서만 표시. 백엔드 `GET /sources/stats` 엔드포인트 연동. 소스별 성공률, 지연, 에러 추이를 시각화. | 3일 |

##### QA 채택 항목 (1건, 6일)

| # | 권고 | 출처 | 결정 | 구현 방법 | 공수 |
|---|------|------|------|----------|------|
| E-9 | 계약 테스트 + 회귀 테스트 | Consultant | **채택** | `respx`로 httpx mock 구현. 7개 소스(Tavily, Naver, YouTube, Google CSE, NewsData, NewsAPI, Perplexity) 각각에 대해 계약 테스트(응답 스키마 검증) + 회귀 테스트(기존 동작 보장) 작성. CI 통합(GitHub Actions)으로 PR 시 자동 실행. | 6일 |

#### 거부 항목

| # | 권고 | 출처 | 거부 사유 |
|---|------|------|----------|
| R-4 | SSE 데이터 스트림 압축 | Red Team | 현재 SSE 이벤트 크기가 소규모(JSON 수십~수백 바이트)이며, 이벤트 빈도도 6회(10%→100%) 수준. 압축 오버헤드가 이득보다 클 수 있음. 대규모 스트리밍이 필요해지는 시점에 재검토. |
| R-5 | Cross-Language 검색 브릿지 | Red Team | 법률 뉴스 특성상 한국어 키워드의 영어 번역이 법률 용어 정확도를 보장하기 어려움. 번역 API 호출 비용/지연이 추가되며, 한국 법률 뉴스 검색에서 영어 소스의 가치가 제한적. |
| E-7 | 원본 발행시각/수집시각 분리 표기 | Consultant | GoogleSource의 `published_at=now()` 문제가 해결(Section 9.2)된 후에야 의미 있는 기능. 현재는 신뢰할 수 없는 시각을 표시하게 되어 오히려 혼란 유발. 소스 정합성 수정 완료 후 재검토. |
| E-8 | 중복 기사 병합 + 출처 다양성 점수 | Consultant | 중복 탐지 알고리즘(유사도 비교, 클러스터링)이 필요하며, 현재 가용 소스 3개에서는 중복 발생 빈도가 낮음. 소스 확장 후 실제 중복 데이터를 확보한 뒤 설계하는 것이 합리적. |

---

### 11.6 기획서 반영 요약

#### 이미 반영 확인 (변경 없음)

기획서 Section 2~9에서 이미 다루고 있어 추가 수정이 불필요한 항목:

- C-1: `safe_fetch_with_status()` 전환 (Section 3.4)
- C-2: NewsData `status` 필드 검증 (Section 9.1, 9.2)
- C-4: YouTube timezone-aware 변환 (Section 9.2)
- C-7: Perplexity 모델명 config 분리 (Section 9.2)
- C-8: NewsAPI `language=ko` 수정 (Section 9.1, 9.2)
- C-9: GoogleSource `published_at` 메타 태그 파싱 (Section 9.2)

#### v1.1 채택 반영 사항

| 반영 대상 Section | 추가 내용 | 근거 |
|------------------|----------|------|
| Section 3.1 (SSE 프록시) | `AbortController` + 120초 타임아웃 추가 | R-1 채택 |
| Section 3.2 (캐시 클리어) | Rate Limiter (분당 5회) 적용 + `window.confirm()` 확인 | C-3, E-5 채택 |
| Section 3.3 (소스 상태 UI) | "성공 N개 / 전체 M개" 부분 성공 텍스트 추가 | E-2 채택(축소) |
| Section 3.4 (뉴스 에러 추적) | 소스별 타임아웃 값 튜닝 (소스 특성 기반) | R-3 채택(축소) |

#### v1.2 전체 구현 일정 (보류 11건 전체 채택)

v1.1에서 보류된 11건을 파트별 분석을 통해 전체 채택하고, 3트랙 병렬 실행 구조로 총 **8일** 일정을 수립하였다.

| 트랙 | 담당 | 포함 항목 | 공수 | 비고 |
|------|------|----------|------|------|
| **트랙 A (백엔드)** | 백엔드 개발자 | 기존 4이슈(2.9일) + E-6, C-5, E-11, C-6, E-3, R-2+E-10, E-4 BE(5일) | **7.9일** | 크리티컬 패스 |
| **트랙 B (프론트엔드)** | 프론트엔드 개발자 | E-1(2.5일) + E-4 FE(3일) | **5.5일** | 트랙 A와 병렬 |
| **트랙 C (QA)** | QA 엔지니어 | E-9(6일) | **6일** | 트랙 A/B와 병렬 |

```
총 공수(순차): 2.9일 + 5일 + 5.5일 + 6일 = 19.4일
총 일정(병렬): 최장 트랙(A) 기준 = 약 8일
병렬화 효율: 59% 단축 (19.4일 → 8일)
```

#### 트랙 간 의존 관계

```
트랙 A (백엔드)                  트랙 B (프론트엔드)         트랙 C (QA)
├─ E-3: 파이프라인 분리 ────────→ E-1: 단계형 진행 UX        │
│  (SSE 이벤트 스키마 확장)        (phase/source_status 소비)  │
│                                                            │
├─ E-4 BE: Prometheus 메트릭 ──→ E-4 FE: 관측성 대시보드     │
│  (GET /sources/stats 제공)      (stats API 소비)            │
│                                                            │
├─ 소스 구현 완료 ──────────────────────────────────────────→ E-9: 계약+회귀 테스트
│  (7개 소스 인터페이스 안정화)                                (소스별 mock+검증)
```

> **참고**: 트랙 B의 E-1은 트랙 A의 E-3(파이프라인 분리)에서 SSE 이벤트 스키마가 확장된 후 연동 가능. 트랙 B는 E-1의 UI 프레임워크 구축을 먼저 진행하고, 백엔드 스키마 확정 후 연동 작업을 수행한다.

---

### 11.7 PM 종합 판단

#### v1.1 판단 (채택 5건 + 보류 11건)

**Red Team 평가**: 기획서가 이미 백엔드 소스별 버그를 Section 9에서 상세히 분석하고 있어, Red Team의 Critical 지적(NewsData 필드 미검증, 에러 소실)은 **이미 해결 계획이 수립된 상태**이다. 캐시 클리어 API의 Rate Limiting 부재는 유효한 보안 지적으로 채택하였다.

**External Consultant 평가**: 업계 격차 분석이 체계적이며, Circuit Breaker, 관측성, 파이프라인 분리 제안은 아키텍처 고도화의 핵심 방향으로 적합하다.

#### v1.2 판단 (보류 11건 전체 채택)

**전체 채택 결정 배경**:

1. **파트별 분석으로 실현 가능성 확인**: 백엔드/프론트엔드/QA 각 파트에서 구체적 구현 방법과 공수를 산출하여, v1.1에서 "인프라 변경 필요" "별도 기획 필요"로 보류했던 항목들이 실제로는 합리적인 공수(각 0.5~3일) 내에 구현 가능함이 확인되었다.

2. **Graceful Degradation 설계**: 핵심 인프라 의존 항목들이 미설치/미설정 시에도 정상 동작하도록 설계되었다.
   - R-2+E-10 (Redis 캐시): Redis 미설정 시 L1(인메모리)만 자동 사용
   - E-4 (Prometheus): `prometheus_client` 미설치 시 no-op 폴백
   - E-4 FE (관측성 대시보드): `NEXT_PUBLIC_SHOW_DEV_TOOLS`로 숨김 처리

3. **3트랙 병렬화로 일정 효율 극대화**: 순차 실행 시 19.4일이 필요한 작업을 3트랙 병렬로 8일로 단축. 크리티컬 패스는 트랙 A(백엔드 7.9일)이며, 프론트엔드(5.5일)와 QA(6일)는 백엔드와 동시 진행.

4. **통합 구현의 시너지 효과**: 보류 항목을 개별 스프린트로 분산하면 매번 컨텍스트 전환 비용이 발생하지만, 현재 스프린트에서 통합 구현하면 다음과 같은 시너지가 있다:
   - C-5(Circuit Breaker) + E-11(Bulkhead+Backoff): 통합 구현으로 0.5일씩 총 1일 (개별 시 각 1일 이상 소요 예상)
   - R-2(Redis) + E-10(다층 캐시): 통합 구현으로 1일 (개별 시 각 1일 이상)
   - E-3(파이프라인 분리) → E-1(단계형 UX): 파이프라인 분리가 선행되어야 단계형 UX가 가능하므로, 같은 스프린트에서 연속 구현이 효율적

**총 공수 요약**:

| 구분 | 공수 |
|------|------|
| 기존 4가지 이슈 + v1.1 채택 5건 | 2.9일 |
| 백엔드 보류 7건 | 5일 |
| 프론트엔드 보류 2건 | 5.5일 |
| QA 보류 1건 | 6일 |
| **순차 합계** | **19.4일** |
| **병렬 실행 총 일정** | **약 8일** (최장 트랙 A 기준) |

**최종 결론**: 보류 11건 전체를 채택하여 현재 스프린트에 통합한다. Graceful Degradation 설계로 인프라 미준비 환경에서도 기본 동작이 보장되며, 3트랙 병렬 실행으로 8일 내 완료 가능하다. 이로써 키워드 탐색 기능이 단순 버그 수정 수준을 넘어, 소스 안정성(Circuit Breaker/Bulkhead), 관측성(Prometheus), 테스트 커버리지(계약/회귀 테스트), UX 고도화(단계형 진행)까지 달성하는 포괄적인 품질 개선이 된다.

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2026-03-01 | 초안 작성. 4가지 이슈 분석, 해결 방안 수립, 기존 소스 수정 기획(v2.0)과의 관계 정리, 코드 현행 분석 반영 | PM (Product Manager) |
| 1.1 | 2026-03-01 | Red Team/External Consultant 피드백 통합 (Section 11 추가). Rate Limiting, SSE 타임아웃, 부분 성공 표시, 확인 모달 채택. 향후 로드맵(Phase 2~4) 수립. | PM (Product Manager) |
| 1.2 | 2026-03-01 | 보류 11건 전체 채택. 백엔드(7건/5일), 프론트엔드(2건/5.5일), QA(1건/6일) 파트별 분석 기반 구현 방법 및 공수 확정. 3트랙 병렬 실행으로 총 일정 2.9일→8일 확장. Section 7.1 전체 일정 재구성, Section 11.5 보류→채택 전환, Section 11.6 구현 일정으로 교체, Section 11.7 전체 채택 판단 추가. | PM (Product Manager) |
