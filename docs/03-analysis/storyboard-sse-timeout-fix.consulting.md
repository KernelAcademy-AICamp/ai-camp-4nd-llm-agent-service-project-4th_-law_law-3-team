# 외부 컨설팅 보고서: 스토리보드 SSE 타임아웃 수정

> **검증 대상**: `docs/01-plan/features/storyboard-sse-timeout-fix.plan.md`
> **검증 도구**: Codex CLI (External Consultant)
> **검증일**: 2026-03-01

---

## 1. 격차 분석 (현재 vs 목표)

- **진단 정확**: "웹툰 SSE만 API Route 누락"은 코드베이스와 일치
- 업계 최고 수준 대비 주요 격차:
  1. **타임아웃 전략**: `AbortSignal.timeout(300s)`는 연결 성공 후 스트림 진행 중에도 300초 도달 시 강제 종료 → 장기 SSE에 부적합
  2. **재연결 복원성**: 클라이언트는 fetch 예외 시에만 재시도, 정상 EOF가 `all_done` 없이 발생하면 복구 불가
  3. **이벤트 재개(Resume)**: `id`/`Last-Event-ID` 기반 재개 전략 없음 → 재연결 시 이벤트 유실/중복
  4. **운영 관측성**: stream 생존시간, TTFE, 비정상 종료율, 재시도 횟수 지표 설계 없음
  5. **Next.js 14 정합성**: `params: Promise<{ jobId: string }>`는 Next.js 14 기준 권장 시그니처와 차이 (동작은 가능)

## 2. 기술 개선 제안

1. **타임아웃 2단계화**: connect timeout(15~30초)만 적용, 연결 성립 후 타임아웃 해제 → 키워드 SSE 패턴 참조
2. **클라이언트 비정상 EOF 복구**: `done=true`인데 `all_done` 미수신 → 오류 간주, backoff 재연결 + `getWebtoonJobStatus()` 동기화
3. **프록시 취소 전파**: `request.signal` 결합 → 클라이언트 이탈 시 업스트림 즉시 중단
4. **에러 응답 규격 통일**: 504/502/5xx에 `error_code`, `retryable`, `job_id` 포함
5. **패턴 공통화**: SSE 프록시 라우트 공통 유틸 추출 (헤더, 스트림 패스스루, 예외 매핑)

## 3. 운영 안정성 제안

### 필수 메트릭
- `sse_open_total`, `sse_close_total`, `sse_abnormal_close_total`
- `sse_ttfb_ms`, `sse_duration_ms`, `sse_retry_count`

### 구조화 로그
- 키: `jobId`, `route`, `close_reason`, `duration_ms`
- 오류 로그는 100% 수집

### 사전 검증 시나리오
- 10패널/20패널/동시 20~50 연결 soak test
- 네트워크 단절 후 재연결 회복 검증
- 브라우저 탭 백그라운드 전환/복귀 검증

## 4. 종합 평가

**조건부 승인 (Approve with changes)**
- 필수 반영: 연결 전용 타임아웃, 비정상 EOF 재연결, 관측성(메트릭/로그)
- 권장 반영: `id`/`Last-Event-ID` 기반 재개, SSE 프록시 공통화

## 참고 소스
- [Next.js Route Handlers](https://nextjs.org/docs/app/building-your-application/routing/route-handlers)
- [Next.js Rewrites](https://nextjs.org/docs/app/api-reference/config/next-config-js/rewrites)
- [MDN SSE](https://developer.mozilla.org/en-US/docs/Web/API/EventSource)
- [MDN AbortSignal.timeout](https://developer.mozilla.org/en-US/docs/Web/API/AbortSignal/timeout_static)
