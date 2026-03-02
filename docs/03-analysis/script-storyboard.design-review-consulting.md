# 외부 컨설팅 설계 리뷰 — Codex CLI

> **검증 대상**: `docs/02-design/features/script-storyboard.design.md` (v1.0)
> **검증 도구**: Codex CLI (gpt-5.3-codex, External Consultant)
> **검증일**: 2026-02-28

---

## 종합 진단

현재 설계는 **MVP~초기 상용 수준**으로 양호.
업계 최고 수준 대비 **운영 신뢰성(워크플로우 내구성), MLOps 평가체계, 보안 거버넌스**에서 격차.

---

## 1. 업계 최고 수준 대비 격차

### 강점
- 이벤트 기반 스트리밍(SSE), 패널 단위 재생성, 상태 스키마 분리
- `scene → prompt → image` 3단계 분리 — 책임 경계 명확

### 핵심 격차
| 항목 | 현재 | 권장 |
|------|------|------|
| 백그라운드 실행 | 앱 프로세스 `asyncio.create_task` | Celery/큐 기반 워커 분리 |
| 상태 머신 | 4개 상태 | 6개+ (`queued/running/retrying/partial_failed/completed/cancelled`) |
| 운영 지표 | 부재 | SLI/SLO + 자동 평가 루프 |
| 파일 제공 | StaticFiles 직접 마운트 | signed URL + 만료 정책 |

## 2. 최신 기술 트렌드

### 반영된 부분
- 스트리밍 UX, 멀티체인 파이프라인, 구조화 이벤트

### 미반영/보완 필요
- 모델명 고정 → **모델 라우터 + capability 기반 선택** 필요
- 관측성 → **OpenTelemetry 표준** 전 구간 계측
- AI 리스크 → **OWASP LLM Top 10, NIST AI RMF** 체계화

## 3. 확장성/유지보수성

### 현재 리스크
- `asyncio.Semaphore(3)` — 단일 인스턴스만 유효, 수평 확장 시 글로벌 제어 불가
- SSE 다중 탭/연결 — HTTP/2 전제, heartbeat/reconnect 설계 필요

### 권고 아키텍처
- API/Worker 분리 + 내구성 큐 (Redis/RabbitMQ)
- DB 기반 명시적 상태전이
- 지수 백오프 + jitter + DLQ
- idempotency key + 패널 단위 체크포인트

## 4. AI/ML 파이프라인 고도화

### 즉시 적용 가능
- 프롬프트 해시 기반 캐시 (동일 입력 재사용)
- 실패 대체(PIL) vs 실제 생성 이미지 품질 플래그 구분

### 다음 단계
- 자동 평가 파이프라인: 스타일 일관성, 텍스트 가독성(OCR), 법률 정확성(HITL)
- 캐릭터 시트/속성 고정 프롬프트 도입
- Temporal/Celery 기반 내구성 워크플로우

## 5. 보안 검토

### 좋은 점
- Pydantic 검증, path traversal 방어, 상태 관리 구조

### 보완 필수
- 프롬프트 인젝션: 패턴 필터만으로 불충분 → 입력 분리(시스템/유저) + 출력 검증 + 정책 게이트
- 이미지 URL: public 노출 최소화 → signed URL + 만료 정책
- 권한 검사: 멀티테넌트 기준 작업 소유권 + 감사로그
- 비용 방지: 사용자별 rate limit, quota, 비용 상한

## 우선순위 로드맵 (권장)

| 시기 | 항목 |
|------|------|
| 0~2주 | 내구성 큐 도입, 상태머신 확장, idempotency, OTel 기본 계측 |
| 2~6주 | signed URL 전환, 권한/감사로그, 재시도/DLQ, 모델 라우터 추상화 |
| 6~12주 | 자동 품질평가 + HITL 승인 플로우, A/B 실험, 비용/품질 최적화 |

---

## 참고 소스
- [FastAPI BackgroundTasks](https://fastapi.tiangolo.com/tutorial/background-tasks/)
- [Celery retries/backoff](https://docs.celeryq.dev/en/v5.4.0/userguide/tasks.html)
- [Gemini API changelog](https://ai.google.dev/gemini-api/docs/changelog)
- [MDN SSE 연결 제한](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events)
- [OpenTelemetry Python](https://opentelemetry.io/docs/languages/python/instrumentation/)
- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [NIST GenAI Profile](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence)

---

## Agent Team 수용 판단

| 피드백 | 수용 여부 | 사유 |
|--------|----------|------|
| Celery/큐 기반 워커 분리 | **기록** | 현재 MVP는 단일 인스턴스 운영, 기존 storyboard 모듈도 `asyncio.create_task` 패턴 사용 중. 향후 트래픽 증가 시 도입 |
| 상태 머신 확장 (6개+) | **부분 수용** | `retrying` 상태를 추가하여 재시도 추적 가능하게 개선. 나머지는 MVP 이후 |
| 프롬프트 해시 캐시 | **수용** | `_get_cache_key()` 함수가 이미 설계에 존재, 캐시 히트 시 스킵 로직 추가 |
| 플레이스홀더 vs 실제 이미지 구분 | **수용** | `is_placeholder` 필드가 이미 반환됨, 프론트엔드에서도 구분 표시 |
| 모델 라우터 추상화 | **기록** | 현재 `STORYBOARD_IMAGE_MODEL` 환경변수로 교체 가능, 런타임 라우터는 향후 과제 |
| signed URL 전환 | **기록** | 현재 로컬 개발 환경, 프로덕션 배포 시 적용 |
| 프롬프트 인젝션 강화 | **부분 수용** | 시스템/유저 입력 분리는 설계에 이미 반영 (Chain 1 프롬프트 구조), 출력 검증 게이트 추가 |
| rate limit / quota | **기록** | 인증 시스템 도입 시 함께 적용 |
| SSE heartbeat/reconnect | **수용** | SSE 30초 heartbeat + 프론트엔드 자동 재연결 로직 추가 |
| 감사로그 | **기록** | 인증 시스템 도입 시 함께 적용 |
| OTel 계측 | **기록** | 향후 과제 (현재 logger 기반 모니터링) |
