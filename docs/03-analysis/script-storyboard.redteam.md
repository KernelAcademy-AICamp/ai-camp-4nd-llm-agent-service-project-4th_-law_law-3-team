# Red Team 검증 보고서 — 대본 + 웹툰 스토리보드

> **검증 대상**: `docs/01-plan/features/script-storyboard.plan.md`
> **검증 도구**: Gemini CLI (Red Team)
> **검증일**: 2026-02-28

---

## 1. 취약점 (Critical/High/Medium/Low)

### [Critical] Resource Exhaustion (DoS) & API Cost Attack
- 하나의 `job_id`가 8~14개의 고비용 이미지 생성(Gemini 3 Pro Image)을 유발
- 공격자가 루프를 통해 `/webtoon` POST 요청을 난사할 경우, API 비용 폭증 및 백엔드 작업 큐 마비 확실시
- Rate Limiting 및 할당량 관리 부재

### [High] Insecure Direct Object Reference (IDOR)
- `job_id`가 유추 가능한 패턴(예: incremental integer)일 경우, 타 사용자의 스토리보드 및 대본 내용 탈취 가능
- 반드시 UUID v4 또는 암호학적으로 안전한 토큰을 사용해야 함

### [High] Prompt Injection (Indirect)
- 사용자가 입력한 `topic`이나 `sections`에 AI 지침을 무력화하는 프롬프트 포함 가능
- 변호사 브랜드에 치명적인 부적절한 이미지 생성 위험

### [Medium] SSE Connection Leak
- HTTP/1.1 기반 브라우저의 도메인당 동시 연결 제한(6개)
- 다수 탭/동시 생성 시 서비스 먹통 가능
- HTTP/2 권장

### [Low] Image URL Exposure
- 생성된 이미지 URL이 Public이면 권한 없는 제3자 접근 가능
- Signed URL 또는 인증 기반 서빙 필요

---

## 2. 아키텍처 개선 제안

- **비동기 워커 기반 작업 분리**: API 서버가 직접 AI 파이프라인 실행 대신 Redis/RabbitMQ + Celery/Task Queue 방식 전환 (작업 상태 영속성 보장)
- **캐릭터 일관성 강화**: 단순 텍스트 주입/1번 패널 참조만으로는 복잡한 각도에서 캐릭터 붕괴. LoRA 학습 또는 Reference-only ControlNet 유사 기법 적용 권고
- **이미지 캐싱 레이어**: 동일/유사 대본 섹션에 대해 Vector Similarity Search 통한 기존 이미지 재사용

---

## 3. 고급 기능 추가 제안

- **Inter-Panel Editing (In-painting)**: 특정 패널의 캐릭터 표정/배경만 수정하는 인페인팅 기능
- **Brand-Specific Style Transfer**: 법무법인 고유 톤앤매너/로고 자동 합성 워터마킹/브랜딩 필터
- **Voice-over Preview**: 스토리보드 패널별 TTS 결합하여 영상 흐름 오디오 프리뷰

---

## 4. 성능 최적화 제안

- **1번 패널 우선 → 나머지 병렬**: 1번 패널(Master Reference)만 우선 생성 → 나머지 패널은 1번을 참조하여 병렬 생성 → 전체 소요 시간 30초 내외로 단축
- **Lazy Loading & Placeholder**: 모든 이미지 완료 대기 대신, `panel_complete` 이벤트마다 개별 로딩 + 고품질 스켈레톤 UI

---

## 5. 운영 안정성 제안

- **Cost Monitoring Dashboard**: 사용자/조직별 AI API 소모 비용 실시간 모니터링, 이상 징후 시 Circuit Breaker
- **Dead Letter Queue (DLQ)**: Safety Filter/오류로 실패한 패널 별도 관리 + 관리자 알림
- **Audit Logging**: 누가/언제/어떤 키워드로 이미지 생성했는지 전수 로그 (저작권/부적절 콘텐츠 소명 자료)
- **Fallback Strategy**: Gemini 3 Pro Image 장애 시 Flux.1 또는 Stable Diffusion으로 Failover
