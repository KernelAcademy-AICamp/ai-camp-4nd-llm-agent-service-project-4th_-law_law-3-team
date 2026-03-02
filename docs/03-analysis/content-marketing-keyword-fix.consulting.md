# [External Consultant] 콘텐츠 마케팅 키워드 탐색 컨설팅 보고서

**대상 프로젝트:** law-3 (Legal President)
**작성자:** External Consultant (Codex CLI / gpt-5.3-codex)
**작성일:** 2026-03-01
**대상 기획서:** `docs/01-plan/features/content-marketing-keyword-fix.plan.md`
**상태:** 개선 권고 (고도화 방안 포함)

---

## 1. 격차 분석 (현재 vs 업계 상위)

### 1.1 신뢰성(Resilience) 격차
- **현재:** 소스별 실패가 전체 검색 품질 저하로 직결되고, `safe_fetch()`에서 에러가 소실되어 관측 가능성이 낮음.
- **업계 상위권:** 소스 단위 Circuit Breaker, Timeout Budget, Retry with Jitter, Fallback 체계가 기본.
- **격차:** 장애 격리/복구 자동화가 부족해 "부분 장애가 전체 UX 장애"로 전이됨.

### 1.2 데이터 품질/정합성 격차
- **현재:** `language=ko` 미지원 호출, `status` 미검증, naive datetime, `published_at=now()` 강제 등으로 신뢰도 저하.
- **업계 상위권:** 수집-정규화-검증(Validation)-품질점수(Quality Score) 파이프라인이 분리 운영됨.
- **격차:** 스키마 검증·시간 정규화·소스별 계약 테스트(contract test)가 미흡.

### 1.3 운영성(Ops) 격차
- **현재:** 캐시 클리어 기능은 추가되었으나 Rate Limiting/권한 통제가 핵심 리스크.
- **업계 상위권:** Admin API 보호(ACL, audit log, rate limit, idempotency key) 기본 적용.
- **격차:** 운영 API 보안·감사 체계 부족.

### 1.4 제품 경쟁력 격차
- **현재:** 7개 중 3개만 안정 동작, 모델명 하드코딩, 소스 가중치 정적.
- **업계 상위권:** 실시간 성능 기반 동적 라우팅, 공급자 추상화, A/B 자동 최적화.
- **격차:** "고정 로직" 중심이라 품질/비용 최적화가 어려움.

---

## 2. 기술 고도화 제안

### 2.1 안정성 아키텍처 (최우선)
- 소스별 Circuit Breaker(half-open 포함), Bulkhead, Exponential Backoff 도입.
- `safe_fetch()`를 "에러 삼킴"에서 "구조화 에러 반환(Result[T, E])"으로 변경.
- 표준 에러 코드 체계(`SOURCE_TIMEOUT`, `INVALID_PAYLOAD`, `AUTH_FAILED`) 정의.

### 2.2 공급자 추상화/동적 라우팅
- Perplexity 모델명 하드코딩 제거: 설정 레지스트리 + Feature Flag로 전환.
- 동적 소스 가중치: 최근 성공률, p95 지연, 최신성, 중복률 기반 점수화.
- 점수 기반 Top-N 팬아웃 + 조기종료(early cutoff)로 비용/속도 최적화.

### 2.3 관측성(Observability)
- OpenTelemetry + Prometheus: 소스별 성공률, latency, 빈 결과율(0-hit) 대시보드화.
- SLO 예시: "검색 요청 성공률 99.5%, p95 2.5s".
- 알림: 특정 소스 0건 연속 N회, auth 오류 급증, 스키마 검증 실패율 급증.

### 2.4 보안/운영 제어
- 캐시 클리어 API: 사용자 권한 분리(Admin only), IP+사용자 기준 Rate Limiting, 감사로그 필수.
- 키 미설정 사전검증(boot-time health check)으로 "실행 중 장애"를 "배포 전 실패"로 이동.

---

## 3. UX 개선 제안

### 3.1 진행 상태 UX 개선
- SSE 진행바 0% 고정은 해결 방향 적절(API Route 프록시).
- **추가 권고:** 단계형 진행 상태(소스 수집→정규화→랭킹→완료) + 예상 남은 시간(ETA) 표시.
- 부분 성공 시 "3/7 소스 완료" 명시로 실패를 숨기지 않음.

### 3.2 실패 커뮤니케이션
- "결과 없음"과 "소스 장애"를 분리 표기.
- 소스별 상태 배지(정상/지연/오류) 제공, 재시도 CTA 제공.

### 3.3 사용자 신뢰 강화
- 뉴스 항목에 원본 발행시각(UTC→로컬 변환)과 수집시각 분리 표기.
- 중복 기사 병합 및 출처 다양성 점수 노출(편향 완화).

### 3.4 운영 UX
- 캐시 클리어 버튼은 확인 모달 + 영향 범위 표시(전체/소스별/키별) + 롤백 옵션 권장.

---

## 4. 데이터 파이프라인 고도화 제안

### 4.1 수집 계층
- 커넥터별 표준 인터페이스(`fetch()`, `validate()`, `normalize()`)로 통일.
- 키/권한/엔드포인트 사전 헬스체크 자동화.

### 4.2 정규화/검증 계층
- `published_at`는 원본값 보존 + 파싱 실패 시 별도 필드(`parsed_at`, `parse_error`) 저장.
- timezone-aware datetime 강제(naive 금지).
- 스키마 검증 실패 시 격리 큐(Dead Letter Queue)로 이동.

### 4.3 품질 평가/랭킹 계층
- 소스 신뢰도, 최신성, 중복도, 본문 충실도 기반 품질 스코어 계산.
- 동적 가중치 시스템을 온라인 러닝 또는 주기적 배치로 업데이트.

### 4.4 제공/캐시 계층
- 다층 캐시(L1 메모리, L2 Redis) + 소스별 TTL 차등.
- 캐시 무효화는 태그 기반(주제/소스/시간창)으로 세분화.

### 4.5 거버넌스
- 계약 테스트(소스 응답 스키마), 회귀 테스트(대표 쿼리셋), 카나리 릴리스 적용.
- 핵심 KPI: 소스 가용률, 빈결과율, 중복률, 사용자 클릭률, 응답 p95.

---

## [우선순위 권장]

1. Circuit Breaker + 구조화 에러 + Rate Limiting
2. datetime/스키마 정합성 수정
3. 동적 소스 가중치 + 관측성 대시보드
4. 단계형 진행 UX + 부분 성공 표시
