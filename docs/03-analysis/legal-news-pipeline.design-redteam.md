# Red Team 설계 리뷰 보고서: Legal News Pipeline Design

> **검증 도구**: Gemini CLI (Red Team)
> **검증 대상**: `docs/02-design/features/legal-news-pipeline.design.md` (v0.1.0)
> **검증 일시**: 2026-02-26

---

## 1. 아키텍처 일관성

- **[확인]** `app/tools/` 배치 + `app/services/` 오케스트레이터 구조 적절
- **[보완]** 뉴스 소비 API 모듈 필요 (`app/modules/legal_news`) — 프론트엔드 조회용 라우터/스키마
- **[확인]** 기존 인프라 재사용 (`get_chat_model`, `create_query_embedding`, `async_session_factory`)

## 2. 누락된 컴포넌트/엣지 케이스

- **[보완]** 운영 알림 체계 — Slack Webhook/Email 등 실패 즉시 알림 수단 필요
- **[보완]** 인덱싱 복구 로직 — `is_indexed=false`인 문서 일괄 재처리 CLI 커맨드
- **[보완]** HTML 크롤링 구조 변경 대비 — Selector/Metadata-driven parser 패턴 도입

## 3. 확장성

- **[확인]** Strategy 패턴 (`BaseNewsSource`) 소스 확장성 적절
- **[대안]** 소스 10개+ 확장 시 TaskIQ/Celery 비동기 큐 분리 고려
- **[확인]** 멱등성 (`ON CONFLICT DO UPDATE`) 안정적

## 4. 대안 접근 방식

- **[대안]** Fuzzy Deduplication — SimHash/MinHash 활용 유사도 95% 이상 중복 제거
- **[보완]** 법령/판례 ID 링크 자동화 — `ReferenceValidator`에서 유효 참조 시 ID 저장

## 5. 보안

- **[확인]** PII 마스킹 적절
- **[보완]** SSRF 방어 — httpx 클라이언트에서 내부망 IP 요청 차단 로직
