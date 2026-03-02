# 외부 컨설팅 보고서: 법률 뉴스 통계 대시보드 개선

> **Feature**: `legal-news-stats-enhance`
> **검증일**: 2026-02-28
> **검증 도구**: Codex CLI (gpt-5.3-codex, read-only)

---

## 외부 컨설팅 보고서

### 1. 격차 분석 (현재 vs 목표)

| 영역 | 현재 | 목표 (업계 상위) | 격차 |
|------|------|-----------------|------|
| 제품 역량 | 모듈형 API + 멀티에이전트 | Agentic Workflow, 대량 문서 처리, 워크플로우 빌더 | 질문응답 중심 → 워크플로우 전환 필요 |
| 신뢰성 | RAG 기반 답변 | 권위 출처 인용 검증, 감사 이력, 정책 통제 | 근거 신뢰성 UX 노출 필요 |
| 운영 성숙도 | 에이전트 오케스트레이션 | Durable execution, human-in-the-loop, 관측성 | 품질 측정 자동화 미흡 |

### 2. 기술 고도화 제안

- LangGraph 체크포인터(Postgres) 기반 재시작/중단/승인/재개 표준화
- 태스크별 모델 라우팅 (요약/분류는 경량, 법리 추론은 고성능)
- 답변을 "주장-근거" 구조로 강제, 근거 없는 문장 차단
- OpenTelemetry 기반 전구간 관측

### 3. UX 개선 제안

- **성능 체감 개선**: Next.js App Router 스트리밍/Suspense로 초기 응답 개선
- 긴 작업은 백그라운드 실행 + 알림 복귀 (타임아웃 체감 제거)

### 4. AI/ML 고도화 제안

- Hybrid retrieval(BM25+벡터)+재정렬(reranker)+질문 분해(query decomposition) — *이미 구현 완료*
- 관계형 지식 그래프 병행 — *PostgreSQL Recursive CTE로 이관 완료*
- 평가 체계: grounded answer rate, citation precision, hallucination rate KPI

### 5. 비즈니스 전략 제안

- 상품 계층화: Starter / Pro / Enterprise
- 수익화: 좌석 + 워크플로우 실행 + 고급 검증팩 부가과금
- 확장 우선순위: 계약검토 → 소송문서 → eDiscovery

---

## PM 분석: 현재 Feature 기획에 대한 반영 판단

> Codex CLI는 기획 보고서 원문이 stdin으로 불완전하게 전달되어 플랫폼 전반에 대한 전략적 리뷰를 제공함.
> 본 Feature(통계 대시보드 개선)에 직접 해당하는 제안만 선별 반영.

| # | 제안 | 채택 | 사유 |
|---|------|------|------|
| 1 | 스트리밍/Suspense 체감 개선 | 미반영 | 본 Feature 스코프 외 (향후 전체 개선) |
| 2 | 태스크별 모델 라우팅 | 미반영 | 본 Feature와 무관 |
| 3 | 평가 체계 KPI | 참고 | RAG 기여도 인포그래픽에서 간접적으로 "데이터 기여" 지표 제공 |
| 4 | 상품 계층화 | 미반영 | 비즈니스 전략 차원, 본 Feature 스코프 외 |
