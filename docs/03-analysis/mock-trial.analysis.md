# Mock Trial (모의 법정) Gap Analysis Report

> **Analysis Type**: Gap Analysis (Design vs Implementation)
>
> **Project**: law-3-team (법률 서비스 플랫폼)
> **Date**: 2026-02-21 (v0.2 보강)
> **Baseline**: Plan v0.5 + Design v0.3.0 기준 분석 (v0.1), Design v0.4.0 반영 후 재분석 (v0.2)
> **Design Doc**: [mock-trial.design.md](../02-design/features/mock-trial.design.md)
> **Plan Doc**: [mock-trial.plan.md](../01-plan/features/mock-trial.plan.md)
> **Match Rate**: 91%

---

## 1. 전체 점수

| 카테고리 | 검증 항목 수 | 일치 수 | 점수 | 상태 |
|----------|:----------:|:------:|:----:|:----:|
| 데이터 모델 (Section 3) | 20 | 19 | 95% | ✅ |
| API 명세 (Section 4) | 8 | 8 | 100% | ✅ |
| UI/UX 설계 (Section 5) | 20 | 18 | 90% | ✅ |
| Backend 상세 설계 (Section 6) | 25 | 23 | 92% | ✅ |
| Frontend 상세 설계 (Section 7) | 15 | 14 | 93% | ✅ |
| Error Handling (Section 8) | 10 | 7 | 70% | ⚠️ |
| Security (Section 9) | 10 | 8 | 80% | ⚠️ |
| 파일 구조 (Section 12) | 20 | 19 | 95% | ✅ |
| 시스템 통합 | 5 | 5 | 100% | ✅ |
| **전체** | **133** | **121** | **91%** | **✅** |

### 1.1 점수 산출 근거

각 카테고리의 점수는 **검증 항목 수 대비 일치 항목 수 비율**로 산출한다.

- **데이터 모델 (95%)**: MockTrialState 20개 필드 중 19개 일치. `excluded_evidence` 필드(부동의 증거 ID) 미구현 → -5%
- **API 명세 (100%)**: 4개 엔드포인트 × 2(요청+응답 형식) = 8항목 전체 일치
- **UI/UX 설계 (90%)**: 20개 컴포넌트/레이아웃 중 18개 일치. 픽셀아트 에셋 미확보(-5%), 로비 씬 면책 동의 모달 미구현(-5%)
- **Backend 상세 설계 (92%)**: 25개 항목 중 23개 일치. useTrialState 훅 미구현(-4%), 서브그래프 노드 async 시그니처 불일치(-4%) → Design v0.4에서 수정 완료
- **Frontend 상세 설계 (93%)**: 15개 항목 중 14개 일치. useTrialState 훅 미분리(-7%)
- **Error Handling (70%)**: 10개 에러 시나리오 중 7개 처리. LLM 타임아웃(-10%), Canvas 폴백(-10%), EventBus 큐(-10%) 미구현
- **Security (80%)**: 10개 보안 요구사항 중 8개 충족. Rate Limiting(-10%), RAG 환각 방지 프롬프트(-10%) 미구현
- **파일 구조 (95%)**: 20개 파일/디렉토리 중 19개 존재. 픽셀아트 에셋 디렉토리 미확보
- **시스템 통합 (100%)**: modules.ts, api.ts, next.config.js, router.py, nodes.py 5곳 전체 일치

---

## 2. 완벽 일치 항목 (100%)

- 4개 API 엔드포인트 + 응답 형식
- 11개 서브그래프 노드 (형사 6단계 + 민사 6단계 + setup/evidence/verdict 공통)
- 14개 Frontend 컴포넌트
- 4곳 모듈 동기화 (modules.ts, api.ts, next.config.js, router)
- 10개 INTENT_PATTERNS
- TypeScript 타입/상수 전체
- EventBus 10개 이벤트

---

## 3. 누락 항목 (7건) — 구체적 수정 권고 포함

| # | 항목 | Design 위치 | 영향도 | 구체적 수정 방법 |
|---|------|-------------|:------:|----------------|
| 1 | 픽셀아트 에셋 (타일맵, 스프라이트시트) | Section 12.1 | Medium | `public/assets/mock-trial/` 에 Tiled 타일맵 JSON + 5종 스프라이트시트 PNG 추가. 무료 에셋(itch.io) 또는 AI 생성 도구 활용 |
| 2 | LLM 타임아웃 에러 처리 (408) | Section 8.1 | Medium | `_generate_with_timeout()` 래퍼 함수 추가. `asyncio.wait_for(timeout=30)` 적용 → Design v0.4 Section 13.3.1 설계 반영 |
| 3 | Rate Limiting (세션당 50회) | Section 9.6 | Medium | `MockTrialState.llm_call_count` 필드 추가 + 각 노드에서 `_check_rate_limit()` 호출 → 초과 시 verdict_node 강제 이동 |
| 4 | Canvas 렌더링 실패 React fallback | Section 8.2 | Low | `MockTrialGame.tsx`에 `canvasFailed` 상태 추가 → `MockTrialTextMode` 폴백 컴포넌트 렌더링 |
| 5 | EventBus 메시지 큐 + 재전송 | Section 8.3 | Low | `CourtEventBus` 버퍼링 메커니즘 구현 (Design v0.3 Section 8.3 설계대로). 큐 최대 크기 100 설정 |
| 6 | WebGL 미지원 안내 | Section 8.2 | Low | `Phaser.AUTO` → Canvas 2D 자동 폴백 활용 + 미지원 브라우저 감지 시 안내 문구 표시 |
| 7 | RAG 환각 방지 프롬프트 | Section 9.9 | Low | 모든 에이전트 시스템 프롬프트에 "검색 결과에 없는 판례번호/법조문을 임의 생성 금지" 지시 추가 |

---

## 4. 의도적 변경 (3건, 기능 동일)

| # | 항목 | Design (v0.3) | Implementation | 이유 | Design 반영 |
|---|------|---------------|----------------|------|:-----------:|
| 1 | LLM 호출 | `get_solar_response_stream()` | `get_chat_model()` | langchain 통합 클라이언트 사용 | ✅ v0.4 반영 완료 |
| 2 | RAG 함수 | `search_pipeline()` | `search_relevant_documents_async()` | 기존 RAG 서비스 직접 활용 | ✅ v0.4 반영 완료 |
| 3 | Phaser physics | `{ default: 'arcade' }` | 제거 | 물리엔진 불필요 (정적 캐릭터) | ✅ v0.4 반영 완료 |

---

## 5. 추가 구현 (8건, Design에 없는 개선)

| # | 항목 | 위치 | 설명 | Design 반영 |
|---|------|------|------|:-----------:|
| 1 | `game/config.ts` 상수 분리 | Frontend | Phaser 게임 설정을 별도 상수 파일로 분리 | ✅ v0.4 Section 14 |
| 2 | `game/sprites/characters.ts` 캐릭터 설정 | Frontend | 5종 캐릭터 위치/크기/애니메이션을 설정 객체 관리 | ✅ v0.4 Section 14 |
| 3 | `MockTrialAgent` 폴백 에이전트 | Backend | 서브그래프 진입 실패 시 안내 메시지 폴백 | ✅ v0.4 Section 14 |
| 4 | `get_evidence_searcher()` 싱글톤 | Backend | EvidenceSearcher 인스턴스 재사용 | ✅ v0.4 Section 14 |
| 5 | Loading fallback UI | Frontend | Phaser.js 로드 중 스켈레톤 UI | ✅ v0.4 Section 14 |
| 6 | `isMounted` guard | Frontend | React strict mode 이중 초기화 방지 | ✅ v0.4 Section 14 |
| 7 | Pydantic input validation | Backend | 입력 모델 min/max_length, pattern 검증 추가 | ✅ v0.4 Section 14 |
| 8 | 카테고리 확장 | Backend | criminal_embezzlement, _other 등 추가 | ✅ v0.4 Section 14 |

---

## 6. 미달 영역 원인 분석

### 6.1 Error Handling (70%) — 원인 분석

| 미달 항목 | 원인 | 영향 |
|-----------|------|------|
| LLM 타임아웃 처리 | CourtAgent.generate()에 타임아웃 미적용. 외부 LLM API 장애 시 무한 대기 | 사용자 경험 저하, 세션 행 |
| Canvas 폴백 UI | Phaser.js 실패 시 빈 화면. 에러 바운더리만 존재 | 저사양 브라우저에서 사용 불가 |
| EventBus 큐 | 씬 전환 중 SSE 응답 도착 시 이벤트 유실. 현재 구독자 없으면 버림 | 간헐적 AI 발언 누락 |

### 6.2 Security (80%) — 원인 분석

| 미달 항목 | 원인 | 영향 |
|-----------|------|------|
| Rate Limiting | `llm_call_count` 필드와 체크 로직 미구현 | 악의적 세션에서 LLM 비용 폭발 |
| RAG 환각 방지 | 프롬프트에 "검색 결과만 인용" 지시 누락 | 판결문에 존재하지 않는 판례 인용 가능 |

---

## 7. Act 단계 권고사항 — 개선 로드맵

### 7.1 Error Handling 70% → 90% 개선 계획

| 우선순위 | 작업 | 예상 공수 | 목표 점수 기여 |
|:--------:|------|:--------:|:-------------:|
| 1 | `_generate_with_timeout()` 래퍼 함수 구현 (30초 타임아웃) | 0.5일 | +10% |
| 2 | `MockTrialGame.tsx` Canvas 폴백 UI 추가 | 0.5일 | +5% |
| 3 | `CourtEventBus` 버퍼링 메커니즘 구현 (Design 8.3) | 1일 | +5% |

**구현 순서**: 1 → 2 → 3 (사용자 영향도 순)

### 7.2 Security 80% → 95% 개선 계획

| 우선순위 | 작업 | 예상 공수 | 목표 점수 기여 |
|:--------:|------|:--------:|:-------------:|
| 1 | 세션 Rate Limiting 구현 (`MAX_LLM_CALLS_PER_SESSION = 50`) | 0.5일 | +10% |
| 2 | RAG 환각 방지 프롬프트 추가 (모든 에이전트) | 0.5일 | +5% |
| 3 | 프롬프트 인젝션 테스트 세트 작성 + 방어율 측정 | 1일 | (검증) |

**구현 순서**: 1 → 2 → 3

### 7.3 전체 개선 후 예상 Match Rate

| 카테고리 | 현재 | 개선 후 예상 |
|----------|:----:|:----------:|
| Error Handling | 70% | 90% |
| Security | 80% | 95% |
| **전체** | **91%** | **95%** |

---

## 8. 문서 동기화 현황 (v0.2 추가)

### 8.1 Plan ↔ Analysis 동기화

| 항목 | 동기화 상태 | 비고 |
|------|:----------:|------|
| FR 상태 (Pending→Done/In Progress) | ✅ | Plan v0.6에서 38개 FR 상태 업데이트 완료 |
| Implementation Step 상태 | ✅ | Plan v0.6에서 Step 3~21 상태 + Phase E 추가 완료 |
| FR-29~38 Implementation Step | ✅ | Plan v0.6 Phase E (Step 22~25) 추가 완료 |

### 8.2 Design ↔ Analysis 동기화

| 항목 | 동기화 상태 | 비고 |
|------|:----------:|------|
| 의도적 변경 3건 역반영 | ✅ | Design v0.4 Section 2.3, 6.5, 7.1 수정 완료 |
| async 키워드 추가 | ✅ | Design v0.4 Section 6.3 노드 함수 8개 수정 완료 |
| 추가 구현 8건 역반영 | ✅ | Design v0.4 Section 14 추가 완료 |
| FR-29~38 설계 명세 | ✅ | Design v0.4 Section 13 추가 완료 |

### 8.3 Plan ↔ Design 동기화

| 항목 | 동기화 상태 | 비고 |
|------|:----------:|------|
| §42 조문 수정 → §43+§39 | ✅ | 양쪽 문서 수정 완료 |
| closing 노드명 통일 | ✅ | Plan: criminal_closing_node/civil_closing_node로 통일 |
| Design 버전 업데이트 | ✅ | v0.3.0 → v0.4.0 |

---

## 9. 결론

Match Rate **91%** — 90% 기준 충족, **Check 단계 통과**.

핵심 기능(데이터 모델, API, 서브그래프, UI 컴포넌트)은 92-100% 일치하여 기능적으로 안정적이다.

**주요 미비 영역**:
- Error Handling(70%): LLM 타임아웃, Canvas 폴백, EventBus 큐 — 3개 항목 보강 필요
- Security(80%): Rate Limiting, RAG 환각 방지 — 2개 항목 보강 필요

**문서 부채 해소**: Plan v0.6, Design v0.4, Analysis v0.2 동기화 완료. 3문서 간 정합성 확보.

**다음 단계**: Act 단계에서 Section 7의 개선 로드맵에 따라 Error Handling/Security 미달 항목을 구현하고, gap-detector로 재분석하여 95% 이상 달성을 목표로 한다.

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-02-19 | 초기 Gap 분석 (Plan v0.4 + Design v0.3.0 기준) | Claude |
| 0.2 | 2026-02-21 | 종합 분석 반영 보강: (1) 점수 산출 근거 추가 (Section 1.1) — 카테고리별 검증 항목 수/일치 수 명시. (2) 누락 항목에 구체적 수정 방법 추가 (Section 3). (3) 미달 영역 원인 분석 추가 (Section 6) — Error Handling/Security 각 항목별 원인/영향 분석. (4) Act 단계 개선 로드맵 추가 (Section 7) — Error Handling 70%→90%, Security 80%→95% 달성 계획. (5) 문서 동기화 현황 추가 (Section 8) — Plan/Design/Analysis 3문서 간 동기화 상태 추적. (6) 의도적 변경/추가 구현에 Design 역반영 상태 컬럼 추가. (7) 분석 기준 버전 명시 (Baseline 필드) | Claude |
