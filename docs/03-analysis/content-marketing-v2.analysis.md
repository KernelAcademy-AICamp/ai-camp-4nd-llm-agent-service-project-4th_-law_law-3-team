# Gap Analysis Report: content-marketing v2.0

> **분석일**: 2026-02-22
> **설계 문서**: `docs/02-design/features/content-marketing-v2.design.md` (Final — Red Team 검증 완료)
> **이전 분석**: `content-marketing.analysis.md` (v0.1.0 기준, 93% PASS)
> **Match Rate**: **3.9%** (FAIL — v2.0은 신규 구현 단계)

---

## 종합 점수

| 카테고리 | 점수 | 상태 |
|---------|:----:|:----:|
| API 설계 일치율 | 22% | FAIL |
| 데이터 모델 일치율 | 0% | FAIL |
| Backend 클래스 구현율 | 0% | FAIL |
| Frontend 컴포넌트 구현율 | 0% | FAIL |
| Frontend 타입 일치율 | 0% | FAIL |
| 환경변수 일치율 | 14% | FAIL |
| 보안 요구사항 (Auth Dependency) | 0% | FAIL |
| **전체 Match Rate** | **3.9%** | **FAIL** |

---

## 항목 집계

| 카테고리 | 설계 항목 | 구현됨 | 부분 구현 | 미구현 |
|---------|:--------:|:-----:|:-------:|:-----:|
| Backend API 엔드포인트 | 9 | 2 | 0 | 7 |
| Backend Pydantic Schema (신규) | 14 | 0 | 0 | 14 |
| Backend 기존 스키마 변경 | 5 | 0 | 0 | 5 |
| Backend 신규 파일/클래스 | 10 | 0 | 0 | 10 |
| Backend ORM 모델 | 2 | 0 | 0 | 2 |
| Backend Alembic 마이그레이션 | 1 | 0 | 0 | 1 |
| Backend 환경변수 | 7 | 1 | 1 | 5 |
| Frontend 신규 컴포넌트 | 8 | 0 | 0 | 8 |
| Frontend 기존 컴포넌트 변경 | 6 | 0 | 0 | 6 |
| Frontend 신규 타입 | 15 | 0 | 0 | 15 |
| Frontend 기존 타입 변경 | 5 | 0 | 0 | 5 |
| Frontend hooks 신규/변경 | 3 | 0 | 0 | 3 |
| Frontend 서비스 함수 추가 | 5 | 0 | 0 | 5 |
| **합계** | **90** | **3** | **1** | **86** |

---

## Gap 목록 (우선순위별)

### High Priority — 핵심 아키텍처

| # | 항목 | 설계 섹션 | 미구현 근거 |
|---|------|---------|-----------|
| H-01 | Auth Dependency 적용 (IDOR 방지) | §4.1, §2.1.1 | router에 `Depends(get_current_user)` 없음 |
| H-02 | `tools/persona/` 전체 디렉토리 | §5.1~§5.2, §6 | 디렉토리 없음 |
| H-03 | `MetadataExtractor` (PII 원천 차단) | §5.1 | 파일 없음 |
| H-04 | `PersonaAnalyzer` (Track 1) | §5.2 | 파일 없음 |
| H-05 | `LegalGateScorer` (TrendScorer 확장) | §5.3 | 2차원→5차원 미확장 |
| H-06 | `BackgroundTrendWorker` | §2.3.1 | 파일 없음 |
| H-07 | `PromptChainExecutor` | §5.4 | 파일 없음 |
| H-08 | Persona Pydantic 스키마 8개 | §3.1 | schema에 없음 |
| H-09 | `models/lawyer_persona.py` ORM | §3.4 | 파일 없음 |
| H-10 | Alembic 마이그레이션 | §9.1 | 마이그레이션 없음 |

### Medium Priority — 기능 완성

| # | 항목 | 설계 섹션 | 미구현 근거 |
|---|------|---------|-----------|
| M-01 | Persona API 5개 엔드포인트 | §4.1 | router에 없음 |
| M-02 | `persona_db_service.py` | §6 | 파일 없음 |
| M-03 | `OnboardingProcessor` (Track 2) | §5, §6 | 파일 없음 |
| M-04 | `TrendScoreDetail` 스키마 | §3.2, §3.3 | schema에 없음 |
| M-05 | `TrendRequest`에 `persona_id` 추가 | §3.3 | 필드 없음 |
| M-06 | `ScriptRequest`에 `persona_id` + 하위호환 | §3.3 | 필드 없음 |
| M-07 | `ScriptStreamEvent`에 `stage_update` | §3.3 | 이벤트 없음 |
| M-08 | v2.0 환경변수 7개 | §10 | config.py에 미존재 |
| M-09 | `PersonaGate.tsx` | §7, §8.1 | 파일 없음 |
| M-10 | `PersonaOnboarding.tsx` | §7, §8.2 | 파일 없음 |
| M-11 | `hooks/usePersona.ts` | §7 | 파일 없음 |
| M-12 | Frontend 타입 v2.0 전체 | §3.5 | types/index.ts에 없음 |

### Low Priority — UI 개선

| # | 항목 | 설계 섹션 | 미구현 근거 |
|---|------|---------|-----------|
| L-01 | `PersonaBanner.tsx` | §7 | 파일 없음 |
| L-02 | `PersonaEditor.tsx` | §7 | 파일 없음 |
| L-03 | `PersonaConfirmation.tsx` | §7, §8.2 | 파일 없음 |
| L-04 | `FeedbackPanel.tsx` | §7, §8.4 | 파일 없음 |
| L-05 | `RAGPreview.tsx` | §7 | 파일 없음 |
| L-06 | `StageProgress.tsx` | §7 | 파일 없음 |
| L-07~L-09 | 기존 컴포넌트 v2.0 변경 | §7, §8.3 | 미반영 |
| L-10 | `services/index.ts` persona 함수 5개 | §7 | 없음 |
| L-11 | `TrendDashboard.tsx` Legal Gate UI | §8.3 | 미반영 |

---

## 결론

현재 코드는 **v0.1.0 설계를 충실히 구현한 상태**(이전 분석 93%)이나, v2.0 설계 기준으로는 **86개 항목이 미구현**입니다. v2.0은 기존 기능 위에 3개 대형 시스템을 추가하는 구조:

1. **페르소나 시스템** (Track 1/2, 5 API, 8 UI, DB) — 완전 신규
2. **Legal Gate 스코어링** (5차원, BackgroundWorker, 글로벌 캐시) — 기존 확장
3. **PromptChainExecutor** (3단계 RAG 체인, stage_update SSE) — 완전 신규

**권장**: 설계 문서 §11 구현 우선순위에 따라 Phase 1(페르소나)부터 순차 이행. Red Team 보안 지적([심각 1] IDOR, [심각 3] PII)을 최우선 적용.
