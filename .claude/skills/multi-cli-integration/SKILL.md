---
name: multi-cli-integration
description: Claude, Gemini CLI, Codex CLI를 조합하여 복합 작업을 수행하는 통합 패턴. 보안 감사+수정, 리팩토링+리뷰, 아키텍처 분석+구현 등 여러 CLI 강점을 결합하는 워크플로우에 사용.
---

# Multi-CLI Integration 패턴

## 1. 개요

### 목적
Claude, Gemini CLI, Codex CLI의 강점을 조합하여 단일 도구로는 어려운 복합 작업을 수행합니다.

### 참조 스킬
이 스킬은 다음 스킬들의 상위 조합 패턴입니다:

| 스킬 | 역할 | 위치 |
|------|------|------|
| `gemini-cli-delegation` | Gemini CLI 단독 사용 패턴 | `.claude/skills/gemini-cli-delegation/SKILL.md` |
| `codex-cli-delegation` | Codex CLI 단독 사용 패턴 | `.claude/skills/codex-cli-delegation/SKILL.md` |

### 도구별 강점 매트릭스

| 역할 | Claude | Gemini CLI | Codex CLI |
|------|--------|-----------|-----------|
| 계획 수립 | **최적** | - | - |
| 정밀 코드 작성 | **최적** | - | 보조 |
| 대규모 분석 (10+ 파일) | 제한적 | **최적** | - |
| 코드 리뷰 | 가능 | - | **최적** |
| 샌드박스 실행 | 불가 | - | **최적** |
| 웹 검색 연동 | 가능 | - | **최적** |
| 결과 통합/검증 | **최적** | - | - |
| 최종 판단 | **최적** | - | - |

---

## 2. 설치 상태별 조합 모드

### 설치 확인
```bash
# 한 번에 두 CLI 확인
GEMINI=$(which gemini 2>/dev/null && echo "Y" || echo "N")
CODEX=$(which codex 2>/dev/null && echo "Y" || echo "N")
echo "Gemini: $GEMINI, Codex: $CODEX"
```

### 조합 모드 결정

| Gemini | Codex | 모드 | 설명 |
|--------|-------|------|------|
| O | O | **Full Orchestration** | 모든 CLI 활용 가능 |
| O | X | **Gemini Only** | 분석은 Gemini, 리뷰/실행은 Claude |
| X | O | **Codex Only** | 분석은 Claude, 리뷰/실행은 Codex |
| X | X | **Claude Solo** | Claude가 모든 작업 직접 수행 |

---

## 3. 워크플로우 템플릿

### 3.1 Analyze-Then-Act (분석 → 실행)

대규모 분석 후 코드 수정이 필요한 작업에 사용합니다.

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Phase 1     │    │  Phase 2     │    │  Phase 3     │
│  Gemini CLI  │ →  │  Claude      │ →  │  Codex CLI   │
│  (분석)       │    │  (수정)       │    │  (리뷰)       │
└──────────────┘    └──────────────┘    └──────────────┘
```

**적용 시나리오**: 보안 감사, 리팩토링, 마이그레이션

**Full Orchestration 모드**:
```bash
# Phase 1: Gemini CLI로 대규모 분석
gemini "backend/app/ 전체의 보안 취약점을 분석해줘. OWASP Top 10 기준으로 분류하고 파일 경로와 라인 번호를 포함해줘." -y -o text

# Phase 2: Claude가 분석 결과 기반으로 수정
# (Claude가 직접 Edit 도구로 수정)

# Phase 3: Codex CLI로 수정 사항 리뷰
codex "$(git diff --cached) 이 보안 수정사항을 리뷰해줘. 수정이 적절한지, 새로운 취약점이 없는지 확인해줘."
```

**Fallback (Gemini Only)**:
```bash
# Phase 1: Gemini CLI로 분석
gemini "..." -y -o text

# Phase 2: Claude가 수정
# Phase 3: Claude가 직접 git diff로 리뷰 (Codex 대체)
```

**Fallback (Codex Only)**:
```bash
# Phase 1: Claude가 Task(Explore)로 분석 (Gemini 대체)
# Phase 2: Claude가 수정
# Phase 3: Codex CLI로 리뷰
codex "$(git diff --cached) ..."
```

**Fallback (Claude Solo)**:
```bash
# Phase 1: Claude가 Task(Explore) + Glob/Grep으로 분석
# Phase 2: Claude가 수정
# Phase 3: Claude가 git diff로 자체 리뷰
```

---

### 3.2 Review-Then-Fix (리뷰 → 수정)

코드 리뷰 후 발견된 문제를 수정하는 작업에 사용합니다.

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Phase 1     │    │  Phase 2     │    │  Phase 3     │
│  Codex CLI   │ →  │  Claude      │ →  │  검증         │
│  (리뷰)       │    │  (수정)       │    │  (lint/test) │
└──────────────┘    └──────────────┘    └──────────────┘
```

**적용 시나리오**: PR 리뷰 후 수정, 코드 품질 개선

**Full Orchestration 모드**:
```bash
# Phase 1: Codex CLI로 코드 리뷰
codex "$(git diff dev..HEAD) 이 브랜치의 모든 변경사항을 리뷰해줘. 버그, 성능, 보안, 코드 품질 관점에서 분석해줘."

# Phase 2: Claude가 리뷰 결과 기반 수정 (Edit 도구)

# Phase 3: 검증
# Backend: uv run ruff check . && uv run mypy .
# Frontend: npm run build
```

**Fallback (Claude Solo)**:
```bash
# Phase 1: Claude가 git diff 직접 분석
# Phase 2: Claude가 수정
# Phase 3: 검증 동일
```

---

### 3.3 Deep-Scan (심층 분석)

프로젝트 전체를 여러 관점에서 분석하는 작업에 사용합니다.

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Gemini CLI  │    │  Codex CLI   │    │  Claude      │
│  (구조 분석)  │ ┐  │  (CVE 검색)  │ ┐  │  (통합 보고)  │
│              │ ├→ │              │ ├→ │              │
│  (병렬 실행)  │ ┘  │  (병렬 실행)  │ ┘  │              │
└──────────────┘    └──────────────┘    └──────────────┘
```

**적용 시나리오**: 종합 보안 감사, 기술 부채 분석, 출시 전 점검

**Full Orchestration 모드**:
```bash
# 병렬 Phase: Gemini와 Codex 동시 실행
# Gemini: 코드 구조/아키텍처 분석
gemini "프로젝트 전체 아키텍처를 분석해줘. 계층 위반, 순환 의존성, 코드 중복을 찾아줘." -y -o text

# Codex: 외부 취약점/최신 정보 검색
codex "이 프로젝트에서 사용하는 주요 패키지의 알려진 보안 취약점을 검색하고 보고해줘"

# Claude: 두 결과를 통합하여 최종 보고서 작성
```

**Fallback (Claude Solo)**:
```bash
# Claude가 순차적으로 수행
# 1. Task(Explore)로 아키텍처 분석
# 2. WebSearch로 취약점 검색
# 3. 결과 통합 보고
```

---

### 3.4 Migrate-And-Verify (마이그레이션 → 검증)

라이브러리/프레임워크 마이그레이션 작업에 사용합니다.

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐
│  Gemini CLI  │    │  Codex CLI   │    │  Claude      │    │  Codex   │
│  (영향 분석)  │ →  │  (가이드 검색)│ →  │  (마이그레이션)│ →  │  (검증)   │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────┘
```

**적용 시나리오**: 패키지 업그레이드, API 변경, 프레임워크 전환

**Full Orchestration 모드**:
```bash
# Phase 1: Gemini CLI로 영향 범위 분석
gemini "Pydantic v1 패턴을 사용하는 모든 파일과 사용 패턴을 분석해줘." -y -o text

# Phase 2: Codex CLI로 마이그레이션 가이드 검색
codex "Pydantic v2 마이그레이션 가이드에서 주요 변경점과 자동 변환 도구를 찾아줘"

# Phase 3: Claude가 마이그레이션 수행 (Edit 도구)

# Phase 4: Codex CLI로 변경사항 검증
codex --full-auto --sandbox read-only "마이그레이션 후 테스트를 실행하고 결과를 보고해줘"
```

---

## 4. 에러 처리 및 충돌 해결

### CLI 실행 실패 시

```
CLI 실행 실패
    │
    ├─ 타임아웃 → 작업 범위 축소 후 재시도
    │
    ├─ 네트워크 오류 → 로컬 분석으로 전환 (Fallback)
    │
    ├─ 인증 실패 → 사용자에게 설정 확인 안내
    │
    └─ 알 수 없는 오류 → Fallback + 에러 내용 보고
```

### CLI 간 결과 충돌 시

여러 CLI가 상반되는 결과를 제공하는 경우:

1. **실제 코드 확인**: Claude가 직접 해당 코드를 읽어 확인
2. **코드가 진실**: CLI 출력보다 실제 코드 상태를 우선
3. **불확실한 경우**: 양쪽 의견을 모두 사용자에게 보고
4. **보안 관련**: 보수적으로 판단 (취약점 가능성 있으면 보고)

### 결과 통합 원칙

| 상황 | 원칙 |
|------|------|
| Gemini와 Codex 결과 일치 | 높은 신뢰도로 채택 |
| Gemini와 Codex 결과 불일치 | Claude가 직접 확인하여 판단 |
| 한쪽만 문제 발견 | Claude가 해당 코드 확인 후 판단 |
| 양쪽 모두 오탐 의심 | 실제 코드 기반으로 독립 판단 |

---

## 5. 복합 시나리오 예시

### 시나리오 1: 프로젝트 보안 감사

```
[사용자] "프로젝트 전체 보안 감사 해줘"

[Claude - 오케스트레이션]
1. CLI 설치 확인
2. 모드 결정 (Full / Partial / Solo)

[Phase 1 - 분석] (Gemini CLI 또는 Claude)
├─ API 엔드포인트 인증/인가 패턴 분석
├─ SQL 인젝션 가능성 스캔
├─ XSS 취약점 패턴 스캔
└─ 환경 변수/비밀 관리 점검

[Phase 2 - 외부 정보] (Codex CLI 또는 WebSearch)
├─ 사용 패키지 CVE 검색
├─ 최신 보안 모범 사례 확인
└─ OWASP Top 10 체크리스트 대조

[Phase 3 - 통합 보고] (Claude)
├─ 결과 교차 검증
├─ 심각도별 분류 (Critical / High / Medium / Low)
├─ 수정 방안 제시
└─ 최종 보고서 작성
```

### 시나리오 2: 대규모 리팩토링

```
[사용자] "서비스 레이어를 리팩토링하고 싶어"

[Phase 1 - 현황 분석] (Gemini CLI 또는 Claude)
├─ 서비스 파일 구조 분석
├─ 함수간 의존성 매핑
├─ 코드 스멜 탐지
└─ 리팩토링 후보 목록화

[Phase 2 - 계획 수립] (Claude)
├─ 분석 결과 검증
├─ 리팩토링 계획 수립
├─ 영향 범위 정의
└─ 사용자 승인 요청

[Phase 3 - 코드 수정] (Claude)
├─ 계획에 따른 코드 수정
├─ 린트/타입 체크
└─ 테스트 실행

[Phase 4 - 리뷰] (Codex CLI 또는 Claude)
├─ 수정 사항 리뷰
├─ 새로운 문제 탐지
└─ 최종 검증
```

### 시나리오 3: 신규 기능 개발

```
[사용자] "사용자 인증 기능을 추가해줘"

[Phase 1 - 조사] (Codex CLI 또는 WebSearch)
├─ 최신 인증 패턴 검색 (JWT, OAuth 등)
├─ FastAPI 보안 모범 사례 확인
└─ 관련 패키지 평가

[Phase 2 - 아키텍처 분석] (Gemini CLI 또는 Claude)
├─ 기존 코드와의 통합 지점 분석
├─ 영향받는 모듈 파악
└─ 설계 방안 도출

[Phase 3 - 구현] (Claude)
├─ 코드 작성
├─ 테스트 작성
└─ 문서 업데이트

[Phase 4 - 검증] (Codex CLI + Claude)
├─ 코드 리뷰
├─ 보안 점검
└─ 통합 테스트
```

---

## 6. 성능 최적화 팁

### 병렬 실행
- Gemini CLI와 Codex CLI는 서로 독립적이므로 병렬 실행 가능
- Claude의 Bash 도구에서 백그라운드 실행 활용

```bash
# 병렬 실행 예시 (결과를 파일로 저장)
gemini "아키텍처 분석..." -y -o text > /tmp/gemini_result.txt &
codex -q "CVE 검색..." > /tmp/codex_result.txt &
wait
```

### 컨텍스트 효율화
- Gemini CLI 결과는 요약만 Claude에 전달
- Codex CLI 결과 중 관련 부분만 선택 활용
- 중간 결과는 scratchpad 디렉토리에 저장

### 재시도 전략
- CLI 실패 시 최대 1회 재시도
- 재시도 실패 시 즉시 Fallback 전환
- Fallback 전환 시 사용자에게 안내

---

## 7. 보고 형식

### 통합 워크플로우 보고

```
─────────────────────────────────────────────────
📊 Multi-CLI 워크플로우 보고
─────────────────────────────────────────────────
🔄 워크플로우: [Analyze-Then-Act / Review-Then-Fix / Deep-Scan / Migrate-And-Verify]
📋 실행 모드: [Full Orchestration / Gemini Only / Codex Only / Claude Solo]

Phase 1: [도구명] - [작업 설명] - [결과 요약]
Phase 2: [도구명] - [작업 설명] - [결과 요약]
Phase 3: [도구명] - [작업 설명] - [결과 요약]

✅ 검증 결과: [통과/실패 항목]
⚠️ 제한 사항: [있으면 기술]
─────────────────────────────────────────────────
```
