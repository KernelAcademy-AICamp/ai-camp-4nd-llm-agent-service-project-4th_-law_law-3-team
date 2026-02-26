# Triple Review Workflow (3중 검증 워크플로우)

기획 및 구현 작업 시 이 워크플로우를 **항상(ALWAYS)** 따라야 합니다.

## 팀 구성

| 역할 | 담당 | 모델/도구 | 책임 |
|------|------|----------|------|
| **Lead Manager** | Claude Agent #1 | Opus | 최종 확인, 의사결정, 팀 조율 |
| **Agent Team** (4명) | Claude Agent #2~#5 | Sonnet | 기획/설계/분석/구현 실행 |
| **Red Team** | Gemini CLI | 외부 | 보고서/코드 검증, 취약점 분석, 고급 기능 제안 |
| **External Consultant** | Codex CLI | 외부 | 최종 검증, 컨설팅 보고서, 최고 수준 기능 제안 |

## Phase 1: 기획/계획 단계 워크플로우

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Claude Agent Team → 단계별 보고서 작성             │
│          (Lead Manager: Opus, Workers: Sonnet x4)           │
├─────────────────────────────────────────────────────────────┤
│  Step 2: Gemini CLI (Red Team) → 보고서 검증                │
│          - 취약점 분석                                       │
│          - 현업 수준 고급 기술/기능 추가 제안                 │
│          - Red Team 보고서 작성                               │
├─────────────────────────────────────────────────────────────┤
│  Step 3: Claude Agent Team → Red Team 보고서 분석            │
│          - 계획 수정 및 보고서 업데이트                       │
│          - Lead Manager 최종 확인                             │
├─────────────────────────────────────────────────────────────┤
│  Step 4: Codex CLI (External Consultant) → 보고서 검증       │
│          - 취약점 분석                                       │
│          - 최고 수준 서비스를 위한 고급 기술/기능 추가 제안   │
│          - 컨설팅 보고서 작성                                 │
├─────────────────────────────────────────────────────────────┤
│  Step 5: Claude Agent Team → 컨설팅 보고서 분석              │
│          - 계획 수정 및 보고서 최종 업데이트                  │
│          - Lead Manager 최종 확인 및 결정                     │
├─────────────────────────────────────────────────────────────┤
│  Step 6: 사용자 승인 요청                                    │
└─────────────────────────────────────────────────────────────┘
```

## Phase 2: 기술 구현/코딩 단계 워크플로우

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Claude Agent Team → 코드 구현                      │
├─────────────────────────────────────────────────────────────┤
│  Step 2: Gemini CLI (Red Team) → 코드 리뷰                  │
│          - 오류/취약점 발견 → Agent Team이 수정               │
├─────────────────────────────────────────────────────────────┤
│  Step 3: Codex CLI (External Consultant) → 코드 리뷰 + 직접 수정 │
│          - 오류/개선점 발견 시 Codex가 직접 코드 수정          │
│          - `codex review --uncommitted` (리뷰 전용) 또는      │
│            `codex exec --full-auto "..."` (분석+수정 권한)     │
│          - 수정 결과는 git diff로 추적 가능                   │
│          ⚠ Claude Code Bash 환경은 TTY 없음 →                 │
│            반드시 `codex exec` 또는 `codex review` 서브커맨드 사용 │
├─────────────────────────────────────────────────────────────┤
│  Step 4: Claude Agent Team → Codex 수정 코드 분석 및 평가    │
│          - `git diff`로 Codex 변경 사항 확인                  │
│          - 변경 적절성, 코드 품질, 아키텍처 정합성 평가       │
│          - 부적절한 변경은 revert, 유효한 변경은 채택          │
│          - 구현 계획 및 기획 보고서 업데이트                  │
│          - Lead Manager 최종 확인                             │
└─────────────────────────────────────────────────────────────┘
```

## 팀 내 합의 불가 시 조율 프로토콜

에이전트 팀 내에서 합의가 되지 않는 토픽/문제 발생 시:

1. **Lead Manager**가 쟁점 정리
2. **Gemini CLI**에 교차 질문 → 의견 수렴
3. **Codex CLI**에 교차 질문 → 의견 수렴
4. 3개 소스의 의견을 종합하여 **Lead Manager**가 최종 결정

```bash
# Gemini CLI 교차 질문 템플릿
gemini "다음 기술적 쟁점에 대해 분석하고 추천 방향을 제시해주세요.
쟁점: {논쟁_주제}
Option A: {옵션A_설명}
Option B: {옵션B_설명}
프로젝트 컨텍스트: FastAPI + Next.js + LangGraph 법률 서비스 플랫폼
평가 기준: 성능, 확장성, 유지보수성, 보안, 사용자 경험" -y -o text

# Codex CLI 교차 질문 템플릿 (비대화형 exec 서브커맨드 사용)
codex exec "다음 기술적 쟁점에 대해 분석하고 추천 방향을 제시해주세요.
쟁점: {논쟁_주제}
Option A: {옵션A_설명}
Option B: {옵션B_설명}
프로젝트 컨텍스트: FastAPI + Next.js + LangGraph 법률 서비스 플랫폼
평가 기준: 성능, 확장성, 유지보수성, 보안, 사용자 경험" --ephemeral -s read-only
```

## CLI 호출 규칙

### Gemini CLI (Red Team) 보고서 요청 템플릿

```bash
gemini "당신은 Red Team 보안/기술 전문가입니다.
다음 보고서를 검증하고 Red Team 보고서를 작성해주세요.

검토 관점:
1. 보안 취약점 분석 (OWASP Top 10 기준)
2. 아키텍처 약점 및 단일 장애점
3. 현업 수준 서비스를 위한 고급 기술/기능 추가 제안
4. 성능 병목 지점 및 확장성 이슈
5. 운영 관점의 모니터링/로깅/알림 부재 항목

출력 형식:
## Red Team 검증 보고서
### 1. 취약점 (Critical/High/Medium/Low)
### 2. 아키텍처 개선 제안
### 3. 고급 기능 추가 제안
### 4. 성능 최적화 제안
### 5. 운영 안정성 제안

---
$(cat {보고서_경로})
---" -y -o text
```

### Codex CLI (External Consultant) 보고서 요청 템플릿

```bash
# ⚠ Claude Code Bash 환경에서는 반드시 `codex exec` 서브커맨드 사용
codex exec "당신은 외부 기술 컨설턴트입니다.
다음 보고서를 검증하고 컨설팅 보고서를 작성해주세요.

검토 관점:
1. 업계 최고 수준 서비스 대비 격차 분석
2. 최신 기술 트렌드 반영 여부
3. 사용자 경험(UX) 최적화 방안
4. 데이터 파이프라인 및 AI/ML 고도화 방안
5. 비즈니스 확장성 및 수익화 전략

출력 형식:
## 외부 컨설팅 보고서
### 1. 격차 분석 (현재 vs 목표)
### 2. 기술 고도화 제안
### 3. UX 개선 제안
### 4. AI/ML 고도화 제안
### 5. 비즈니스 전략 제안

---
$(cat {보고서_경로})
---" --ephemeral -s read-only
```

### Gemini CLI (Red Team) 코드 리뷰 템플릿

```bash
gemini "당신은 시니어 코드 리뷰어이자 보안 전문가입니다.
다음 코드를 리뷰하고 상세한 리뷰 보고서를 작성해주세요.

리뷰 관점:
1. 보안 취약점 (인젝션, XSS, CSRF, 인증/인가 취약점)
2. 버그 및 논리적 오류
3. 성능 이슈 및 최적화 기회
4. 코드 품질 및 유지보수성
5. 에러 핸들링 및 엣지 케이스

$(for f in {파일_목록}; do echo '--- '$f' ---'; cat $f; done)
" -y -o text
```

### Codex CLI (External Consultant) 코드 리뷰 + 직접 수정 템플릿

**방법 A: uncommitted 변경 자동 리뷰 + 수정** (권장)
```bash
# Codex가 uncommitted 변경사항을 리뷰하고 직접 수정
codex review --uncommitted
```

**방법 B: 특정 파일 대상 리뷰 + 수정** (비대화형)
```bash
# codex exec --full-auto: 비대화형 자동 실행 (파일 수정 허용)
codex exec --full-auto "당신은 외부 기술 컨설턴트이자 코드 품질 전문가입니다.
다음 파일들을 리뷰하고, 발견된 오류와 개선점을 직접 수정해주세요.

리뷰 관점:
1. 아키텍처 패턴 및 설계 원칙 준수
2. 테스트 가능성 및 테스트 커버리지
3. 타입 안전성 및 API 계약 정합성
4. 확장성 및 모듈화 수준
5. 업계 모범 사례 대비 개선점

수정 규칙:
- 버그, 타입 오류, 보안 취약점은 직접 수정
- 성능 개선, 리팩토링은 직접 수정
- 아키텍처 변경 등 대규모 수정은 코멘트만 남기고 수정하지 않음
- 수정한 내용을 요약하여 출력

대상 파일: {파일_목록}"
```

**방법 C: 리뷰만 (수정 없이)**
```bash
# codex review: 비대화형 리뷰 전용 서브커맨드
codex review --uncommitted

# codex exec + 읽기 전용 샌드박스
codex exec "리뷰만 수행. 코드 수정하지 말 것. ..." --ephemeral -s read-only
```

### Codex 수정 후 Agent Team 평가 프로토콜

```bash
# 1. Codex 수정 사항 확인
git diff

# 2. Agent Team이 변경 사항 분석 (code-analyzer 에이전트 활용)
# 평가 기준:
#   - 변경이 기존 아키텍처 패턴과 일관성 있는가?
#   - 코딩 스타일 규칙(coding-style.md)을 준수하는가?
#   - 불필요한 변경이나 과도한 리팩토링이 포함되지 않았는가?
#   - 새로운 버그나 타입 오류를 도입하지 않았는가?

# 3. 부적절한 변경 revert
git checkout -- {부적절한_파일}

# 4. 유효한 변경만 스테이징
git add {유효한_파일들}
```

## 보고서 저장 경로

| 보고서 유형 | 저장 경로 |
|------------|----------|
| 기획 보고서 (Agent Team) | `docs/01-plan/features/{feature}.plan.md` |
| 설계 보고서 (Agent Team) | `docs/02-design/features/{feature}.design.md` |
| Red Team 보고서 | `docs/03-analysis/{feature}.redteam.md` |
| 컨설팅 보고서 | `docs/03-analysis/{feature}.consulting.md` |
| 최종 분석 보고서 | `docs/03-analysis/{feature}.analysis.md` |
| 완료 보고서 | `docs/04-report/{feature}.report.md` |
| 코드 리뷰 (Red Team) | `docs/03-analysis/{feature}.code-review-redteam.md` |
| 코드 리뷰 (Consultant) | `docs/03-analysis/{feature}.code-review-consulting.md` |

## Agent Team 구성 (5명)

| Agent # | 역할 | 모델 | subagent_type 예시 |
|---------|------|------|-------------------|
| #1 | **Lead Manager** (조율/결정) | `opus` | `bkit:cto-lead` |
| #2 | Product/기획 분석 | `sonnet` | `bkit:product-manager` |
| #3 | 아키텍처/설계 | `sonnet` | `bkit:frontend-architect` 또는 `bkit:enterprise-expert` |
| #4 | 보안/품질 검증 | `sonnet` | `bkit:security-architect` 또는 `bkit:code-analyzer` |
| #5 | QA/갭 분석 | `sonnet` | `bkit:qa-strategist` 또는 `bkit:gap-detector` |

역할은 작업 성격에 따라 유연하게 조정 가능. Lead Manager가 팀 구성을 결정.

## 외부 CLI 실행 실패 시 처리 (절대 건너뛰기 금지)

Red Team(Gemini CLI) 또는 External Consultant(Codex CLI) 실행이 실패할 경우:

1. **자동 건너뛰기/Fallback 금지** — 실패한 단계를 조용히 넘어가지 않음
2. **사용자에게 즉시 보고** — 실패한 CLI, 원인, 에러 메시지를 명확히 설명
3. **사용자 지침에 따라 진행** — `AskUserQuestion`으로 다음 선택지 제시:
   - CLI 문제 해결 후 재시도
   - Claude Agent Team이 해당 역할을 대체 수행
   - 해당 단계 건너뛰기
   - 작업 중단

> 상세 절차: `cli-tool-routing.md`의 "CLI 실행 실패 시 필수 절차" 참조

## 필수 원칙

1. **각 단계별 보고서 작성 필수** - 단계를 건너뛸 수 없음
2. **Lead Manager 최종 확인** - 모든 단계 전환 시 Lead Manager 승인 필요
3. **외부 CLI 결과 무조건 수용 금지** - Agent Team이 분석 후 유효한 제안만 채택
4. **Codex 수정은 Agent Team 평가 필수** - Codex가 직접 수정한 코드는 반드시 Agent Team이 diff 분석 → 적절성 평가 → 선별 채택/revert
5. **사용자 승인** - 기획 완료 후 구현 전환 시 반드시 사용자 승인
6. **보고서 버전 관리** - 수정 시 이전 내용 덮어쓰되, 주요 변경 이력은 보고서 하단에 기록
7. **외부 CLI 실패 시 사용자 보고 필수** - 자동 건너뛰기 금지, 반드시 사유 설명 후 사용자 지침 대기
