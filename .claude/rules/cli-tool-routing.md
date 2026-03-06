# CLI Tool Routing Rules

Claude는 외부 AI CLI 도구(Gemini CLI, Codex CLI)를 활용할 때 이 규칙들을 **항상(ALWAYS)** 따라야 합니다.

> **상세 가이드**: `.claude/skills/multi-cli-integration/SKILL.md` 참조

## Gemini CLI 인증

Gemini CLI는 **Google Auth (OAuth) 인증**을 사용합니다. API 키가 아닌 `gemini auth` 명령으로 인증합니다.

```bash
# 인증 상태 확인 (세션 첫 Gemini 호출 전)
gemini auth status 2>/dev/null || gemini auth login
```

- `GEMINI_API_KEY` 환경변수는 사용하지 않음
- 인증 실패 시 `gemini auth login` 안내 후 Fallback 전환

## 도구 선택 기준

| 작업 유형 | 도구 | Fallback |
|----------|------|----------|
| **설계 (PDCA Design)** | **Claude + Gemini CLI 협업** | Claude 단독 |
| 대규모 코드 분석 (10+ 파일) | Gemini CLI | Task(Explore) |
| 멀티모달 (이미지/스크린샷) | Gemini CLI | 대체 불가 |
| Deep Think 추론 | Gemini CLI | Claude |
| 코드 리뷰 (PR/커밋) | Codex CLI | git diff 분석 |
| 샌드박스 실행 | Codex CLI | 실행 불가 안내 |
| 정밀 코드 작성 / 5개 이하 파일 | Claude | - |

## 설계 시 Claude + Gemini 협업 워크플로우 (필수)

**PDCA Design 단계** (`/pdca design` 또는 설계 문서 작성)에서는 Claude와 Gemini가 **공동 설계**합니다.

### 워크플로우

```
Step 1: Claude가 설계 초안 작성
    │   → 요구사항 분석, 아키텍처 초안, 컴포넌트 구조
    │
    ▼
Step 2: Gemini CLI에 설계 리뷰 + 보완 요청
    │   → 기존 코드베이스 맥락 포함하여 설계 검토
    │   → 누락된 관점, 대안 아키텍처, 엣지 케이스 제안
    │
    ▼
Step 3: Claude가 Gemini 피드백 검증 및 통합
    │   → 유효한 제안만 필터링
    │   → 최종 설계 문서 확정
    │
    ▼
Step 4: 최종 설계 문서 저장
    → docs/02-design/features/{feature}.design.md
```

### Gemini 설계 리뷰 명령어 템플릿

```bash
gemini "당신은 시니어 소프트웨어 아키텍트입니다.
다음 설계를 검토하고 개선 제안을 해주세요.

검토 관점:
1. 아키텍처 일관성 (기존 모듈 패턴과의 정합성)
2. 누락된 컴포넌트/인터페이스/엣지 케이스
3. 확장성 및 유지보수성
4. 대안 접근 방식 (더 나은 방법이 있다면)
5. 보안 고려사항

형식:
- [보완] 추가/수정이 필요한 항목
- [대안] 더 나은 접근 방식 제안
- [확인] 설계가 적절한 항목

프로젝트 기술 스택: FastAPI + Next.js + LangGraph, 모듈형 아키텍처
기존 코드 구조: @backend/app/core/registry.py @backend/app/modules/ @frontend/src/lib/modules.ts

---
$(cat docs/02-design/features/{feature}.design.md)
---" -y -o text
```

### Gemini 설계 참여 트리거 조건

| 조건 | Gemini 협업 |
|------|------------|
| `/pdca design` 실행 시 | **필수** |
| 새 모듈/서비스 설계 | **필수** |
| API 스키마 설계 | **필수** |
| 아키텍처 변경 설계 | **필수** |
| 단순 UI 컴포넌트 설계 | 선택적 |
| 버그 수정 설계 | 불필요 |

## Codex CLI 실행 모드 (필수)

Claude Code Bash 도구는 **TTY(터미널)를 제공하지 않습니다**.
Codex CLI의 대화형 모드는 TUI 렌더링을 위해 TTY가 필수이므로, 반드시 비대화형 서브커맨드를 사용합니다.

| 사용 가능 (비대화형) | 사용 금지 (대화형, TTY 필수) |
|---------------------|---------------------------|
| `codex exec "prompt" --ephemeral -s read-only` | `codex "prompt"` |
| `codex review --uncommitted` | `codex -a on-failure "prompt"` |
| `codex review --base main` | `codex --full-auto "prompt"` |
| `stdin \| codex exec - --ephemeral -s read-only` | `stdin \| codex "prompt"` |

## Codex CLI 안전 실행 규칙 (2026-02-27 추가)

> 근거: `.claude/FAILURE_LOG.md` FAIL-001, `docs/03-analysis/codex-cli-error-investigation.report.md`

### 필수 플래그

- **`--ephemeral` 필수** — 세션을 디스크에 저장하지 않아 병렬 실행 시 세션 충돌 방지 ([#11435](https://github.com/openai/codex/issues/11435))
- **`-s read-only` 기본** — 파일 수정이 불필요한 리뷰/검토 작업

### 금지 조합 (알려진 Codex CLI 버그)

| 금지 패턴 | 이유 | GitHub 이슈 |
|-----------|------|------------|
| `codex exec --full-auto` | 무한 대기 + 좀비 프로세스 | [#7852](https://github.com/openai/codex/issues/7852) |
| 200줄 초과 프롬프트 | 출력 형식 붕괴 | [#11122](https://github.com/openai/codex/issues/11122) |
| 병렬 `codex exec` 동시 실행 | 세션 파일 오염 | [#11435](https://github.com/openai/codex/issues/11435) |
| `codex review --base main` | 작업 트리 전체 포함 + 환각 | [#8404](https://github.com/openai/codex/issues/8404) |

### 프롬프트 준비 규칙

- 기획서/코드 전체 대신 **핵심 섹션만 추출** (200줄 이하)
- 프롬프트 간소화는 **이 규칙에 명시된 방법**으로만 수행
- **PM이 임의로 프롬프트를 변경/간소화하는 것은 금지** — 반드시 사용자에게 보고 후 승인
- 프롬프트 추출 예: `sed -n '1,200p' docs/01-plan/features/{feature}.plan.md`

### 순차 실행 보장

에이전트 팀에서 Codex CLI를 호출할 때 **동시 호출을 금지**합니다.
이전 Codex 세션이 완전히 종료된 후 다음 호출을 시작합니다.

### 실패 시 Failure Log 기록 (필수)

CLI 실행 실패 시 `.claude/FAILURE_LOG.md`에 즉시 기록합니다.
상세: `.claude/rules/failure-log-protocol.md`

## CLI 실행 실패 시 필수 절차 (절대 건너뛰기 금지)

외부 CLI(Gemini CLI, Codex CLI) 실행이 실패하면 **자동 Fallback이나 건너뛰기를 하지 않고**, 반드시 아래 절차를 따릅니다:

1. **즉시 중단**: 해당 단계를 건너뛰지 않음
2. **사유 보고**: 사용자에게 다음을 명확히 설명
   - 어떤 CLI가 실패했는지 (Gemini CLI / Codex CLI)
   - 실패 원인 (미설치, 인증 실패, TTY 문제, 타임아웃, 네트워크 오류 등)
   - 시도한 명령어와 에러 메시지 원문
3. **사용자 지침 대기**: `AskUserQuestion`으로 사용자에게 다음 중 선택 요청
   - CLI 문제를 해결 후 재시도
   - 해당 단계를 Claude 단독으로 대체 수행
   - 해당 단계를 건너뛰기
   - 작업 중단
4. **사용자 선택에 따라 진행**: 사용자가 명시적으로 지시한 방향으로만 진행

> **주의**: "도구 선택 기준" 테이블의 Fallback 열은 **사용자가 대체를 승인한 경우**에만 사용합니다. 자동 전환하지 않습니다.

## 필수 규칙

1. **설치 확인**: 매 세션 첫 CLI 호출 전 `which gemini`/`which codex` 실행
2. **인증 확인**: Gemini CLI는 `gemini auth` 기반 인증 사용 (`GEMINI_API_KEY` 사용 금지)
3. **Codex 비대화형**: Claude Code에서 Codex CLI 호출 시 반드시 `codex exec` 또는 `codex review` 서브커맨드 사용
4. **결과 검증**: CLI 출력의 파일 경로/함수명을 Glob/Grep으로 존재 확인
5. **최종 책임**: CLI는 보조, 최종 코드 작성과 검증은 항상 Claude 담당
6. **설계 협업**: PDCA Design 단계에서는 Gemini CLI 리뷰를 필수로 실행
7. **실패 시 사용자 보고 필수**: → 상세: 위 "CLI 실행 실패 시 필수 절차" 참조

## 금지 사항

- 민감 파일 전달 (`.env`, `*.pem`, `credentials.json`)
- 검증 없이 CLI 결과를 코드에 반영
- 사용자 동의 없이 CLI 실행
- 파일 수정 모드로 CLI 실행
- `GEMINI_API_KEY` 환경변수 사용 (auth 인증만 허용)
- **Codex CLI 대화형 모드 사용 (`codex "prompt"`, `codex -a ...`)** — TTY 미지원으로 실패
