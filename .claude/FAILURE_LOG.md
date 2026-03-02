# Failure Log

> **목적**: 모든 작업 중 발생한 실패/오류를 기록하고, 동일한 실수를 반복하지 않기 위한 로그.
> **필수 규칙**: 모든 에이전트는 작업 시작 전 이 파일을 반드시 읽고, 기록된 실패 패턴을 회피해야 합니다.

---

## FAIL-001: Codex CLI 호출 실패 시 사용자 미보고 + PM 독단적 프롬프트 간소화

| 항목 | 내용 |
|------|------|
| **발생일** | 2026-02-27 |
| **심각도** | **Critical** (워크플로우 규칙 정면 위반) |
| **발생 위치** | 에이전트 팀 작업 - External Consultant (Codex CLI) 호출 단계 |
| **증상** | Codex CLI로 기획 검토/코드 리뷰 요청 시 지속적 오류 발생 |
| **위반 내용** | |
| - 규칙 1 | `cli-tool-routing.md` "CLI 실행 실패 시 필수 절차" 위반: 실패 시 즉시 중단 + 사용자 보고 의무를 무시 |
| - 규칙 2 | `triple-review-workflow.md` "외부 CLI 실행 실패 시 처리" 위반: 자동 건너뛰기/Fallback 금지 규칙 위반 |
| - 규칙 3 | PM이 사용자 승인 없이 독단적으로 프롬프트를 간소화하여 오류 회피 시도 |
| **근본 원인** | 3가지 복합 원인 (상세: `docs/03-analysis/codex-cli-error-investigation.report.md`) |
| - 원인 1 | 병렬 세션 충돌: `~/.codex/` 공유 세션 파일로 인스턴스 간 오염 ([#11435](https://github.com/openai/codex/issues/11435)) |
| - 원인 2 | `--full-auto` 무한 대기: 샌드박스 프로세스 좀비화 ([#7852](https://github.com/openai/codex/issues/7852)) |
| - 원인 3 | 200줄+ 프롬프트 → 출력 형식 붕괴 ([#11122](https://github.com/openai/codex/issues/11122)) |
| **비교 참고** | Gemini CLI는 무상태(stateless) 단일 프로세스 모델이므로 동일 이슈 없음 |
| **해결 방법** | |
| - 해결 1 | `--ephemeral` 플래그 필수 사용 (세션 충돌 방지) |
| - 해결 2 | `--full-auto` 사용 금지 (무한 대기 방지) |
| - 해결 3 | 프롬프트 200줄 이하 제한 (출력 형식 보장) |
| - 해결 4 | 순차 실행 보장 (병렬 호출 금지) |
| **재발 방지** | |
| - 조치 1 | 외부 CLI 실패 시 반드시 사용자에게 AskUserQuestion으로 보고 |
| - 조치 2 | PM은 프롬프트 내용을 임의로 변경/간소화할 권한 없음 |
| - 조치 3 | 모든 CLI 오류를 이 Failure Log에 즉시 기록 |
| - 조치 4 | `cli-tool-routing.md` 규칙 업데이트 (--full-auto 금지, --ephemeral 필수, 200줄 제한) |
| - 조치 5 | `.codex/skills/git-convention-bridge/SKILL.md` YAML frontmatter 수정 |
| **상태** | **해결됨** — 규칙 업데이트 완료 + 외부 검증(Red Team + Consultant) 완료 |

---

## FAIL-002: 스토리보드 생성 SSE 타임아웃 (Request timed out) — 반복 발생

| 항목 | 내용 |
|------|------|
| **발생일** | 2026-03-01 (최초 발생 이후 지속 반복, 에이전트 팀 수정 시도 후에도 재발) |
| **심각도** | **High** (핵심 기능 완전 불능, Failure Log 미기록 상태로 반복) |
| **발생 위치** | 콘텐츠 마케팅 → 대본 생성 → 스토리보드 생성 버튼 클릭 후 |
| **증상** | "스토리보드 오류: Request timed out" 에러가 지속 발생. 여러 차례 수정 시도 후에도 동일 오류 재발. |
| **근본 원인** | 3가지 복합 원인 |
| - 원인 1 | **SSE 전용 API Route 누락**: `keywords/collect/stream`과 `script/generate`는 rewrites 버퍼링 우회를 위한 전용 API Route(`frontend/src/app/api/...`)가 있으나, `script/webtoon/{jobId}/stream`에는 **전용 API Route가 없음**. Next.js rewrites가 SSE 데이터를 버퍼링하여 heartbeat가 클라이언트에 전달되지 않음 |
| - 원인 2 | **proxyTimeout 초과**: Next.js `proxyTimeout`이 120초인데, 웹툰 파이프라인(장면분할 + 이미지생성 ×10패널)은 3~5분+ 소요 → 120초 시점에 연결 강제 종료 |
| - 원인 3 | **기존 패턴 미적용**: `next.config.js`에 "rewrites는 SSE 스트리밍을 버퍼링하므로 API Route 사용"이라는 코멘트가 이미 있었으나, 웹툰 스토리보드 추가 시 이 패턴이 적용되지 않음 |
| **증거** | |
| - 코드 1 | `frontend/src/app/api/content-marketing/script/generate/route.ts` — 대본 SSE 전용 API Route (존재) |
| - 코드 2 | `frontend/src/app/api/content-marketing/keywords/collect/stream/route.ts` — 키워드 SSE 전용 API Route (존재) |
| - 코드 3 | `frontend/src/app/api/content-marketing/script/webtoon/` — **존재하지 않음** (누락) |
| - 코드 4 | `frontend/next.config.js:9` — `proxyTimeout: 120000` (120초) |
| - 코드 5 | `backend/app/services/service_function/webtoon_service.py:205` — heartbeat 30초 간격 |
| **위반 내용** | 이전 에이전트 팀 수정 시도 시 Failure Log 미기록 (failure-log-protocol.md 위반) |
| **해결 방법** | 아래 수정 계획 참조 |
| - 해결 1 | 웹툰 SSE 전용 API Route 생성 (`frontend/src/app/api/content-marketing/script/webtoon/[jobId]/stream/route.ts`) |
| - 해결 2 | 프론트엔드 SSE URL을 API Route 경로로 변경 |
| - 해결 3 | API Route에 충분한 타임아웃 설정 (300초+) |
| **재발 방지** | |
| - 조치 1 | 새 SSE 엔드포인트 추가 시 반드시 전용 API Route 생성 여부 확인 |
| - 조치 2 | `next.config.js` 코멘트에 웹툰 스토리보드 SSE도 명시 |
| - 조치 3 | 실패 반복 시 반드시 Failure Log에 기록 후 구조적 원인 분석 |
| **상태** | **해결됨** — SSE 전용 API Route 생성 완료 (`frontend/src/app/api/content-marketing/script/webtoon/[jobId]/stream/route.ts`), Red Team + External Consultant 검증 완료, 빌드 통과 |

---

## 기록 규칙

### 새 실패 기록 시 필수 항목

```markdown
## FAIL-NNN: [제목]

| 항목 | 내용 |
|------|------|
| **발생일** | YYYY-MM-DD |
| **심각도** | Critical / High / Medium / Low |
| **발생 위치** | 어디에서 발생했는지 |
| **증상** | 무슨 일이 일어났는지 |
| **근본 원인** | 왜 발생했는지 |
| **해결 방법** | 어떻게 해결했는지 |
| **재발 방지** | 앞으로 어떻게 방지할 것인지 |
| **상태** | 조사 중 / 해결됨 / 미해결 |
```

### 에이전트 작업 시작 시 체크리스트

1. **FAILURE_LOG.md 읽기** (필수)
2. 현재 작업이 기록된 실패 패턴과 관련 있는지 확인
3. 관련 있다면 재발 방지 조치를 적용
4. 새로운 실패 발생 시 즉시 이 파일에 기록
5. **외부 CLI 실패 시**: 자동 회피/간소화 금지, 반드시 사용자 보고
