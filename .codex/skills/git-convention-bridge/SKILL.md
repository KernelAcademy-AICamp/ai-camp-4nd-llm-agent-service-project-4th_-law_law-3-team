---
name: git-convention-bridge
description: "Git convention bridge for Codex CLI - maps .claude rules to Codex runtime"
---
# Git Convention Bridge (Codex)

Codex가 이 저장소에서 Git 작업을 수행할 때 `.claude` 규칙을 안전하게 적용하기 위한 브리지 스킬입니다.

## 목적

- `.claude/rules/git-convention.md`를 Git 작업의 단일 기준(SSOT)으로 사용
- Codex 런타임의 브랜치 prefix 제약(`codex/`)과 프로젝트 브랜치 규칙을 매핑
- 브랜치 생성/커밋 전에 확인할 체크리스트 제공

## 참조 우선순위

1. `.claude/rules/git-convention.md` (브랜치/커밋 규칙 SSOT)
2. `CLAUDE.md` (기본 브랜치 `dev`, PR/운영 규칙)
3. `AGENTS.md` (프로젝트 요약 규칙, 커밋 메시지 형식)

충돌 시 위 우선순위를 따르고, Codex 런타임 제약으로 인해 브랜치명만 아래 매핑 규칙을 적용합니다.

## 브랜치 명명 매핑 규칙 (Codex 제약 대응)

프로젝트 규칙 예시:
- `feature/<desc>`
- `fix/<desc>`
- `docs/<desc>`
- `refactor/<desc>`
- `release/<version>`
- `hotfix/<desc>`

Codex에서 생성할 실제 브랜치명:
- `feature/foo` -> `codex/feature-foo`
- `fix/bar` -> `codex/fix-bar`
- `docs/baz` -> `codex/docs-baz`
- `refactor/qux` -> `codex/refactor-qux`
- `hotfix/x` -> `codex/hotfix-x`
- `release/1.2.3` -> `codex/release-1-2-3`

규칙:
- 소문자만 사용
- 케밥 케이스 유지
- 슬래시(`/`)는 첫 prefix 이후 `-`로 변환
- 버전의 점(`.`)은 `-`로 변환

## 브랜치 생성 절차

1. 현재 브랜치와 워크트리 상태 확인 (`git branch --show-current`, `git status --short`)
2. 기준 브랜치가 `dev`인지 확인 (dirty worktree면 필요 시 `git worktree` 사용)
3. 원래 의도 브랜치 타입(`docs`, `feature`, `fix` 등)을 결정
4. Codex 매핑 브랜치명 생성 (`codex/<type>-<desc>`)
5. 이름 검증:
   - 소문자
   - 공백 없음
   - 설명은 케밥 케이스
6. 브랜치 생성/전환

## 커밋 메시지 절차

`.claude/rules/git-convention.md` Section 1~3을 따릅니다.

필수 형식:

```text
<타입>(<범위>): <제목>

[context] ...
[changes]
- ...
- ...
```

추가 규칙:
- 제목은 한국어, 72자 이내, 마침표 금지
- 타입은 영어 (`feat`, `fix`, `docs`, `refactor`, `test` 등)
- 변경 파일이 3개 이상이면 `[impact]`, `[files]` 포함

## 금지사항 (요약)

- `git push --force` (특히 `main`, `dev`, `release`)
- 민감 파일 커밋 (`.env`, `*.pem`, `credentials.json`)
- 공유 브랜치에서 `reset --hard`, `rebase -i`, `amend` (push 후)
- 대용량 파일(100MB+) 직접 커밋

## 실행 예시

### 문서 작업

- 의도 브랜치: `docs/lawyer-exam-ingest-plan`
- Codex 브랜치: `codex/docs-lawyer-exam-ingest-plan`

### 기능 작업

- 의도 브랜치: `feature/lawyer-exam-parser`
- Codex 브랜치: `codex/feature-lawyer-exam-parser`

### 버그 수정

- 의도 브랜치: `fix/ingest-source-path`
- Codex 브랜치: `codex/fix-ingest-source-path`

## 운영 메모

- 현재 작업 트리가 dirty이면 기존 변경을 건드리지 않기 위해 `git worktree`를 우선 고려합니다.
- `.claude/rules/git-convention.md`가 변경되면 이 브리지 스킬은 매핑/예시만 갱신하고 규칙 본문은 복제하지 않습니다.
