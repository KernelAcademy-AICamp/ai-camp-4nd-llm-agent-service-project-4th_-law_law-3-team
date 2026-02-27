---
name: project-commit
description: 법률 서비스 플랫폼의 커밋 워크플로우. 변경 분석 → 정적 검증 → 커밋 메시지 생성 → 커밋 수행. 코드 변경 후 커밋 시 사용.
---

# Project Commit Skill

법률 서비스 플랫폼의 커밋 워크플로우.
변경 분석 → 검증 → 커밋 메시지 생성 → 커밋을 수행합니다.

## 실행 절차

### 1. 변경 분석
```bash
git status -u
git diff --cached --stat
git diff
git log --oneline -5
```

### 2. 정적 검증 (필수)

변경된 파일에 따라 해당 검증 실행:

| 변경 대상 | 검증 명령어 | 실행 위치 |
|-----------|------------|-----------|
| Backend Python | `uv run ruff check backend/app/` | `backend/` |
| Backend Python | `uv run mypy backend/app/` | `backend/` |
| Frontend TS/TSX | `npm run build` | `frontend/` |

**검증 실패 시 커밋하지 않음. 수정 후 재검증.**

### 3. 커밋 메시지 작성

Conventional Commits (한국어) + 구조화 태그:
```
<타입>(<범위>): <제목 72자 이내>

[context] <변경 동기/배경>
[changes]
- <변경 항목 1>
- <변경 항목 2>
[impact] <영향받는 모듈/레이어> (3개 이상 파일 변경 시)
[files] <핵심 변경 파일 경로> (3개 이상 파일 변경 시)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

타입: feat, fix, docs, style, refactor, perf, test, build, ci, chore
영향 태그 (해당 시): `BREAKING CHANGE:`, `Migration:`, `API-Change:`

상세 규칙: `.claude/rules/git-convention.md` Section 1, 2 참조

### 4. 커밋 실행

```bash
# 관련 파일만 선택적으로 staging (git add -A 금지)
git add <specific-files>

# HEREDOC으로 커밋
git commit -m "$(cat <<'EOF'
타입(범위): 제목

[context] 변경 동기
[changes]
- 변경 항목

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"

# 검증
git status
```

### 5. 금지 사항
- `.env`, `*.pem`, `credentials.json` 커밋 금지
- `git add -A` 또는 `git add .` 사용 금지
- 검증 미통과 상태에서 커밋 금지
- `--force` push 금지
- `--amend`는 push 전 로컬에서만

---

## 머지 및 충돌 해결

### 머지 전 체크리스트

```bash
git fetch origin
git rebase origin/dev          # 최신 코드 동기화
git status                     # 충돌 확인
uv run pytest                  # Backend 테스트
npm run build                  # Frontend 빌드
```

### 충돌 해결 가이드

```bash
# 1. 충돌 파일 확인
git status

# 2. 충돌 마커 해결
<<<<<<< HEAD
현재 브랜치의 코드
=======
병합하려는 브랜치의 코드
>>>>>>> feature/xxx

# 3. 해결 후
git add <resolved-file>
git rebase --continue   # 또는 git merge --continue

# 4. 반드시 테스트 실행
uv run pytest
```

### 충돌 해결 원칙

1. **코드 이해 우선**: 양쪽 코드의 의도를 먼저 파악
2. **기능 유지**: 양쪽의 기능이 모두 동작하도록 통합
3. **테스트 필수**: 해결 후 반드시 테스트 실행
4. **의심 시 질문**: 확실하지 않으면 원 작성자에게 확인

---

## 실수 복구 가이드

### 커밋 메시지 수정 (push 전)
```bash
git commit --amend -m "새로운 메시지"
```

### 커밋 취소 (push 전)
```bash
git reset --soft HEAD~1   # 변경사항 유지 (staged)
git reset HEAD~1          # 변경사항 유지 (unstaged)
git reset --hard HEAD~1   # 변경사항도 삭제 (주의!)
```

### 잘못된 파일 커밋 제거 (push 전)
```bash
git reset HEAD~1 -- path/to/file
git commit --amend
```

### 이미 Push한 경우
```bash
git revert <commit-hash>  # 안전한 방법 (공유 브랜치)
git push
```
