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
