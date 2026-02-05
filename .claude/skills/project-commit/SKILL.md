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

Conventional Commits (한국어):
```
<타입>(<범위>): <명령형 제목 50자 이내>

<본문: 무엇을, 왜 변경했는지>

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

타입: feat, fix, docs, style, refactor, perf, test, build, ci, chore

### 4. 커밋 실행

```bash
# 관련 파일만 선택적으로 staging (git add -A 금지)
git add <specific-files>

# HEREDOC으로 커밋
git commit -m "$(cat <<'EOF'
타입(범위): 제목

본문

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
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
