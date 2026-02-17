# Claude Code Hooks 설정

프로젝트에 `Edit`/`Write` 후 자동 린트 검증 훅이 포함되어 있습니다.
훅 스크립트(`post-write-verify.sh`)는 git에 포함되지만, 활성화 설정은 **각자 로컬에서** 해야 합니다.

## 설정 방법

`.claude/settings.local.json` 파일에 아래 내용을 추가하세요 (파일이 없으면 새로 생성):

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "\"$CLAUDE_PROJECT_DIR\"/.claude/hooks/post-write-verify.sh",
            "timeout": 30
          }
        ]
      }
    ]
  }
}
```

## 동작 방식

| 파일 타입 | 검증 도구 | 실패 시 |
|-----------|----------|---------|
| `*.py` (backend/) | `uv run ruff check` | 수정 후 재검증 |
| `*.ts, *.tsx` (frontend/) | `npx tsc --noEmit` | 수정 후 재검증 |

## 참고
- `.claude/settings.local.json`은 `.gitignore`에 포함되어 있으므로 커밋되지 않습니다
- 훅이 실패하면 Claude가 자동으로 코드를 수정하고 재검증합니다
- `jq`가 설치되어 있어야 합니다 (`brew install jq`)
