#!/bin/bash
# SessionStart hook: 현재 git 브랜치, worktree 경로, 작업 컨텍스트를 Claude에게 주입

BRANCH=$(git branch --show-current 2>/dev/null)
if [ -z "$BRANCH" ]; then
  BRANCH="(detached HEAD)"
fi

WORKTREE=$(git rev-parse --show-toplevel 2>/dev/null)
if [ -z "$WORKTREE" ]; then
  WORKTREE="$(pwd)"
fi

echo "[git-context] branch=$BRANCH worktree=$WORKTREE"

# worktree 작업 컨텍스트 (있으면 주입)
CONTEXT_FILE="$WORKTREE/.claude/worktree-context.md"
if [ -f "$CONTEXT_FILE" ]; then
  echo ""
  cat "$CONTEXT_FILE"
fi
