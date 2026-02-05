#!/bin/bash
# PostToolUse hook: Edit|Write 후 자동 린트 검증
# Python → ruff check, TypeScript → tsc --noEmit

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

# 파일 경로가 없으면 종료
if [[ -z "$FILE_PATH" || ! -f "$FILE_PATH" ]]; then
  exit 0
fi

PROJECT_DIR="$CLAUDE_PROJECT_DIR"

# Python 파일 린트 (ruff)
if [[ "$FILE_PATH" =~ \.py$ ]]; then
  if [[ "$FILE_PATH" =~ /backend/ ]]; then
    cd "$PROJECT_DIR/backend" 2>/dev/null || exit 0
    RESULT=$(uv run ruff check "$FILE_PATH" 2>&1)
    if [[ $? -ne 0 ]]; then
      echo "$RESULT" >&2
      exit 1
    fi
  fi
fi

# TypeScript/TSX 파일 검증
if [[ "$FILE_PATH" =~ \.(ts|tsx)$ ]]; then
  if [[ "$FILE_PATH" =~ /frontend/ ]]; then
    cd "$PROJECT_DIR/frontend" 2>/dev/null || exit 0
    RESULT=$(npx tsc --noEmit 2>&1 | head -20)
    if [[ $? -ne 0 ]]; then
      echo "$RESULT" >&2
      exit 1
    fi
  fi
fi

exit 0
