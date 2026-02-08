#!/bin/bash
# Plan Review Script: Gemini CLI + Codex CLI 병렬 플랜 검토
#
# 사용법: bash review-plan.sh <context_file>
# 결과:   /tmp/plan_review_result.md
#
# SKILL 참조: .claude/skills/plan-review/SKILL.md

set -euo pipefail

CONTEXT_FILE="${1:-}"
RESULT_FILE="/tmp/plan_review_result.md"
GEMINI_RESULT="/tmp/plan_review_gemini.md"
CODEX_RESULT="/tmp/plan_review_codex.md"

# ─── 입력 검증 ───
if [[ -z "$CONTEXT_FILE" ]]; then
  echo "Error: 컨텍스트 파일 경로를 지정해주세요." >&2
  echo "Usage: bash review-plan.sh <context_file>" >&2
  exit 1
fi

if [[ ! -f "$CONTEXT_FILE" ]]; then
  echo "Error: 파일을 찾을 수 없습니다: $CONTEXT_FILE" >&2
  exit 1
fi

CONTEXT=$(cat "$CONTEXT_FILE")

# ─── CLI 설치 확인 ───
GEMINI_OK=false
CODEX_OK=false

if command -v gemini &>/dev/null; then
  GEMINI_OK=true
fi

if command -v codex &>/dev/null; then
  CODEX_OK=true
fi

if [[ "$GEMINI_OK" == false && "$CODEX_OK" == false ]]; then
  cat > "$RESULT_FILE" <<'EOF'
# 리뷰 결과 없음

Gemini CLI와 Codex CLI가 모두 설치되지 않아 외부 리뷰를 수행할 수 없습니다.

설치 방법:
- Gemini CLI: https://github.com/google-gemini/gemini-cli
- Codex CLI: https://github.com/openai/codex
EOF
  echo "$RESULT_FILE"
  exit 0
fi

# ─── 리뷰 프롬프트 정의 ───
GEMINI_PROMPT="당신은 소프트웨어 아키텍트입니다. 다음 구현 플랜을 검토해주세요.

검토 관점:
1. 기존 아키텍처와의 일관성 (프로젝트 제약 조건 참고)
2. 누락된 파일/의존성/영향 범위
3. 모듈 간 결합도 문제
4. 확장성/유지보수성 우려

형식: 각 항목에 대해 [통과/주의/문제] 판정과 1줄 근거를 제시하세요.
문제가 없으면 '검토 통과'라고만 답하세요. 반드시 한국어로 답변하세요.

---
${CONTEXT}
---"

CODEX_PROMPT="당신은 시니어 개발자입니다. 다음 구현 플랜의 실행가능성을 검토해주세요.

검토 관점:
1. 구현 순서의 논리적 정확성 (의존성 순서 올바른지)
2. 타입/인터페이스 호환성 (Backend↔Frontend 계약)
3. 엣지 케이스 누락
4. 테스트 전략의 충분성

형식: 각 항목에 대해 [통과/주의/문제] 판정과 1줄 근거를 제시하세요.
문제가 없으면 '검토 통과'라고만 답하세요. 반드시 한국어로 답변하세요.

---
${CONTEXT}
---"

# ─── 결과 파일 초기화 ───
cat > "$RESULT_FILE" <<EOF
# 플랜 외부 리뷰 결과

검토 시각: $(date '+%Y-%m-%d %H:%M:%S')
사용 도구: Gemini=${GEMINI_OK}, Codex=${CODEX_OK}

EOF

# ─── 병렬 실행 ───
PIDS=()

if [[ "$GEMINI_OK" == true ]]; then
  (
    REVIEW=$(gemini "$GEMINI_PROMPT" -y -o text 2>&1) || REVIEW="[오류] Gemini CLI 실행 실패: $?"
    {
      echo "## Gemini CLI 리뷰 (구조/아키텍처)"
      echo ""
      echo "$REVIEW"
      echo ""
    } > "$GEMINI_RESULT"
  ) &
  PIDS+=($!)
fi

if [[ "$CODEX_OK" == true ]]; then
  (
    REVIEW=$(codex --full-auto "$CODEX_PROMPT" 2>&1) || REVIEW="[오류] Codex CLI 실행 실패: $?"
    {
      echo "## Codex CLI 리뷰 (실행가능성)"
      echo ""
      echo "$REVIEW"
      echo ""
    } > "$CODEX_RESULT"
  ) &
  PIDS+=($!)
fi

# 모든 프로세스 완료 대기
for PID in "${PIDS[@]}"; do
  wait "$PID" 2>/dev/null || true
done

# ─── 결과 병합 ───
if [[ -f "$GEMINI_RESULT" ]]; then
  cat "$GEMINI_RESULT" >> "$RESULT_FILE"
fi

if [[ -f "$CODEX_RESULT" ]]; then
  cat "$CODEX_RESULT" >> "$RESULT_FILE"
fi

# ─── 정리 ───
rm -f "$GEMINI_RESULT" "$CODEX_RESULT"

echo "$RESULT_FILE"
