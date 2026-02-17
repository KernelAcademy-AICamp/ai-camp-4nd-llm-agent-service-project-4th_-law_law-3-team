# CLI Tool Routing Rules

Claude는 외부 AI CLI 도구(Gemini CLI, Codex CLI)를 활용할 때 이 규칙들을 **항상(ALWAYS)** 따라야 합니다.

> **상세 가이드**: `.claude/skills/multi-cli-integration/SKILL.md` 참조

## 도구 선택 기준

| 작업 유형 | 도구 | Fallback |
|----------|------|----------|
| 대규모 코드 분석 (10+ 파일) | Gemini CLI | Task(Explore) |
| 멀티모달 (이미지/스크린샷) | Gemini CLI | 대체 불가 |
| Deep Think 추론 | Gemini CLI | Claude |
| 코드 리뷰 (PR/커밋) | Codex CLI | git diff 분석 |
| 샌드박스 실행 | Codex CLI | 실행 불가 안내 |
| 정밀 코드 작성 / 5개 이하 파일 | Claude | - |

## 필수 규칙

1. **설치 확인**: 매 세션 첫 CLI 호출 전 `which gemini`/`which codex` 실행
2. **결과 검증**: CLI 출력의 파일 경로/함수명을 Glob/Grep으로 존재 확인
3. **최종 책임**: CLI는 보조, 최종 코드 작성과 검증은 항상 Claude 담당

## 금지 사항

- 민감 파일 전달 (`.env`, `*.pem`, `credentials.json`)
- 검증 없이 CLI 결과를 코드에 반영
- 사용자 동의 없이 CLI 실행
- 파일 수정 모드로 CLI 실행
