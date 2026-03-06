# Code Verification Protocol

Claude는 코드를 변경한 후 이 검증 프로토콜을 **항상(ALWAYS)** 따라야 합니다.

> **상세 가이드**: `.claude/skills/code-verification/SKILL.md` 참조

## 1. 변경 후 정적 검증 (필수)

| 변경 대상 | 검증 명령어 | 실행 위치 |
|-----------|------------|-----------|
| Backend Python | `uv run ruff check backend/app/` | `backend/` |
| Backend Python | `uv run mypy backend/app/` | `backend/` |
| Frontend TS/TSX | `npm run build` | `frontend/` |

**검증 통과하지 못한 상태로 작업 완료 보고 금지**

## 2. API 변경 시 런타임 검증 (필수)

API 엔드포인트 변경 시 dev 서버에서 `curl`로 실제 응답 확인.
응답 JSON 필드명/타입이 Frontend 타입과 일치하는지 검증.

## 3. 프론트-백엔드 API 계약 동기화

- **snake_case를 양쪽에서 그대로 사용** (camelCase 변환 없음)
- `backend/app/modules/<module>/schema/` 변경 → `frontend/src/features/<module>/types/` 확인
- → 모듈별 매핑: CLAUDE.md "모듈 매핑" 참조

## 4. 모듈 동기화 (4곳 동시 확인)

1. `frontend/src/lib/modules.ts` — `enabled` 플래그
2. `frontend/src/lib/api.ts` — `endpoints` 객체
3. `frontend/next.config.js` — `rewrites` 프록시 규칙
4. `backend/app/modules/<module>/router/__init__.py` — 라우터 구현

## 5. 검증 실패 시

에러 읽기 → 원인 파악 → 수정 → 재검증 → **통과할 때까지 반복**
