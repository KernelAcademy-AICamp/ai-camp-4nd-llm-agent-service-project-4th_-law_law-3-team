# Project Review Skill

법률 서비스 플랫폼에 특화된 코드 리뷰 스킬.
staged 변경 사항을 분석하여 구조화된 리뷰 리포트를 생성합니다.

## 실행 절차

### 1. 변경 사항 수집
```bash
git diff --cached --stat
git diff --cached
```

### 2. 검증 항목

#### Backend (Python/FastAPI)
- [ ] `uv run ruff check backend/app/` 린트 통과
- [ ] `uv run mypy backend/app/` 타입 체크 통과
- [ ] Pydantic 스키마 변경 시 Frontend 타입과 동기화 확인
- [ ] API 경로 변경 시 `frontend/next.config.js` rewrites 확인
- [ ] 새 모듈 추가 시 4곳 동기화 확인 (modules.ts, api.ts, next.config.js, router)
- [ ] LLM 프롬프트에 사용자 입력 직접 삽입 여부 (Prompt Injection)
- [ ] `services/` → `modules/` 방향 import 금지 확인

#### Frontend (TypeScript/Next.js)
- [ ] `npm run build` 빌드 통과 (tsc + ESLint)
- [ ] `dangerouslySetInnerHTML` 사용 여부
- [ ] API 응답 타입이 Backend 스키마와 일치하는지
- [ ] `features/` 간 교차 import 없는지

#### 보안
- [ ] 하드코딩된 API 키, 비밀번호 없는지
- [ ] `console.log`, `print()` 디버그 코드 없는지
- [ ] 새 엔드포인트에 입력 검증(max_length 등) 있는지

### 3. 리포트 형식

```
## Code Review Report

### 변경 요약
- 변경 파일: N개
- 추가/삭제: +X/-Y lines

### 검증 결과
| 항목 | 상태 | 비고 |
|------|------|------|
| Backend 린트 | ✅/❌ | |
| Backend 타입 | ✅/❌ | |
| Frontend 빌드 | ✅/❌ | |
| 보안 이슈 | ✅/❌ | |
| API 동기화 | ✅/❌ | |

### 발견된 이슈
1. [심각도] 파일:라인 - 설명

### 권장 사항
- ...
```
