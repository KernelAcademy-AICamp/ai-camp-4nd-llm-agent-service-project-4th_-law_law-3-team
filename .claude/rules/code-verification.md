# Code Verification Protocol

Claude는 코드를 변경한 후 이 검증 프로토콜을 **항상(ALWAYS)** 따라야 합니다.

> 상세 가이드: `.claude/skills/code-verification/SKILL.md` 참조

---

## 1. 변경 후 정적 검증 (필수)

코드 수정 후 **반드시** 아래 검증 명령어를 실행하고, 실패 시 수정 후 재검증한다.

| 변경 대상 | 검증 명령어 | 실행 위치 |
|-----------|------------|-----------|
| Backend Python | `uv run ruff check backend/app/` | `backend/` |
| Backend Python | `uv run mypy backend/app/` | `backend/` |
| Frontend TS/TSX | `npm run build` | `frontend/` |
| 양쪽 모두 | 위 3개 모두 | 각각 |

- `npm run build`가 TypeScript 타입 체크 + ESLint + 번들링을 한번에 수행
- **검증 통과하지 못한 상태로 작업 완료 보고 금지**

---

## 2. API 변경 시 런타임 검증 (필수)

API 엔드포인트(요청/응답 구조)를 변경한 경우, **정적 검증 외에 추가로** dev 서버를 실행하여 실제 응답을 확인한다.

### 절차
1. Backend dev 서버 실행 (이미 실행 중이면 스킵)
2. `curl` 또는 `Invoke-WebRequest`로 변경된 엔드포인트 호출
3. 응답 JSON의 필드명/타입이 Frontend 타입과 일치하는지 확인
4. 불일치 발견 시 즉시 수정

---

## 3. 프론트-백엔드 API 계약 동기화 (필수)

API 스키마(요청/응답)를 변경할 때 양쪽을 함께 확인한다.

### Backend → Frontend 방향
- `backend/app/modules/<module>/schema/` 변경 → `frontend/src/features/<module>/types/` 확인
- 필드명 매핑: 이 프로젝트는 camelCase 변환 없이 **snake_case를 양쪽에서 그대로 사용**

### Frontend → Backend 방향
- endpoint 호출 변경 → Backend 라우터 경로 존재 확인
- `frontend/src/lib/api.ts` endpoints + `frontend/next.config.js` rewrites 일치 확인

### 동기화 대상 파일 매핑

| Backend 스키마 | Frontend 타입 |
|---------------|--------------|
| `modules/lawyer_finder/schema/` | `features/lawyer-finder/types/` |
| `modules/lawyer_stats/schema/` | `features/lawyer-stats/types/` |
| `modules/case_precedent/schema/` | `features/case-precedent/types/` |
| `modules/small_claims/schema/` | `features/small-claims/types/` |
| `modules/storyboard/schema/` | `features/storyboard/types/` |
| `modules/law_study/schema/` | `features/law-study/types/` (존재 시) |

---

## 4. 모듈 활성화 상태 동기화 (필수)

모듈 추가/삭제/비활성화 시 **4곳 동시 확인**:

1. `frontend/src/lib/modules.ts` — `enabled` 플래그
2. `frontend/src/lib/api.ts` — `endpoints` 객체
3. `frontend/next.config.js` — `rewrites` 프록시 규칙 (또는 API Route 프록시)
4. `backend/app/modules/<module>/router/__init__.py` — 라우터 구현 여부

### 현재 모듈 매핑 현황

| 모듈 (Backend) | modules.ts | api.ts | next.config.js rewrites |
|----------------|-----------|--------|------------------------|
| `lawyer_finder` | `lawyer-finder` ✅ | `lawyerFinder` ✅ | `/api/lawyer-finder` ✅ |
| `lawyer_stats` | `lawyer-stats` ✅ | `lawyerStat` ✅ | `/api/lawyer-stats` ✅ |
| `case_precedent` | `case-precedent` ✅ | `casePrecedent` ✅ | `/api/case-precedent` ✅ |
| `small_claims` | `small-claims` ✅ | `smallClaims` ✅ | `/api/small-claims` ✅ |
| `storyboard` | `storyboard` ✅ | `storyboard` ✅ | API Route 프록시 |
| `law_study` | `law-study` ✅ | `lawStudy` ✅ | `/api/law-study` ✅ |

**핵심**: Frontend `enabled: true`인 모듈은 Backend에 **동작하는** 라우터가 있어야 한다.

---

## 5. import/의존성 검증

- **Backend**: `modules/` → `services/` import OK, 반대 방향 금지
- **Frontend**: `features/` 간 교차 import 금지, 공유 코드는 `lib/` 또는 `components/`
- **스키마 중복 금지**: 같은 Pydantic 모델을 2곳에 정의하지 않음

---

## 6. Zod 런타임 검증 권장 패턴

API 서비스 함수에서 응답을 받을 때 Zod 스키마로 파싱하는 패턴을 **권장**한다.

- 새로운 API 서비스 함수 작성 시 Zod 스키마 정의를 함께 작성
- 기존 서비스 함수는 점진적으로 적용 (강제하지 않음)
- Zod 스키마에서 TypeScript 타입을 추출 (`z.infer<>`)하여 타입 정의 중복 제거

```typescript
// 예시: Zod 스키마 → 타입 추출
import { z } from 'zod';

export const LawyerSchema = z.object({
  id: z.number(),
  name: z.string(),
  latitude: z.number().nullable(),
  longitude: z.number().nullable(),
});

export type Lawyer = z.infer<typeof LawyerSchema>;
```

---

## 7. 검증 실패 시 행동

1. 에러 메시지 읽기
2. 원인 파악 및 수정
3. 재검증
4. **통과할 때까지 반복**

검증 미통과 상태로 작업 완료 보고 **절대 금지**.

---

**중요**: 이 규칙들은 코드 변경 시 예외 없이 적용됩니다. 검증을 건너뛸 합당한 이유가 있다면 사용자에게 명시적으로 설명해야 합니다.
