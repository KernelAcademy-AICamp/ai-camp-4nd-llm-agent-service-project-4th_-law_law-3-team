# Code Verification Skill

코드 변경 시 빌드/린트/타입체크 및 API 계약 동기화를 검증하는 상세 가이드.

> 규칙 요약: `.claude/rules/code-verification.md` 참조

---

## 1. 변경 유형별 정적 검증 체크리스트

### Backend만 변경

```bash
cd backend

# 1. 린트 검증
uv run ruff check backend/app/

# 2. 타입 체크
uv run mypy backend/app/

# 3. (선택) 테스트 실행
uv run pytest
```

**정상 출력 예시:**
```
# ruff - 에러 없음
All checks passed!

# mypy - 에러 없음
Success: no issues found in N source files
```

**에러 예시:**
```
# ruff
backend/app/modules/lawyer_finder/router/__init__.py:5:1: F401 `os` imported but unused

# mypy
backend/app/services/service_function/lawyer_service.py:42: error: Argument 1 to "func" has incompatible type "str"; expected "int"
```

### Frontend만 변경

```bash
cd frontend

# TypeScript 타입 체크 + ESLint + 번들링을 한번에 수행
npm run build
```

**정상 출력 예시:**
```
✓ Linting and checking validity of types
✓ Creating an optimized production build
✓ Compiled successfully
```

**에러 예시:**
```
Type error: Property 'latitude' does not exist on type 'LawyerCluster'.
  Did you mean 'lat'?
```

### 양쪽 모두 변경

위 Backend + Frontend 검증을 모두 실행하고, 추가로 **Section 3 API 계약 동기화**를 확인한다.

---

## 2. API 변경 시 런타임 검증 상세

API 엔드포인트의 요청/응답 구조를 변경한 경우, 정적 검증 외에 추가로 dev 서버에서 실제 응답을 확인한다.

### 구체적 절차

```bash
# 1. Backend 서버 실행 (백그라운드)
cd backend && uv run uvicorn app.main:app --reload

# 2. 변경한 엔드포인트를 curl로 확인
# Windows PowerShell에서는 curl이 Invoke-WebRequest의 alias이므로 주의
curl.exe http://localhost:8000/api/lawyer-finder/nearby?latitude=37.5&longitude=127.0

# 3. 응답 JSON 필드를 Frontend 타입과 대조
```

### 검증 포인트

- 응답 필드명이 Frontend 타입의 필드명과 **정확히** 일치하는지
- `null`이 올 수 있는 필드가 Frontend에서 `| null` 또는 `?`로 선언되어 있는지
- 중첩 객체의 구조가 일치하는지
- 배열 응답의 아이템 타입이 일치하는지
- HTTP 상태 코드가 예상과 일치하는지

### 서버가 실행 중이 아닌 경우

- 백그라운드로 서버를 실행하고 검증 후 종료
- 또는 `pytest`로 API integration 테스트 실행 (테스트가 있는 경우)

### 예시: 응답 필드 불일치 발견

```json
// Backend 실제 응답
{ "lawyer_id": 1, "latitude": 37.5, "longitude": 127.0 }

// Frontend 타입 정의
interface Lawyer {
  lawyer_id: number;
  lat: number;    // ← 불일치! latitude여야 함
  lng: number;    // ← 불일치! longitude여야 함
}
```

→ Frontend 타입을 Backend 응답에 맞춰 수정하거나, Backend 스키마를 조정한다.

---

## 3. 프론트-백엔드 타입 동기화 패턴

### Pydantic → TypeScript 타입 매핑

| Pydantic | TypeScript | 비고 |
|----------|-----------|------|
| `str` | `string` | |
| `int` | `number` | |
| `float` | `number` | |
| `bool` | `boolean` | |
| `Optional[str]` | `string \| null` | `?`는 undefined 허용, `\| null`은 null 허용 |
| `list[str]` | `string[]` | |
| `dict[str, Any]` | `Record<string, unknown>` | `any` 금지 |
| `datetime` | `string` | ISO 8601 문자열로 직렬화됨 |
| `Enum` | `string` (union type) | `type Status = 'active' \| 'inactive'` |

### 필드명 규칙

이 프로젝트는 camelCase 변환 없이 **snake_case를 양쪽에서 그대로 사용**한다.

```python
# Backend (Pydantic)
class LawyerResponse(BaseModel):
    lawyer_id: int
    office_name: str
    total_count: int
```

```typescript
// Frontend (TypeScript) - snake_case 그대로 사용
interface LawyerResponse {
  lawyer_id: number;
  office_name: string;
  total_count: number;
}
```

### 동기화 검증 절차

1. Backend 스키마 파일 변경 감지
2. 대응하는 Frontend 타입 파일 확인 (Section 3 매핑 테이블 참조)
3. 필드명, 타입, Optional 여부 비교
4. 불일치 시 수정

### 동기화 대상 파일 매핑

| Backend 스키마 | Frontend 타입 |
|---------------|--------------|
| `modules/lawyer_finder/schema/` | `features/lawyer-finder/types/` |
| `modules/lawyer_stats/schema/` | `features/lawyer-stats/types/` |
| `modules/case_precedent/schema/` | `features/case-precedent/types/` |
| `modules/small_claims/schema/` | `features/small-claims/types/` |
| `modules/storyboard/schema/` | `features/storyboard/types/` |
| `modules/review_price/schema/` | `features/review-price/types/` (존재 시) |
| `modules/law_study/schema/` | `features/law-study/types/` (존재 시) |

---

## 4. Zod 런타임 검증 도입 가이드

### 왜 Zod가 필요한가

- TypeScript 타입은 **컴파일 시점**에만 존재하고, 런타임에는 사라진다
- Backend 응답이 예상과 다르면 (`latitude` vs `lat`) 런타임에 `undefined` 발생
- Zod는 **런타임에** 실제 데이터를 검증하여 불일치를 즉시 발견한다

### 설치

```bash
cd frontend && npm install zod
```

### 패턴: Zod 스키마 → TypeScript 타입 추출

```typescript
// features/lawyer-finder/types/index.ts
import { z } from 'zod';

// Zod 스키마 정의 (Backend Pydantic 모델과 1:1 매핑)
export const LawyerSchema = z.object({
  id: z.number(),
  name: z.string(),
  latitude: z.number().nullable(),
  longitude: z.number().nullable(),
  specialty: z.string().nullable(),
  distance: z.number().optional(),
});

// TypeScript 타입은 스키마에서 추출 (중복 정의 방지)
export type Lawyer = z.infer<typeof LawyerSchema>;

// 목록 응답 스키마
export const LawyerListResponseSchema = z.object({
  lawyers: z.array(LawyerSchema),
  total_count: z.number(),
});
export type LawyerListResponse = z.infer<typeof LawyerListResponseSchema>;
```

### 패턴: API 서비스에서 Zod 파싱

```typescript
// features/lawyer-finder/services/index.ts
import { api, endpoints } from '@/lib/api';
import { LawyerListResponseSchema } from '../types';

export async function fetchNearbyLawyers(lat: number, lng: number) {
  const response = await api.get(`${endpoints.lawyerFinder}/nearby`, {
    params: { latitude: lat, longitude: lng },
  });

  // 런타임 검증: Backend 응답이 스키마와 다르면 여기서 에러 발생
  const validated = LawyerListResponseSchema.parse(response.data);
  return validated;
}
```

### Zod 에러 발생 시

```typescript
// parse()는 에러를 throw
try {
  const data = LawyerListResponseSchema.parse(response.data);
} catch (error) {
  if (error instanceof z.ZodError) {
    // 어떤 필드가 불일치인지 상세 에러 메시지 제공
    console.error('API 응답 스키마 불일치:', error.issues);
    // [{ code: 'invalid_type', expected: 'number', received: 'undefined', path: ['latitude'] }]
  }
}

// safeParse()는 에러를 throw하지 않음 (프로덕션 권장)
const result = LawyerListResponseSchema.safeParse(response.data);
if (!result.success) {
  console.warn('API 응답 스키마 불일치:', result.error.issues);
  // fallback 처리 가능
}
```

### 점진적 도입 전략

1. **새로운 API 서비스 함수**에는 Zod 스키마를 필수 적용
2. **기존 서비스 함수**는 버그 발생 시 또는 리팩토링 시 점진적 적용
3. Zod 스키마가 source of truth가 되어 별도 `interface` 중복 정의 불필요

---

## 5. 모듈 동기화 상세

### 모듈 추가 시 체크리스트

1. **Backend**: `backend/app/modules/<module_name>/router/__init__.py`에 `router` 변수 정의
2. **Frontend modules.ts**: `frontend/src/lib/modules.ts`에 모듈 메타데이터 추가
3. **Frontend api.ts**: `frontend/src/lib/api.ts`의 `endpoints` 객체에 추가
4. **Frontend next.config.js**: `rewrites`에 프록시 규칙 추가 (또는 API Route 프록시)
5. **Frontend page**: `src/app/<module-name>/page.tsx` 페이지 생성
6. **Frontend feature**: `src/features/<module-name>/` 구조 생성

### API 경로 변환 규칙

| Backend 모듈명 | API 경로 | Frontend endpoint key |
|----------------|---------|----------------------|
| `snake_case` | `/api/kebab-case` | `camelCase` |

예시:
- `lawyer_finder` → `/api/lawyer-finder` → `lawyerFinder`
- `case_precedent` → `/api/case-precedent` → `casePrecedent`
- `small_claims` → `/api/small-claims` → `smallClaims`

### 현재 모든 모듈의 4곳 매핑 현황

| Backend 모듈 | modules.ts id | api.ts key | next.config.js rewrite | Backend router |
|-------------|--------------|-----------|----------------------|----------------|
| `lawyer_finder` | `lawyer-finder` | `lawyerFinder` | `/api/lawyer-finder/:path*` | ✅ |
| `lawyer_stats` | `lawyer-stats` | `lawyerStat` | `/api/lawyer-stats/:path*` | ✅ |
| `case_precedent` | `case-precedent` | `casePrecedent` | `/api/case-precedent/:path*` | ✅ |
| `small_claims` | `small-claims` | `smallClaims` | `/api/small-claims/:path*` | ✅ |
| `storyboard` | `storyboard` | `storyboard` | API Route 프록시 | ✅ |
| `review_price` | `review-price` | `reviewPrice` | `/api/review-price/:path*` | ✅ |
| `law_study` | `law-study` | `lawStudy` | `/api/law-study/:path*` | ✅ |

---

## 6. 일반적인 코드 꼬임 패턴과 예방법

### 1. 스키마 중복 정의

**문제**: 같은 Pydantic 모델을 `schema/`와 `router/` 양쪽에 정의
**예방**: `schema/`에만 정의하고 router에서 import

```python
# ✅ Good
from ..schema import LawyerResponse

# ❌ Bad - router 안에서 별도 정의
class LawyerResponse(BaseModel):  # 중복!
    ...
```

### 2. Frontend 타입이 Backend에 없는 필드 기대

**문제**: Frontend에서 `lat`/`lng`를 기대하는데 Backend는 `latitude`/`longitude` 반환
**예방**: Backend Pydantic 스키마를 기준으로 Frontend 타입 작성 + Zod 검증

### 3. 미완성 모듈을 enabled로 설정

**문제**: Backend 라우터가 미구현인데 Frontend에서 `enabled: true`
**예방**: 구현 완료 전까지 `enabled: false`로 설정

### 4. 에러 응답 형식 불일치

**문제**: Frontend가 `{ error: string }` 기대, Backend는 `{ detail: string }` 반환
**예방**: `backend/app/core/errors.py` 커스텀 예외 활용, 에러 응답 형식 통일

### 5. import 경로 변경 후 참조 누락

**문제**: 파일 이동 후 다른 파일의 import가 깨짐
**예방**: `ruff check`(F401, E402)와 `mypy`가 감지, `npm run build`가 TS import 에러 감지

### 6. rewrites/endpoints 누락

**문제**: 모듈 추가 시 `next.config.js` rewrites를 빠뜨려 404 발생
**예방**: 모듈 추가 시 Section 5 체크리스트의 4곳 동시 확인

### 7. API 응답 런타임 불일치

**문제**: TypeScript 빌드는 통과하지만 실제 API 응답 필드가 다름
**예방**: Zod `.parse()` 적용 + dev 서버 `curl` 검증

---

## 7. 에러 발생 시 디버깅 체크리스트

### ruff 에러 코드별 해결법

| 코드 | 설명 | 해결 |
|------|------|------|
| `F401` | import했지만 사용하지 않음 | 불필요한 import 제거 |
| `F811` | 같은 이름으로 재정의 | 변수명 변경 또는 중복 제거 |
| `E402` | 모듈 레벨 import가 파일 상단에 없음 | import를 파일 상단으로 이동 |
| `I001` | import 정렬 위반 | `isort` 또는 `ruff format`으로 자동 정렬 |
| `N801` | 클래스명이 PascalCase가 아님 | 클래스명을 PascalCase로 변경 |

### mypy 에러 유형별 해결법

| 에러 유형 | 예시 | 해결 |
|----------|------|------|
| `incompatible type` | `Argument has incompatible type "str"; expected "int"` | 타입 일치시키기 또는 캐스트 |
| `missing return` | `Missing return statement` | 모든 분기에서 return 추가 |
| `name not defined` | `Name "foo" is not defined` | import 추가 또는 변수 정의 |
| `has no attribute` | `"Model" has no attribute "field"` | 모델 필드 확인 |

### Frontend build 에러 유형별 해결법

| 에러 유형 | 예시 | 해결 |
|----------|------|------|
| Type error | `Property 'x' does not exist on type 'Y'` | 타입 정의 확인, 필드명 일치시키기 |
| Module not found | `Cannot find module '@/lib/foo'` | import 경로 확인, 파일 존재 여부 |
| ESLint error | `'var' is not allowed` | ESLint 규칙에 맞게 수정 |
| Unused variable | `'x' is declared but never used` | 사용하거나 제거 |

### API 연동 문제 체크리스트

| 증상 | 원인 | 확인 |
|------|------|------|
| 404 Not Found | rewrites 누락 또는 경로 불일치 | `next.config.js` rewrites 확인 |
| CORS 에러 | Backend CORS 설정 누락 | `backend/app/main.py` CORS 미들웨어 확인 |
| undefined 필드 | 응답 필드명 불일치 | Backend 실제 응답 vs Frontend 타입 비교 |
| 500 Server Error | Backend 런타임 에러 | Backend 로그 확인 |
| 타임아웃 | AI 처리 지연 | `api.ts` timeout, `next.config.js` proxyTimeout 확인 |

### Zod 파싱 에러 해석법

```typescript
// ZodError 구조
{
  issues: [
    {
      code: 'invalid_type',      // 에러 종류
      expected: 'number',        // 기대한 타입
      received: 'undefined',     // 실제 받은 값
      path: ['lawyers', 0, 'latitude'],  // 에러 위치
      message: 'Required'        // 에러 메시지
    }
  ]
}
```

**해석**: `lawyers[0].latitude` 필드가 `number`여야 하는데 `undefined`가 왔다 → Backend 응답에 `latitude` 필드가 없거나, 다른 이름으로 내려오고 있음.

---

## 8. Windows 환경 참고사항

### 명령어 구분자

```bash
# Windows에서도 && 사용 가능 (PowerShell, cmd 모두)
cd backend && uv run ruff check backend/app/

# ; 는 PowerShell에서만 동작 (cmd에서는 안 됨)
```

### 경로 구분자

```bash
# ✅ 슬래시 사용 권장 (Windows에서도 대부분 동작)
uv run ruff check backend/app/modules/lawyer_finder/

# ⚠️ 백슬래시는 이스케이프 문제 발생 가능
```

### PowerShell curl 주의

```powershell
# PowerShell에서 curl은 Invoke-WebRequest의 alias
# 실제 curl.exe를 사용하려면:
curl.exe http://localhost:8000/api/lawyer-finder/nearby?latitude=37.5&longitude=127.0

# 또는 Invoke-RestMethod 사용 (JSON 자동 파싱)
Invoke-RestMethod http://localhost:8000/api/lawyer-finder/nearby?latitude=37.5`&longitude=127.0
# 주의: & 앞에 ` (backtick) 필요
```

### uv, npm 호환성

- `uv run`, `uv sync`: Windows에서 정상 동작
- `npm run build`, `npm run dev`: Windows에서 정상 동작
- Python 경로 주의: `python3` 대신 `python` 사용 (Windows 기본)

---

## 관련 규칙/스킬

| 파일 | 역할 | 이 스킬과의 관계 |
|------|------|-----------------|
| `.claude/rules/code-verification.md` | 검증 프로토콜 규칙 (요약) | 이 스킬의 규칙 버전 |
| `.claude/rules/coding-style.md` | 코드 스타일 | 스타일 vs 변경 후 검증 |
| `.claude/rules/git-convention.md` | Git 커밋 규칙 | 커밋 전 vs 코드 변경 직후 |
| `.claude/skills/tdd-methodology/` | 테스트 작성 방법론 | TDD vs 빌드/린트/타입체크 |
| `.claude/skills/react-nextjs-frontend/` | 컴포넌트 패턴 | UI 패턴 vs API 계약 동기화 |
