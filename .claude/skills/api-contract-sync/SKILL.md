---
name: api-contract-sync
description: 프론트엔드-백엔드 API 계약 자동 동기화 검증. Pydantic 스키마 ↔ TypeScript 타입 매칭, endpoint 경로 일치, next.config.js rewrites 확인. API 스키마 변경, 모듈 추가/삭제, 프론트-백 연동 작업 시 사용.
---

# API Contract Sync Skill

프론트엔드-백엔드 API 계약의 자동 동기화 검증 가이드.

> **참조**: `code-verification.md` (Section 3-4), `CLAUDE.md` (모듈 매핑 현황)

## 1. 동기화 대상 매핑

### 모듈별 파일 매핑

| 모듈 | Backend 스키마 | Frontend 타입 | API 경로 |
|------|---------------|-------------- |----------|
| lawyer_finder | `modules/lawyer_finder/schema/` | `features/lawyer-finder/types/` | `/api/lawyer-finder` |
| lawyer_stats | `modules/lawyer_stats/schema/` | `features/lawyer-stats/types/` | `/api/lawyer-stats` |
| case_precedent | `modules/case_precedent/schema/` | `features/case-precedent/types/` | `/api/case-precedent` |
| small_claims | `modules/small_claims/schema/` | `features/small-claims/types/` | `/api/small-claims` |
| storyboard | `modules/storyboard/schema/` | `features/storyboard/types/` | `/api/storyboard` |
| law_study | `modules/law_study/schema/` | `features/law-study/types/` | `/api/law-study` |

### 4곳 동기화 체크포인트

| # | 파일 | 확인 항목 |
|---|------|----------|
| 1 | `frontend/src/lib/modules.ts` | `enabled` 플래그 |
| 2 | `frontend/src/lib/api.ts` | `endpoints` 객체 |
| 3 | `frontend/next.config.js` | `rewrites` 프록시 규칙 |
| 4 | `backend/app/modules/<module>/router/__init__.py` | 라우터 구현 |

## 2. 자동 검증 명령어

### 2.1 스키마 필드 비교

```bash
# Backend Pydantic 모델의 필드 추출
grep -rn "class.*BaseModel\|class.*Schema" backend/app/modules/*/schema/*.py

# Frontend TypeScript 타입의 필드 추출
grep -rn "interface\|type.*=" frontend/src/features/*/types/*.ts

# 비교: Backend 스키마 필드명이 Frontend 타입에 존재하는지
# 이 프로젝트는 snake_case를 양쪽에서 그대로 사용
```

### 2.2 Endpoint 경로 일치 확인

```bash
# Backend 라우터 경로 추출
grep -rn "@router\.\(get\|post\|put\|delete\|patch\)" backend/app/modules/*/router/*.py

# Frontend API 엔드포인트 추출
grep -n "endpoints" frontend/src/lib/api.ts

# next.config.js rewrites 추출
grep -A2 "source.*destination" frontend/next.config.js
```

### 2.3 모듈 활성화 상태 비교

```bash
# Frontend modules.ts에서 enabled 모듈 추출
grep -B1 "enabled.*true" frontend/src/lib/modules.ts

# Backend에서 라우터 존재 확인
ls backend/app/modules/*/router/__init__.py
```

## 3. 타입 매핑 규칙

### Python → TypeScript 타입 변환

| Python (Pydantic) | TypeScript | 비고 |
|-------------------|------------|------|
| `str` | `string` | |
| `int` | `number` | |
| `float` | `number` | |
| `bool` | `boolean` | |
| `Optional[T]` | `T \| null` | |
| `list[T]` | `T[]` | |
| `dict[str, T]` | `Record<string, T>` | |
| `datetime` | `string` | ISO 8601 형식 |
| `Enum` | `string` (유니온 리터럴) | `"A" \| "B" \| "C"` |

### 필드명 규칙

이 프로젝트는 **snake_case를 양쪽에서 그대로 사용**합니다.

```python
# Backend
class LawyerResponse(BaseModel):
    lawyer_name: str
    office_address: Optional[str]
    specialties: list[str]
```

```typescript
// Frontend (snake_case 그대로 사용)
interface LawyerResponse {
  lawyer_name: string;
  office_address: string | null;
  specialties: string[];
}
```

## 4. 변경 시 검증 절차

### 4.1 Backend 스키마 변경 시

1. Pydantic 모델 변경 내용 확인
2. 대응하는 Frontend TypeScript 타입 파일 찾기
3. 필드 추가/삭제/타입변경을 Frontend에도 반영
4. Frontend에서 해당 타입을 사용하는 컴포넌트 확인
5. `npm run build`로 타입 에러 검증

### 4.2 Frontend 타입 변경 시

1. TypeScript 타입 변경 내용 확인
2. 대응하는 Backend Pydantic 모델 확인
3. API 호출 코드에서 올바른 필드명 사용 확인
4. `uv run ruff check`로 Backend 검증

### 4.3 새 모듈 추가 시

**4곳 동시 생성 체크리스트**:

- [ ] `frontend/src/lib/modules.ts` — 모듈 메타데이터 추가
- [ ] `frontend/src/lib/api.ts` — endpoints 객체에 추가
- [ ] `frontend/next.config.js` — rewrites 프록시 규칙 추가
- [ ] `backend/app/modules/<module>/router/__init__.py` — 라우터 구현

### 4.4 API 경로 변경 시

```
Backend: snake_case 모듈명 → kebab-case API 경로
예: lawyer_finder → /api/lawyer-finder

확인:
1. backend/app/core/registry.py 자동 등록 경로 확인
2. frontend/next.config.js rewrites 경로 일치 확인
3. frontend/src/lib/api.ts endpoints 경로 일치 확인
```

## 5. 불일치 유형별 해결

### 필드 누락

```
Backend에 있지만 Frontend에 없는 필드
→ Frontend 타입에 optional 필드로 추가 (기존 코드 호환성)

Frontend에 있지만 Backend에 없는 필드
→ 사용 여부 확인 후 제거 또는 Backend에 추가
```

### 타입 불일치

```
Backend: Optional[str], Frontend: string (null 미처리)
→ Frontend를 string | null로 수정, 사용처에 null 체크 추가

Backend: list[str], Frontend: string (배열 미처리)
→ Frontend를 string[]로 수정
```

### 경로 불일치

```
Backend 등록: /api/lawyer-finder
next.config.js: /api/lawyer-search (오타)
→ next.config.js를 /api/lawyer-finder로 수정
```

## 6. 보고 형식

```
## API 계약 동기화 검증 결과

### 모듈별 상태

| 모듈 | 스키마 | 타입 | Endpoint | Rewrites | 상태 |
|------|--------|------|----------|----------|------|
| lawyer_finder | ✅ | ✅ | ✅ | ✅ | OK |
| lawyer_stats | ✅ | ✅ | ✅ | ✅ | OK |
| case_precedent | ✅ | ⚠️ | ✅ | ✅ | 타입 불일치 |

### 불일치 상세

1. **case_precedent**: `decision_date` 필드
   - Backend: `Optional[str]`
   - Frontend: `string` (null 미처리)
   - 영향: null 값 수신 시 런타임 에러
   - 수정: Frontend 타입 수정 필요

### 총 결과: N/M 모듈 동기화 완료
```
