# 판례 검색 필터링 기능 계획

## 요약
1. 사이드바에서 판례 검색 화면에 직접 진입했을 때 **사건종류명 + 기간 필터**로 PostgreSQL에서 판례를 검색하는 기능을 추가한다.
2. 채팅 진입 경로(aiReferences 기반)는 **변경하지 않는다**.
3. LanceDB 벡터 검색을 사용하지 않고 **PostgreSQL만 사용**한다.
4. 기존 `GET /precedents` 엔드포인트는 건드리지 않고, 새 엔드포인트 `GET /precedents/filter`를 추가한다.

## 현재 문제
1. **사이드바 직접 진입 시 빈 화면**: `sessionData.aiReferences`가 없어 "챗봇에게 질문해보세요" 안내만 표시
2. **필터 UI 부재**: `SearchPanel.tsx`에 필터 컴포넌트 없음 — 사건종류, 기간 필터가 전혀 없음
3. **PostgreSQL 필터 쿼리 미활용**: `case_type`(B-tree 인덱스), `decision_date`(B-tree 인덱스) 이미 존재하나 검색에 미사용
4. **직접 검색 불가**: 모든 검색이 LanceDB 벡터 검색에만 의존, 키워드+필터 기반 DB 직접 검색 없음

## MVP 범위
1. 백엔드: PostgreSQL 필터 쿼리 서비스 함수 + 새 API 엔드포인트 2개
2. 프론트엔드: 필터 UI(사건종류 select + 기간 pill + 직접입력 date) + 결과 목록 + 상세 뷰 연결
3. 사이드바 직접 진입 시만 필터 모드 활성화 (page.tsx 분기)

## MVP 제외
1. LanceDB 스키마에 case_type 컬럼 추가 (재인제스트 필요)
2. 채팅 진입 경로에 필터 적용
3. 법원명(court_name) 필터 (추후 확장 가능)
4. 전문 검색(FTS/BM25) 연동
5. 정렬 옵션 (현재는 선고일 내림차순 고정)

## 진입 모드 분기 기준

| 진입 경로 | 판단 기준 | 동작 |
|----------|----------|------|
| 채팅 진입 | `sessionData.aiReferences` 존재 | 기존 LawyerView / UserView (변경 없음) |
| 사이드바 직접 진입 | `aiReferences` 없음 + `agent=case_search` + `id` 미지정 | **FilterablePrecedentView** (신규) |
| URL로 특정 판례 진입 | `?id=xxx` 존재 | 기존 LawyerView(initialCaseId) (변경 없음) |

## 목표 파일 구조

### 백엔드 (기존 2파일 수정)
```
backend/app/
├── services/service_function/
│   └── precedent_service.py           # [수정] search_by_filter(), get_case_types() 추가
└── modules/case_precedent/router/
    └── __init__.py                    # [수정] GET /precedents/filter, GET /precedents/case-types 추가
```

### 프론트엔드 (기존 3파일 수정 + 신규 4파일)
```
frontend/src/
├── app/case-precedent/
│   └── page.tsx                       # [수정] isFilterMode 분기 추가
└── features/case-precedent/
    ├── types/
    │   └── index.ts                   # [수정] 필터 관련 타입 추가
    ├── services/
    │   └── index.ts                   # [수정] filterPrecedents(), getCaseTypes() 추가
    ├── hooks/
    │   └── usePrecedentFilter.ts      # [신규] 필터 상태 + API 호출 훅
    └── components/
        ├── FilterPanel.tsx            # [신규] 필터 UI (select + pill + date)
        ├── FilteredResultList.tsx     # [신규] 필터 결과 목록
        └── FilterablePrecedentView.tsx # [신규] 필터 모드 최상위 뷰
```

## 변경 상세

### 1단계: 백엔드 서비스 함수 (`precedent_service.py`)

**`PrecedentService.search_by_filter()`** 추가:
- 파라미터: `keyword`(선택), `case_type`(선택), `date_from`(선택), `date_to`(선택), `offset`, `limit`
- SQLAlchemy 동적 조건 조합: `case_type ==`, `decision_date >=`, `decision_date <=`, `case_name/summary ILIKE`
- 반환: `{"total": int, "precedents": [dict]}`
- `decision_date DESC` 정렬, offset/limit 페이지네이션
- 기존 패턴 (`async_session_factory`, `select`, `SQLAlchemyError`) 재사용

**`PrecedentService.get_case_types()`** 추가:
- `SELECT DISTINCT case_type FROM precedent_documents WHERE case_type IS NOT NULL ORDER BY case_type`
- 반환: `list[str]` (예: `["민사", "형사", "행정", ...]`)

### 2단계: 백엔드 라우터 (`router/__init__.py`)

**인라인 스키마 추가** (기존 패턴 따름):
- `FilteredPrecedentItem(BaseModel)`: id, serial_number, case_name, case_number, case_type, court_name, decision_date, summary
- `FilteredPrecedentListResponse(BaseModel)`: keyword, total, offset, limit, precedents
- `CaseTypesResponse(BaseModel)`: case_types

**엔드포인트 추가** (L290 `GET /precedents` 위에, `GET /precedents/{precedent_id}` L346 위에):
- `GET /precedents/filter` — case_type + date 범위 + keyword ILIKE + offset/limit
- `GET /precedents/case-types` — 사건종류 DISTINCT 목록

**경로 순서 주의**: `/precedents/filter`와 `/precedents/case-types`를 `/precedents/{precedent_id}`보다 **위에** 정의 (FastAPI 경로 매칭 순서)

### 3단계: 프론트엔드 타입 (`types/index.ts`)

```typescript
export type DatePreset = 'all' | '3y' | '5y' | '10y' | 'custom'

export interface PrecedentFilterParams {
  keyword: string
  case_type: string        // "" = 전체
  date_preset: DatePreset
  date_from: string        // YYYY-MM-DD, custom일 때만
  date_to: string          // YYYY-MM-DD, custom일 때만
  offset: number
  limit: number
}

export interface FilteredPrecedentItem {
  id: string
  serial_number: string
  case_name: string | null
  case_number: string | null
  case_type: string | null
  court_name: string | null
  decision_date: string | null
  summary: string | null
}

export interface FilteredPrecedentListResponse {
  keyword: string
  total: number
  offset: number
  limit: number
  precedents: FilteredPrecedentItem[]
}
```

### 4단계: 프론트엔드 서비스 (`services/index.ts`)

`casePrecedentService`에 메서드 추가:
- `filterPrecedents(params)` → `GET /case-precedent/precedents/filter?...`
- `getCaseTypes()` → `GET /case-precedent/precedents/case-types`

### 5단계: 프론트엔드 훅 (`usePrecedentFilter.ts` 신규)

- 필터 상태: keyword, case_type, date_preset, date_from, date_to
- 날짜 프리셋 계산: `3y` → 현재 날짜에서 3년 전 ~ 현재
- 마운트 시 `getCaseTypes()` 자동 호출 → caseTypes 배열
- `search()`: offset=0 리셋 후 API 호출
- `loadMore()`: offset += limit 후 결과 append
- `selectItem(id)`: 선택 ID 저장
- 초기 상태: 검색 전까지 빈 결과 표시

### 6단계: 프론트엔드 컴포넌트

**`FilterPanel.tsx`** (신규):
```
┌─ 필터 (border-b px-4 py-3 space-y-3) ─────────────────┐
│ [검색어 입력]  (Enter 검색)                             │
│ 사건종류: [전체 ▼]  (select 드롭다운)                  │
│ 기간: [전체] [3년] [5년] [10년] [직접입력]  (pill 버튼) │
│ [직접입력 시만] 시작일: [____] ~ 종료일: [____]        │
│                                [검색] 버튼              │
└─────────────────────────────────────────────────────────┘
```
- 스타일: 기존 패턴 (select: lawyer-finder, pill: lawyer-stats)

**`FilteredResultList.tsx`** (신규):
- 결과 카드 목록 (기존 `CaseCard` 재사용)
- FilteredPrecedentItem → PrecedentItem 변환 (similarity=0, doc_type="precedent")
- "더 보기" 버튼, 빈 상태/에러/로딩 처리

**`FilterablePrecedentView.tsx`** (신규):
- 좌측(w-96): FilterPanel + FilteredResultList
- 우측(flex-1): CaseDetailPanel (기존 재사용)
- 아이템 선택 시 `getPrecedentDetail(serial_number)` → 상세 표시

### 7단계: page.tsx 분기

```tsx
const hasChatReferences = sessionData.aiReferences && Array.isArray(sessionData.aiReferences) && sessionData.aiReferences.length > 0
const isFilterMode = !hasChatReferences && agentType === 'case_search' && !initialCaseId

// 렌더링 분기
{initialCaseId ? (
  <LawyerView initialCaseId={initialCaseId} />
) : isFilterMode ? (
  <FilterablePrecedentView />
) : userRole === 'lawyer' ? (
  <LawyerView />
) : (
  <UserView />
)}
```

## 기존 인프라 활용

| 항목 | 상태 | 비고 |
|------|------|------|
| `case_type` 컬럼 | 존재 | `idx_precedent_docs_case_type` B-tree |
| `decision_date` 컬럼 | 존재 | `idx_precedent_docs_date` B-tree |
| `court_name` 컬럼 | 존재 | `idx_precedent_docs_court` B-tree |
| `precedent_documents` 테이블 | 92,055건 | ORM 모델 완비 |
| `fetch_precedent_details()` | 존재 | serial_number IN 쿼리 |
| `CaseCard` 컴포넌트 | 존재 | `PrecedentItem` 타입 기반 |
| `CaseDetailPanel` 컴포넌트 | 존재 | 상세 뷰 재사용 |

## 구현 순서

1. 백엔드 서비스 함수 → `precedent_service.py`
2. 백엔드 라우터 → `router/__init__.py` (스키마 + 엔드포인트)
3. 프론트 타입 → `types/index.ts`
4. 프론트 서비스 → `services/index.ts`
5. 프론트 훅 → `usePrecedentFilter.ts`
6. 프론트 컴포넌트 → FilterPanel, FilteredResultList, FilterablePrecedentView
7. page.tsx → 진입 분기 추가
8. 정적 검증 → ruff + mypy + npm run build

## 검증 계획

| 검증 | 명령어 |
|------|--------|
| 백엔드 린트 | `cd backend && uv run ruff check app/` |
| 백엔드 타입 | `cd backend && uv run mypy app/` |
| 프론트 빌드 | `cd frontend && npm run build` |
| API 테스트 (case-types) | `curl localhost:8000/api/case-precedent/precedents/case-types` |
| API 테스트 (filter) | `curl "localhost:8000/api/case-precedent/precedents/filter?case_type=민사&limit=5"` |
| API 테스트 (기간) | `curl "localhost:8000/api/case-precedent/precedents/filter?date_from=2023-01-01&date_to=2026-03-04"` |
