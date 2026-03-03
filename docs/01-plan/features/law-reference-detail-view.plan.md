# 법령/판례 참조자료 상세 뷰 분리 + 백엔드 보강 계획

## 요약
1. 참조자료 상세 뷰를 **문서유형(법령/판례) × 역할(일반인/변호사) = 4개 파일**로 분리한다.
2. 공통 헤더(로고, 배지, 제목)를 `ReferenceDetailHeader`로 추출한다.
3. 백엔드 `format_law_sources()`에 `law_type`, `article_number`, `article_title`, `ministry` 필드를 추가한다.
4. 법령 메타데이터 DB 보강 함수 `_populate_law_metadata()`를 신설한다.
5. `UserView.tsx`를 목록 뷰 + 라우팅만 담당하도록 475줄 → ~200줄로 축소한다.
6. **`LawyerView` 우측 패널에 법령 상세 뷰를 추가**한다. (`CaseDetailPanel` → doc_type 분기)
7. **`SearchPanel`에서 미사용 props(`onSearch`, `onFilterChange`, `filters`) 제거**한다.

## 현재 문제
1. **법령 상세 뷰가 사실상 없음**: `UserView.tsx:342-351`에서 `content` 하나만 ReactMarkdown으로 표시
2. **법령 메타데이터 누락**: `format_law_sources()`가 5개 필드만 전송 (`law_type`, `article_number` 등 미포함)
3. **단일 파일 비대화**: `UserView.tsx` 475줄에 목록+상세+판례+법령+역할분기 혼재
4. **LawyerView 우측 패널이 판례 전용**: `CaseDetailPanel` → `PrecedentFullTextViewer`만 렌더링. 챗봇이 법령을 `aiReferences`로 보내도 법령 상세를 표시할 수 없음
5. **SearchPanel 미사용 props**: `onSearch`, `onFilterChange`, `filters`를 props로 받지만 검색 입력 UI가 없어 실제 사용되지 않음 (검색 결과는 챗봇 `aiReferences`에서 주입)

## MVP 범위
1. 백엔드 법령 메타데이터 보강 (`_populate_law_metadata()`, `format_law_sources()` 확장)
2. 프론트엔드 UserView 상세 뷰 4파일 분리 (LawDetailUser, LawDetailLawyer, PrecedentDetailUser, PrecedentDetailLawyer)
3. 공통 헤더 컴포넌트 추출 (`ReferenceDetailHeader`)
4. 공유 유틸 분리 (`docTypeUtils.ts`)
5. `UserView.tsx` 리팩토링 (목록 뷰 + 상세 라우팅만 담당)
6. `CaseDetailPanel` doc_type 분기 (판례: 기존 `PrecedentFullTextViewer`, 법령: `LawDetailLawyer` 재사용)
7. `SearchPanel` 미사용 props 정리
8. `LawyerView` → `useCaseSearch` 훅에서 불필요한 search/filters 제거

## MVP 제외
1. 법령 체계도(`StatuteHierarchyView`)와의 연동 강화
2. 법령 개정 이력 표시
3. 관련 판례 자동 추천
4. 법령 비교 뷰

## 목표 파일 구조

### 프론트엔드 (신규 6파일 + 기존 5파일 수정)
```
frontend/src/features/case-precedent/
├── components/
│   ├── UserView.tsx                   # [수정] 목록 뷰 + 상세 라우팅 (composition root)
│   ├── ReferenceDetailHeader.tsx      # [신규] 공통 헤더 (로고, 배지, 제목, 부제목)
│   ├── LawDetailUser.tsx              # [신규] 일반인 법령 상세
│   ├── LawDetailLawyer.tsx            # [신규] 변호사 법령 상세 (UserView + CaseDetailPanel 양쪽에서 사용)
│   ├── PrecedentDetailUser.tsx        # [신규] 일반인 판례 상세 (기존 UserView에서 추출)
│   ├── PrecedentDetailLawyer.tsx      # [신규] 변호사 판례 상세 (PrecedentFullTextViewer 래퍼)
│   ├── LawyerView.tsx                 # [수정] useCaseSearch에서 search/filters 제거
│   ├── SearchPanel.tsx                # [수정] onSearch, onFilterChange, filters props 제거
│   ├── CaseDetailPanel.tsx            # [수정] doc_type 분기 (판례/법령)
│   └── CaseCard.tsx                   # (변경 없음)
├── hooks/
│   └── useCaseSearch.ts               # [수정] search/filters 관련 코드 제거
├── utils/
│   └── docTypeUtils.ts                # [신규] docTypeLabels, getDocTypeBadgeColor 공유 유틸
└── types/
    └── index.ts                       # [수정] ChatSource에 3개 필드 추가
```

### 백엔드 (3파일 수정)
```
backend/app/services/rag/
├── retrieval.py     # [수정] _populate_law_metadata() 함수 추가
├── pipeline.py      # [수정] _populate_law_metadata() 호출 추가
└── format_utils.py  # [수정] format_law_sources() 필드 추가
```

## 공개 API/타입 변경

### 백엔드 SSE sources 이벤트 필드 추가 (법령)
```json
// 기존
{ "doc_id": "...", "doc_type": "law", "law_name": "...", "similarity": 0.89, "content": "..." }

// 보강 후
{ "doc_id": "...", "doc_type": "law", "law_name": "...", "law_type": "법률",
  "ministry": "법무부", "article_number": "750", "article_title": "불법행위의 내용",
  "similarity": 0.89, "content": "..." }
```

### 프론트엔드 ChatSource 타입 추가
```typescript
article_number?: string   // 조문번호
article_title?: string    // 조문제목
ministry?: string         // 소관부처
```

## 구현 순서

### Phase 1: 백엔드 법령 메타데이터 보강

**Step 1-1.** `retrieval.py`에 `_populate_law_metadata()` 추가
- `_populate_precedent_metadata()` (517줄) 직후에 신설
- `law_documents` 테이블에서 `law_type`, `ministry` 배치 조회
- `law_articles` 테이블에서 `article_title` 배치 조회
- 패턴: `_populate_precedent_metadata()`와 동일 (in-place metadata 주입)

**Step 1-2.** `pipeline.py`에서 호출 추가
- `_apply_law_article_content(reranked)` 직후에 `_populate_law_metadata(reranked)` 삽입
- 동기(`execute` 283줄 이후) + 비동기(`_rerank_and_fetch` 549줄 이후) 양쪽
- import 블록에 `_populate_law_metadata` 추가

**Step 1-3.** `format_utils.py`의 `format_law_sources()` 필드 추가
- 기존 5개 → 9개 필드: `law_type`, `ministry`, `article_number`, `article_title` 추가
- `metadata`에서 해당 필드 추출하여 dict에 포함

### Phase 2: 프론트엔드 타입 + 유틸 준비

**Step 2-1.** `types/index.ts`의 `ChatSource`에 필드 추가
- `article_number?: string`, `article_title?: string`, `ministry?: string`

**Step 2-2.** `utils/docTypeUtils.ts` 신규 생성
- `UserView.tsx`에서 `docTypeLabels`, `getDocTypeLabel`, `getDocTypeBadgeColor` 이동
- UserView(목록 뷰)와 ReferenceDetailHeader(상세 헤더) 양쪽에서 import

### Phase 3: 공통 헤더 컴포넌트

**Step 3-1.** `ReferenceDetailHeader.tsx` 신규 생성
- `UserView.tsx:172-233` 헤더 섹션 추출
- Props: `selectedRef: ChatSource`, `isLawyer: boolean`
- 법령 subtitle: `제N조 (조문제목)` 또는 `법령종류`
- 판례 subtitle: `사건번호`
- 법령일 때 `law_type` 배지 추가 표시

### Phase 4: 판례 상세 컴포넌트 분리

**Step 4-1.** `PrecedentDetailUser.tsx` 신규 생성
- `UserView.tsx:240-317` 일반인 판례 상세 추출
- 판결요지(blue) + 주문(green) + 참조조문 아코디언 + 판결문 전체보기 아코디언 + 그래프 보강정보
- `isProvisionsOpen` 상태를 이 컴포넌트 내부로 이동

**Step 4-2.** `PrecedentDetailLawyer.tsx` 신규 생성
- `PrecedentFullTextViewer mode="direct"` 래퍼
- 그래프 보강 정보(인용 법령, 유사 판례) 포함

### Phase 5: 법령 상세 컴포넌트 신규 설계

**Step 5-1.** `LawDetailUser.tsx` 신규 생성 (일반인)
- 법령 정보 배너 (소관부처, 법령종류) — 초록색
- 조문 헤더 (제N조 + 조문제목) — 파란색
- 조문 본문 (ReactMarkdown)
- 법령 활용 안내 (변호사 상담 권장 안내)

**Step 5-2.** `LawDetailLawyer.tsx` 신규 생성 (변호사)
- 법령 메타정보 테이블 (dl/dt/dd 구조)
- 조문 원문 (법원 스타일: 헤더바 + 흰 배경)
- 관련 법령 (그래프 보강 정보)
- **Props는 ChatSource | PrecedentDetail 양쪽 수용** (UserView + CaseDetailPanel에서 공용)

### Phase 6: UserView.tsx 리팩토링

**Step 6-1.** Detail View 섹션(144-355줄) 제거 → 컴포넌트 라우팅으로 교체
```tsx
{selectedRef.doc_type === 'law' ? (
  isLawyer ? <LawDetailLawyer /> : <LawDetailUser />
) : (
  isLawyer ? <PrecedentDetailLawyer /> : <PrecedentDetailUser />
)}
```
- `docTypeLabels`, `getDocTypeBadgeColor`를 `docTypeUtils`에서 import
- `CollapsibleSection` 컴포넌트는 사용처 확인 후 불필요하면 제거

### Phase 7: LawyerView 우측 패널 법령 대응

**Step 7-1.** `CaseDetailPanel.tsx` 수정 — doc_type 분기 추가
- 기존: 무조건 `PrecedentFullTextViewer mode="direct"` 렌더링
- 변경: `case_.doc_type === 'law'` → `LawDetailLawyer` 렌더링, 나머지 → 기존 유지
```tsx
// 기존
<PrecedentFullTextViewer data={case_} mode="direct" />

// 변경
{case_.doc_type === 'law' ? (
  <LawDetailLawyer source={case_} />
) : (
  <PrecedentFullTextViewer data={case_} mode="direct" />
)}
```
- 헤더바도 법령/판례에 따라 분기 (법령: `법령명 제N조`, 판례: `법원 날짜 사건번호`)
- 빈 선택 상태 메시지를 "판례를 선택하세요" → "문서를 선택하세요"로 변경

**Step 7-2.** `SearchPanel.tsx` 수정 — 미사용 props 제거
- 제거할 props: `onSearch`, `onFilterChange`, `filters`
- `filters.keyword` 사용처 → `results.length`로 대체 (결과 유무만 판단)
- 참조조문(`provisions`)은 `selectedCase` props에서만 추출하므로 유지

**Step 7-3.** `useCaseSearch.ts` 수정 — search/filters 관련 코드 제거
- 제거: `filters` state, `setFilters`, `search` 함수, `SearchFilters` import
- 제거: `searchError` state (검색 입력이 없으므로 불필요)
- 유지: `searchResults` (챗봇 `aiReferences`에서 주입), `selectCase`, `selectedCase`
- 유지: `aiReferences` useEffect (챗봇 결과 자동 주입 로직)

**Step 7-4.** `LawyerView.tsx` 수정 — 제거된 props 반영
- `useCaseSearch`에서 `filters`, `setFilters`, `search`, `searchError` 제거
- `SearchPanel`에 전달하는 해당 props 제거

### Phase 8: 정적 검증

**Step 8-1.** 백엔드: `cd backend && uv run ruff check app/services/rag/`
**Step 8-2.** 프론트: `cd frontend && npm run build`

## 데이터 흐름

### 일반인 (UserView)
```
[벡터 검색] → reranked docs
    ↓
[_apply_law_article_content] → 법령 조문 content 주입
    ↓
[_populate_law_metadata] ★신규 → law_type, ministry, article_title 주입
    ↓
[_populate_precedent_metadata] → 판례 메타데이터 주입
    ↓
[format_law_sources / format_precedent_sources] → SSE sources 이벤트 구성
    ↓
[SSE 전송] → 프론트엔드 ChatSource[]
    ↓
[UserView] → 문서유형 + 역할 분기 → 4개 상세 컴포넌트
```

### 변호사 (LawyerView)
```
[챗봇 대화] → SSE sources 이벤트 → sessionData.aiReferences
    ↓
[useCaseSearch useEffect] → PrecedentItem[] 변환 → searchResults
    ↓
┌──────────────────────────┬────────────────────────────────┐
│ SearchPanel (좌측)       │ CaseDetailPanel (우측)         │
│  CaseCard 목록 표시      │  doc_type 분기:                │
│  (검색 입력 UI 없음)     │  ├ law → LawDetailLawyer      │
│                          │  └ 그 외 → PrecedentFullText   │
└──────────────────────────┴────────────────────────────────┘
```

## 테스트 케이스
1. 법령 검색 후 참조자료에 `law_type`, `article_number` 표시 확인
2. 일반인 모드: 법령 카드 클릭 → 조문 헤더 + 본문 + 법령 정보 배너 표시
3. 변호사 모드 (UserView): 법령 카드 클릭 → 메타정보 테이블 + 조문 원문 표시
4. **변호사 모드 (LawyerView): 챗봇이 법령을 aiReferences로 전달 → 좌측 목록에 법령 카드 표시 → 클릭 시 우측에 법령 상세 표시**
5. **변호사 모드 (LawyerView): 챗봇이 판례를 aiReferences로 전달 → 기존과 동일 동작**
6. 일반인 모드: 판례 카드 클릭 → 기존과 동일 (판결요지 + 주문 + 아코디언)
7. 변호사 모드: 판례 카드 클릭 → 기존과 동일 (PrecedentFullTextViewer direct)
8. keyword-only 법령 (article_number 없음): "조문 내용을 불러올 수 없습니다" 표시
9. **SearchPanel에 onSearch/filters props가 없어도 빌드 에러 없음**
10. `npm run build` 성공, `ruff check` 통과

## 가정/기본값
1. `law_documents` 테이블에 `law_type`, `ministry` 컬럼이 이미 존재한다.
2. `law_articles` 테이블에 `article_title` 컬럼이 이미 존재한다.
3. DB 스키마 변경(마이그레이션)은 불필요하다.
4. `PrecedentFullTextViewer` 컴포넌트는 기존 그대로 재사용한다.
5. SearchPanel의 검색 입력 UI는 설계상 의도적으로 없음 (결과는 챗봇에서 주입).
6. `CaseCard.tsx`는 이미 `doc_type === 'law'`를 처리하고 있으므로 변경 불필요.
