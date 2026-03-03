# 법령/판례 참조자료 상세 뷰 분리 + 백엔드 보강 설계서

> **Feature**: law-reference-detail-view
> **Plan**: `docs/01-plan/features/law-reference-detail-view.plan.md`
> **Date**: 2026-03-03
> **Status**: Draft

---

## 1. Executive Summary

챗봇 참조자료의 법령 상세 뷰를 문서유형(법령/판례) × 역할(일반인/변호사) = 4개 파일로 분리하고, 백엔드에서 법령 메타데이터를 보강하며, LawyerView 우측 패널에도 법령 상세를 추가한다.

**핵심 변경 3가지:**
1. 백엔드 `_populate_law_metadata()` 신설 → SSE sources에 `law_type`, `ministry`, `article_number`, `article_title` 추가
2. UserView 상세 뷰를 4개 전용 컴포넌트로 분리 (475줄 → ~200줄)
3. LawyerView의 `CaseDetailPanel`에 doc_type 분기 추가 + `SearchPanel` 미사용 props 정리

**영향 범위:**
- Backend: `services/rag/` 3파일 수정 (retrieval, pipeline, format_utils)
- Frontend: 신규 6파일 + 기존 5파일 수정

---

## 2. Backend 설계

### 2.1 `_populate_law_metadata()` — retrieval.py

`_populate_precedent_metadata()`와 동일 패턴으로 법령 메타데이터를 in-place 주입.

```python
def _populate_law_metadata(docs: list[dict[str, Any]]) -> None:
    """법령 문서의 law_type·ministry·article_title 보강 (in-place).

    벡터 검색 결과의 법령 문서에 law_documents/law_articles 테이블에서
    메타데이터를 배치 조회하여 주입.
    """
```

**조회 대상:**

| 테이블 | 조회 키 | 가져올 컬럼 |
|--------|---------|------------|
| `law_documents` | `law_id` | `law_type`, `ministry` |
| `law_articles` | `(law_id, article_number)` | `article_title` |

**로직:**
1. `docs`에서 `data_type == "법령"`인 문서 필터링
2. 고유 `doc_id` (= `law_id`) 수집 → `law_documents` 배치 조회
3. `(doc_id, article_number)` 쌍 수집 → `law_articles` 배치 조회
4. 각 doc의 `metadata`에 in-place 주입

```python
# 법령 문서 필터링
law_docs = [
    (idx, doc)
    for idx, doc in enumerate(docs)
    if doc.get("metadata", {}).get("data_type") == "법령"
]
if not law_docs:
    return

# 1) law_documents 배치 조회 (law_type, ministry)
law_ids = list({doc.get("metadata", {}).get("doc_id", "") for _, doc in law_docs})
law_meta_map: dict[str, dict[str, str]] = {}
with sync_session_factory() as session:
    rows = session.execute(
        text(
            "SELECT law_id, law_type, ministry "
            "FROM law_documents WHERE law_id = ANY(:ids)"
        ),
        {"ids": law_ids},
    ).fetchall()
    for row in rows:
        law_meta_map[str(row[0])] = {
            "law_type": row[1] or "",
            "ministry": row[2] or "",
        }

# 2) law_articles 배치 조회 (article_title)
article_queries = [
    (doc.get("metadata", {}).get("doc_id", ""), doc.get("metadata", {}).get("article_number", ""))
    for _, doc in law_docs
    if doc.get("metadata", {}).get("article_number")
]
article_title_map: dict[tuple[str, str], str] = {}
if article_queries:
    conditions = [
        and_(LawArticle.law_id == lid, LawArticle.article_number == anum)
        for lid, anum in article_queries
    ]
    with sync_session_factory() as session:
        rows = session.execute(
            select(LawArticle.law_id, LawArticle.article_number, LawArticle.article_title)
            .where(or_(*conditions))
        ).fetchall()
        for row in rows:
            article_title_map[(str(row[0]), str(row[1]))] = row[2] or ""

# 3) in-place 주입
for _, doc in law_docs:
    meta = doc.get("metadata", {})
    lid = meta.get("doc_id", "")
    if lid in law_meta_map:
        extra = law_meta_map[lid]
        if not meta.get("law_type"):
            meta["law_type"] = extra["law_type"]
        if not meta.get("ministry"):
            meta["ministry"] = extra["ministry"]
    anum = meta.get("article_number", "")
    if anum and (lid, anum) in article_title_map:
        meta["article_title"] = article_title_map[(lid, anum)]
```

### 2.2 pipeline.py 호출 지점

동기/비동기 양쪽에서 `_apply_law_article_content` 직후에 삽입:

```python
# pipeline.py 동기 (execute, ~283줄)
_populate_content(reranked, contents)
_populate_precedent_metadata(reranked)
_apply_law_article_content(reranked)
_populate_law_metadata(reranked)        # ← 추가

# pipeline.py 비동기 (_rerank_and_fetch, ~549줄)
_populate_content(reranked, contents)
_populate_precedent_metadata(reranked)
_apply_law_article_content(reranked)
_populate_law_metadata(reranked)        # ← 추가
```

import 추가:
```python
from app.services.rag.retrieval import (
    ...,
    _populate_law_metadata,  # 추가
)
```

### 2.3 format_law_sources() 필드 확장 — format_utils.py

```python
# 기존 (5개 필드)
def format_law_sources(documents):
    sources.append({
        "doc_id": ..., "doc_type": "law", "law_name": ...,
        "similarity": ..., "content": ...
    })

# 변경 후 (9개 필드)
def format_law_sources(documents):
    sources.append({
        "doc_id": metadata.get("doc_id", ""),
        "doc_type": "law",
        "law_name": metadata.get("case_name", "") or metadata.get("title", ""),
        "law_type": metadata.get("law_type", ""),           # 추가
        "ministry": metadata.get("ministry", ""),           # 추가
        "article_number": metadata.get("article_number", ""),  # 추가
        "article_title": metadata.get("article_title", ""),    # 추가
        "similarity": round(doc.get("similarity", 0), 3),
        "content": doc.get("content", ""),
    })
```

---

## 3. Frontend 설계

### 3.1 타입 변경 — `types/index.ts`

```typescript
export interface ChatSource {
  // ... 기존 필드 유지 ...
  law_name?: string
  law_type?: string          // 이미 존재
  // 추가 필드
  article_number?: string    // 조문번호
  article_title?: string     // 조문제목
  ministry?: string          // 소관부처
}
```

### 3.2 공유 유틸 — `utils/docTypeUtils.ts` (신규)

`UserView.tsx`에서 추출:

```typescript
export const docTypeLabels: Record<string, string> = {
  law: '법령', precedent: '판례', constitutional: '헌재결정',
  committee: '위원회 결정', administration: '행정심판',
  legislation: '입법예고',
}

export function getDocTypeLabel(docType: string): string {
  return docTypeLabels[docType] || docType
}

export function getDocTypeBadgeColor(docType: string): string {
  switch (docType) {
    case 'law': return 'bg-green-50 text-green-600'
    case 'precedent': return 'bg-blue-50 text-blue-600'
    case 'constitutional': return 'bg-purple-50 text-purple-600'
    case 'committee': return 'bg-orange-50 text-orange-600'
    case 'administration': return 'bg-yellow-50 text-yellow-700'
    case 'legislation': return 'bg-teal-50 text-teal-600'
    default: return 'bg-gray-50 text-gray-600'
  }
}
```

### 3.3 ReferenceDetailHeader (신규)

`UserView.tsx:172-233` 헤더 영역 추출.

```typescript
interface ReferenceDetailHeaderProps {
  selectedRef: ChatSource
  isLawyer: boolean
}

export function ReferenceDetailHeader({ selectedRef, isLawyer }: ReferenceDetailHeaderProps)
```

**렌더링 내용:**
- 발행기관/법원 로고 (기존 lawTypeLogo 유틸 사용)
- doc_type 배지 + law_type 배지 (법령일 때만)
- 제목: 법령명 또는 사건명
- 부제목: `제{article_number}조 ({article_title})` 또는 사건번호
- 역할 배지 (변호사/일반인)

### 3.4 PrecedentDetailUser (신규)

`UserView.tsx:240-317` 일반인 판례 상세 추출.

```typescript
interface PrecedentDetailUserProps {
  source: ChatSource
}
```

**렌더링 내용:** 판결요지(blue box) → 주문(green box) → 참조조문 아코디언 → 판결문 전체보기 아코디언 → 그래프 보강정보(인용 법령, 유사 판례)

`isProvisionsOpen` 상태를 이 컴포넌트 내부로 이동.

### 3.5 PrecedentDetailLawyer (신규)

```typescript
interface PrecedentDetailLawyerProps {
  source: ChatSource
}
```

**렌더링 내용:** `PrecedentFullTextViewer mode="direct"` (ChatSource → PrecedentDetail 어댑터) + 그래프 보강정보

### 3.6 LawDetailUser (신규)

```typescript
interface LawDetailUserProps {
  source: ChatSource
}
```

**렌더링 구조:**
```
┌─────────────────────────────────────┐
│ 법령 정보 배너 (초록색 bg)          │
│ 📋 법령종류: 법률  |  소관부처: 법무부 │
├─────────────────────────────────────┤
│ 조문 헤더 (파란색 bg)               │
│ 제750조 (불법행위의 내용)           │
├─────────────────────────────────────┤
│ 조문 본문 (ReactMarkdown)           │
│ 고의 또는 과실로 인한 위법행위로    │
│ 타인에게 손해를 가한 자는 ...       │
├─────────────────────────────────────┤
│ 💡 법령 활용 안내 (회색 border)     │
│ 이 법령의 적용 여부는 구체적인      │
│ 사실관계에 따라 달라질 수 있습니다.  │
│ 전문가 상담을 권장합니다.            │
└─────────────────────────────────────┘
```

**article_number 없는 경우 (keyword-only 법령):**
```
┌─────────────────────────────────────┐
│ ⚠️ 조문 정보 없음 (노란색 bg)       │
│ 해당 법령의 조문 내용을 불러올 수    │
│ 없습니다. 챗봇에게 구체적인 조문     │
│ 번호를 포함하여 질문해보세요.        │
└─────────────────────────────────────┘
```

### 3.7 LawDetailLawyer (신규)

```typescript
interface LawDetailLawyerProps {
  source: ChatSource | PrecedentDetail
}
```

**UserView + CaseDetailPanel 양쪽에서 사용**하므로 Props를 유니온 타입으로 설계.

**렌더링 구조:**
```
┌─────────────────────────────────────┐
│ 법령 메타정보 테이블 (dl/dt/dd)     │
│ 법령종류   법률                     │
│ 소관부처   법무부                   │
│ 조문번호   제750조                  │
│ 조문제목   불법행위의 내용          │
├─────────────────────────────────────┤
│ ■ 조문 원문 (헤더바 + 흰 bg)       │
│ article_content (ReactMarkdown)     │
├─────────────────────────────────────┤
│ 관련 법령 (그래프 보강정보)          │
│ cited_statutes[]                    │
└─────────────────────────────────────┘
```

### 3.8 UserView.tsx 리팩토링

**변경 전 (475줄):** 목록 뷰 + 상세 헤더 + 판례 상세(일반인/변호사) + 법령 상세 + 역할 분기 혼재

**변경 후 (~200줄):** 목록 뷰 + 컴포넌트 라우팅만 담당

```tsx
// Detail View 분기 (기존 144-355줄 대체)
if (selectedRef) {
  const isLaw = selectedRef.doc_type === 'law'
  return (
    <div className="h-full flex flex-col bg-white">
      {/* 뒤로가기 + 역할 배지 (헤더 바) */}
      <div className="p-4 border-b ...">
        <button onClick={() => setSelectedRef(null)}>←</button>
        <span>{isLawyer ? '변호사 모드' : '일반인 모드'}</span>
      </div>
      {/* 상세 컨텐츠 */}
      <div className="flex-1 overflow-y-auto p-6">
        <ReferenceDetailHeader selectedRef={selectedRef} isLawyer={isLawyer} />
        {isLaw ? (
          isLawyer ? <LawDetailLawyer source={selectedRef} />
                   : <LawDetailUser source={selectedRef} />
        ) : (
          isLawyer ? <PrecedentDetailLawyer source={selectedRef} />
                   : <PrecedentDetailUser source={selectedRef} />
        )}
      </div>
    </div>
  )
}
```

삭제 대상:
- `docTypeLabels`, `getDocTypeBadgeColor` (→ `docTypeUtils.ts`로 이동)
- 상세 헤더 렌더링 (→ `ReferenceDetailHeader`)
- 판례 상세 렌더링 (→ `PrecedentDetailUser/Lawyer`)
- 법령 상세 렌더링 (→ `LawDetailUser/Lawyer`)
- `isProvisionsOpen` state (→ `PrecedentDetailUser` 내부로 이동)

### 3.9 CaseDetailPanel.tsx 수정

**변경 전:** 무조건 `PrecedentFullTextViewer`
**변경 후:** doc_type 분기

```tsx
export function CaseDetailPanel({ case_, isLoading, error }: CaseDetailPanelProps) {
  // ... 로딩/에러/빈 상태 처리 (기존 유지, 메시지만 "문서를 선택하세요"로 변경)

  const isLaw = case_.doc_type === 'law'

  return (
    <div className="flex-1 flex flex-col bg-white overflow-hidden">
      {/* 헤더바 */}
      <div className="px-4 py-2 border-b border-gray-200 bg-gray-50 text-sm">
        {isLaw ? (
          // 법령 헤더: 법령명 + 조문번호
          <span className="font-medium">
            {case_.law_name}
            {case_.article_number && ` 제${case_.article_number}조`}
          </span>
        ) : (
          // 판례 헤더: 법원 날짜 사건번호 (기존 유지)
          <span className="font-medium">
            {court} {date} 선고 {caseNumber} 판결
          </span>
        )}
      </div>

      {/* 본문 */}
      <div className="flex-1 overflow-y-auto">
        {isLaw ? (
          <LawDetailLawyer source={case_} />
        ) : (
          <PrecedentFullTextViewer data={case_} mode="direct" />
        )}
      </div>
    </div>
  )
}
```

### 3.10 SearchPanel.tsx 정리

**제거할 props:**
```diff
 interface SearchPanelProps {
   results: PrecedentItem[]
   totalResults: number
   isSearching: boolean
   error: string | null
-  filters: SearchFilters
   selectedCaseId: string | null
   selectedCase?: PrecedentDetail | null
-  onFilterChange: (filters: Partial<SearchFilters>) => void
-  onSearch: () => void
   onCaseSelect: (id: string) => void
 }
```

**결과 유무 판단 변경:**
```diff
- ) : filters.keyword ? (
+ ) : results.length === 0 ? (
    <span>검색 결과가 없습니다</span>
- ) : (
-   <span>검색어를 입력하세요</span>
  )
```

빈 결과 표시 메시지: "챗봇에게 질문하면 관련 문서가 표시됩니다"

### 3.11 useCaseSearch.ts 정리

**제거:**
- `filters` state, `setFiltersState`, `DEFAULT_FILTERS`
- `search()` 함수, `searchError` state
- `SearchFilters` type import

**유지:**
- `searchResults` (aiReferences에서 주입)
- `selectCase`, `selectedCase`
- `aiReferences` useEffect
- `askAI`, `aiResponse`

### 3.12 LawyerView.tsx 수정

```diff
 export function LawyerView({ initialCaseId }: LawyerViewProps) {
   const {
     searchResults,
     totalResults,
     isSearching,
-    searchError,
     selectedCase,
     isLoadingDetail,
     detailError,
-    filters,
-    setFilters,
-    search,
     selectCase,
   } = useCaseSearch(initialCaseId)

   return (
     <div className="h-full flex overflow-hidden">
       <SearchPanel
         results={searchResults}
         totalResults={totalResults}
         isSearching={isSearching}
-        error={searchError}
-        filters={filters}
+        error={null}
         selectedCaseId={selectedCase?.id || null}
         selectedCase={selectedCase}
-        onFilterChange={setFilters}
-        onSearch={search}
         onCaseSelect={selectCase}
       />
       <CaseDetailPanel ... />
     </div>
   )
 }
```

---

## 4. 구현 순서

| Phase | 작업 | 파일 |
|-------|------|------|
| 1 | 백엔드 `_populate_law_metadata()` + pipeline 호출 + `format_law_sources()` 확장 | retrieval.py, pipeline.py, format_utils.py |
| 2 | 프론트 타입 + 유틸 준비 | types/index.ts, utils/docTypeUtils.ts |
| 3 | 공통 헤더 컴포넌트 | ReferenceDetailHeader.tsx |
| 4 | 판례 상세 분리 | PrecedentDetailUser.tsx, PrecedentDetailLawyer.tsx |
| 5 | 법령 상세 신규 | LawDetailUser.tsx, LawDetailLawyer.tsx |
| 6 | UserView 리팩토링 | UserView.tsx |
| 7 | LawyerView 우측 패널 + SearchPanel 정리 | CaseDetailPanel.tsx, SearchPanel.tsx, useCaseSearch.ts, LawyerView.tsx |
| 8 | 정적 검증 | ruff check + npm run build |

---

## 5. 테스트 케이스

| # | 시나리오 | 기대 결과 |
|---|---------|----------|
| 1 | 법령 검색 → SSE sources 확인 | `law_type`, `article_number`, `article_title`, `ministry` 필드 포함 |
| 2 | 일반인 + 법령 카드 클릭 | 조문 헤더 + 본문 + 법령 정보 배너 + 활용 안내 |
| 3 | 변호사 + 법령 카드 클릭 (UserView) | 메타정보 테이블 + 조문 원문 + 관련 법령 |
| 4 | 변호사 + 법령 카드 클릭 (LawyerView) | CaseDetailPanel에서 LawDetailLawyer 렌더링 |
| 5 | 변호사 + 판례 카드 클릭 (LawyerView) | 기존과 동일 (PrecedentFullTextViewer) |
| 6 | 일반인 + 판례 카드 클릭 | 기존과 동일 (판결요지 + 주문 + 아코디언) |
| 7 | keyword-only 법령 (article_number 없음) | "조문 내용을 불러올 수 없습니다" 표시 |
| 8 | SearchPanel 빌드 | onSearch/filters props 없이 빌드 통과 |
| 9 | ruff check + npm run build | 에러 없음 |
