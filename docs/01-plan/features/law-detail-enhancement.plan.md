# 법령 상세 뷰 개선 — 메타 블록 · 법령 계층 · 조문 트리 · 판례 연결

> PDCA Plan | Feature: law-detail-enhancement | Created: 2026-03-03

## Context

법령 상세 뷰(`LawDetailUser`, `LawDetailLawyer`)는 현재 법령종류/소관부처 + 조문 본문 + 전문 아코디언 + LLM 요약만 표시한다.
시행일·공포번호·법령 계층 등 핵심 메타데이터가 보이지 않고, 100조 이상 법령의 장/절 네비게이션이 없으며, 해당 법령을 인용한 판례로의 연결이 없다.

### 현재 상태 요약

| 영역 | 현재 | 목표 |
|------|------|------|
| 메타데이터 | 법령종류, 소관부처만 표시 | + 시행일, 공포일자, 공포번호 |
| 법령 계층 | 없음 (체계도 페이지에서만 확인) | 상세 뷰 내에서 법 > 시행령 > 시행규칙 표시 |
| 조문 탐색 | 단순 목록 (평면) | 장/절/조 트리 구조 + 접기/펼치기 |
| 판례 연결 | 없음 | "이 법령을 인용한 판례" 버튼 |

### 제외 항목

- **현행 여부**: 데이터가 현행 법령만 수록 → 표시 불필요
- **연혁 비교**: 법령당 최신 버전 1건만 존재 → 과거 개정본 없음
- **제정일**: 별도 필드 없음 (간접 추론만 가능)

---

## 데이터 가용성 확인

| 필요 데이터 | DB 필드 | 상태 |
|------------|---------|------|
| 시행일 | `law_documents.enforcement_date` (Date) | **있음** |
| 공포일자 | `law_documents.promulgation_date` (String, YYYYMMDD) | **있음** |
| 공포번호 | `law_documents.promulgation_no` (String) | **있음** |
| 법령 계층 | `statute_hierarchy` (child_id → parent_id FK) | **있음** + API 구현 완료 |
| 장/절 구조 | JSON `조문번호=null` 항목에 "제N장"/"제N절" 텍스트 | **있음** (비구조적) |
| 판례→법령 인용 | `case_statute_citations` (law_doc_id 인덱스) | **있음** (역방향 API만 미구현) |

---

## MVP 범위

### Phase 1 (우선): 메타 블록 + 법령 계층

1. `LawFullTextResponse` 스키마에 날짜/공포번호 필드 추가
2. 컴포넌트 마운트 시 `getLawFullText` 자동 호출 (기존: 아코디언 클릭 시만)
3. 메타 블록 UI 추가 (시행일, 공포일자, 공포번호)
4. 법령 계층 표시 (기존 `getStatuteHierarchy` API 재사용)

### Phase 2: 조문 트리 + 판례 연결

5. 조문 배열을 장/절/조 트리로 파싱하는 프론트엔드 유틸
6. 법령 전문 아코디언 내 트리 UI (접기/펼치기)
7. `PgGraphService.get_cases_citing_statute()` 신규 메서드
8. `GET /statutes/{statute_id}/citing-cases` 엔드포인트
9. 프론트엔드 판례 연결 버튼 UI

---

## Phase 1: 메타 블록 + 법령 계층

### 1-1. 백엔드: LawFullTextResponse 스키마 확장

**파일**: `backend/app/modules/case_precedent/router/__init__.py`

`LawFullTextResponse`에 3개 필드 추가:

```python
class LawFullTextResponse(BaseModel):
    """법령 전문 응답 (조문 단위)"""
    law_id: str
    law_name: str
    law_type: Optional[str] = None
    ministry: Optional[str] = None
    ai_summary: Optional[str] = None
    supplementary: Optional[str] = None
    articles: List[LawArticleItem]
    total_articles: int
    # Phase 1 추가
    enforcement_date: Optional[str] = None   # ISO "YYYY-MM-DD"
    promulgation_date: Optional[str] = None  # "YYYYMMDD" 원본
    promulgation_no: Optional[str] = None
```

`get_law_full_text` 핸들러 반환값 수정:

```python
return LawFullTextResponse(
    ...기존...,
    enforcement_date=law.enforcement_date.isoformat() if law.enforcement_date else None,
    promulgation_date=law.promulgation_date,
    promulgation_no=law.promulgation_no,
)
```

### 1-2. 프론트엔드: LawFullText 타입 확장

**파일**: `frontend/src/features/case-precedent/types/index.ts`

```typescript
export interface LawFullText {
  law_id: string
  law_name: string
  law_type?: string
  ministry?: string
  ai_summary?: string
  supplementary?: string
  articles: LawArticleItem[]
  total_articles: number
  // Phase 1 추가
  enforcement_date?: string    // "YYYY-MM-DD"
  promulgation_date?: string   // "YYYYMMDD"
  promulgation_no?: string
}
```

### 1-3. 프론트엔드: 날짜 포맷 유틸

**파일**: `frontend/src/features/case-precedent/utils/dateUtils.ts` (신규)

```typescript
/** "YYYYMMDD" → "YYYY. MM. DD." */
export function formatPromulgationDate(yyyymmdd: string): string {
  if (yyyymmdd.length !== 8) return yyyymmdd
  return `${yyyymmdd.slice(0, 4)}. ${yyyymmdd.slice(4, 6)}. ${yyyymmdd.slice(6, 8)}.`
}

/** "YYYY-MM-DD" → "YYYY. MM. DD." */
export function formatIsoDate(isoDate: string): string {
  const [year, month, day] = isoDate.split('-')
  return `${year}. ${month}. ${day}.`
}
```

### 1-4. 프론트엔드: 전문 자동 로딩 + 메타 블록 UI

**파일**: `LawDetailUser.tsx`, `LawDetailLawyer.tsx`

**전문 자동 로딩**: 기존 `handleToggleFullText`는 아코디언 버튼 클릭 시만 호출됨.
`useEffect`를 추가하여 `source.doc_id`가 있으면 컴포넌트 마운트 시 즉시 `getLawFullText` 호출.
`fullText` state에 캐시되므로 아코디언 클릭 시 추가 API 호출 없이 즉시 표시.

```tsx
// 컴포넌트 마운트 시 전문 자동 로딩
useEffect(() => {
  if (!source.doc_id || fullText) return
  const load = async () => {
    setIsLoading(true)
    try {
      const data = await casePrecedentService.getLawFullText(source.doc_id!)
      setFullText(data)
    } catch {
      // 아코디언에서 재시도 가능
    } finally {
      setIsLoading(false)
    }
  }
  load()
}, [source.doc_id]) // eslint-disable-line react-hooks/exhaustive-deps
```

**메타 블록 UI** (기존 법령 정보 배너 아래, 조문 본문 위):

LawDetailUser — 기존 녹색 배너 아래에 추가:
```tsx
{fullText && (fullText.enforcement_date || fullText.promulgation_date || fullText.promulgation_no) && (
  <div className="grid grid-cols-3 gap-3 text-sm">
    {fullText.enforcement_date && (
      <div className="bg-gray-50 rounded-lg p-3 border border-gray-100">
        <dt className="text-gray-500 text-xs mb-1">시행일</dt>
        <dd className="text-gray-800 font-medium">{formatIsoDate(fullText.enforcement_date)}</dd>
      </div>
    )}
    {fullText.promulgation_date && (
      <div className="bg-gray-50 rounded-lg p-3 border border-gray-100">
        <dt className="text-gray-500 text-xs mb-1">공포일자</dt>
        <dd className="text-gray-800 font-medium">{formatPromulgationDate(fullText.promulgation_date)}</dd>
      </div>
    )}
    {fullText.promulgation_no && (
      <div className="bg-gray-50 rounded-lg p-3 border border-gray-100">
        <dt className="text-gray-500 text-xs mb-1">공포번호</dt>
        <dd className="text-gray-800 font-medium">제{fullText.promulgation_no}호</dd>
      </div>
    )}
  </div>
)}
```

LawDetailLawyer — 기존 `<dl>` 메타 테이블에 동일 필드 추가.

### 1-5. 프론트엔드: 법령 계층 표시

**파일**: `LawDetailUser.tsx`, `LawDetailLawyer.tsx`

기존 서비스 함수 `casePrecedentService.getStatuteHierarchy` 재사용 (신규 API 불필요).

```tsx
const [hierarchy, setHierarchy] = useState<StatuteHierarchyResponse | null>(null)

useEffect(() => {
  if (!source.doc_id) return
  casePrecedentService.getStatuteHierarchy(source.doc_id)
    .then(setHierarchy)
    .catch(() => {}) // silent fail
}, [source.doc_id])
```

계층 UI (메타 블록 아래, 조문 본문 위):

```tsx
{hierarchy && (hierarchy.upper.length > 0 || hierarchy.lower.length > 0) && (
  <div className="bg-blue-50 rounded-xl border border-blue-100 p-4">
    <h4 className="text-xs font-semibold text-blue-600 mb-2">법령 단계 구조</h4>
    <div className="flex items-center gap-2 flex-wrap text-sm">
      {hierarchy.upper.map((node) => (
        <a key={node.id} href={`/statute-hierarchy?id=${node.id}&name=${node.name}&type=${node.type}`}
           className="text-blue-700 hover:underline">
          {node.name}
        </a>
      ))}
      {hierarchy.upper.length > 0 && <span className="text-gray-400">›</span>}
      <span className="font-bold text-blue-900 bg-blue-100 px-2 py-0.5 rounded">
        {source.law_name}
      </span>
      {hierarchy.lower.length > 0 && <span className="text-gray-400">›</span>}
      {hierarchy.lower.map((node) => (
        <a key={node.id} href={`/statute-hierarchy?id=${node.id}&name=${node.name}&type=${node.type}`}
           className="text-blue-700 hover:underline">
          {node.name}
        </a>
      ))}
    </div>
  </div>
)}
```

---

## Phase 2: 조문 트리 + 판례 연결

### 2-1. 프론트엔드: 조문 트리 파서

**파일**: `frontend/src/features/case-precedent/utils/articleTreeParser.ts` (신규)

DB 스키마 변경 없이 프론트엔드 순수 함수로 구현.
`articles` 배열에서 `article_number` 빈 문자열 + `article_content`에 "제N장"/"제N절" 패턴인 항목을 장/절 헤더로 분류.

```typescript
export type ArticleNodeType = 'chapter' | 'section' | 'article'

export interface ArticleTreeNode {
  type: ArticleNodeType
  label: string                    // "제1장 총칙", "제1절 통칙", "제5조"
  article?: LawArticleItem        // article 타입만 해당
  children: ArticleTreeNode[]
}

const CHAPTER_RE = /^제\d+장/
const SECTION_RE = /^제\d+절/

export function buildArticleTree(articles: LawArticleItem[]): ArticleTreeNode[] {
  const roots: ArticleTreeNode[] = []
  let currentChapter: ArticleTreeNode | null = null
  let currentSection: ArticleTreeNode | null = null

  for (const article of articles) {
    const isHeader = !article.article_number || article.article_number.trim() === ''
    if (isHeader) {
      const label = (article.article_title || article.article_content || '').trim()
      if (CHAPTER_RE.test(label)) {
        currentChapter = { type: 'chapter', label, children: [] }
        currentSection = null
        roots.push(currentChapter)
      } else if (SECTION_RE.test(label)) {
        currentSection = { type: 'section', label, children: [] }
        if (currentChapter) currentChapter.children.push(currentSection)
        else roots.push(currentSection)
      }
      continue
    }
    const node: ArticleTreeNode = {
      type: 'article',
      label: `제${article.article_number}조`,
      article,
      children: [],
    }
    if (currentSection) currentSection.children.push(node)
    else if (currentChapter) currentChapter.children.push(node)
    else roots.push(node)
  }
  return roots
}

/** 장/절 구조가 있는 법령인지 판별 */
export function hasTreeStructure(nodes: ArticleTreeNode[]): boolean {
  return nodes.some(n => n.type === 'chapter' || n.type === 'section')
}
```

**타입 추가**: `ArticleTreeNode`, `ArticleNodeType`을 `types/index.ts`에 추가하거나 파서 파일에서 export.

### 2-2. 프론트엔드: 트리 UI

**파일**: `LawDetailUser.tsx`, `LawDetailLawyer.tsx`

기존 아코디언 내 단순 `articles.map()` 목록을 트리 구조로 교체.

동작:
- `hasTreeStructure(tree)` = true → 장/절 접기/펼치기 트리 렌더링
- `hasTreeStructure(tree)` = false → 기존 단순 목록 유지 (fallback)
- 현재 검색된 조문(`source.article_number`)이 포함된 장/절은 기본 펼침
- 장/절 헤더 클릭으로 토글

```tsx
// useMemo로 트리 생성
const articleTree = useMemo(
  () => fullText ? buildArticleTree(fullText.articles) : [],
  [fullText]
)
const isTree = hasTreeStructure(articleTree)
```

장/절 접기/펼치기 상태 관리: `Set<string>` (라벨 기반 키).

### 2-3. 백엔드: 판례 연결 API

**파일**: `backend/app/tools/graph/pg_graph_service.py`

신규 메서드:

```python
async def get_cases_citing_statute(
    self,
    law_id: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """이 법령을 인용한 판례 목록 (역방향 쿼리, idx_csc_statute 활용)"""
    async with async_session_factory() as session:
        # law_id → PK 변환
        law_pk = (await session.execute(
            select(LawDocument.id).where(LawDocument.law_id == law_id)
        )).scalar_one_or_none()
        if not law_pk:
            return []

        stmt = (
            select(
                PrecedentDocument.serial_number,
                PrecedentDocument.case_number,
                PrecedentDocument.case_name,
                PrecedentDocument.decision_date,
                PrecedentDocument.court_name,
            )
            .join(CaseStatuteCitation, CaseStatuteCitation.case_doc_id == PrecedentDocument.id)
            .where(CaseStatuteCitation.law_doc_id == law_pk)
            .order_by(desc(PrecedentDocument.decision_date))
            .limit(limit)
        )
        result = await session.execute(stmt)
        return [dict(row._mapping) for row in result]
```

### 2-4. 백엔드: citing-cases 엔드포인트

**파일**: `backend/app/modules/case_precedent/router/__init__.py`

스키마:

```python
class CitingCaseItem(BaseModel):
    serial_number: Optional[str] = None
    case_number: Optional[str] = None
    case_name: Optional[str] = None
    decision_date: Optional[str] = None
    court_name: Optional[str] = None

class CitingCasesResponse(BaseModel):
    statute_id: str
    total: int
    cases: List[CitingCaseItem]
```

엔드포인트:

```python
@router.get("/statutes/{statute_id}/citing-cases", response_model=CitingCasesResponse)
async def get_citing_cases(
    statute_id: str,
    limit: int = Query(10, ge=1, le=50),
) -> CitingCasesResponse:
```

**라우터 순서**: `get_statute_children` 이후, `get_statute_graph` 이전에 배치 (FastAPI 경로 매칭 순서).

### 2-5. 프론트엔드: 판례 연결 버튼

**파일**: `types/index.ts`, `services/index.ts`, `LawDetailUser.tsx`, `LawDetailLawyer.tsx`

타입 추가:

```typescript
export interface CitingCaseItem {
  serial_number?: string
  case_number?: string
  case_name?: string
  decision_date?: string
  court_name?: string
}

export interface CitingCasesResponse {
  statute_id: string
  total: number
  cases: CitingCaseItem[]
}
```

서비스 함수 추가:

```typescript
getCitingCases: async (statuteId: string, limit = 10): Promise<CitingCasesResponse> => {
  const params = new URLSearchParams({ limit: limit.toString() })
  const response = await api.get(`${endpoints.casePrecedent}/statutes/${statuteId}/citing-cases?${params}`)
  return response.data
},
```

UI: 버튼 클릭 시 lazy loading → 판례 목록 표시.
판례 없으면 "인용 판례가 없습니다" 안내.

---

## 구현 순서 (총 8 Step)

```
Phase 1 — 메타 블록 + 법령 계층
  Step 1: LawFullTextResponse 스키마 + 핸들러 확장 (백엔드)
  Step 2: LawFullText 타입 확장 + dateUtils.ts (프론트엔드)
  Step 3: 전문 자동 로딩 + 메타 블록 UI (LawDetailUser → LawDetailLawyer)
  Step 4: 법령 계층 표시 (두 컴포넌트, 기존 API 재사용)

Phase 2 — 조문 트리 + 판례 연결
  Step 5: articleTreeParser.ts + ArticleTreeNode 타입 (프론트엔드)
  Step 6: 아코디언 내 트리 UI 교체 (두 컴포넌트)
  Step 7: PgGraphService.get_cases_citing_statute + API 엔드포인트 (백엔드)
  Step 8: 판례 연결 버튼 UI (프론트엔드)
```

## 수정 대상 파일 요약

| # | 파일 | 작업 |
|---|------|------|
| 1 | `backend/app/modules/case_precedent/router/__init__.py` | LawFullTextResponse 확장 + CitingCases 스키마/엔드포인트 |
| 2 | `backend/app/tools/graph/pg_graph_service.py` | `get_cases_citing_statute` 메서드 추가 |
| 3 | `frontend/src/features/case-precedent/types/index.ts` | LawFullText 확장 + ArticleTreeNode + CitingCasesResponse |
| 4 | `frontend/src/features/case-precedent/services/index.ts` | `getCitingCases` 서비스 함수 추가 |
| 5 | `frontend/src/features/case-precedent/components/LawDetailUser.tsx` | 메타블록 + 계층 + 트리 + 판례버튼 |
| 6 | `frontend/src/features/case-precedent/components/LawDetailLawyer.tsx` | 동일 |
| 7 | `frontend/src/features/case-precedent/utils/dateUtils.ts` | **신규** — 날짜 포맷 |
| 8 | `frontend/src/features/case-precedent/utils/articleTreeParser.ts` | **신규** — 조문 트리 파서 |

재사용 기존 코드:
- `casePrecedentService.getStatuteHierarchy()` — 법령 계층 API (기존)
- `casePrecedentService.getLawFullText()` — 법령 전문 API (기존, 자동 호출로 변경)
- `StatuteHierarchyResponse` 타입 (기존 `types/hierarchy.ts`)

## 검증 계획

1. **백엔드 정적 검증**: `cd backend && uv run ruff check app/modules/case_precedent/ app/tools/graph/`
2. **백엔드 타입 검증**: `cd backend && uv run mypy app/modules/case_precedent/ app/tools/graph/`
3. **프론트엔드 빌드**: `cd frontend && npm run build`
4. **API 응답 확인** (dev 서버):
   - `curl localhost:8000/api/case-precedent/laws/{law_id}/full-text | jq '{enforcement_date, promulgation_date, promulgation_no}'`
   - `curl localhost:8000/api/case-precedent/statutes/{id}/citing-cases | jq '.total'`
5. **UI 확인**:
   - 장/절 있는 법령(민법 등) → 트리 구조 표시
   - 장/절 없는 법령 → 단순 목록 fallback
   - 현재 조문 강조 + 해당 장/절 자동 펼침
   - 계층 없는 법령 → 계층 블록 미표시

## 가정

1. `law_documents` 테이블에 `enforcement_date`, `promulgation_date`, `promulgation_no` 컬럼이 이미 존재한다. → **확인 완료**
2. `statute_hierarchy` 테이블 + API가 이미 구현되어 있다. → **확인 완료**
3. `case_statute_citations` 테이블에 `idx_csc_statute(law_doc_id)` 인덱스가 있다. → **확인 완료**
4. DB 스키마 변경(Alembic 마이그레이션)은 불필요하다.
5. `law_articles` 테이블에 장/절 헤더도 저장되어 있다 (조문번호 빈 문자열). → **실제 데이터 확인 필요**
