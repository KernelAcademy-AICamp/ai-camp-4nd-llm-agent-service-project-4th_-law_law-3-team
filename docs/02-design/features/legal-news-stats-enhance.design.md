# 설계 문서: 법률 뉴스 통계 대시보드 개선

> **Feature**: `legal-news-stats-enhance`
> **작성일**: 2026-02-28
> **기획 보고서**: `docs/01-plan/features/legal-news-stats-enhance.plan.md`
> **상태**: Draft

---

## 1. 백엔드 설계

### 1-1. 스키마 변경 (`schema/__init__.py`)

#### 기존 `NewsCategoryStats` 확장

```python
class NewsCategoryStats(BaseModel):
    """카테고리 분포 통계 응답"""
    items: list[CategoryStatItem]
    total: int = Field(description="전체 기사 수")
    period_days: int | None = Field(default=None, description="조회 기간 (일, None=전체)")
```

- `period_days` 필드 추가 (Optional, 하위 호환 유지)
- `days` 미지정 시 `period_days=None` 반환

#### 신규 `RagContributionStats`

```python
class RagSourceItem(BaseModel):
    """RAG 데이터 소스별 건수"""
    source_name: str = Field(description="데이터 소스명 (law_documents|news_articles)")
    count: int = Field(description="문서 건수")
    description: str = Field(description="소스 설명")


class RagContributionStats(BaseModel):
    """RAG 기여도 통계 응답"""
    law_documents_count: int = Field(description="PostgreSQL 법령 문서 수")
    news_articles_count: int = Field(description="PostgreSQL 뉴스 기사 수")
    news_indexed_count: int = Field(description="LanceDB 임베딩 완료 뉴스 수")
    total_rag_documents: int = Field(description="전체 RAG 문서 수 (법령 + 뉴스)")
    news_contribution_percent: float = Field(description="뉴스 데이터 기여 비율 (%)")
    sources: list[RagSourceItem] = Field(description="소스별 상세 통계")
```

### 1-2. 서비스 함수 변경 (`service.py`)

#### `get_news_category_stats` 수정

```python
async def get_news_category_stats(
    db: AsyncSession,
    *,
    days: int | None = None,  # None이면 전체 기간
) -> NewsCategoryStats:
    """카테고리 분포 통계 조회 (SQL 집계)"""
    category_expr = _build_category_expression()

    query = (
        select(
            category_expr.label("category"),
            func.count().label("cnt"),
        )
        .group_by(category_expr)
        .order_by(func.count().desc())
    )

    # 기간 필터 (SQLAlchemy ORM, SQL Injection 안전)
    if days is not None:
        kst = timezone(timedelta(hours=9))
        since = (datetime.now(kst) - timedelta(days=days)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        query = query.where(NewsArticle.published_at >= since)

    rows = (await db.execute(query)).all()
    items = [CategoryStatItem(category=row.category, count=row.cnt) for row in rows]
    total = sum(row.cnt for row in rows)
    return NewsCategoryStats(items=items, total=total, period_days=days)
```

핵심: `timedelta(days=days)` 사용 → SQL Injection 불가 (Red Team 피드백 반영)

#### `get_rag_contribution_stats` 신규

```python
from app.models.law_document import LawDocument

async def get_rag_contribution_stats(
    db: AsyncSession,
) -> RagContributionStats:
    """RAG 기여도 통계 조회"""
    # PostgreSQL COUNT 쿼리 (ORM 기반, 안전)
    law_result = await db.execute(select(func.count()).select_from(LawDocument))
    law_count = law_result.scalar_one()

    news_result = await db.execute(select(func.count()).select_from(NewsArticle))
    news_count = news_result.scalar_one()

    indexed_result = await db.execute(
        select(func.count()).select_from(NewsArticle).where(
            NewsArticle.is_indexed == True  # noqa: E712
        )
    )
    news_indexed = indexed_result.scalar_one()

    total_rag = law_count + news_count
    contribution_pct = (news_count / total_rag * 100) if total_rag > 0 else 0.0

    return RagContributionStats(
        law_documents_count=law_count,
        news_articles_count=news_count,
        news_indexed_count=news_indexed,
        total_rag_documents=total_rag,
        news_contribution_percent=round(contribution_pct, 1),
        sources=[
            RagSourceItem(
                source_name="law_documents",
                count=law_count,
                description="법령 문서 (법률/시행령/시행규칙)",
            ),
            RagSourceItem(
                source_name="news_articles",
                count=news_count,
                description="법률 뉴스 기사 (로타임즈 + 네이버)",
            ),
        ],
    )
```

### 1-3. 라우터 변경 (`router/__init__.py`)

#### `/stats/category` 수정

```python
@router.get("/stats/category", response_model=NewsCategoryStats, summary="카테고리 분포 통계")
async def get_news_category_stats(
    days: int | None = Query(None, ge=1, le=90, description="조회 기간 (일, 미지정 시 전체)"),
    db: AsyncSession = Depends(get_db),
) -> NewsCategoryStats:
    """뉴스 기사 카테고리 분포 통계 (기간 필터 지원)"""
    from app.modules.legal_news.service import get_news_category_stats as svc
    return await svc(db, days=days)
```

#### `/stats/rag-contribution` 신규 (/{article_id} 앞에 위치)

```python
@router.get(
    "/stats/rag-contribution",
    response_model=RagContributionStats,
    summary="RAG 기여도 통계",
)
async def get_rag_contribution_stats(
    db: AsyncSession = Depends(get_db),
) -> RagContributionStats:
    """법령 DB vs 뉴스 데이터 비교 — RAG 시스템 기여도"""
    from app.modules.legal_news.service import get_rag_contribution_stats as svc
    return await svc(db)
```

라우터 등록 순서:
1. `/list`
2. `/stats/daily`
3. `/stats/category` (수정)
4. `/stats/rag-contribution` (신규)
5. `/{article_id}` (기존, 반드시 마지막)
6. `/search` (POST)

---

## 2. 프론트엔드 설계

### 2-1. 타입 추가 (`types/index.ts`)

```typescript
// NewsCategoryStats 확장
export interface NewsCategoryStats {
  items: CategoryStatItem[]
  total: number
  period_days: number | null  // 추가
}

// RAG 기여도 신규
export interface RagSourceItem {
  source_name: string
  count: number
  description: string
}

export interface RagContributionStats {
  law_documents_count: number
  news_articles_count: number
  news_indexed_count: number
  total_rag_documents: number
  news_contribution_percent: number
  sources: RagSourceItem[]
}
```

### 2-2. 서비스 함수 변경 (`services/index.ts`)

```typescript
/** 카테고리 분포 통계 조회 (기간 필터) */
export async function fetchNewsCategoryStats(
  days?: number,
  signal?: AbortSignal,
): Promise<NewsCategoryStats> {
  const { data } = await api.get<NewsCategoryStats>(`${BASE}/stats/category`, {
    params: days ? { days } : undefined,
    signal,
  })
  return data
}

/** RAG 기여도 통계 조회 */
export async function fetchRagContributionStats(
  signal?: AbortSignal,
): Promise<RagContributionStats> {
  const { data } = await api.get<RagContributionStats>(
    `${BASE}/stats/rag-contribution`,
    { signal },
  )
  return data
}
```

### 2-3. 훅 변경 (`useNewsStats.ts`)

```typescript
interface UseNewsStatsReturn {
  // 기존
  dailyStats: NewsStatsDaily | null
  categoryStats: NewsCategoryStats | null
  loading: boolean
  error: string | null
  periodDays: number
  setPeriodDays: (days: number) => void
  // 추가
  categoryPeriodDays: number
  setCategoryPeriodDays: (days: number) => void
  categoryLoading: boolean
  ragStats: RagContributionStats | null
  ragLoading: boolean
}
```

**상태 흐름**:
- `categoryPeriodDays` (기본 7) → 변경 시 `loadCategoryStats(days)` 재호출
- `ragStats` → 마운트 시 1회 조회 (변경 없음)
- 각 AbortController 독립 관리 (daily, category, rag 별도)

### 2-4. NewsBarChart 수정

**변경 사항**: Legend 아래에 소스별 누계 건수 표시

```tsx
export function NewsBarChart({ items, periodDays, onPeriodChange }: NewsBarChartProps) {
  const data = transformData(items, periodDays)

  // useMemo로 누계 계산 (Red Team 피드백 반영)
  const totals = useMemo(() => ({
    lawtimes: data.reduce((sum, row) => sum + row.lawtimes, 0),
    naver: data.reduce((sum, row) => sum + row.naver, 0),
  }), [data])

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-4 shadow-sm">
      {/* 헤더 + 기간 선택 (기존) */}
      {/* ... */}

      {/* 차트 (기존) */}
      {/* ... */}

      {/* 누계 건수 (신규) */}
      <div className="flex items-center justify-center gap-6 mt-2 pt-2 border-t border-gray-100">
        <div className="flex items-center gap-1.5 text-xs">
          <span className="w-2.5 h-2.5 rounded-full bg-blue-500" />
          <span className="text-gray-600">로타임즈</span>
          <span className="font-semibold text-gray-800">
            {totals.lawtimes.toLocaleString()}건
          </span>
        </div>
        <div className="flex items-center gap-1.5 text-xs">
          <span className="w-2.5 h-2.5 rounded-full bg-green-500" />
          <span className="text-gray-600">네이버</span>
          <span className="font-semibold text-gray-800">
            {totals.naver.toLocaleString()}건
          </span>
        </div>
      </div>
    </div>
  )
}
```

### 2-5. NewsDonutChart 수정

**변경 사항**:
1. Props 확장 (`periodDays`, `onPeriodChange`)
2. 기간 선택 버튼 추가
3. 커스텀 Legend로 교체 (카테고리 건수 포함)
4. 파이 라벨은 퍼센트만 표시 (건수는 Legend에서 확인)

```tsx
interface NewsDonutChartProps {
  items: CategoryStatItem[]
  total: number
  periodDays: number           // 추가
  onPeriodChange: (days: number) => void  // 추가
  loading?: boolean            // 추가: 기간 변경 시 로딩
}

const PERIOD_OPTIONS = [
  { value: 7, label: '7일' },
  { value: 14, label: '14일' },
  { value: 30, label: '30일' },
]

export function NewsDonutChart({
  items, total, periodDays, onPeriodChange, loading,
}: NewsDonutChartProps) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-4 shadow-sm">
      {/* 헤더: 제목 + 기간 선택 */}
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-gray-800">카테고리 분포</h3>
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500">총 {total.toLocaleString()}건</span>
          <div className="flex gap-1">
            {PERIOD_OPTIONS.map((opt) => (
              <button
                key={opt.value}
                onClick={() => onPeriodChange(opt.value)}
                className={`px-2.5 py-1 text-xs rounded-md transition-colors ${
                  periodDays === opt.value
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {loading ? (
        <div className="flex items-center justify-center h-48">
          <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600" />
        </div>
      ) : items.length === 0 ? (
        <div className="flex items-center justify-center h-48 text-sm text-gray-400">
          데이터 없음
        </div>
      ) : (
        <>
          {/* 도넛 차트 (Legend 제거, 파이 라벨은 퍼센트만) */}
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie
                data={items}
                dataKey="count"
                nameKey="category"
                cx="50%"
                cy="50%"
                innerRadius={55}
                outerRadius={80}
                paddingAngle={2}
                label={({ percent }) => {
                  if (Number(percent) < 0.05) return ''
                  return `${(Number(percent) * 100).toFixed(0)}%`
                }}
                labelLine={{ stroke: '#9ca3af', strokeWidth: 1 }}
              >
                {items.map((_, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip
                formatter={(value) => [`${Number(value).toLocaleString()}건`]}
                contentStyle={{ fontSize: 12, borderRadius: 8 }}
              />
            </PieChart>
          </ResponsiveContainer>

          {/* 커스텀 Legend: 카테고리별 건수 포함 */}
          <ul className="grid grid-cols-2 gap-x-4 gap-y-1.5 mt-2 pt-2 border-t border-gray-100">
            {items.map((item, index) => (
              <li key={item.category} className="flex items-center gap-1.5 text-xs">
                <span
                  className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                  style={{ backgroundColor: COLORS[index % COLORS.length] }}
                />
                <span className="text-gray-700 truncate">{item.category}</span>
                <span className="text-gray-500 ml-auto font-medium">
                  {item.count.toLocaleString()}건
                </span>
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  )
}
```

### 2-6. RagContributionCard 신규 컴포넌트

```tsx
// frontend/src/features/legal-news/components/RagContributionCard.tsx

interface RagContributionCardProps {
  stats: RagContributionStats | null
  loading: boolean
  error?: string | null
}

export function RagContributionCard({ stats, loading, error }: RagContributionCardProps) {
  // Graceful Degradation (Red Team 피드백 반영)
  if (error) {
    return (
      <div className="bg-white rounded-xl border border-gray-200 p-4 shadow-sm">
        <h3 className="text-sm font-semibold text-gray-800 mb-2">RAG 데이터 기여도</h3>
        <div className="text-xs text-gray-400 text-center py-4">
          일시적으로 조회할 수 없습니다
        </div>
      </div>
    )
  }

  if (loading || !stats) {
    return (
      <div className="bg-white rounded-xl border border-gray-200 p-4 shadow-sm">
        <h3 className="text-sm font-semibold text-gray-800 mb-2">RAG 데이터 기여도</h3>
        <div className="flex items-center justify-center h-32">
          <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600" />
        </div>
      </div>
    )
  }

  const { law_documents_count, news_articles_count, news_contribution_percent, total_rag_documents } = stats
  const lawPercent = total_rag_documents > 0 ? (law_documents_count / total_rag_documents * 100) : 0
  const newsPercent = total_rag_documents > 0 ? (news_articles_count / total_rag_documents * 100) : 0

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-4 shadow-sm">
      <h3 className="text-sm font-semibold text-gray-800 mb-3">RAG 데이터 기여도</h3>

      {/* 법령 데이터 */}
      <div className="mb-3">
        <div className="flex items-center justify-between text-xs mb-1">
          <span className="text-gray-600">법령 데이터</span>
          <span className="font-medium text-gray-800">
            {law_documents_count.toLocaleString()}건
          </span>
        </div>
        <div className="w-full bg-gray-100 rounded-full h-2">
          <div
            className="bg-blue-500 h-2 rounded-full transition-all"
            style={{ width: `${Math.min(lawPercent, 100)}%` }}
          />
        </div>
      </div>

      {/* 뉴스 데이터 */}
      <div className="mb-3">
        <div className="flex items-center justify-between text-xs mb-1">
          <span className="text-gray-600">뉴스 데이터</span>
          <span className="font-medium text-gray-800">
            {news_articles_count.toLocaleString()}건
          </span>
        </div>
        <div className="w-full bg-gray-100 rounded-full h-2">
          <div
            className="bg-green-500 h-2 rounded-full transition-all"
            style={{ width: `${Math.min(newsPercent, 100)}%` }}
          />
        </div>
      </div>

      {/* 구분선 */}
      <div className="border-t border-gray-100 pt-3 mt-1">
        <div className="flex items-center justify-between text-xs mb-1">
          <span className="text-gray-600">뉴스 기여 비율</span>
          <span className="font-semibold text-blue-600">
            {news_contribution_percent.toFixed(1)}%
          </span>
        </div>
        <div className="w-full bg-gray-100 rounded-full h-2.5">
          <div
            className="bg-gradient-to-r from-blue-500 to-green-500 h-2.5 rounded-full transition-all"
            style={{ width: `${Math.min(news_contribution_percent, 100)}%` }}
          />
        </div>
        <p className="text-xs text-gray-400 mt-2 text-center">
          법률 뉴스 {news_articles_count.toLocaleString()}건이 RAG 검색에 기여
        </p>
      </div>
    </div>
  )
}
```

### 2-7. 페이지 수정 (`page.tsx`)

```tsx
// useNewsStats 훅 확장 사용
const {
  dailyStats, categoryStats, loading: statsLoading, error: statsError,
  periodDays, setPeriodDays,
  categoryPeriodDays, setCategoryPeriodDays, categoryLoading,
  ragStats, ragLoading,
} = useNewsStats()

// 통계 사이드바에 추가
{!statsLoading && categoryStats && (
  <NewsDonutChart
    items={categoryStats.items}
    total={categoryStats.total}
    periodDays={categoryPeriodDays}
    onPeriodChange={setCategoryPeriodDays}
    loading={categoryLoading}
  />
)}
<RagContributionCard
  stats={ragStats}
  loading={ragLoading}
  error={statsError}
/>
```

---

## 3. API 계약 동기화 매트릭스

| Backend 스키마 | Frontend 타입 | 동기화 |
|---------------|--------------|--------|
| `NewsCategoryStats.period_days: int \| None` | `NewsCategoryStats.period_days: number \| null` | snake_case 동일 |
| `RagSourceItem` | `RagSourceItem` | snake_case 동일 |
| `RagContributionStats` | `RagContributionStats` | snake_case 동일 |

---

## 4. 구현 단계별 검증 계획

| 단계 | 작업 | 검증 |
|------|------|------|
| 1 | Backend: schema 변경 | `uv run ruff check backend/app/` + `uv run mypy backend/app/` |
| 2 | Backend: service 변경 | ruff + mypy |
| 3 | Backend: router 변경 | ruff + mypy + `curl` 테스트 |
| 4 | Frontend: types 변경 | `npm run build` |
| 5 | Frontend: services 변경 | build |
| 6 | Frontend: useNewsStats 변경 | build |
| 7 | Frontend: NewsBarChart 수정 | build + 브라우저 확인 |
| 8 | Frontend: NewsDonutChart 수정 | build + 브라우저 확인 |
| 9 | Frontend: RagContributionCard 신규 | build + 브라우저 확인 |
| 10 | Frontend: page.tsx 수정 | build + 전체 통합 확인 |

---

## 5. 파일 변경 목록 (최종)

### Backend (3파일 수정)

| 파일 | 라인 영향 | 변경 내용 |
|------|----------|----------|
| `schema/__init__.py` | +25줄 | `period_days` 필드, `RagSourceItem`, `RagContributionStats` |
| `service.py` | +35줄 | `get_news_category_stats(days)`, `get_rag_contribution_stats()` |
| `router/__init__.py` | +20줄 | `/stats/category` days 파라미터, `/stats/rag-contribution` |

### Frontend (6파일 수정 + 1파일 신규)

| 파일 | 변경 내용 |
|------|----------|
| `types/index.ts` | `period_days`, `RagSourceItem`, `RagContributionStats` 추가 |
| `services/index.ts` | `fetchNewsCategoryStats(days?)`, `fetchRagContributionStats()` |
| `hooks/useNewsStats.ts` | `categoryPeriodDays`, `ragStats`, 별도 로딩 상태 |
| `components/NewsBarChart.tsx` | `useMemo` 누계 + 누계 표시 div |
| `components/NewsDonutChart.tsx` | Props 확장, 기간 선택, 커스텀 Legend |
| `components/RagContributionCard.tsx` | **신규** (Tailwind Progress Bar) |
| `app/legal-news/page.tsx` | 훅 확장 사용, RagContributionCard 배치 |
