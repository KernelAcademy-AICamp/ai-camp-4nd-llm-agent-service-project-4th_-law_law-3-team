# 기획 보고서: 법률 뉴스 통계 대시보드 개선

> **Feature**: `legal-news-stats-enhance`
> **작성일**: 2026-02-28
> **상태**: Draft (기획 단계)
> **담당**: Agent Team (PM 조율)

---

## 1. 배경 및 목적

법률 뉴스 수집·요약 및 하이브리드 검색 기능의 통계 대시보드를 개선하여, 사용자에게 더 풍부한 데이터 인사이트를 제공한다.

### 현재 상태

| 항목 | 현재 | 개선 후 |
|------|------|---------|
| 일별 수집 건수 | 그래프만 표시 | 그래프 + 소스별 누계 건수 |
| 카테고리 분포 | 전체 기간 고정 | 7/14/30일 기간 선택 가능 |
| 카테고리 범례 | 이름만 표시 | 이름 + 건수 표시 |
| RAG 기여도 | 없음 | 법령 DB vs 뉴스 비교 인포그래픽 |

---

## 2. 요구사항

### FR-01: 일별 수집 건수 그래프 누계 표시

**설명**: 일별 수집 건수 막대 차트(NewsBarChart)의 범례 아래에 선택된 기간 내 소스별 누계 수집 건수를 표시한다.

**상세**:
- 로타임즈(lawtimes) 누계 건수
- 네이버뉴스(naver) 누계 건수
- 기간(7/14/30일) 변경 시 누계도 자동 갱신
- 프론트엔드에서 기존 데이터(`items`)를 집계하여 계산 (추가 API 불필요)

**UI 레이아웃**:
```
┌───────────────────────────────────────┐
│ 일별 수집 건수          [7일][14일][30일] │
│                                       │
│           [Bar Chart 영역]             │
│                                       │
│  ● 로타임즈  ● 네이버                  │  ← 기존 Legend
│  로타임즈 123건 · 네이버 456건          │  ← 신규 누계
└───────────────────────────────────────┘
```

**변경 파일**: `frontend/src/features/legal-news/components/NewsBarChart.tsx`

---

### FR-02: 카테고리 분포 기간 선택

**설명**: 카테고리 분포 도넛 차트(NewsDonutChart)에 7일/14일/30일 기간 선택 버튼을 추가한다.

**상세**:
- 기간 선택 UI는 일별 수집 건수와 동일한 디자인 ([7일][14일][30일] 버튼)
- 기간 변경 시 백엔드 API 재호출 → 해당 기간 카테고리 분포 표시
- 기본값: 7일

**백엔드 변경**:
- `GET /api/legal-news/stats/category` 엔드포인트에 `days` Query 파라미터 추가
  - `days: int | None = Query(None, ge=1, le=90)`
  - `None`이면 전체 기간 (하위 호환성 유지)
- `NewsCategoryStats` 스키마에 `period_days: int | None` 필드 추가

**프론트엔드 변경**:
- `NewsDonutChart` Props에 `periodDays`, `onPeriodChange` 추가
- `useNewsStats` 훅에 `categoryPeriodDays`, `setCategoryPeriodDays` 상태 추가
- `fetchNewsCategoryStats(days?, signal?)` 함수 시그니처 확장

**변경 파일**:
- Backend: `service.py`, `schema/__init__.py`, `router/__init__.py`
- Frontend: `NewsDonutChart.tsx`, `useNewsStats.ts`, `services/index.ts`, `types/index.ts`, `page.tsx`

---

### FR-03: 카테고리 분포 범례 건수 표시

**설명**: 카테고리 분포 도넛 차트 범례 아래에 각 카테고리별 뉴스 건수를 표시한다.

**상세**:
- recharts 내장 `<Legend>` 대신 커스텀 Legend 렌더러 사용
- 각 카테고리에 색상 점 + 카테고리명 + 건수 표시
- 2열 그리드 레이아웃으로 가독성 확보

**UI 레이아웃**:
```
┌───────────────────────────────────────┐
│ 카테고리 분포          [7일][14일][30일] │
│                    총 1,234건          │
│                                       │
│           [Donut Chart 영역]           │
│                                       │
│  ● 판결 큐레이션   345건  ● 법조계 인사  289건 │
│  ● 법령 동향       178건  ● 형사·검찰   156건 │
│  ● 소송·재판       134건  ● 법조계 동향  98건 │
│  ● 기타            34건                │
└───────────────────────────────────────┘
```

**변경 파일**: `frontend/src/features/legal-news/components/NewsDonutChart.tsx`

---

### FR-04: RAG 기여도 인포그래픽

**설명**: 카테고리 분포 차트 아래에 RAG 시스템 기여도를 보여주는 인포그래픽 카드를 추가한다.

**상세**:
- PostgreSQL `law_documents` 테이블의 법령 문서 수
- PostgreSQL `news_articles` 테이블의 뉴스 기사 수
- 뉴스 데이터가 전체 RAG 데이터 풀 대비 차지하는 비율
- LanceDB `legal_chunks` 건수 (조회 가능 시)
- LanceDB `news_chunks` 건수 (조회 가능 시)

**백엔드 변경**:
- 새 엔드포인트: `GET /api/legal-news/stats/rag-contribution`
- 새 스키마: `RagSourceItem`, `RagContributionStats`
- 새 서비스 함수: `get_rag_contribution_stats(db)`

**프론트엔드 변경**:
- 새 컴포넌트: `RagContributionCard.tsx`
- Tailwind CSS 기반 Progress Bar (recharts 불필요)
- `useNewsStats` 훅에 RAG 통계 로드 추가
- 새 타입: `RagSourceItem`, `RagContributionStats`
- 새 서비스: `fetchRagContributionStats(signal?)`

**UI 레이아웃**:
```
┌───────────────────────────────────────┐
│  RAG 데이터 기여도                     │
│                                       │
│  📚 법령 데이터                        │
│  ██████████████████████████████  N건  │
│                                       │
│  📰 뉴스 데이터                        │
│  ████████                        N건  │
│                                       │
│  ─────────────────────────────────    │
│  뉴스 데이터 기여 비율                  │
│  ██████████████░░░░░░░░░░  XX.X%     │
│                                       │
│  "법률 뉴스 N건이 RAG 검색에 기여"     │
└───────────────────────────────────────┘
```

**변경 파일**:
- Backend: `service.py`, `schema/__init__.py`, `router/__init__.py`
- Frontend: 새 `RagContributionCard.tsx`, `useNewsStats.ts`, `services/index.ts`, `types/index.ts`, `page.tsx`

---

## 3. 기술 설계

### 3-1. 백엔드 API 변경

#### `/stats/category` 기간 필터 추가

| 항목 | 변경 전 | 변경 후 |
|------|---------|---------|
| 라우터 파라미터 | 없음 | `days: int \| None = Query(None, ge=1, le=90)` |
| 서비스 시그니처 | `get_news_category_stats(db)` | `get_news_category_stats(db, *, days=None)` |
| 스키마 필드 | items + total | items + total + `period_days: int \| None` |
| SQL WHERE | 없음 | `published_at >= NOW() - INTERVAL 'N days'` (days 지정 시) |

하위 호환성: `days` 미지정 시 전체 기간 조회 (기존 동작 유지).

#### `/stats/rag-contribution` 신규 엔드포인트

```python
# 응답 스키마
class RagSourceItem(BaseModel):
    source_name: str        # "law_documents" | "news_articles" 등
    count: int
    description: str

class RagContributionStats(BaseModel):
    law_documents_count: int          # 법령 문서 수
    news_articles_count: int          # 뉴스 기사 수
    news_indexed_count: int           # 임베딩 완료 뉴스 수
    total_rag_documents: int          # 전체 RAG 문서 수 (법령 + 뉴스)
    news_contribution_percent: float  # 뉴스 기여 비율 (%)
    sources: list[RagSourceItem]
```

SQL 쿼리:
- `SELECT COUNT(*) FROM law_documents` → 법령 문서 수
- `SELECT COUNT(*) FROM news_articles` → 뉴스 기사 수
- `SELECT COUNT(*) FROM news_articles WHERE is_indexed = TRUE` → 임베딩 완료 수
- 기여 비율 = `news_articles_count / (law_documents_count + news_articles_count) * 100`

라우터 위치: `/stats/rag-contribution`은 `/{article_id}` 경로보다 **반드시 앞**에 위치.

### 3-2. 프론트엔드 변경

#### useNewsStats 훅 확장

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
}
```

#### 컴포넌트 변경 요약

| 컴포넌트 | 변경 |
|---------|------|
| `NewsBarChart` | Legend 아래 누계 건수 div 추가 |
| `NewsDonutChart` | Props 확장 (periodDays, onPeriodChange), 커스텀 Legend, 건수 표시 |
| `RagContributionCard` | 신규 생성 (Tailwind Progress Bar) |
| `page.tsx` | 훅 반환값 확장, RagContributionCard 추가 |

---

## 4. 구현 순서

| 순서 | 작업 | 의존성 | 백엔드 필요 |
|------|------|--------|------------|
| 1 | FR-01: NewsBarChart 누계 표시 | 없음 | 없음 |
| 2 | FR-03: NewsDonutChart 커스텀 Legend + 건수 | 없음 | 없음 |
| 3 | FR-02: 카테고리 기간 선택 (Backend + Frontend) | 없음 | **있음** |
| 4 | FR-04: RAG 기여도 인포그래픽 (Backend + Frontend) | 없음 | **있음** |

- 1~2번: 백엔드 변경 없이 프론트엔드만으로 즉시 구현 가능
- 3~4번: 백엔드 API 변경 선행 → 프론트엔드 연동 순서

---

## 5. 변경 파일 목록

### Backend (4파일)

| 파일 | 변경 유형 | 내용 |
|------|----------|------|
| `backend/app/modules/legal_news/schema/__init__.py` | 수정 | `NewsCategoryStats.period_days` 추가, `RagSourceItem`/`RagContributionStats` 신규 |
| `backend/app/modules/legal_news/service.py` | 수정 | `get_news_category_stats(days)` 파라미터 추가, `get_rag_contribution_stats()` 신규 |
| `backend/app/modules/legal_news/router/__init__.py` | 수정 | `/stats/category` days 파라미터, `/stats/rag-contribution` 신규 |
| (import) | 수정 | `service.py`에 `LawDocument` import 추가 |

### Frontend (7파일)

| 파일 | 변경 유형 | 내용 |
|------|----------|------|
| `frontend/src/features/legal-news/types/index.ts` | 수정 | `NewsCategoryStats.period_days`, `RagSourceItem`, `RagContributionStats` 추가 |
| `frontend/src/features/legal-news/services/index.ts` | 수정 | `fetchNewsCategoryStats(days?)`, `fetchRagContributionStats()` |
| `frontend/src/features/legal-news/hooks/useNewsStats.ts` | 수정 | `categoryPeriodDays`, `ragStats` 상태 추가 |
| `frontend/src/features/legal-news/components/NewsBarChart.tsx` | 수정 | 누계 건수 표시 추가 |
| `frontend/src/features/legal-news/components/NewsDonutChart.tsx` | 수정 | 기간 선택, 커스텀 Legend + 건수 |
| `frontend/src/features/legal-news/components/RagContributionCard.tsx` | **신규** | RAG 기여도 인포그래픽 |
| `frontend/src/app/legal-news/page.tsx` | 수정 | 훅 확장, RagContributionCard 추가 |

---

## 6. 리스크 분석

| 리스크 | 심각도 | 완화 방안 |
|--------|--------|---------|
| 카테고리 SQL CASE WHEN + 기간 필터 성능 | Low | `published_at` 인덱스 존재 (idx_news_published_at), 현재 데이터량 소규모 |
| LawDocument 모델 import 누락 | Low | service.py에 import 명시적 추가 |
| 하위 호환 깨짐 (category API) | Low | `days=None` 기본값으로 기존 동작 유지 |
| RAG 통계 응답 시간 | Low | PostgreSQL COUNT 쿼리 2개, 소규모 데이터 |

---

## 7. 고위험 영역 확인 (Fast-track 판단)

| 고위험 영역 | 해당 여부 |
|------------|----------|
| 인증/권한 | 해당 없음 |
| 개인정보/데이터 접근 | 해당 없음 (통계 집계만) |
| 프롬프트 체인 | 해당 없음 |
| 외부 연동 | 해당 없음 |
| 새로운 아키텍처 패턴 | 해당 없음 (기존 패턴 확장) |

→ **Fast-track 대상 가능**, 단 사용자 요청에 따라 정규 워크플로우(3중 검증) 진행.
