# Lawyer Stats Bubble Chart Plan

> **Summary**: 변호사 분포 시각화 — Plotly.js 2D 버블 차트로 지역 x 전문분야 분포를 표현하고, 연도 슬라이더로 사건 수요 변화를 탐색하며, 유저(변호사)가 본인 위치를 마킹할 수 있는 인터랙티브 시각화
>
> **Project**: law-3-team (법률 서비스 플랫폼)
> **Author**: Claude
> **Date**: 2026-02-27
> **Status**: Draft (v1.0)
> **Branch**: feat/lawyer-stats

---

## 1. Overview

### 1.1 Purpose

**핵심 대상**: 신입 변호사 / 개업 준비 중인 변호사
**핵심 가치**: 시장 내 본인의 포지션을 시각적으로 파악하고, 시간에 따른 수요 변화를 탐색

현재 변호사 통계 대시보드에는 지역별 막대 차트, 히트맵 등 **정적 통계**만 있다. 이를 보완하여:

1. **지역 x 전문분야** 2차원 평면에 변호사 수를 버블 크기로 표현
2. **연도 슬라이더** (2015~2024)로 사건 수요 변화를 시각적 탐색
3. **유저 마커**로 본인의 지역/전문분야 위치를 표시하여 경쟁 구도 파악

### 1.2 Background

- 기존 `CrossAnalysisHeatmap`: 지역 x 전문분야 데이터를 히트맵 테이블로 표현 (정적)
- `lawyers_2010_2025.csv`: 법원별 변호사 수 시계열 데이터 (2010~2025, 63개 법원)
- `trial_statistics` 테이블: 법원별/카테고리별/연도별 사건 처리 건수 (2015~2024)
- Plotly.js (`plotly.js-basic-dist-min` ~1MB): scatter 버블, 슬라이더, 애니메이션 내장

### 1.3 핵심 사용자 시나리오

```
시나리오 1 (시장 탐색): 사용자가 버블 차트를 보며 "서울 민사 버블이 크고, 제주 행정은 작다"를 한눈에 파악

시나리오 2 (시간 변화): 연도 슬라이더를 2015→2024로 드래그하며 "가사 사건이 전국적으로 증가했구나"를 관찰

시나리오 3 (내 위치 확인): 지역=부산, 전문분야=형사를 선택하면 ★ 마커가 해당 버블 위에 표시되어 "내 위치의 경쟁 강도"를 파악

시나리오 4 (상세 정보): 버블에 마우스를 올리면 "서울 민사: 변호사 5,231명, 사건 수 12,340건, 부담지수 2.36" 툴팁 표시
```

### 1.4 데이터 구성

| 축/요소 | 데이터 소스 | 설명 |
|---------|-----------|------|
| **X축: 지역** | `lawyers` 테이블 `province` (17개 시/도) | 교차분석에서 사용하는 지역 단위와 동일 |
| **Y축: 전문분야** | `lawyers` 테이블 `specialties` → 12대분류 매핑 | 기존 `SPECIALTY_CATEGORIES` 활용 |
| **버블 크기** | `CrossAnalysisCell.count` | 해당 지역+분야의 변호사 수 |
| **버블 색상** | 전문분야 카테고리별 고정 색상 | 12대분류별 구분 |
| **연도 슬라이더** | `trial_statistics` (2015~2024) | 사건 수요 변화 — 배경 히트맵 또는 보조 버블로 표현 |
| **유저 마커 ★** | 사용자 입력 (지역 드롭다운 + 전문분야 드롭다운) | 해당 좌표에 별도 마커 표시 |

### 1.5 데이터 한계 및 대응

| 한계 | 영향 | 대응 |
|------|------|------|
| 변호사 분포 = 현재 시점 스냅샷 | 과거 연도별 지역x분야 분포 없음 | **버블 크기는 현재 고정**, 슬라이더는 사건 수요만 변동 |
| 사건 수요 카테고리 ≠ 전문분야 12대분류 | 1:1 매핑 불가 (민사/형사/가사/행정/소년보호/가정보호 6종) | 수요는 **지역 단위 총량**으로 표현 (분야별 수요 오버레이 아님) |
| 시/도 17개 단위 | 구 단위 불가 | X축 17개 포인트로 충분한 가독성 |

---

## 2. 아키텍처 설계

### 2.1 시각화 구성

```
┌───────────────────────────────────────────────────────┐
│  변호사 분포 시각화                                       │
│                                                         │
│  X축: 지역 (서울, 경기, 부산, 대구, ...)                  │
│  Y축: 전문분야 (민사, 형사, 가사, 행정, ...)              │
│  버블 크기: 변호사 수                                     │
│  버블 색상: 전문분야 카테고리                              │
│                                                         │
│     ●●●  ●●   ○       ○   ●                            │
│     ○○   ●    ●●      ●   ○                            │
│     ●    ○    ○  ★    ○   ●                            │
│     서울  경기  부산    대구  광주 ...                      │
│                  ↑ 내 위치                                │
│                                                         │
│  ◀ [2015] ═══════●══════════ [2024] ▶  연도 슬라이더     │
│                 2020                                     │
│  → 슬라이더 변경 시: 배경에 해당 연도 사건 수요 밀도 표시   │
│                                                         │
│  [내 정보: 지역 ▼  전문분야 ▼]    범례: ● 버블크기 = 변호사수 │
└───────────────────────────────────────────────────────┘
```

### 2.2 데이터 흐름

```
[Backend]
  │
  ├─ GET /api/lawyer-stats/cross-analysis  (기존)
  │   → 지역 x 전문분야 교차 데이터 (버블 크기)
  │
  ├─ GET /api/lawyer-stats/demand-by-region?category=전체&year=2020  (기존 확장)
  │   → 해당 연도의 지역별 사건 수요 (슬라이더 연동)
  │
  └─ GET /api/lawyer-stats/bubble-data  (신규 — 통합 엔드포인트)
      → { cross: CrossAnalysisCell[], demand_by_year: { [year]: DemandStat[] } }
      → 프론트에서 한 번의 호출로 전체 데이터 확보

[Frontend]
  │
  ├─ BubbleChart.tsx (Plotly.js)
  │   ├─ scatter trace: X=지역, Y=전문분야, size=변호사수, color=카테고리
  │   ├─ 유저 마커 trace: X=선택지역, Y=선택분야, marker=★
  │   └─ 슬라이더: Plotly `sliders` 또는 별도 React 컴포넌트
  │
  └─ page.tsx
      └─ cross-section 아래에 bubble-section 추가
```

### 2.3 연도 슬라이더 동작 상세

연도 슬라이더 변경 시 **버블 크기는 변하지 않고** (현재 변호사 분포 고정), 대신:

**방안: 배경 히트맵 오버레이**
- 각 지역 열에 해당 연도의 총 사건 수를 배경 색상 강도로 표현
- 슬라이더 변경 → 배경 색상만 트랜지션
- 변호사 분포(버블)와 사건 수요(배경)의 미스매치를 시각적으로 드러냄

```
예시: 2024년 선택 시
- 서울 열 배경: 진한 빨강 (사건 수 최다)
- 제주 열 배경: 연한 빨강 (사건 수 적음)
- 버블은 동일 크기 유지
→ "사건은 많은데 변호사가 적은 지역" = 기회 지역 시각적 파악
```

### 2.4 신규 통합 API 설계

기존 API를 각각 호출하면 N번 요청이 필요하므로, **버블 차트 전용 통합 엔드포인트**를 추가:

```
GET /api/lawyer-stats/bubble-data

Response:
{
  "cross": [                          // 교차분석 데이터 (버블)
    { "region": "서울", "category_name": "민사", "count": 5231 },
    ...
  ],
  "regions": ["서울", "경기", ...],    // X축 라벨
  "categories": ["민사", "형사", ...], // Y축 라벨
  "demand_by_year": {                  // 연도별 수요 (슬라이더)
    "2015": [{ "region": "서울", "total_cases": 45000 }, ...],
    "2016": [...],
    ...
    "2024": [...]
  },
  "available_years": [2015, 2016, ..., 2024]
}
```

이렇게 하면 프론트에서 **1회 호출**로 전체 데이터를 확보하고, 슬라이더 변경 시 추가 API 호출 없이 클라이언트에서 처리.

---

## 3. 변경 파일 목록

### 3.1 Backend 변경

| # | 파일 | 변경 유형 | 설명 |
|---|------|---------|------|
| B1 | `backend/app/modules/lawyer_stats/schema/__init__.py` | 추가 | `BubbleDataResponse`, `RegionDemandByYear` 스키마 |
| B2 | `backend/app/modules/lawyer_stats/router/__init__.py` | 추가 | `GET /bubble-data` 엔드포인트 |
| B3 | `backend/app/services/service_function/lawyer_stats_service.py` | 추가 | `calculate_bubble_data()` 함수 |
| B4 | `backend/app/services/service_function/lawyer_stats_db_service.py` | 추가 | `calculate_bubble_data_db()` 함수 |

### 3.2 Frontend 변경

| # | 파일 | 변경 유형 | 설명 |
|---|------|---------|------|
| F1 | `frontend/src/features/lawyer-stats/types/index.ts` | 추가 | `BubbleDataResponse`, `RegionDemandByYear` 타입 |
| F2 | `frontend/src/features/lawyer-stats/services/index.ts` | 추가 | `fetchBubbleData()` API 함수 |
| F3 | `frontend/src/features/lawyer-stats/components/BubbleChart.tsx` | **신규 생성** | Plotly.js 버블 차트 컴포넌트 |
| F4 | `frontend/src/features/lawyer-stats/components/index.ts` | 수정 | `BubbleChart` export 추가 |
| F5 | `frontend/src/features/lawyer-stats/components/StickyTabNav.tsx` | 수정 | `TabType`에 `'bubble'` 추가 |
| F6 | `frontend/src/app/lawyer-stats/page.tsx` | 수정 | bubble-section 추가, dynamic import, ref/observer 연동 |
| F7 | `package.json` | 수정 | `react-plotly.js`, `plotly.js-basic-dist-min` 의존성 추가 |

---

## 4. 상세 구현 계획

### 4.1 [B1] 스키마 추가

```python
class RegionDemandByYear(BaseModel):
    """연도별 지역 사건 수요 총량"""
    region: str
    total_cases: int

class BubbleDataResponse(BaseModel):
    """버블 차트 통합 데이터"""
    cross: list[CrossAnalysisCell]           # 지역 x 전문분야 교차 (버블)
    regions: list[str]                       # X축 라벨
    categories: list[str]                    # Y축 라벨
    demand_by_year: dict[str, list[RegionDemandByYear]]  # 연도별 수요
    available_years: list[int]               # 사용 가능 연도
```

### 4.2 [B2] 엔드포인트 추가

```python
@router.get("/bubble-data", response_model=BubbleDataResponse)
async def get_bubble_data(db: AsyncSession = Depends(get_db)):
    """버블 차트용 통합 데이터"""
    if settings.USE_DB_LAWYERS:
        return await calculate_bubble_data_db(db)
    return calculate_bubble_data()
```

### 4.3 [B3/B4] 서비스 함수

```python
async def calculate_bubble_data_db(db: AsyncSession) -> dict:
    """버블 차트 통합 데이터 계산 (DB 버전)"""
    # 1. 교차분석 데이터 (기존 함수 재활용)
    cross_result = await calculate_cross_analysis_db(db)

    # 2. 연도별 지역 수요 총량 (2015~2024)
    demand_by_year = {}
    for year in range(2015, 2025):
        rows = await db.execute(
            select(
                TrialStatistics.court_name,
                func.sum(TrialStatistics.case_count).label("total_cases"),
            )
            .where(TrialStatistics.year == year)
            .group_by(TrialStatistics.court_name)
        )
        # court_name → province 매핑 후 집계
        demand_by_year[str(year)] = [...]

    return {
        "cross": cross_result["data"],
        "regions": cross_result["regions"],
        "categories": cross_result["categories"],
        "demand_by_year": demand_by_year,
        "available_years": list(range(2015, 2025)),
    }
```

### 4.4 [F3] BubbleChart 컴포넌트 핵심 구조

```tsx
// Plotly.js 2D scatter bubble
import dynamic from 'next/dynamic'

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false })

interface BubbleChartProps {
  data: BubbleDataResponse
}

export function BubbleChart({ data }: BubbleChartProps) {
  const [selectedYear, setSelectedYear] = useState(2024)
  const [userRegion, setUserRegion] = useState<string | null>(null)
  const [userSpecialty, setUserSpecialty] = useState<string | null>(null)

  // 버블 trace: 지역 x 전문분야
  const bubbleTrace = {
    type: 'scatter',
    mode: 'markers',
    x: data.cross.map(d => d.region),
    y: data.cross.map(d => d.category_name),
    marker: {
      size: data.cross.map(d => Math.sqrt(d.count) * scaleFactor),
      color: data.cross.map(d => categoryColorMap[d.category_name]),
      opacity: 0.7,
    },
    text: data.cross.map(d => `${d.region} ${d.category_name}: ${d.count}명`),
    hoverinfo: 'text',
  }

  // 유저 마커 trace
  const userTrace = userRegion && userSpecialty ? {
    type: 'scatter',
    mode: 'markers+text',
    x: [userRegion],
    y: [userSpecialty],
    marker: { size: 16, symbol: 'star', color: '#FF6B6B' },
    text: ['내 위치'],
    textposition: 'top center',
  } : null

  // 배경 히트맵: 선택 연도의 수요
  // → Plotly shapes 또는 heatmap trace로 구현

  return (
    <div>
      {/* 연도 슬라이더 */}
      <input
        type="range"
        min={2015} max={2024}
        value={selectedYear}
        onChange={e => setSelectedYear(Number(e.target.value))}
      />

      {/* 유저 입력 */}
      <select onChange={e => setUserRegion(e.target.value)}>...</select>
      <select onChange={e => setUserSpecialty(e.target.value)}>...</select>

      {/* Plotly 차트 */}
      <Plot
        data={[bubbleTrace, userTrace].filter(Boolean)}
        layout={{ ... }}
        config={{ responsive: true }}
      />
    </div>
  )
}
```

### 4.5 [F5] StickyTabNav 확장

```typescript
export type TabType = 'region' | 'cross' | 'bubble'

const TABS: { id: TabType; label: string }[] = [
  { id: 'region', label: '지역별' },
  { id: 'cross', label: '지역 × 전문분야' },
  { id: 'bubble', label: '분포 시각화' },
]
```

### 4.6 [F6] page.tsx 변경

```tsx
// dynamic import (Plotly ~1MB 번들 분리)
const BubbleChart = dynamic(
  () => import('@/features/lawyer-stats/components/BubbleChart').then(m => m.BubbleChart),
  { ssr: false, loading: DynamicLoadingFallback }
)

// ref 추가
const bubbleSectionRef = useRef<HTMLDivElement>(null)

// useQuery 추가
const bubbleQuery = useQuery({
  queryKey: ['lawyer-stats', 'bubble-data'],
  queryFn: fetchBubbleData,
  enabled: activeTab === 'bubble',  // 탭 활성화 시에만 호출
})

// IntersectionObserver에 bubble-section 추가
// TabType에 'bubble' 추가

// JSX: cross-section 아래에
<section id="bubble-section" ref={bubbleSectionRef} className="scroll-mt-16">
  {bubbleQuery.data && <BubbleChart data={bubbleQuery.data} />}
</section>
```

---

## 5. 팀 구성 및 작업 분배

### 5.1 경량 팀 (B)

| 역할 | 모델 | 담당 | 작업 |
|------|------|------|------|
| **Lead Manager** | Opus | 조율 + 스키마 설계 + 통합 검증 | B1, F1, 최종 검증 |
| **Backend 워커** | Sonnet | API + 서비스 구현 | B2, B3, B4 |
| **Frontend 워커** | Sonnet | 컴포넌트 + 페이지 통합 | F2, F3, F4, F5, F6, F7 |

### 5.2 작업 흐름

```
Phase 1: Lead가 스키마 설계 (B1 + F1)
           ↓ 동시 배포
Phase 2: ┌─ Backend 워커: B2, B3, B4 (API + 서비스)
          └─ Frontend 워커: F2, F3, F4, F5, F6, F7 (컴포넌트 + 페이지)
           ↓ 합류
Phase 3: Lead가 통합 검증
          - Backend: ruff check + mypy
          - Frontend: npm run build
          - 타입 동기화 확인 (B1 ↔ F1)
```

---

## 6. 의존성

### 6.1 NPM 패키지 (신규)

```bash
cd frontend
npm install react-plotly.js plotly.js-basic-dist-min
npm install -D @types/react-plotly.js
```

- `plotly.js-basic-dist-min`: scatter 차트만 포함한 경량 번들 (~1MB)
- `react-plotly.js`: React wrapper

### 6.2 Backend 의존성

추가 패키지 없음. 기존 SQLAlchemy + Pydantic으로 충분.

---

## 7. 검증 계획

### 7.1 정적 검증

```bash
cd backend && uv run ruff check backend/app/modules/lawyer_stats/ backend/app/services/service_function/
cd backend && uv run mypy backend/app/modules/lawyer_stats/ backend/app/services/service_function/
cd frontend && npm run build
```

### 7.2 API 검증

```bash
# 통합 데이터 엔드포인트
curl -s http://localhost:8000/api/lawyer-stats/bubble-data | jq '.regions | length'
# 기대값: 17 (시/도)

curl -s http://localhost:8000/api/lawyer-stats/bubble-data | jq '.available_years'
# 기대값: [2015, 2016, ..., 2024]

curl -s http://localhost:8000/api/lawyer-stats/bubble-data | jq '.cross | length'
# 기대값: 17 x 12 = 204 (지역 x 전문분야)
```

### 7.3 프론트엔드 시각 검증

| # | 시나리오 | 기대 동작 |
|---|---------|---------|
| 1 | 페이지 로드 → '분포 시각화' 탭 클릭 | 버블 차트 렌더링 (17개 지역 x 12개 분야) |
| 2 | 슬라이더 2015→2024 드래그 | 배경 색상 강도 변화 (사건 수요 반영) |
| 3 | 유저 정보 입력 (서울, 민사) | ★ 마커가 서울-민사 위치에 표시 |
| 4 | 버블 호버 | 툴팁: "서울 민사: 5,231명" |
| 5 | 스크롤 → bubble-section 도달 | StickyTabNav '분포 시각화' 탭 활성화 |
| 6 | 모바일 반응형 | 차트 크기 조정, 터치 인터랙션 |

---

## 8. 구현 순서

1. **[B1 + F1]** 스키마/타입 정의 (Lead)
2. **[B2, B3, B4]** Backend API + 서비스 구현 (Backend 워커)
3. **[F7]** NPM 의존성 설치 (Frontend 워커)
4. **[F3]** BubbleChart 컴포넌트 구현 (Frontend 워커)
5. **[F2, F4]** API 서비스 + export (Frontend 워커)
6. **[F5, F6]** StickyTabNav 확장 + page.tsx 통합 (Frontend 워커)
7. Backend 정적 검증 (ruff + mypy)
8. Frontend 빌드 검증 (npm run build)

---

## 변경 이력

| 버전 | 일자 | 변경 내용 |
|------|------|---------|
| v1.0 | 2026-02-27 | 초안 — Plotly.js 2D 버블 차트 + 연도 슬라이더(사건 수요) + 유저 마커 |
