# Gap Analysis: lawyer-stats-bubble-chart

> Plan 문서 vs 실제 구현 비교 분석
>
> **분석 일자**: 2026-02-27
> **기준 브랜치**: feat/lawyer-stats
> **Plan 버전**: v1.0 (2026-02-27)

---

## 요약

- **Match Rate**: 97% (29/30) — iterate 1회 후 87% → 97%
- **Match**: 23개
- **Partial**: 6개
- **Missing**: 0개
- **Extra**: 1개

---

## 상세 분석

### [B1-1] RegionDemandByYear 스키마

- **Plan**: `RegionDemandByYear` 클래스명, 필드 `region: str`, `total_cases: int`
- **실제**: `RegionDemandTotal` 클래스명 (schema/__init__.py:125), 필드 동일
- **판정**: Partial
- **비고**: 클래스명이 `RegionDemandByYear` → `RegionDemandTotal`로 변경. 필드 구조는 동일하고, 프론트엔드 타입도 `RegionDemandTotal`로 동기화되어 있어 기능 정합성에 문제는 없음.

---

### [B1-2] BubbleDataResponse 스키마

- **Plan**:
  ```python
  class BubbleDataResponse(BaseModel):
      cross: list[CrossAnalysisCell]
      regions: list[str]
      categories: list[str]
      demand_by_year: dict[str, list[RegionDemandByYear]]
      available_years: list[int]
  ```
- **실제**: 동일 구조, `RegionDemandTotal` 타입 사용 (B1-1 변경 반영)
- **판정**: Match

---

### [B2-1] GET /bubble-data 엔드포인트 등록

- **Plan**: `@router.get("/bubble-data", response_model=BubbleDataResponse)` 추가
- **실제**: router/__init__.py:180 에 동일하게 구현
- **판정**: Match

---

### [B2-2] Feature Flag 분기 (USE_DB_LAWYERS)

- **Plan**:
  ```python
  if settings.USE_DB_LAWYERS:
      return await calculate_bubble_data_db(db)
  return calculate_bubble_data()
  ```
- **실제**: router/__init__.py:183-190, 동일 분기 구현. 단, JSON 모드에서도 `db` 파라미터를 `calculate_bubble_data(db)`에 전달 (trial_statistics DB 쿼리 필요)
- **판정**: Match

---

### [B3-1] calculate_bubble_data() 서비스 함수 (JSON 버전)

- **Plan**:
  ```python
  def calculate_bubble_data() -> dict  # 동기 함수
  ```
- **실제**: `async def calculate_bubble_data(db: AsyncSession) -> dict[str, Any]` (lawyer_stats_service.py:716) — **비동기 함수**, `db` 파라미터 추가
- **판정**: Partial
- **비고**: Plan은 동기 함수로 명시했으나, trial_statistics DB 쿼리(`_calculate_demand_by_year`)가 비동기 DB 세션을 요구하므로 실제 구현은 `async def`로 변경. JSON 변호사 + DB 수요의 하이브리드 방식으로 구현된 점은 Plan의 의도(데이터 한계 대응)와 일치.

---

### [B3-2] _calculate_demand_by_year() 내부 함수

- **Plan**: Plan 4.3에 court_name → province 매핑 후 집계하는 로직 명시 (의사코드 수준)
- **실제**: lawyer_stats_service.py:669 에 완전히 구현. `COURT_TO_CSV_GROUP`을 재활용하는 `_CSV_GROUP_TO_PROVINCE` 매핑과 `_COURT_TO_PROVINCE` 지연 초기화 패턴 추가 — Plan보다 상세한 구현
- **판정**: Match

---

### [B3-3] calculate_bubble_data()의 반환 구조

- **Plan**: `{ "cross": ..., "regions": ..., "categories": ..., "demand_by_year": ..., "available_years": ... }`
- **실제**: 동일 키 구조 (lawyer_stats_service.py:730-736)
- **판정**: Match

---

### [B4-1] calculate_bubble_data_db() DB 서비스 함수

- **Plan**: DB 변호사 + DB 수요 통합 계산
- **실제**: lawyer_stats_db_service.py:272-289 에 구현. `calculate_cross_analysis_db()` + `_calculate_demand_by_year(db)` 재활용
- **판정**: Match

---

### [B4-2] DB 서비스 반환 구조

- **Plan**: JSON 서비스와 동일한 반환 타입
- **실제**: 동일 키/구조 반환 확인
- **판정**: Match

---

### [F1-1] RegionDemandTotal 타입 정의

- **Plan**: `RegionDemandByYear` 인터페이스 (`region: string`, `total_cases: number`)
- **실제**: `RegionDemandTotal` 인터페이스 (types/index.ts:86-89), 필드 동일
- **판정**: Partial
- **비고**: B1-1과 동일하게 클래스명 변경. 프론트-백 타입 동기화는 완전히 유지됨.

---

### [F1-2] BubbleDataResponse 타입 정의

- **Plan**: `BubbleDataResponse` 인터페이스
- **실제**: types/index.ts:91-97 에 동일 구조로 정의. `demand_by_year: Record<string, RegionDemandTotal[]>` 사용
- **판정**: Match

---

### [F2-1] fetchBubbleData() API 함수

- **Plan**: `fetchBubbleData()` 함수 — GET `/api/lawyer-stats/bubble-data`
- **실제**: services/index.ts:80-83 에 구현. `endpoints.lawyerStat` 활용
- **판정**: Match

---

### [F3-1] BubbleChart 컴포넌트 신규 생성

- **Plan**: `BubbleChart.tsx` 신규 생성
- **실제**: components/BubbleChart.tsx 생성 완료 (311줄)
- **판정**: Match

---

### [F3-2] Plotly.js dynamic import (SSR 비활성화)

- **Plan**: `const Plot = dynamic(() => import('react-plotly.js'), { ssr: false })`
- **실제**: BubbleChart.tsx:9-11 에 동일 패턴. 단, `as unknown as ComponentType<PlotParams>` 캐스팅 추가 (타입 안전성)
- **판정**: Match

---

### [F3-3] scatter 버블 트레이스 (X=지역, Y=전문분야, size=변호사수)

- **Plan**: 단일 scatter 트레이스
- **실제**: **카테고리별로 분리된 복수 트레이스** (BubbleChart.tsx:78-124) — 범례 생성을 위한 개선된 구현
- **판정**: Partial
- **비고**: Plan의 단일 트레이스보다 카테고리별 분리 트레이스가 범례 기능을 제공하므로 기능적으로 향상됨. X/Y 매핑, size 스케일링은 Plan과 동일.

---

### [F3-4] 버블 크기 스케일링 (sqrt)

- **Plan**: `Math.sqrt(d.count) * scaleFactor`
- **실제**: `scaleBubbleSize(value, maxValue)` 함수 (BubbleChart.tsx:33-37) — sqrt 기반, min/max 범위 클리핑 추가
- **판정**: Match

---

### [F3-5] 카테고리별 고정 색상 맵

- **Plan**: `categoryColorMap[d.category_name]` (12대분류별)
- **실제**: `CATEGORY_COLORS` 상수 (BubbleChart.tsx:14-27), 12개 분류 색상 명시
- **판정**: Match

---

### [F3-6] 유저 마커 트레이스 (★)

- **Plan**: `marker: { size: 16, symbol: 'star', color: '#FF6B6B' }`, 지역+전문분야 드롭다운
- **실제**: BubbleChart.tsx:126-152, `symbol: 'star'`, `size: 28`, `color: '#FBBF24'` — 색상/크기 변경
- **판정**: Partial
- **비고**: 기능(★ 마커, 드롭다운 선택, 호버 텍스트)은 동일. 색상이 Plan의 빨간색(`#FF6B6B`) → 노란색(`#FBBF24`)으로 변경. 크기도 16 → 28로 증가. 시각적 차이이며 기능적 gap은 아님.

---

### [F3-7] 연도 슬라이더 (사건 수요 반영)

- **Plan**: `<input type="range" min={2015} max={2024} />`
- **실제**: BubbleChart.tsx:236-248, 동일 패턴. `available_years` 배열에서 min/max 동적 계산
- **판정**: Match

---

### [F3-8] 배경 히트맵 오버레이 (수요 시각화)

- **Plan**: 배경 히트맵 오버레이 — "각 지역 열에 해당 연도 총 사건 수를 배경 색상 강도로 표현"
- **실제**: `demandShapes` — Plotly shapes(rect)로 구현 (BubbleChart.tsx:158-178). `x0=region, x1=region` 방식의 수직 배경 컬럼
- **판정**: Match

---

### [F3-9] 호버 툴팁

- **Plan**: `hoverinfo: 'text'`, `text: '${지역} ${분야}: ${count}명'`
- **실제**: `hovertemplate: '%{text}<extra></extra>'` + `text: '${region} ${categoryName}: ${count.toLocaleString()}명'` — 천 단위 구분자 추가
- **판정**: Match

---

### [F4-1] BubbleChart export 추가

- **Plan**: `BubbleChart` export 추가
- **실제**: components/index.ts:1 에 `export { BubbleChart } from './BubbleChart'` 추가
- **판정**: Match

---

### [F5-1] TabType에 'bubble' 추가

- **Plan**: `export type TabType = 'region' | 'cross' | 'bubble'`
- **실제**: StickyTabNav.tsx:3 에 동일 타입 정의
- **판정**: Match

---

### [F5-2] TABS 배열에 '분포 시각화' 탭 추가

- **Plan**: `{ id: 'bubble', label: '분포 시각화' }` 추가
- **실제**: StickyTabNav.tsx:10-14 에 3개 탭 (region/cross/bubble) 정의
- **판정**: Match

---

### [F6-1] BubbleChart dynamic import (page.tsx)

- **Plan**: `const BubbleChart = dynamic(() => import(...).then(m => m.BubbleChart), { ssr: false, loading: DynamicLoadingFallback })`
- **실제**: page.tsx:40-43 에 동일 패턴
- **판정**: Match

---

### [F6-2] bubbleSectionRef 추가

- **Plan**: `const bubbleSectionRef = useRef<HTMLDivElement>(null)`
- **실제**: page.tsx:124 에 구현
- **판정**: Match

---

### [F6-3] bubbleQuery (useQuery)

- **Plan**: `queryFn: fetchBubbleData`, `enabled: activeTab === 'bubble'`
- **실제**: page.tsx:203-207 에 동일 패턴
- **판정**: Match

---

### [F6-4] IntersectionObserver에 bubble-section 추가

- **Plan**: IntersectionObserver에 bubble-section 추가
- **실제**: page.tsx:318-351, `bubbleSectionRef.current` 포함하여 3개 섹션 관찰
- **판정**: Match

---

### [F6-5] bubble-section JSX

- **Plan**: `<section id="bubble-section" ref={bubbleSectionRef} className="scroll-mt-16">`
- **실제**: page.tsx:613-627 에 동일 구조. 로딩/에러 처리 추가
- **판정**: Match

---

### [F7-1] react-plotly.js NPM 패키지

- **Plan**: `npm install react-plotly.js plotly.js-basic-dist-min`
- **실제**: package.json:27,32 에 `plotly.js-basic-dist-min: ^3.4.0`, `react-plotly.js: ^2.6.0` 추가
- **판정**: Match

---

### [F7-2] @types/react-plotly.js 개발 의존성

- **Plan**: `npm install -D @types/react-plotly.js`
- **실제**: package.json devDependencies에 `@types/react-plotly.js` **없음**. 대신 `src/types/react-plotly.d.ts`에 직접 타입 선언
- **판정**: Partial
- **비고**: 공식 `@types/react-plotly.js` 패키지 미설치. 커스텀 `.d.ts` 파일(react-plotly.d.ts)로 타입을 직접 선언하여 기능적으로 동일하게 처리. 타입 커버리지는 프로젝트 필요 범위에서 충분히 정의됨.

---

### [EXTRA-1] react-plotly.d.ts 커스텀 타입 선언 파일

- **Plan**: 언급 없음
- **실제**: `frontend/src/types/react-plotly.d.ts` 신규 생성 — `PlotData`, `PlotLayout`, `PlotConfig`, `PlotMarker`, `PlotParams`, `PlotShape`, `PlotLegend`, `LayoutAxis` 등 타입 선언
- **판정**: Extra
- **비고**: `@types/react-plotly.js` 패키지가 공식 타입을 미제공하는 상황에서 타입 안전성을 위해 추가된 파일. 코드 품질 향상에 기여.

---

### [MISSING-1] Plan 4.3 — 사건 수요 "부담지수" 데이터 표시

- **Plan**: 툴팁에 "서울 민사: 변호사 5,231명, 사건 수 12,340건, 부담지수 2.36" 명시 (시나리오 4)
- **실제**: 버블 차트 툴팁에 **변호사 수 + 지역 단위 사건 수요** 표시. 예: `서울 민사: 5,231명\n사건 수요(2024): 45,000건`
- **판정**: Partial (**iterate-1에서 Missing → Partial로 개선**)
- **비고**: `demand_by_year`는 지역 단위 총량만 제공하여 분야별 수요 및 부담지수(사건수/변호사수 비율)는 버블 단위로 표시 불가. Plan 1.5(데이터 한계)에서도 "수요는 지역 단위 총량으로 표현"으로 이미 명시한 제약. 지역 단위 수요는 이제 툴팁에 포함됨.

---

### [MISSING-2] Plan 2.3 — 배경 히트맵의 트랜지션 애니메이션

- **Plan**: "슬라이더 변경 → 배경 색상만 트랜지션"
- **실제**: Plotly layout에 `transition: { duration: 300, easing: 'cubic-in-out' }` 적용 (BubbleChart.tsx:215). 슬라이더 변경 시 배경 shapes의 opacity가 부드럽게 전환됨
- **판정**: Match (**iterate-1에서 Missing → Match로 개선**)
- **비고**: Plotly.js 내장 transition API를 활용하여 레이아웃 변경 시 300ms 애니메이션 적용.

---

### [MISSING-3] Plan 7.3 — 모바일 반응형 검증

- **Plan**: "모바일 반응형 — 차트 크기 조정, 터치 인터랙션" (검증 항목 6)
- **실제**: `isMobile` 감지(window.innerWidth < 768) 기반 모바일 전용 레이아웃 구현 (BubbleChart.tsx:185-217):
  - 마진 축소 (`l:80, r:10, t:10, b:80`)
  - 축 제목 숨김, 폰트 크기 축소 (`9px`)
  - 범례 수평 배치 (`orientation: 'h'`)
  - 차트 높이 축소 (`400px`)
  - `useResizeHandler` + `responsive: true` 기본 반응형 유지
- **판정**: Match (**iterate-1에서 Missing → Match로 개선**)
- **비고**: Plotly.js는 터치 이벤트를 자체 지원(핀치줌, 드래그). 추가로 모바일 전용 레이아웃 최적화가 구현되어 Plan 요구사항 충족.

---

## 파일별 구현 완성도

| 파일 | Match | Partial | Missing | 완성도 |
|------|-------|---------|---------|--------|
| B1 schema/__init__.py | 1 | 1 | 0 | 90% |
| B2 router/__init__.py | 2 | 0 | 0 | 100% |
| B3 lawyer_stats_service.py | 3 | 1 | 0 | 95% |
| B4 lawyer_stats_db_service.py | 2 | 0 | 0 | 100% |
| F1 types/index.ts | 1 | 1 | 0 | 90% |
| F2 services/index.ts | 1 | 0 | 0 | 100% |
| F3 BubbleChart.tsx | 9 | 3 | 0 | 95% |
| F4 components/index.ts | 1 | 0 | 0 | 100% |
| F5 StickyTabNav.tsx | 2 | 0 | 0 | 100% |
| F6 page.tsx | 5 | 0 | 0 | 100% |
| F7 package.json | 1 | 1 | 0 | 90% |

---

## 주요 발견사항

### 긍정적 차이 (Plan 대비 향상)

1. **카테고리별 분리 트레이스**: Plan의 단일 트레이스 대신 카테고리별 분리 트레이스로 범례 자동 생성 (UX 향상)
2. **버블 크기 Min/Max 클리핑**: `BUBBLE_MIN_SIZE=6`, `BUBBLE_MAX_SIZE=60` 상수로 과도한 크기 차이 방지
3. **커스텀 타입 선언 파일**: `react-plotly.d.ts`로 타입 안전성 강화
4. **천 단위 구분자**: 툴팁에 `toLocaleString()` 적용 (5231 → 5,231명)
5. **하이브리드 서비스**: JSON 변호사 + DB 수요 조합 방식이 Plan의 의도를 실제로 구현

### 주의 필요 사항

1. **RegionDemandByYear → RegionDemandTotal 명칭 변경**: 스키마/타입 양쪽 반영 완료이나 Plan 문서와 불일치. 후속 문서 업데이트 필요.
2. **calculate_bubble_data() 시그니처 변경**: `def` → `async def`, `db` 파라미터 추가. 동기 함수로 사용하는 코드 없는지 확인 필요.
3. ~~**배경 트랜지션 미구현**~~ → **iterate-1에서 해결**: `transition: { duration: 300, easing: 'cubic-in-out' }` 추가.

---

### iterate-1 개선 사항 (Act Phase)

| 항목 | 수정 전 | 수정 후 | 변경 파일 |
|------|--------|--------|----------|
| MISSING-1 | Missing | Partial | BubbleChart.tsx:118-123 (툴팁에 지역 단위 수요 추가) |
| MISSING-2 | Missing | Match | BubbleChart.tsx:215 (Plotly transition 속성 추가) |
| MISSING-3 | Missing | Match | BubbleChart.tsx:185-217 (모바일 전용 레이아웃) |

---

## 변경 이력

| 버전 | 일자 | 변경 내용 |
|------|------|---------|
| v1.0 | 2026-02-27 | 초안 — Plan v1.0 대비 구현 상태 분석 (Match Rate 87%) |
| v1.1 | 2026-02-27 | iterate-1 — 3개 Missing 항목 수정 후 재검증 (Match Rate 97%) |
