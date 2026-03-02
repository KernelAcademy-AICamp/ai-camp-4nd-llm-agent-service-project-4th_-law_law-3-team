# 법률 뉴스 통계 그래프 수정 기획서

**작성일**: 2026-02-27
**기준 문서**: `docs/03-analysis/legal-news-stats.inspection-report.md`
**수정 출처**: 팀 교차 리뷰 (4건) + 사용자 요구사항 (2건) + 3중 검증 피드백 (2건)

---

## 1. 수정 항목 총괄

| # | 항목 | 출처 | 우선순위 | 대상 파일 |
|---|------|------|:--------:|----------|
| F1 | 막대 그래프: Stacked → Grouped (겹침 해소) | 사용자 | **P0** | `NewsBarChart.tsx` |
| F2 | 막대 그래프: 각 막대 위에 건수 숫자 라벨 표시 | 사용자 | **P0** | `NewsBarChart.tsx` |
| F3 | 도넛 차트: 각 카테고리 슬라이스에 건수 라벨 표시 | 사용자 | **P0** | `NewsDonutChart.tsx` |
| F4 | `get_news_stats_daily` total 중복 쿼리 제거 | 팀 리뷰 | P1 | `service.py` |
| F5 | 라우터 `/{article_id}` 경로 순서 주석 추가 | 팀 리뷰 | P1 | `router/__init__.py` |
| F6 | 막대 그래프 툴팁 `labelFormatter`에 `formatDate` 적용 | 팀 리뷰 | P1 | `NewsBarChart.tsx` |
| F7 | `transformData` 빈 날짜 보간 시 KST 기준 날짜 계산 | 팀 리뷰 | P1 | `NewsBarChart.tsx` |
| F8 | Backend `since` 날짜 정규화 (시각 00:00:00) | Red Team | P1 | `service.py` |
| F9 | 30일 조회 시 막대 라벨 자동 숨김 | Red Team + Consultant | P1 | `NewsBarChart.tsx` |

---

## 2. 상세 수정 명세

### F1. 막대 그래프: Stacked → Grouped (P0)

**현재 상태**: `<Bar stackId="a" .../>` 로 두 소스(로타임즈/네이버) 막대가 겹쳐서 표시됨.
**문제점**: 소스별 수집량 비교가 어렵고, 뒤에 있는 막대가 가려짐.
**수정 방안**: `stackId` prop 제거 → recharts가 자동으로 Grouped 배치.

```tsx
// Before (Stacked)
<Bar dataKey="lawtimes" stackId="a" fill="#3b82f6" radius={[0, 0, 0, 0]} />
<Bar dataKey="naver" stackId="a" fill="#22c55e" radius={[4, 4, 0, 0]} />

// After (Grouped)
<Bar dataKey="lawtimes" fill="#3b82f6" radius={[4, 4, 0, 0]} />
<Bar dataKey="naver" fill="#22c55e" radius={[4, 4, 0, 0]} />
```

### F2. 막대 그래프: 건수 숫자 라벨 (P0)

**현재 상태**: 호버 시 툴팁에서만 건수 확인 가능.
**수정 방안**: recharts `<Bar>` 내 `<LabelList>` 컴포넌트 사용하여 막대 위에 건수 표시.

```tsx
import { LabelList } from 'recharts'

<Bar dataKey="lawtimes" fill="#3b82f6" radius={[4, 4, 0, 0]}>
  <LabelList
    dataKey="lawtimes"
    position="top"
    style={{ fontSize: 10, fill: '#374151' }}
    formatter={(value: number) => (value > 0 ? value : '')}
  />
</Bar>
<Bar dataKey="naver" fill="#22c55e" radius={[4, 4, 0, 0]}>
  <LabelList
    dataKey="naver"
    position="top"
    style={{ fontSize: 10, fill: '#374151' }}
    formatter={(value: number) => (value > 0 ? value : '')}
  />
</Bar>
```

**주의사항**:
- 0건인 경우 빈 문자열 반환하여 라벨 숨김 (시각적 노이즈 방지)
- `position="top"` → Grouped 배치에서 각 막대 바로 위에 표시
- 차트 높이 여유 확보를 위해 `margin.top` 조정 필요 (5 → 15)

### F3. 도넛 차트: 건수 라벨 (P0)

**현재 상태**: 호버 시 툴팁에서만 건수 확인 가능. 범례에 카테고리명만 표시.
**수정 방안**: recharts `<Pie>` 의 `label` prop 사용하여 각 슬라이스에 건수 표시.

```tsx
<Pie
  data={items}
  dataKey="count"
  nameKey="category"
  cx="50%"
  cy="45%"
  innerRadius={55}
  outerRadius={80}
  paddingAngle={2}
  label={({ category, count, percent }) => {
    if (percent < 0.05) return null  // 5% 미만 슬라이스는 라벨 숨김
    return `${category} ${count}건`
  }}
  labelLine={{ stroke: '#9ca3af', strokeWidth: 1 }}
>
```

**주의사항**:
- 비율이 너무 작은 슬라이스(5% 미만)는 라벨이 겹칠 수 있으므로 숨김 처리
- `labelLine` 추가하여 슬라이스와 라벨 연결선 표시
- 차트 높이를 260 → 300으로 조정하여 라벨 공간 확보
- `outerRadius`를 90 → 80으로 줄여 라벨 공간 추가 확보 (Consultant 필수 권고)

### F4. total 중복 쿼리 제거 (P1)

**현재 상태**: `get_news_stats_daily()` 에서 일별 통계 쿼리 + 별도 count 쿼리 → DB 라운드트립 2회.
**수정 방안**: items 결과의 count 합산으로 total 계산 (DB 쿼리 1회로 축소).

```python
# Before
total_query = select(func.count()).select_from(
    select(NewsArticle.id).where(NewsArticle.published_at >= since).subquery()
)
total = (await db.execute(total_query)).scalar_one()

# After
total = sum(item.count for item in items)
```

### F5. 라우터 경로 순서 주석 (P1)

**현재 상태**: `/{article_id}` path parameter 라우트가 `/stats/daily`, `/stats/category` 뒤에 위치하지만 주석 없음.
**수정 방안**: FastAPI path parameter 매칭 순서 관련 주석 추가.

```python
# NOTE: /{article_id} 는 반드시 정적 경로(/list, /stats/*, /search) 뒤에 위치해야 함.
# FastAPI는 경로를 등록 순서대로 매칭하므로, 이 경로가 앞에 있으면
# /stats/daily 등이 article_id="stats"로 잘못 매칭됨.
@router.get("/{article_id}", ...)
```

### F6. 툴팁 labelFormatter 날짜 포맷 (P1)

**현재 상태**: 툴팁 헤더에 `String(label)` → 원본 날짜 문자열(2026-02-24) 그대로 표시.
**수정 방안**: `formatDate` 함수를 `labelFormatter`에 적용.

```tsx
// Before
labelFormatter={(label) => String(label)}

// After
labelFormatter={formatDate}
```

### F7. KST 기준 날짜 보간 (P1)

**현재 상태**: `new Date()` → UTC 기준 → `toISOString().slice(0, 10)` → UTC 날짜.
  백엔드 KST(UTC+9) 기준과 불일치 가능 (특히 자정~09시 구간).
**수정 방안**: KST 오프셋을 명시적으로 적용.

```typescript
// Before
const today = new Date()
// ...
const dateStr = d.toISOString().slice(0, 10)

// After (KST 명시 적용)
function toKSTDateString(date: Date): string {
  const kst = new Date(date.getTime() + 9 * 60 * 60 * 1000)
  return kst.toISOString().slice(0, 10)
}

const today = new Date()
// ...
const dateStr = toKSTDateString(d)
```

### F8. Backend `since` 날짜 정규화 (P1) — Red Team 피드백

**현재 상태**: `since = datetime.now(kst) - timedelta(days=days)` → 현재 시각 기준 슬라이딩 윈도우.
  오후 2시에 조회 시 첫 날짜가 오후 2시 이후 데이터만 포함 → 수집량 급감으로 오해.
**수정 방안**: `since`를 해당 날짜 00:00:00으로 정규화.

```python
# Before
since = datetime.now(kst) - timedelta(days=days)

# After
since = (datetime.now(kst) - timedelta(days=days)).replace(
    hour=0, minute=0, second=0, microsecond=0
)
```

### F9. 30일 조회 시 막대 라벨 숨김 (P1) — Red Team + Consultant 합의

**현재 상태**: F2 적용 시 30일 조회 → 60개 막대(30일 × 2소스) → 라벨 겹침.
**수정 방안**: `periodDays > 14`일 때 `LabelList` 렌더링 비활성화.

```tsx
// periodDays를 props로 전달받아 조건부 렌더링
<Bar dataKey="lawtimes" fill="#3b82f6" radius={[4, 4, 0, 0]}>
  {periodDays <= 14 && (
    <LabelList
      dataKey="lawtimes"
      position="top"
      style={{ fontSize: 10, fill: '#374151' }}
      formatter={(value: number) => (value > 0 ? value : '')}
    />
  )}
</Bar>
```

---

## 3. 변경 파일 요약

| 파일 | 수정 항목 | 변경 규모 |
|------|----------|----------|
| `frontend/src/features/legal-news/components/NewsBarChart.tsx` | F1, F2, F6, F7, F9 | 중 (5건 통합 수정) |
| `frontend/src/features/legal-news/components/NewsDonutChart.tsx` | F3 | 소 (라벨 추가) |
| `backend/app/modules/legal_news/service.py` | F4, F8 | 소 (6줄 삭제, 3줄 추가) |
| `backend/app/modules/legal_news/router/__init__.py` | F5 | 소 (주석 3줄 추가) |

---

## 4. 검증 계획

| 단계 | 검증 내용 | 명령어/방법 |
|------|----------|-----------|
| 1 | Backend 정적 분석 | `cd backend && uv run ruff check app/ && uv run mypy app/` |
| 2 | Frontend 빌드 | `cd frontend && npm run build` |
| 3 | API 응답 확인 | `curl localhost:8000/api/legal-news/stats/daily?days=7` |
| 4 | 막대 그래프 7일 | Playwright: Grouped 배치 + 숫자 라벨 표시 확인 |
| 5 | 막대 그래프 30일 | Playwright: 라벨 숨김 + 막대 가독성 확인 |
| 6 | 도넛 차트 라벨 | Playwright: 카테고리별 건수 라벨 + 5% 미만 숨김 확인 |
| 7 | 모바일 뷰 (390px) | 도넛 라벨 겹침 없는지 확인 |

---

## 5. 리스크 분석

| 리스크 | 영향도 | 대응 |
|--------|:------:|------|
| Grouped 배치 시 30일 기간에 막대가 좁아짐 | 중 | F9: 14일 초과 시 라벨 숨김으로 가독성 유지 |
| 도넛 라벨이 작은 슬라이스에서 겹침 | 중 | 5% 미만 슬라이스 라벨 숨김 + outerRadius 축소 |
| 첫 날짜 데이터 왜곡 (슬라이딩 윈도우) | 중 | F8: since를 00:00:00으로 정규화 |
| KST 변환이 서머타임 없는 한국에서만 유효 | 저 | 한국 전용 서비스이므로 문제 없음 |
| LabelList import 추가 시 번들 크기 | 저 | recharts 기존 import에서 추가, tree-shaking 적용됨 |

---

## 6. 3중 검증 피드백 반영 내역

### Red Team (Gemini CLI) 피드백

| 지적 사항 | 심각도 | 반영 |
|----------|:------:|------|
| `since` 날짜 정규화 (시각 00:00:00) 필요 | Medium | **F8로 추가** |
| 30일 조회 시 60개 막대 라벨 겹침 | Medium | **F9로 추가** |
| `/{article_id}` 경로에 regex 제약 추가 | Low | 백로그 (기존 F5 주석으로 1차 대응) |
| LabelList value threshold | Low | F2/F9에서 0건 숨김 + 14일 초과 숨김으로 대응 |

### External Consultant (Codex CLI) 피드백

| 지적 사항 | 심각도 | 반영 |
|----------|:------:|------|
| 모바일 도넛 라벨 겹침 위험 → height 300, outerRadius 80 | Medium | **F3에 반영** |
| 30일 구간 `margin.top`, `barSize`, `minTickGap` 조정 | Medium | **F9에 반영** |
| 차트 접근성 (role="img", aria-label, aria-pressed) | Low | 백로그 (P2) |
| 날짜 포맷 함수 통합 (formatChartDate) | Low | 백로그 (P2) |
| 라벨 충돌 테스트 기준 명시 | Medium | 검증 계획에 반영 (단계 5, 7) |

### 백로그 (본 수정에서 제외, 별도 추적)

| 항목 | 우선순위 | 출처 |
|------|:--------:|------|
| `/{article_id}` 경로 regex/UUID 제약 | P2 | Red Team |
| 차트 접근성 ARIA 속성 | P2 | Consultant |
| 날짜 포맷 함수 통합 (`formatChartDate`) | P2 | Consultant |
| 카테고리 분류 Python/SQL SSOT 리팩토링 | P1 | 팀 리뷰 |
| `func.date(published_at)` 함수형 인덱스 | P2 | Red Team + Consultant |
| 통계 API `sources` 메타 추가 | P2 | Consultant |
