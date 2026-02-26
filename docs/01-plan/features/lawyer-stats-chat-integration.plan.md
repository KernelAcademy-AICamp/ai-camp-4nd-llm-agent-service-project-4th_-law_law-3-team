# Lawyer Stats Chat Integration Plan

> **Summary**: 신입 변호사 대상 시장 진입 인사이트 제공 — 채팅으로 "어디에 개업할지, 무슨 전문직종이 좋을지" 질문하면, 다중 통계 데이터를 조합하여 전략적 추천 + 대시보드 시각화 연동
>
> **Project**: law-3-team (법률 서비스 플랫폼)
> **Author**: Claude
> **Date**: 2026-02-26
> **Status**: Draft (v2.1)
> **Branch**: feat/lawyer-stats

---

## 1. Overview

### 1.1 Purpose

**핵심 대상**: 신입 변호사 (변호사 시험 합격자, 개업 준비 중인 변호사)
**핵심 가치**: 단순 통계 조회가 아닌, **시장 진입 전략 추천** (어디에, 무슨 분야로)

현재 `LawyerStatsAgent`는 단순 네비게이션 에이전트다. 이를 **시장 분석 어드바이저**로 업그레이드하여:

1. 사용자의 자연어 질문을 LLM으로 분석 (단순 통계 조회 vs 전략적 추천)
2. **다중 데이터 소스를 동시 조합** (밀도 + 전문분야 + 수요 + 교차분석)
3. 데이터 기반 **비즈니스 인사이트/추천** 생성 ("경쟁 낮고 수요 높은 지역은...")
4. 추천 결과에 맞는 대시보드 필터를 자동 연동

### 1.2 Background

- 기존 `LawyerStatsAgent`: RAG/LLM 미사용, `/lawyer-stats`로 NAVIGATE만 반환
- 통계 API 및 서비스 함수 완성 (overview, region, density, specialty, cross, demand)
- 프론트엔드 대시보드 컴포넌트 완성 (지도, 히트맵, 바차트 등)
- **데이터 한계**: 지역 단위는 시/도(17개)까지 가능, 구 단위는 불가

### 1.3 핵심 사용자 시나리오

```
시나리오 1 (추천형): "새로 변호사 시험 합격했어. 어디에 개업할지, 무슨 전문직종이 좋을지 추천해줘"
→ query_type: "recommend_market"
→ 다중 데이터 조회: density + specialty + demand + cross
→ LLM이 조합 분석:
  "현재 시장 분석 결과를 바탕으로 추천드립니다.
   **경쟁 대비 수요가 높은 지역 TOP 3:**
   1. 제주 — 변호사 밀도 XX, 사건 부담지수 YY (가장 높음)
   2. 강원 — ...
   3. 충남 — ...
   **블루오션 전문분야:**
   - 행정법: 전체 변호사 중 X%만 전문, 행정소송 건수 증가 추세
   - 가사: ..."
→ 대시보드: 밀도 모드로 전환, 추천 지역 하이라이트

시나리오 2 (지역 특화): "서울에 진입하고 싶은데 무슨 전문직종이 좋을까"
→ query_type: "recommend_specialty"
→ 다중 데이터 조회: cross_analysis(서울) + demand(서울, 전분야) + density(서울)
→ LLM이 분석:
  "서울 시장 분석 결과입니다.
   **서울 변호사 현황:** 총 XX명, 인구 10만명당 YY명 (전국 1위)
   **포화 분야 (경쟁 과열):** 민사(XX%), 형사(XX%)
   **추천 분야 (상대적 기회):**
   1. 행정법 — 서울 변호사 중 X%만 전문, 행정소송 건수 XX건
   2. 가사 — 가사소송 증가 추세, 전문 변호사 비율 낮음
   3. ..."
→ 대시보드: 서울 선택 + 교차분석 탭 + 서울 하이라이트

시나리오 3 (단순 조회): "서울 변호사 밀도 알려줘"
→ query_type: "density" (기존과 동일)
→ 단일 데이터 조회: density(서울)
→ "서울의 인구 10만명당 변호사 수는 XX명입니다..."
→ 대시보드: 서울 + 밀도 모드

시나리오 4 (수요 분석): "형사 사건이 많은 지역은?"
→ query_type: "demand"
→ 데이터 조회: demand(형사)
→ "형사 사건 접수 기준 상위 지역은..."
→ 대시보드: 수요 모드 + 형사
```

---

## 2. 아키텍처 설계

### 2.1 핵심 개념: 질문 유형 2계층 분류

```
사용자 질문
  │
  ├── 추천형 (Recommendation) ────── 다중 데이터 조합 + 인사이트 생성
  │   ├── recommend_market: "어디에 개업?" → 전 지역 비교
  │   ├── recommend_specialty: "무슨 분야?" (지역 지정) → 특정 지역 분야 분석
  │   └── recommend_region: "이 분야로 어디?" (분야 지정) → 특정 분야 지역 분석
  │
  └── 조회형 (Query) ────── 단일 데이터 조회 + 요약 서술
      ├── overview: 전체 현황
      ├── region: 지역별 수
      ├── density: 밀도
      ├── specialty: 전문분야별
      ├── cross: 교차분석
      ├── demand: 수요
      └── prediction: 예측
```

### 2.2 데이터 흐름

```
사용자 메시지
  │
  ▼
[router_node] → lawyer_stats_node
  │
  ▼
[LawyerStatsAgent.process()]
  │
  ├─ Step 1: LLM 의도 분석 → StatsIntent
  │   { query_type: "recommend_market", regions: [], ... }
  │
  ├─ Step 2: 데이터 조회 (query_type에 따라 단일 or 다중)
  │   ├── 추천형: 여러 서비스 함수 동시 호출 (asyncio.gather)
  │   │   → { overview, density, specialty, cross, demand }
  │   └── 조회형: 해당 서비스 함수 1개 호출
  │       → { density_data } 등
  │
  ├─ Step 3: LLM 응답 생성
  │   ├── 추천형: 시장 분석 + 전략적 추천 프롬프트
  │   └── 조회형: 데이터 요약 서술 프롬프트
  │
  └─ Step 4: AgentResult 구성
      ├─ message: 분석/추천 텍스트
      ├─ session_data.stats_filter: 대시보드 필터 상태
      └─ actions: NAVIGATE → /lawyer-stats
  │
  ▼
[Frontend] ChatWidget → sessionData → lawyer-stats page
  └─ stats_filter 적용 → 대시보드 상태 변경
```

### 2.3 StatsIntent 구조

```python
class StatsIntent(BaseModel):
    """LLM이 파싱한 통계 요청 의도"""

    # 질문 유형 (추천형 3 + 조회형 7 = 10가지)
    query_type: str
    # "recommend_market" | "recommend_specialty" | "recommend_region"
    # "overview" | "region" | "density" | "specialty" | "cross" | "demand" | "prediction"

    # 지역 필터
    regions: list[str] = []          # 관심 지역 (약칭: "서울", "부산")
    province: str | None = None      # 시/도 (약칭)

    # 분야 필터
    specialty_interest: str | None = None  # 관심 전문분야 (예: "행정", "가사")

    # 뷰 설정
    view_mode: str | None = None
    indicator_group: str | None = None
    active_tab: str | None = None
    prediction_year: int | None = None
    demand_category: str | None = None
    demand_year: int | None = None
```

### 2.4 추천형 데이터 조합 전략

| query_type | 필요 데이터 | 서비스 함수 (동시 호출) | 분석 관점 |
|------------|-----------|------|---------|
| `recommend_market` | 밀도 + 전문분야 + 수요 | `density()` + `specialty()` + `demand()` + `cross()` | 경쟁 낮고 수요 높은 지역+분야 |
| `recommend_specialty` | 특정 지역의 교차분석 + 수요 | `cross_by_province(지역)` + `demand(전분야)` + `density()` | 해당 지역에서 기회 분야 |
| `recommend_region` | 특정 분야의 지역별 분포 + 수요 | `cross()` + `density()` + `demand(분야)` | 해당 분야에서 기회 지역 |

### 2.5 StatsFilter 구조 (프론트엔드 전달)

```typescript
interface StatsFilter {
  viewMode?: 'count' | 'density' | 'prediction' | 'case_count' | 'burden_index'
  indicatorGroup?: 'supply' | 'demand'
  activeTab?: 'region' | 'cross'
  selectedProvince?: string | null
  predictionYear?: 2030 | 2035 | 2040
  demandCategory?: string
  demandYear?: number
  crossRegions?: string[]
}
```

---

## 3. 변경 파일 목록

### 3.1 Backend 변경

| # | 파일 | 변경 유형 | 설명 |
|---|------|---------|------|
| B1 | `backend/app/multi_agent/agents/lawyer_stats_agent.py` | **전면 재작성** | 시장 분석 어드바이저 에이전트 (2계층 의도분석 + 다중 데이터 조합 + 추천/조회 응답) |
| B2 | `backend/app/multi_agent/nodes.py` | 소폭 수정 | `lawyer_stats_node`: nonstreaming → streaming 전환 |
| B3 | `backend/app/modules/lawyer_stats/schema/__init__.py` | 추가 | `StatsIntent` Pydantic 모델 |

### 3.2 Frontend 변경

| # | 파일 | 변경 유형 | 설명 |
|---|------|---------|------|
| F1 | `frontend/src/features/lawyer-stats/types/index.ts` | 추가 | `StatsFilter` 인터페이스 |
| F2 | `frontend/src/app/lawyer-stats/page.tsx` | 수정 | `sessionData.stats_filter` 감지 → 필터 자동 적용 |

---

## 4. 상세 구현 계획

### 4.1 [B1] LawyerStatsAgent 전면 재작성

```python
class LawyerStatsAgent(BaseChatAgent):
    """변호사 시장 분석 어드바이저"""

    name = "lawyer_stats"
    description = "변호사 시장 분석 및 개업 전략 추천"
    supports_streaming = True

    async def process(self, message, history, session_data, user_location):
        # 1. 의도 분석
        intent = await self._parse_intent(message, history)

        # 2. 데이터 조회 (추천형: 다중, 조회형: 단일)
        stats_data = await self._fetch_stats(intent)

        # 3. 응답 생성 (추천형: 인사이트, 조회형: 요약)
        analysis = await self._generate_response(message, intent, stats_data)

        # 4. 대시보드 필터 구성
        stats_filter = self._build_filter(intent)

        return AgentResult(...)

    async def _fetch_stats(self, intent: StatsIntent) -> dict[str, Any]:
        """의도에 따라 단일/다중 데이터 조회"""
        if intent.query_type.startswith("recommend"):
            return await self._fetch_recommendation_data(intent)
        return await self._fetch_query_data(intent)

    async def _fetch_recommendation_data(self, intent) -> dict:
        """추천형: 다중 서비스 함수 동시 호출"""
        # USE_DB_LAWYERS 분기 + AsyncSession 관리
        # asyncio.gather로 병렬 조회
        tasks = []
        tasks.append(self._get_density_data())
        tasks.append(self._get_specialty_data())
        tasks.append(self._get_demand_data(category=intent.demand_category or "민사"))

        if intent.province:
            tasks.append(self._get_cross_by_province(intent.province))
        else:
            tasks.append(self._get_cross_data())

        results = await asyncio.gather(*tasks)
        return {
            "density": results[0],
            "specialty": results[1],
            "demand": results[2],
            "cross": results[3],
        }
```

### 4.2 [B1] LLM 프롬프트 전략 (2종)

#### 추천형 프롬프트 (핵심)

```
당신은 신입 변호사를 위한 시장 분석 어드바이저입니다.
통계 데이터를 바탕으로 개업 전략을 추천하세요.

## 분석 관점
- **경쟁 강도**: 인구 대비 변호사 밀도가 낮은 = 경쟁 적음
- **수요 기회**: 사건 부담지수(사건수/변호사수)가 높은 = 수요 많음
- **분야 블루오션**: 전문 변호사 비율이 낮은데 수요는 있는 분야

## 응답 형식
1. 핵심 추천 (지역 or 분야) — 이유와 함께
2. 경쟁/수요 수치 근거
3. 주의사항 (데이터 한계 언급)

## 주의
- 시/도 단위 데이터입니다 (구 단위 분석 불가)
- 추세/예측은 참고용이며 확정적 조언이 아님을 명시

## 데이터
{stats_data_json}

## 사용자 질문
{message}
```

#### 조회형 프롬프트

```
변호사 통계 데이터를 바탕으로 사용자 질문에 간결하게 답변하세요.
핵심 수치를 먼저 제시하고 상위/하위 비교를 포함하세요.

## 데이터
{stats_data_json}

## 사용자 질문
{message}
```

### 4.3 [B2] nodes.py 수정

```python
async def lawyer_stats_node(state, writer):
    """변호사 시장 분석 노드 (LLM 스트리밍)"""
    from app.multi_agent.agents.lawyer_stats_agent import LawyerStatsAgent
    return await _run_streaming_node(LawyerStatsAgent(), state, writer)
```

### 4.4 [B3] StatsIntent 스키마

`backend/app/modules/lawyer_stats/schema/__init__.py`에 추가.

### 4.5 [F1] StatsFilter 타입

`frontend/src/features/lawyer-stats/types/index.ts`에 추가.

### 4.6 [F2] lawyer-stats page sessionData 연동

```typescript
const { sessionData, setSessionData } = useChat()

useEffect(() => {
  const filter = sessionData?.stats_filter as StatsFilter | undefined
  if (!filter) return

  if (filter.indicatorGroup) setIndicatorGroup(filter.indicatorGroup)
  if (filter.viewMode) setViewMode(filter.viewMode)
  if (filter.activeTab) setActiveTab(filter.activeTab)
  if (filter.selectedProvince !== undefined) setSelectedProvince(filter.selectedProvince)
  if (filter.predictionYear) setPredictionYear(filter.predictionYear)
  if (filter.demandCategory) setDemandCategory(filter.demandCategory)
  if (filter.demandYear) setDemandYear(filter.demandYear)

  // 사용 후 초기화
  setSessionData({ ...sessionData, stats_filter: undefined })
}, [sessionData?.stats_filter])
```

---

## 5. 대시보드 필터 매핑 전략

| query_type | 대시보드 필터 설정 |
|------------|----------------|
| `recommend_market` | `viewMode: 'density'`, `indicatorGroup: 'supply'`, `activeTab: 'region'` |
| `recommend_specialty` | `activeTab: 'cross'`, `selectedProvince: 지역`, `crossRegions: [지역]` |
| `recommend_region` | `viewMode: 'density'`, `activeTab: 'region'` |
| `overview` | 기본값 유지 |
| `region` | `viewMode: 'count'`, `selectedProvince: 지역` |
| `density` | `viewMode: 'density'`, `selectedProvince: 지역` |
| `specialty` | (스크롤만 — 전문분야 섹션으로) |
| `cross` | `activeTab: 'cross'`, `crossRegions: 지역들` |
| `demand` | `indicatorGroup: 'demand'`, `viewMode: 'case_count'`, `demandCategory: 분야` |
| `prediction` | `viewMode: 'prediction'`, `predictionYear: 연도` |

---

## 6. 데이터 한계 및 대응

| 한계 | 영향 | 대응 |
|------|------|------|
| 시/도 단위까지만 가능 (구 단위 X) | "종로구"를 직접 분석 불가 | LLM이 "서울특별시 단위로 분석합니다" 안내 + 서울 데이터 제공 |
| 변호사 수 = 현재 시점 스냅샷 | 신규 진입 동향 불명 | LLM이 "현재 기준 데이터이며 변동 가능" 면책 문구 포함 |
| 수요 = 법원 사건 접수 건수 | 실제 법률 수요와 완전 일치하지 않음 | LLM이 "법원 사건 접수 기준이며 참고용" 명시 |
| 예측 = 인구 추계 기반 | 변호사 공급 변화 미반영 | LLM이 "인구 변화 기반 예측, 공급 변화 미반영" 명시 |

---

## 7. 검증 계획

### 7.1 정적 검증

```bash
cd backend && uv run ruff check backend/app/modules/lawyer_stats/ backend/app/multi_agent/agents/lawyer_stats_agent.py
cd backend && uv run mypy backend/app/modules/lawyer_stats/ backend/app/multi_agent/agents/lawyer_stats_agent.py
cd frontend && npm run build
```

### 7.2 E2E 시나리오 테스트

| # | 입력 메시지 | 기대 동작 |
|---|-----------|---------|
| 1 | "새로 합격했어. 어디에 개업하면 좋을까?" | 추천형 응답 + density 모드 대시보드 |
| 2 | "서울에서 무슨 전문분야가 좋을까?" | 서울 교차분석 + 블루오션 분야 추천 |
| 3 | "행정법 전문으로 가려는데 어디가 좋을까?" | 행정 분야 기준 지역 추천 |
| 4 | "서울 변호사 밀도 알려줘" | 단순 밀도 응답 + density 모드 |
| 5 | "전문분야별 분포 보여줘" | 전문분야 서술 + specialty 섹션 |
| 6 | "2035년 예측은?" | 예측 응답 + prediction 모드 |
| 7 | "형사 사건 수요가 높은 지역?" | 수요 응답 + demand 모드 |

---

## 8. 구현 순서

1. **[B3]** StatsIntent 스키마 추가
2. **[B1]** LawyerStatsAgent 전면 재작성 (의도분석 + 다중데이터 + 추천/조회 응답)
3. **[B2]** nodes.py 스트리밍 전환
4. **[F1]** StatsFilter 타입 추가
5. **[F2]** lawyer-stats page sessionData 연동
6. Backend 정적 검증 (ruff + mypy)
7. Frontend 빌드 검증 (npm run build)

---

## 변경 이력

| 버전 | 일자 | 변경 내용 |
|------|------|---------|
| v1.0 | 2026-02-26 | 초안 (단순 통계 조회 + 필터 연동) |
| v2.0 | 2026-02-26 | 대상을 신입 변호사로 명확화, 추천형 질문 유형 추가, 다중 데이터 조합 전략 추가, LLM 프롬프트 2종 분리, 데이터 한계 명시 |
