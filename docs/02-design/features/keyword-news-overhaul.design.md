# 키워드/뉴스 수집 시스템 전면 재설계 — 상세 설계

> **버전**: 1.2
> **작성일**: 2026-03-01
> **기반**: 기획서 v1.3 (`docs/01-plan/features/keyword-news-overhaul.plan.md`)
> **상태**: Gemini CLI + Codex CLI 리뷰 완료 → PM 최종 확정

---

## 1. 설계 개요

### 1.1 범위

| 영역 | Phase 1 (Critical) | Phase 2 (핵심) | Phase 3 (고도화) |
|------|-------------------|----------------|-----------------|
| Backend | 동적 쿼리 + category 관통 | YouTube 2단계 + RRF + 5차원 스코어링 | Provider Abstraction + KR-WordRank + Semantic Dedup |
| Frontend | 카테고리 선택 UI | 기사 카드 engagement 표시 | convergence 시각화 + Early Signal |

---

## 2. Backend 아키텍처

### 2.1 파일 구조

```
backend/app/tools/trend/
├── circuit_breaker.py          (NEW) — CircuitBreaker 클래스 (점증 backoff)
├── deduplicator.py             (NEW) — SemanticDeduplicator (2단계 병합)
├── category_matcher.py         (NEW) — SemanticCategoryMatcher (KURE-v1 임베딩)
├── rrf.py                      (NEW) — WeightedRRF 소스 통합
├── models.py                   (MOD) — RawTrendItem v3 확장
├── collector.py                (MOD) — 동적 쿼리 + category 관통 + RRF
├── keyword_extractor.py        (MOD) — HybridKeywordExtractor
├── article_scorer.py           (MOD) — 5차원 스코어링 v2
├── sources/
│   ├── google_trends_source.py (NEW) — Provider Abstraction 3계층
│   ├── naver_datalab_source.py (NEW) — Naver DataLab 소스
│   └── youtube_source.py       (MOD) — YouTubeSourceV2 (2단계 API)
```

### 2.2 데이터 모델

#### RawTrendItem v3
```python
@dataclass
class RawTrendItem:
    title: str
    url: str
    snippet: str
    source: TrendSource
    published_at: datetime | None = None
    mention_count: int = 0
    raw_data: dict[str, object] = field(default_factory=dict)
    # v3 NEW
    view_count: int | None = None
    comment_count: int | None = None
    like_count: int | None = None
    engagement_velocity: float | None = None
    is_shorts: bool = False
    convergence_score: float = 0.0
    z_score: float | None = None
    is_early_signal: bool = False
    merged_sources: list[TrendSource] = field(default_factory=list)
```

#### NewsArticle 확장 (Pydantic)
```python
class NewsArticle(BaseModel):
    # 기존 필드 유지
    engagement_score: float = Field(default=0.0, ge=0.0, le=1.0)
    convergence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    view_count: int | None = None
    comment_count: int | None = None
    is_early_signal: bool = False
    # Codex 리뷰 반영: 설명 가능성 필드
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    # 예: {"relevance": 0.8, "legal": 0.6, "recency": 0.9, "engagement": 0.4, "convergence": 0.3}
```

#### SourceConfig 변경
```python
@dataclass
class SourceConfig:
    # 기존 필드 유지
    category: str = "all"  # NEW
```

### 2.3 키워드 파이프라인 4단계

```
Step 1: 멀티소스 수집 (_build_keyword_query(category) → 동적 쿼리)
  └── Tavily + Naver + Google Trends + DataLab (4소스 병렬)
Step 2: KR-WordRank 비지도 추출 (50 후보, 비용 0원)
Step 3: LLM 스코어링 (상위 20개만, 카테고리 컨텍스트 주입)
Step 4: convergence + engagement → 최종 5차원 랭킹
```

### 2.4 뉴스 파이프라인 5단계

```
Step 1: build_news_query(keyword, category) → 동적 쿼리
Step 2: 6소스 병렬 (YouTube 2단계 포함, Semaphore MAX_CONCURRENT_SOURCES=4)
Step 3: URL Canonicalization (UTM 제거, 단축링크 해제) → SemanticDeduplicator (임베딩 2단계)
Step 4: WeightedRRF (소스별 가중치 × 1/(k+rank))
Step 5: score_articles_v2() (5차원)
```

### 2.5 핵심 클래스 시그니처

#### CircuitBreaker (점증 backoff)
```python
class CircuitBreaker:
    _BACKOFF_DURATIONS = [60, 300, 900]  # 1분→5분→15분
    def __init__(self, failure_threshold=3, half_open_max_requests=2): ...
    def is_open(self, service: str) -> bool: ...
    def is_half_open_allowed(self, service: str) -> bool: ...
    def record_failure(self, service: str) -> None: ...
    def record_success(self, service: str) -> None: ...
```

#### SemanticDeduplicator (2단계)
```python
class SemanticDeduplicator:
    AUTO_MERGE_THRESHOLD = 0.90    # 자동 병합
    VERIFY_MERGE_THRESHOLD = 0.82  # 검증 병합
    MIN_ITEMS_FOR_DEDUP = 15
    def deduplicate(self, items: list[RawTrendItem]) -> list[RawTrendItem]: ...
```

#### WeightedRRF
```python
SOURCE_AUTHORITY_WEIGHTS = {
    TrendSource.NAVER: 1.2, TrendSource.GOOGLE_TRENDS: 0.9,
    TrendSource.NEWSAPI: 0.8, TrendSource.YOUTUBE: 0.7,
    TrendSource.NEWSDATA: 0.6, TrendSource.TAVILY: 0.5,
}
def weighted_rrf_merge(source_results) -> list[RawTrendItem]: ...
```

#### 5차원 스코어링
```python
# 가중치: relevance 0.25, legal 0.25, recency 0.15, engagement 0.20, convergence 0.15
def score_articles_v2(keyword, articles, keyword_convergence_score) -> list[NewsArticle]: ...
def _calculate_legal_v2(article, category: str) -> float: ...
    # 가중 결합 (kw 0.2 + RAG 0.4 + LLM 0.4)
    # Gemini 리뷰 반영: category별 Legal Taxonomy를 LLM 프롬프트에 주입하여 정확도 향상
def _calculate_engagement(article) -> float: ...  # log10 정규화 + 댓글 부스트
```

#### GoogleTrendsSource (Provider Abstraction)
```python
class GoogleTrendsProvider(ABC):
    async def fetch_trending(self, geo="KR") -> list[TrendingItem]: ...
    async def fetch_interest_over_time(self, keywords) -> dict[str, list[float]]: ...

class GoogleTrendsSource(BaseTrendSource):
    _providers = [OfficialProvider, SerpApiProvider, FallbackProvider]
    _circuit_breaker = CircuitBreaker()
```

### 2.6 API 변경

| 엔드포인트 | 변경 |
|-----------|------|
| `GET /keywords/collect/stream` | `category` 쿼리 파라미터 추가 |
| `POST /keywords/{id}/news` | 응답에 `engagement_score`, `convergence_score`, `view_count` 추가 |

### 2.7 Phase별 구현 순서

**Phase 1**: schema → collector → service → newsdata/newsapi → router (6작업)
**Phase 2**: models → youtube → schema → scorer → rrf → collector (6작업)
**Phase 3**: datalab → trends → breaker → extractor → dedup → collector → scorer → matcher → sanitize (9작업)

---

## 3. Frontend 아키텍처

### 3.1 타입 변경 (`types/index.ts`)

```typescript
// Phase 1
interface KeywordCollectRequest {
  category?: TrendCategory  // NEW
}

// Phase 2
interface NewsArticle {
  engagement_score: number   // NEW
  view_count: number | null  // NEW
  comment_count: number | null // NEW
  source_weight: number      // NEW
  score_breakdown: Record<string, number>  // NEW (Codex 리뷰)
}

// Phase 3
interface KeywordItem {
  convergence_score: number | null  // NEW
  is_early_signal: boolean          // NEW
}
interface KeywordScoreSchema {
  convergence: number  // NEW
}

type NewsArticleSortKey = 'total_score' | 'engagement_score' | 'recency_score' | 'convergence_score'
```

### 3.2 컴포넌트 변경

#### KeywordCollector (Phase 1)
- 카테고리 드롭다운 추가 (timeRange 왼쪽)
- TrendCategory enum 기반 7개 옵션 + "전체"
- localStorage 저장 (`keyword-category`)

#### KeywordNewsList (Phase 2)
- 정렬 셀렉트박스 (종합/참여도/최신/수렴 순)
- 기사 카드에 engagement 뱃지 (👁 조회수, 💬 댓글수)
- source_weight 기반 소스 뱃지 색상 차별화

#### KeywordCollector + TrendDashboard (Phase 3)
- Early Signal 배너 (is_early_signal === true 키워드 존재 시)
- convergence 뱃지 (score >= 0.5 이상만 표시)

### 3.3 훅 변경

```typescript
// useKeywordFlow — category state + setCategory 추가
// streamKeywordCollect — category 파라미터 전달
// 정렬: 클라이언트 사이드 useMemo (최대 30건이므로 성능 문제 없음)
```

### 3.4 API 계약 동기화

| Backend 필드 | Frontend 타입 | Phase |
|-------------|--------------|-------|
| KeywordCollectRequest.category | KeywordCollectRequest.category | 1 |
| NewsArticle.engagement_score | NewsArticle.engagement_score | 2 |
| NewsArticle.view_count | NewsArticle.view_count | 2 |
| NewsArticle.score_breakdown | NewsArticle.score_breakdown | 2 |
| NewsArticle.convergence_score | NewsArticle.convergence_score | 3 |
| KeywordItem.is_early_signal | KeywordItem.is_early_signal | 3 |

> snake_case 양쪽 동일 사용 (camelCase 변환 없음)

---

## 4. 설계 원칙

1. **하위호환**: 기존 함수/API 유지, v2 신규 추가
2. **점진적 활성화**: Phase별 독립 배포 가능
3. **Graceful Degradation**: 소스 실패, 모델 미로드 시 fallback
4. **기존 패턴 준수**: 싱글턴, BaseTrendSource ABC, safe_fetch_with_status()
5. **타입 힌트 필수**: Python/TypeScript 모두 완전한 타입 정의
6. **모델 경계 정규화**: 내부 처리는 Dataclass(RawTrendItem), 서비스 레이어 출력은 Pydantic(NewsArticle)으로 변환 (Gemini 리뷰 반영)
7. **동시성 제어**: 병렬 소스 수집 시 `asyncio.Semaphore(MAX_CONCURRENT_SOURCES=4)` 적용
8. **Lazy Singleton 로드**: Embedding 모델(KURE-v1)은 첫 호출 시 지연 로드 (기존 패턴 준수)
9. **설정 외부화**: 스코어링 가중치, Dedup 임계치, RRF 가중치 등 핵심 하이퍼파라미터는 `config.py` Settings로 관리 (Codex 리뷰 반영)
10. **URL 정규화**: 수집 데이터의 URL canonicalization (UTM 파라미터 제거, 단축링크 해제, 모바일 도메인 통일)
11. **설명 가능성**: NewsArticle 응답에 `score_breakdown` dict 포함하여 각 차원별 점수 투명하게 노출
12. **LLM 캐시**: 키워드+카테고리+문서 해시 기반 스코어링 결과 캐시로 비용 절감

---

## 변경 이력

| 날짜 | 버전 | 변경 |
|------|------|------|
| 2026-03-01 | 1.0 | 초안 (Backend + Frontend 설계 통합) |
| 2026-03-01 | 1.1 | Gemini CLI 리뷰 피드백 반영 (동시성 제어, Pydantic 정규화, Legal Taxonomy, Lazy Load) |
| 2026-03-01 | 1.2 | Codex CLI 컨설팅 리뷰 반영 (score_breakdown, URL 정규화, 설정 외부화, LLM 캐시) |
