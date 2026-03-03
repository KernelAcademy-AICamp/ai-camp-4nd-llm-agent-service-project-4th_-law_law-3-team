# 유튜브 컨텐츠 키워드/뉴스 수집 시스템 전면 재설계

> **버전**: 1.3 (Iteration 2 Red Team + Consultant 재검증 반영)
> **작성일**: 2026-03-01
> **작성자**: PM (Product Part)
> **상태**: Iteration 2 완료 — 사용자 승인 대기

---

## 1. 목표 및 범위

### 1.1 비전

**네이버 뉴스 카테고리별 베스트를 능가하는 키워드/뉴스 품질**을 달성한다. 현재 시스템은 하드코딩된 고정 쿼리와 단순 URL 중복제거에 의존하여, 카테고리 무관하게 동일한 결과를 반환하는 치명적 결함을 갖고 있다. 이를 멀티소스 실시간 트렌드 수집 + 크로스플랫폼 수렴 탐지 + 5차원 가중 스코어링으로 전면 교체한다.

**Early Signal Capture 전략** (Red Team 제안): 네이버 뉴스에 올라오기 전, 커뮤니티(Tavily) + Google Trends breakout 데이터를 조합하여 **네이버보다 30분~1시간 빠른 이슈 선점**을 목표로 한다. 커뮤니티 언급 급증 + Google Trends Z-score > 2.0 동시 충족 시 "Early Signal" 태그를 부여하여 우선 노출한다.

### 1.2 핵심 목표

| 지표 | 현재 (추정) | 목표 | 측정 방법 |
|------|-----------|------|----------|
| 카테고리 일치율 | 40~50% | 85%+ | 수동 평가 100건 샘플 |
| 뉴스 관련도 평균 점수 | 30~40점 | 65+ | total_score 평균 |
| 네이버 베스트 대비 | 열위 | 동등 이상 | A/B 비교 평가 |
| 키워드 추출 비용 | LLM 1회/수집 | LLM 0~1회/수집 | KR-WordRank fallback 비율 |
| 크로스플랫폼 수렴 탐지 | 미지원 | 3+ 플랫폼 동시 등장 탐지 | convergence_score > 0.5 비율 |

### 1.3 범위

| 영역 | 포함 | 제외 |
|------|------|------|
| Backend | `tools/trend/` 전체, `services/service_function/content_marketing_service.py`, `modules/content_marketing/` | 페르소나 시스템, 대본 생성, 웹툰 |
| Frontend | `KeywordCollector`, `KeywordNewsList`, `TrendDashboard` | 기타 content-marketing 컴포넌트 |
| 신규 의존성 | krwordrank, rank-bm25 | GPU 전용 라이브러리 |
| ~~제거 예정~~ | ~~pytrends~~ (아카이브됨, §7 참조) | - |

---

## 2. 현재 시스템 vs 목표 시스템 격차

### 10대 근본 결함 상세 분석

| # | 결함 | 코드 위치 | 현재 동작 | 목표 상태 | 심각도 |
|---|------|----------|----------|----------|--------|
| 1 | **고정 쿼리 하드코딩** | `collector.py:267,277,337` | `search_query="사건 사고 논란 이슈"` 고정. `collect_community_keywords_with_status()`에서 SourceConfig에 하드코딩된 쿼리 전달. Naver도 `"사건 사고 논란 법률"` 고정. | 카테고리 + 페르소나 기반 동적 쿼리 빌더. 카테고리별 쿼리 템플릿 + 사용자 컨텍스트 반영. | **Critical** |
| 2 | **카테고리 파라미터 미반영** | `content_marketing_service.py:361-414` | `KeywordCollectRequest.category` 필드가 스키마에 없음. `collect_keywords()`에서 `request.time_range`만 사용하고 카테고리를 TrendCollector에 전달하지 않음. | 카테고리를 수집-추출-스코어링 전 파이프라인에 관통 전달. 쿼리 빌더가 카테고리별 쿼리 생성. | **Critical** |
| 3 | **Engagement velocity 부재** | `youtube_source.py:50-101` | YouTube `search.list` API만 호출 (`part: "snippet"`). `videos.list`로 조회수/댓글수/좋아요수 미수집. RawTrendItem에 engagement 필드 없음. | search.list → video ID 추출 → videos.list 2단계 호출. `view_count`, `comment_count`, `like_count` 수집 + `engagement_velocity = views / hours_since_published` 계산. | **High** |
| 4 | **단순 URL 중복제거** | `collector.py:223-240` | `_deduplicate()`가 URL `.rstrip("/").lower()` 정규화 후 set 기반 중복 제거. 동일 사건의 다른 매체 보도(예: 조선일보/한겨레 동일 사건)를 별개로 취급. | Semantic Dedup: 제목+snippet 임베딩 cosine similarity > 0.8이면 동일 기사 판정. 대표 기사 선택 후 소스 메타데이터 병합. | **High** |
| 5 | **크로스플랫폼 수렴 탐지 없음** | 미구현 | 여러 플랫폼(Naver, YouTube, Google, 커뮤니티)에서 동시 등장하는 핵심 이슈 식별 불가. 소스 간 연관성 분석 없음. | trigram-token Jaccard로 다중 플랫폼 동시 등장 키워드 식별. `convergence_score = platforms_appeared / total_platforms` 계산. 3+ 플랫폼 동시 등장 시 가중치 부스트. | **High** |
| 6 | **time_range 하드코딩** | `collector.py:265,279,401` / `newsdata_source.py:19-22` | 키워드 수집: `time_range="48h"` 고정 (SourceConfig). 뉴스 검색: `time_range="7d"` 고정. NewsData.io: 모든 time_range를 48시간으로 매핑 (`_TIMEFRAME_MAP`의 7d→48, 30d→48). | `time_range` 파라미터를 수집-검색 전 과정에 관통 전달. NewsData.io는 API 한계 명시 + 대안 소스(NewsAPI 7d/30d) 자동 보완. | **Medium** |
| 7 | **legal_score RAG 완전 의존** | `article_scorer.py:99-108` / `content_marketing_service.py:671-677` | `_calculate_legal()`이 `legal_issue_label`과 `related_laws` 유무로만 점수 산출. 이 두 필드는 `_enrich_articles()`의 LanceDB 벡터 검색(`_search_related_laws`)에서 채움. 벡터 검색 실패 시 `_safe_law_enrichment()`가 빈 리스트 반환 → 모든 기사 `legal_score = 0.0`. | 3단계 법률 관련도: (1) 키워드 매칭 (법률 키워드 frozenset), (2) RAG 벡터 검색 (기존), (3) LLM fallback (상위 5건만). 한 단계 실패해도 나머지로 보정. | **High** |
| 8 | **Google Trends 실시간 트렌딩 미활용** | `google_source.py` | Google Custom Search Engine (CSE) API만 사용. 실시간 급상승 키워드 탐지 불가. | Provider Abstraction 3계층 (공식 API / SerpAPI / DataLab fallback)으로 Google Trends breakout 키워드 수집 (Z-score > 2.0 = 트렌딩). 기존 CSE와 병행. **pytrends는 사용 금지(아카이브됨)**. | **Medium** |
| 9 | **YouTube videos.list 통계 미수집** | `youtube_source.py:65-98` | `params: {"part": "snippet"}` — snippet만 요청. `videos.list`의 `statistics` (viewCount, commentCount, likeCount) 미수집. 결과적으로 조회수 기반 정렬/필터 불가, 날짜순(`order: "date"`)만 사용. | 2단계 API 호출: (1) search.list → video ID 수집, (2) videos.list?part=statistics → 통계 수집. RawTrendItem에 `view_count`, `comment_count` 필드 추가. | **High** |
| 10 | **멀티소스 가중 통합(RRF) 없음** | `collector.py:182-191` / `search_news_for_keyword:440-456` | `asyncio.gather` 후 `news_items.extend(items)` — 단순 리스트 합산. 소스 간 권위(공신력) 가중 없이 모든 소스를 동등 취급. 네이버 뉴스(공인 매체)와 커뮤니티 게시글이 동일 가중치. | Weighted RRF (Reciprocal Rank Fusion): 소스별 가중치 × 1/(k+rank) 합산. 소스 권위 가중치: Naver=1.2, Google=0.9, NewsAPI=0.8, YouTube=0.7, Perplexity=0.7, NewsData=0.6, Tavily=0.5. | **High** |

---

## 3. 새로운 아키텍처 (파이프라인 구조)

### 3.1 전체 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Keyword Collection Pipeline                       │
│                                                                      │
│  Step 1: 멀티소스 실시간 트렌드 수집 (Provider Abstraction)            │
│    ├── Google Trends (공식 API alpha / Fallback) → breakout 키워드    │
│    ├── 네이버 DataLab → 검색어 트렌드 추이                            │
│    ├── Tavily → 커뮤니티 인기글                                       │
│    └── Naver News → 뉴스 헤드라인                                     │
│                                                                      │
│  Step 2: KR-WordRank 비지도 키워드 추출 (1차 후보 50개)               │
│    ├── 수집 텍스트 → KR-WordRank (비용 0원)                           │
│    └── 후보 키워드 50개 생성                                          │
│                                                                      │
│  Step 3: LLM 법적 관련성 스코어링 (상위 20개)                         │
│    ├── 카테고리 컨텍스트 주입                                         │
│    └── 4차원 점수 (virality, social_impact, legal_relevance,          │
│        content_fitness) 산출                                          │
│                                                                      │
│  Step 4: 크로스플랫폼 수렴 + Engagement → 최종 랭킹                   │
│    ├── convergence_score: 몇 개 플랫폼에서 동시 등장?                 │
│    ├── engagement_velocity: YouTube 조회수/시간                        │
│    └── 최종 랭킹 산출 (5차원 가중합)                                  │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│                    News Search Pipeline                               │
│                                                                      │
│  Step 1: 카테고리 반영 동적 쿼리 빌딩                                 │
│    └── 카테고리 + 키워드 → 최적화된 검색 쿼리 생성                    │
│                                                                      │
│  Step 2: 6소스 병렬 검색 (YouTube 통계 포함)                          │
│    ├── Naver News, Google CSE, NewsAPI, NewsData, YouTube, Perplexity │
│    └── YouTube: search.list → videos.list 2단계                       │
│                                                                      │
│  Step 3: Semantic Dedup (임베딩 중복 제거)                            │
│    └── title+snippet 임베딩 → cosine sim > 0.8 → 병합                │
│                                                                      │
│  Step 4: Weighted RRF 소스 통합                                       │
│    └── 소스별 권위 가중치 × 1/(k+rank) 합산                          │
│                                                                      │
│  Step 5: 5차원 스코어링                                               │
│    └── relevance + legal + recency + engagement + convergence         │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 키워드 수집 파이프라인 상세 (4단계)

#### Step 1: 멀티소스 실시간 트렌드 수집

**현재**: Tavily + Naver 2소스, 고정 쿼리 `"사건 사고 논란 이슈"`

**개선**: 4소스 병렬 수집 + 카테고리 동적 쿼리

```python
# 현재 코드 (collector.py:267)
config = SourceConfig(
    time_range="48h",
    max_results=max_keywords * 3,
    search_query="사건 사고 논란 이슈",  # ← 하드코딩
    include_domains=domains,
)

# 개선 후
def _build_keyword_query(category: TrendCategory, time_range: str) -> str:
    """카테고리별 동적 쿼리 생성"""
    base_queries = {
        TrendCategory.CRIMINAL: "형사 사건 수사 기소 판결",
        TrendCategory.CIVIL: "소송 계약 분쟁 손해배상 부동산",
        TrendCategory.LABOR: "해고 임금 노동 산재 퇴직",
        TrendCategory.FAMILY: "이혼 양육권 상속 가사 위자료",
        TrendCategory.ADMINISTRATIVE: "행정소송 인허가 처분 취소",
        TrendCategory.CORPORATE: "기업 주주 파산 회생 M&A",
        TrendCategory.IP: "특허 상표 저작권 지식재산",
        TrendCategory.ALL: "법률 사건 사고 논란 이슈 판결",
    }
    return base_queries.get(category, base_queries[TrendCategory.ALL])
```

**신규 소스 추가**:

| 소스 | API | 역할 | 비용 |
|------|-----|------|------|
| Google Trends (NEW) | Provider Abstraction (§7.1 참조) | 실시간 급상승 키워드 | 무료/API 비용 |
| 네이버 DataLab (NEW) | Naver DataLab API | 검색어 트렌드 추이 (시간별/일별) | 무료 (일 25,000건) |
| Tavily (기존) | Tavily Search API | 커뮤니티 인기글 | 유료 |
| Naver News (기존) | Naver Search API | 뉴스 헤드라인 | 무료 (일 25,000건) |

> **⚠ CRITICAL**: pytrends 저장소가 **2025-04-17 아카이브(읽기 전용)**되었으며, Google Trends 공식 API는 2025-07-24 알파 공개(접근 제한형). 따라서 Google Trends 소스는 Provider Abstraction 패턴으로 설계하여 3계층(공식 API / 대체 소스 / graceful fallback)으로 구성한다. 상세는 §7.1 참조.

#### Step 2: KR-WordRank 비지도 키워드 추출

**현재**: LLM만 사용하여 키워드 추출 (비용 발생, 속도 느림)

**개선**: KR-WordRank 1차 추출 → LLM은 스코어링에만 사용

```python
# 새로운 키워드 추출 흐름
class HybridKeywordExtractor:
    """KR-WordRank + LLM 하이브리드 키워드 추출"""

    async def extract(self, texts: list[str], max_candidates: int = 50) -> list[str]:
        """
        1. KR-WordRank로 비지도 키워드 추출 (비용 0원, ~100ms)
        2. 불용어/블랙리스트 필터링
        3. 법률 도메인 관련성 순 정렬 (키워드 매칭)
        """
        from krwordrank.word import KRWordRank

        wordrank = KRWordRank(min_count=2, max_length=10)
        keywords, rank, graph = wordrank.extract(texts, beta=0.85, max_iter=10)

        # 법률 키워드 부스트
        boosted = []
        for kw, score in keywords.items():
            legal_boost = 1.5 if any(lk in kw for lk in LEGAL_KEYWORDS) else 1.0
            boosted.append((kw, score * legal_boost))

        boosted.sort(key=lambda x: x[1], reverse=True)
        return [kw for kw, _ in boosted[:max_candidates]]
```

**비용 효과**:
- 현재: LLM 호출 1회/수집 (~$0.01-0.03)
- 개선: KR-WordRank로 후보 추출 (무료) → LLM은 상위 20개 스코어링만

#### Step 3: LLM 법적 관련성 스코어링

**현재**: `keyword_extractor.py`의 `_SCORING_PROMPT_TEMPLATE`이 추출과 스코어링을 동시에 수행

**개선**: Step 2에서 이미 추출된 후보 키워드에 대해 스코어링만 수행

- 카테고리 컨텍스트를 프롬프트에 주입하여 카테고리별 관련성 평가
- `legal_relevance < 0.3` 게이트 유지 (현재와 동일)
- 기존 4차원 점수 체계(virality, social_impact, legal_relevance, content_fitness) 유지

#### Step 4: 크로스플랫폼 수렴 + Engagement → 최종 랭킹

**현재**: 미구현

**개선**:

```python
def calculate_convergence_score(
    keyword: str,
    source_results: dict[str, list[RawTrendItem]],
) -> float:
    """크로스플랫폼 수렴 점수 계산

    여러 플랫폼에서 동시에 등장하는 키워드일수록 높은 점수.
    trigram-token Jaccard 유사도로 키워드 매칭.
    """
    keyword_tokens = set(keyword.split())
    platforms_appeared = 0
    total_platforms = len(source_results)

    for source_name, items in source_results.items():
        for item in items:
            title_tokens = set(item.title.split())
            jaccard = len(keyword_tokens & title_tokens) / max(len(keyword_tokens | title_tokens), 1)
            if jaccard >= 0.3:  # trigram-token Jaccard 임계값
                platforms_appeared += 1
                break

    return platforms_appeared / max(total_platforms, 1)
```

### 3.3 뉴스 검색/스코어링 파이프라인 상세 (5단계)

#### Step 1: 카테고리 반영 동적 쿼리 빌딩 + Query Expansion

**현재**: `search_news_for_keyword()`에서 키워드 그대로 검색

**개선 1 — 카테고리 컨텍스트 주입**:
```python
def build_news_query(keyword: str, category: TrendCategory | None = None) -> str:
    """카테고리 컨텍스트를 포함한 뉴스 검색 쿼리 생성"""
    if category and category != TrendCategory.ALL:
        category_labels = {
            TrendCategory.CRIMINAL: "형사",
            TrendCategory.CIVIL: "민사",
            TrendCategory.LABOR: "노동",
            TrendCategory.FAMILY: "가사",
            TrendCategory.ADMINISTRATIVE: "행정",
            TrendCategory.CORPORATE: "기업",
            TrendCategory.IP: "지식재산",
        }
        return f"{keyword} {category_labels.get(category, '')}"
    return keyword
```

**개선 2 — LLM 기반 Query Expansion** (Red Team 제안, Phase 2 배치):

```python
async def expand_query_with_llm(keyword: str, category: TrendCategory) -> str:
    """LLM으로 검색 쿼리를 법률 전문 용어로 확장

    예시: "음주운전" → "음주운전 OR (혈중알코올농도 AND 면허취소) OR 도로교통법"
    """
    prompt = f"""다음 키워드를 법률 뉴스 검색에 최적화된 불리언 쿼리로 확장하세요.
키워드: {keyword}
카테고리: {category.value}

규칙:
- OR로 동의어/관련 법률 용어를 연결
- AND로 복합 개념을 묶기
- 최대 3개 OR 그룹
- 결과만 출력 (설명 없이)"""

    expanded = await llm_call(prompt)
    # 입력 세니타이징 (Red Team 제안: 특수 연산자 검증)
    return _sanitize_query(expanded)
```

> **적용 범위**: Query Expansion은 Phase 2에서 선택적으로 적용. LLM 호출 비용이 추가되므로
> 사용자가 "심층 검색" 옵션을 선택한 경우에만 활성화. 기본 검색은 카테고리 컨텍스트 주입만 사용.

#### Step 2: 6소스 병렬 검색 (YouTube 통계 포함)

**YouTube 2단계 API 호출** (결함 #3, #9 해결):

```python
class YouTubeSourceV2(BaseTrendSource):
    """YouTube Data API v3 — 2단계 수집 (search + statistics)"""

    async def fetch(self, query: str | None, config: SourceConfig) -> list[RawTrendItem]:
        # Stage 1: search.list → video ID 수집
        search_results = await self._search_videos(query, config)
        video_ids = [r["id"]["videoId"] for r in search_results if r.get("id", {}).get("videoId")]

        if not video_ids:
            return []

        # Stage 2: videos.list → 통계 수집
        statistics = await self._get_video_statistics(video_ids)

        items = []
        for result, stats in zip(search_results, statistics):
            view_count = int(stats.get("viewCount", 0))
            comment_count = int(stats.get("commentCount", 0))
            published_at = _parse_youtube_date(result["snippet"].get("publishedAt"))

            # Engagement Velocity 계산
            hours_since = _hours_since_published(published_at)
            engagement_velocity = view_count / max(hours_since, 1)

            items.append(RawTrendItem(
                title=result["snippet"]["title"],
                url=f"https://www.youtube.com/watch?v={result['id']['videoId']}",
                snippet=result["snippet"].get("description", "")[:300],
                source=TrendSource.YOUTUBE,
                published_at=published_at,
                view_count=view_count,
                comment_count=comment_count,
                engagement_velocity=engagement_velocity,
                raw_data={**result, "statistics": stats},
            ))

        return items

    async def _get_video_statistics(self, video_ids: list[str]) -> list[dict]:
        """videos.list API로 통계 수집 (최대 50개씩 배치)"""
        params = {
            "part": "statistics",
            "id": ",".join(video_ids[:50]),
            "key": settings.YOUTUBE_API_KEY,
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(YOUTUBE_VIDEOS_URL, params=params)
            response.raise_for_status()
            data = response.json()
        return [item.get("statistics", {}) for item in data.get("items", [])]
```

**⚠ YouTube Shorts viewCount 정의 변경 (2025-03-31)** (Consultant 제안 반영):

YouTube Shorts의 `viewCount` 정의가 2025-03-31에 변경되어 "재생 시작" 또는 "재생 포함"으로 카운트됩니다. 이로 인해:
- Shorts와 일반 영상의 조회수를 동일 기준으로 비교할 수 없음
- `engagement_velocity` 계산 시 **Shorts 여부를 식별하여 별도 정규화** 필요

```python
def _normalize_engagement(
    view_count: int, hours: float, is_shorts: bool,
    engaged_views: int | None = None,
) -> float:
    """Shorts/일반 영상 별도 정규화 (v1.3: 동적 Shorts 보정)

    v1.3 변경 (Consultant 권고):
    - 고정 0.3 → engaged_views/views 비율 기반 동적 보정 (0.15~0.60)
    - engaged_views 미수집 시 기본값 0.3 유지
    """
    velocity = view_count / max(hours, 1)
    if is_shorts:
        if engaged_views is not None and view_count > 0:
            # 동적 보정: 실제 시청 비율 기반 (v1.3)
            engagement_ratio = engaged_views / view_count
            shrink_factor = max(0.15, min(engagement_ratio, 0.60))
        else:
            shrink_factor = 0.3  # 기본값 (engaged_views 미수집 시)
        velocity *= shrink_factor
    return min(math.log10(velocity + 1) / 2.0, 1.0)

def _detect_shorts(video_data: dict) -> bool:
    """YouTube Shorts 여부 판별 (duration < 60s 또는 #Shorts 태그)"""
    duration_iso = video_data.get("contentDetails", {}).get("duration", "")
    # PT1M 미만이면 Shorts
    tags = video_data.get("snippet", {}).get("tags", [])
    return "Shorts" in tags or _parse_duration_seconds(duration_iso) < 60
```

> **구현 시점**: Phase 2 (작업 2-1에서 `contentDetails` part를 추가 요청하여 duration 수집)

#### Step 3: Semantic Dedup (임베딩 중복 제거)

**현재**: URL 기반 중복 제거 (`collector.py:223-240`)

**개선**: 임베딩 기반 의미적 중복 제거

```python
class SemanticDeduplicator:
    """임베딩 cosine similarity 기반 중복 제거 (v1.3: 2단계 병합)"""

    # v1.3: 2단계 임계값 (Consultant 권고)
    AUTO_MERGE_THRESHOLD = 0.90    # 자동 병합 (동일 기사 확실)
    VERIFY_MERGE_THRESHOLD = 0.82  # 검증 병합 (출처/시간/엔터티 일치 확인)
    MIN_ITEMS_FOR_DEDUP = 15       # v1.3: 30건 → 15건으로 하향 (Consultant 권고)

    def deduplicate(self, items: list[RawTrendItem]) -> list[RawTrendItem]:
        """
        1. URL 기반 1차 중복 제거 (기존 로직 유지)
        2. title+snippet 임베딩 → 2단계 병합 (v1.3)
           - cosine > 0.90: 자동 병합 (동일 기사)
           - cosine 0.82~0.90: 출처/시간/엔터티 일치 시 병합
        3. 동일 그룹 내 가장 높은 소스 권위 기사를 대표로 선택
        4. 나머지 기사의 소스 메타데이터를 대표 기사에 병합
        """
        # Phase 1: URL 중복 제거 (기존)
        url_deduped = self._url_deduplicate(items)

        # v1.3: 15건 미만 시 URL dedup만 수행
        if len(url_deduped) < self.MIN_ITEMS_FOR_DEDUP:
            return url_deduped

        # Phase 2: 임베딩 2단계 중복 제거 (v1.3)
        texts = [f"{item.title} {item.snippet}" for item in url_deduped]
        embeddings = self._get_embeddings(texts)  # 기존 KURE-v1 모델 활용

        # Stage 1: cosine > 0.90 자동 병합
        auto_clusters = self._cluster_by_similarity(embeddings, self.AUTO_MERGE_THRESHOLD)

        # Stage 2: cosine 0.82~0.90 검증 병합
        result = []
        for cluster_indices in auto_clusters:
            cluster_items = [url_deduped[i] for i in cluster_indices]
            # 자동 병합 클러스터 처리
            representative = self._select_representative(cluster_items)
            representative.merged_sources = [item.source for item in cluster_items]
            result.append(representative)

        # Stage 2: 검증 병합 (출처/시간/엔터티 일치 확인)
        result = self._verify_merge(result, embeddings)

        return result

    def _verify_merge(self, items: list[RawTrendItem], embeddings) -> list[RawTrendItem]:
        """0.82~0.90 구간: 출처/시간/엔터티가 일치할 때만 병합"""
        # 동일 사건의 다른 매체 보도 판별:
        # - 발행 시간 차이 < 24시간
        # - 주요 엔터티(인물/지명) 겹침 > 50%
        # - 소스가 다름 (동일 소스면 유지)
        ...
```

**비용 고려**: 기존 프로젝트의 `KURE-v1` 임베딩 모델(`app/services/rag/embedding.py`)을 재활용하므로 추가 비용 없음. 다만 임베딩 계산 지연(~50ms/10건)을 고려하여, Semantic Dedup은 URL 중복 제거 후 결과가 **15건 이상**일 때만 작동하도록 한다. (v1.3: Consultant 권고에 따라 30건 → 15건으로 하향)

**KURE-v1 fallback 전략** (Backend 분석 반영):
- **모델 로드 성공 시**: 임베딩 기반 Semantic Dedup (cosine similarity > 0.80)
- **모델 미로드 시**: URL 기반 중복 제거로 graceful fallback (Phase 1과 동일 동작)
- 임베딩은 배치 처리(10건 단위)로 GPU/Inference Server 활용도 극대화 (Red Team 제안 반영)

#### Step 4: Weighted RRF 소스 통합

**현재**: `news_items.extend(items)` — 단순 리스트 합산

**개선**:

```python
# 소스별 권위 가중치
SOURCE_AUTHORITY_WEIGHTS: dict[TrendSource, float] = {
    TrendSource.NAVER: 1.2,       # 공인 뉴스 매체 (국내 뉴스 권위 부스트)
    TrendSource.GOOGLE_TRENDS: 0.9,
    TrendSource.NEWSAPI: 0.8,
    TrendSource.NEWSDATA: 0.6,
    TrendSource.YOUTUBE: 0.7,
    TrendSource.PERPLEXITY: 0.7,
    TrendSource.TAVILY: 0.5,      # 커뮤니티 기반
}
# 참고: Naver 가중치를 1.2로 설정 (Backend 분석 반영)
# 국내 법률 뉴스에서 네이버 공인 매체의 권위도를 반영한 값

RRF_K = 60  # Reciprocal Rank Fusion 상수

def weighted_rrf_merge(
    source_results: dict[TrendSource, list[RawTrendItem]],
) -> list[RawTrendItem]:
    """Weighted RRF: 소스별 가중치 × 1/(k+rank) 합산"""
    item_scores: dict[str, float] = {}  # url → score
    item_map: dict[str, RawTrendItem] = {}

    for source, items in source_results.items():
        weight = SOURCE_AUTHORITY_WEIGHTS.get(source, 0.5)
        for rank, item in enumerate(items):
            url_key = item.url.rstrip("/").lower()
            rrf_score = weight * (1.0 / (RRF_K + rank + 1))
            item_scores[url_key] = item_scores.get(url_key, 0.0) + rrf_score
            if url_key not in item_map:
                item_map[url_key] = item

    # RRF 점수 내림차순 정렬
    sorted_urls = sorted(item_scores, key=item_scores.get, reverse=True)
    return [item_map[url] for url in sorted_urls if url in item_map]
```

#### Step 5: 5차원 스코어링

**현재**: `article_scorer.py` — 3차원 (relevance 0.4 + legal 0.3 + recency 0.3)

**개선**: 5차원 스코어링

```python
# 새로운 가중치
_RELEVANCE_WEIGHT = 0.25     # 키워드 관련도
_LEGAL_WEIGHT = 0.25         # 법적 관련도 (3단계 fallback)
_RECENCY_WEIGHT = 0.15       # 최신성
_ENGAGEMENT_WEIGHT = 0.20    # Engagement velocity
_CONVERGENCE_WEIGHT = 0.15   # 크로스플랫폼 수렴

def score_article_v2(
    keyword: str,
    article: NewsArticle,
    convergence_score: float,
) -> NewsArticle:
    """5차원 기사 스코어링"""
    relevance = _calculate_relevance(keyword, article.title, article.snippet)
    legal = _calculate_legal_v2(article)  # 3단계 fallback
    recency = _calculate_recency(article.published_at)
    engagement = _calculate_engagement(article)  # NEW
    convergence = convergence_score  # NEW

    total = (
        relevance * _RELEVANCE_WEIGHT
        + legal * _LEGAL_WEIGHT
        + recency * _RECENCY_WEIGHT
        + engagement * _ENGAGEMENT_WEIGHT
        + convergence * _CONVERGENCE_WEIGHT
    ) * 100

    return article.model_copy(update={
        "relevance_score": round(relevance, 4),
        "legal_score": round(legal, 4),
        "recency_score": round(recency, 4),
        "engagement_score": round(engagement, 4),
        "convergence_score": round(convergence, 4),
        "total_score": round(total, 2),
    })
```

---

## 4. 데이터 소스 전략

### 4.1 기존 6소스 개선

| 소스 | 현재 | 개선 |
|------|------|------|
| **Tavily** | 고정 쿼리 `"사건 사고 논란 이슈"` | 카테고리별 동적 쿼리 + `include_domains` 세분화 |
| **Naver News** | 고정 쿼리 `"사건 사고 논란 법률"` | 카테고리 쿼리 + `sort=sim` (관련도순) 옵션 추가 |
| **YouTube** | `search.list` snippet만 | `search.list` → `videos.list` 2단계, 통계 수집 |
| **Google CSE** | CSE만 | CSE 유지 + Provider Abstraction Google Trends 별도 소스 추가 (§7.1) |
| **NewsData.io** | 모든 time_range → 48h 고정 | API 한계 명시, `time_range > 48h` 시 NewsAPI로 위임 |
| **NewsAPI** | 기본 동작 | `time_range` 파라미터 정확 전달, `sortBy=relevancy` 옵션 추가 |
| **Perplexity** | 심층 분석 유지 | 카테고리 컨텍스트 프롬프트 주입 |

### 4.2 신규 2소스 추가

| 소스 | API | 용도 | 비용 | Rate Limit |
|------|-----|------|------|------------|
| **Google Trends** (NEW) | Provider Abstraction 3계층 (§7.1) | 실시간 급상승 키워드 탐지 | 무료~유료 | 계층별 상이 |
| **네이버 DataLab** (NEW) | Naver DataLab API `v1/datalab/search` | 검색어 트렌드 추이 | 무료 | 일 25,000건 |

**Google Trends 소스 구현 — Provider Abstraction 패턴**:

> **⚠ CRITICAL 의존성 변경**: `pytrends` 저장소가 2025-04-17 아카이브되어 유지보수 중단.
> Google Trends 공식 API는 2025-07-24 알파 공개(접근 제한형).
> 따라서 3계층 Provider Abstraction으로 설계.

```python
class GoogleTrendsProvider(ABC):
    """Google Trends Provider 추상 계층 (Consultant 제안 반영)"""

    @abstractmethod
    async def fetch_trending(self, geo: str = "KR") -> list[TrendingItem]:
        """실시간 트렌딩 키워드 수집"""

    @abstractmethod
    async def fetch_interest_over_time(self, keywords: list[str]) -> dict[str, list[float]]:
        """키워드별 관심도 시계열 (Z-score velocity 계산용)"""


class GoogleTrendsOfficialProvider(GoogleTrendsProvider):
    """계층 1: Google Trends 공식 API (알파, 접근 가능 시)"""

    async def fetch_trending(self, geo: str = "KR") -> list[TrendingItem]:
        # Google Trends API alpha 엔드포인트 사용
        # 접근 권한 획득 시 활성화
        ...

    async def fetch_interest_over_time(self, keywords: list[str]) -> dict[str, list[float]]:
        ...


class SerpApiTrendsProvider(GoogleTrendsProvider):
    """계층 2: SerpAPI/ValueSERP 등 유료 대체 소스"""

    async def fetch_trending(self, geo: str = "KR") -> list[TrendingItem]:
        # SerpAPI Google Trends 엔드포인트 사용
        # 비용: ~$0.01/request
        ...

    async def fetch_interest_over_time(self, keywords: list[str]) -> dict[str, list[float]]:
        ...


class FallbackTrendsProvider(GoogleTrendsProvider):
    """계층 3: Naver DataLab + 기존 소스 조합으로 트렌딩 추정"""

    async def fetch_trending(self, geo: str = "KR") -> list[TrendingItem]:
        # Naver DataLab 인기 검색어 + 기존 소스 교차 분석
        # 비용: 무료
        ...

    async def fetch_interest_over_time(self, keywords: list[str]) -> dict[str, list[float]]:
        # Naver DataLab 시계열 데이터로 대체
        ...


class GoogleTrendsSource(BaseTrendSource):
    """Google Trends 소스 — Provider Abstraction + Circuit Breaker"""

    def __init__(self):
        self._providers: list[GoogleTrendsProvider] = [
            GoogleTrendsOfficialProvider(),  # 1순위: 공식 API
            SerpApiTrendsProvider(),          # 2순위: 유료 대체
            FallbackTrendsProvider(),         # 3순위: DataLab 조합
        ]
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=3, recovery_timeout_seconds=300
        )

    @property
    def name(self) -> TrendSource:
        return TrendSource.GOOGLE_TRENDS_REALTIME

    async def fetch(self, query: str | None, config: SourceConfig) -> list[RawTrendItem]:
        """3계층 순회 — 상위 계층 실패 시 하위로 자동 전환"""
        for provider in self._providers:
            if self._circuit_breaker.is_open(provider.__class__.__name__):
                continue
            try:
                trending = await provider.fetch_trending(geo="KR")
                # Z-score velocity 계산 (Backend 분석 반영)
                enriched = await self._enrich_with_velocity(provider, trending)
                return self._to_raw_items(enriched, config.max_results)
            except Exception:
                self._circuit_breaker.record_failure(provider.__class__.__name__)
                continue
        return []  # 모든 계층 실패 시 graceful degradation

    async def _enrich_with_velocity(
        self, provider: GoogleTrendsProvider, items: list[TrendingItem]
    ) -> list[TrendingItem]:
        """Z-score velocity 계산 (Z-score > 2.0 = 트렌딩)"""
        keywords = [item.keyword for item in items[:20]]
        if not keywords:
            return items
        try:
            time_series = await provider.fetch_interest_over_time(keywords)
            for item in items:
                if item.keyword in time_series:
                    values = time_series[item.keyword]
                    if len(values) >= 3:
                        mean = sum(values) / len(values)
                        std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
                        item.z_score = (values[-1] - mean) / max(std, 0.01)
        except Exception:
            pass  # velocity 실패 시 기존 데이터로 진행
        return items
```

> **Z-score velocity** (Backend 분석 반영): Google Trends 시계열에서 최근 값의 Z-score를 계산하여
> `z_score > 2.0`이면 "급상승 트렌드"로 판정, 키워드 스코어링에 부스트 적용.

---

## 5. 스코어링 알고리즘 상세

### 5.1 키워드 점수 (5차원)

| 차원 | 가중치 | 산출 방법 | 현재 → 개선 |
|------|--------|----------|------------|
| **virality** | 0.20 | 커뮤니티 언급 빈도 + Google Trends Z-score | LLM 판단 → KR-WordRank 빈도 + Trends 데이터 |
| **social_impact** | 0.20 | 사회적 영향도 (LLM) | 유지 (LLM 스코어링) |
| **legal_relevance** | 0.30 | 법적 쟁점 관련도 | LLM 판단 → 키워드 매칭 + LLM 보정 |
| **content_fitness** | 0.15 | 콘텐츠 적합도 | 유지 (LLM 스코어링) |
| **convergence** (NEW) | 0.15 | 크로스플랫폼 수렴 점수 | 미구현 → 플랫폼 동시 등장 비율 |

**총점 공식**: `total = (v*0.20 + s*0.20 + l*0.30 + c*0.15 + conv*0.15) * 100`

### 5.2 뉴스 기사 점수 (5차원)

| 차원 | 가중치 | 산출 방법 | 현재 → 개선 |
|------|--------|----------|------------|
| **relevance** | 0.25 | exact + nospace + token + snippet_freq | 유지 (article_scorer.py) |
| **legal** | 0.25 | legal_issue_label + related_laws | RAG 단독 → 3단계 fallback |
| **recency** | 0.15 | exp(-λ×days), 반감기 7일 | 유지 |
| **engagement** (NEW) | 0.20 | view_velocity + comment_density | 미구현 → YouTube/소스별 통계 |
| **convergence** (NEW) | 0.15 | 해당 키워드의 크로스플랫폼 수렴 점수 | 미구현 → 키워드 수렴 점수 전파 |

**legal_score 3단계 fallback** (결함 #7 해결):

```python
def _calculate_legal_v2(article: NewsArticle) -> float:
    """3단계 법적 관련도 (v1.3: 가중 결합 + confidence 보존)

    v1.3 변경 (Consultant 권고):
    - max 방식 → 가중 결합 (keyword 0.2, RAG 0.4, LLM 0.4)
    - 단계별 confidence를 메타데이터로 보존
    """
    # Stage 1: 키워드 매칭 (비용 0원, 즉시)
    text = f"{article.title} {article.snippet}".lower()
    keyword_hits = sum(1 for kw in LEGAL_KEYWORDS if kw in text)
    keyword_score = min(keyword_hits * 0.15, 0.6)

    # Stage 2: RAG 벡터 검색 결과 (기존)
    rag_score = 0.0
    if article.legal_issue_label:
        rag_score += 0.5
    if article.related_laws:
        rag_score += min(len(article.related_laws) / 5, 0.5)

    # Stage 3: v1.3 가중 결합 (max → weighted combination)
    # LLM fallback은 Phase 3에서 구현 (상위 5건만 호출)
    llm_score = 0.0  # Phase 3에서 구현 시 활성화

    # 가중 결합 (Consultant 권고: keyword 0.2, RAG 0.4, LLM 0.4)
    # LLM 미사용 시: keyword 0.33, RAG 0.67로 재분배
    if llm_score > 0:
        combined = keyword_score * 0.2 + rag_score * 0.4 + llm_score * 0.4
    else:
        combined = keyword_score * 0.33 + rag_score * 0.67

    # 완전 0점 방지 (RAG 실패 시 최소 보장)
    if combined == 0.0:
        combined = 0.1

    # confidence 메타데이터 보존 (근거 추적용, v1.3)
    article._legal_confidence = {
        "keyword_score": round(keyword_score, 4),
        "rag_score": round(rag_score, 4),
        "llm_score": round(llm_score, 4),
        "method": "weighted_combination",
    }

    return min(combined, 1.0)
```

**engagement_score 산출**:

```python
def _calculate_engagement(article: NewsArticle) -> float:
    """Engagement velocity 점수 (0~1)"""
    if not hasattr(article, "view_count") or article.view_count is None:
        return 0.3  # 통계 미수집 소스 기본값

    # view_velocity: 시간당 조회수
    hours = _hours_since_published(article.published_at)
    view_velocity = article.view_count / max(hours, 1)

    # 로그 스케일 정규화 (100 views/hour = 1.0)
    normalized = min(math.log10(view_velocity + 1) / 2.0, 1.0)

    # 댓글 보정 (높은 댓글 비율 = 높은 참여도)
    if article.comment_count and article.view_count > 0:
        comment_ratio = article.comment_count / article.view_count
        comment_boost = min(comment_ratio * 10, 0.2)  # 최대 0.2 부스트
        normalized = min(normalized + comment_boost, 1.0)

    return normalized
```

---

## 6. 카테고리 매칭 개선

### 현재 문제

- `KeywordCollectRequest`에 `category` 필드 자체가 없음
- `TrendRequest.category`는 `TrendCategory` enum이지만 키워드 수집에서 무시됨
- `scorer.py`의 `CATEGORY_KEYWORDS`는 사후 분류만 수행 (수집 쿼리에 반영 안 됨)

### 개선: 임베딩 기반 Semantic Category Matching

```python
class SemanticCategoryMatcher:
    """키워드 → 카테고리 임베딩 기반 매칭"""

    def __init__(self):
        # 카테고리별 대표 문장 (임베딩 미리 계산)
        self.category_descriptions = {
            TrendCategory.CRIMINAL: "형사 사건 수사 기소 재판 판결 범죄 검찰 경찰",
            TrendCategory.CIVIL: "민사 소송 계약 분쟁 손해배상 부동산 채무 채권",
            TrendCategory.LABOR: "노동 해고 임금 퇴직금 산재 부당해고 근로기준법",
            TrendCategory.FAMILY: "가사 이혼 양육권 위자료 재산분할 상속 유산",
            TrendCategory.ADMINISTRATIVE: "행정 인허가 처분 취소 행정소송 규제",
            TrendCategory.CORPORATE: "기업 주주 이사 파산 회생 M&A 상법",
            TrendCategory.IP: "특허 상표 저작권 지식재산 영업비밀 디자인권",
        }
        self._category_embeddings: dict[TrendCategory, list[float]] | None = None

    def match(self, keyword: str) -> tuple[TrendCategory, float]:
        """키워드 → 가장 적합한 카테고리 + 유사도 반환"""
        if self._category_embeddings is None:
            self._build_embeddings()

        keyword_embedding = create_query_embedding(keyword)

        best_category = TrendCategory.ALL
        best_similarity = 0.0

        for category, cat_embedding in self._category_embeddings.items():
            similarity = cosine_similarity(keyword_embedding, cat_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_category = category

        return best_category, best_similarity
```

### API 변경

`KeywordCollectRequest`에 `category` 필드 추가:

```python
class KeywordCollectRequest(BaseModel):
    max_keywords: int = Field(default=10, ge=1, le=30)
    time_range: TimeRange = TimeRange.HOURS_48
    category: TrendCategory = TrendCategory.ALL  # NEW
    community_domains: list[str] | None = None
```

---

## 7. 신규 의존성

### 7.0 의존성 목록

| 패키지 | 버전 | 용도 | 크기 | 라이선스 |
|--------|------|------|------|----------|
| `krwordrank` | ≥1.0 | 한국어 비지도 키워드 추출 (PageRank) | ~30KB | Apache-2.0 |
| `rank-bm25` | ≥0.2 | BM25Okapi 관련도 점수 (향후 확장) | ~20KB | Apache-2.0 |
| ~~`pytrends`~~ | - | ~~Google Trends wrapper~~ (**아카이브됨, 사용 금지**) | - | - |

**설치**:
```bash
cd backend
uv add krwordrank rank-bm25
```

### 7.1 pytrends 의존성 전략 변경 (CRITICAL)

> **근거**: External Consultant 보고서 — pytrends 저장소가 **2025-04-17 아카이브(읽기 전용)**됨.
> Google Trends 공식 API는 2025-07-24 알파 공개(접근 제한형).

| 계층 | Provider | 조건 | 비용 | 우선순위 |
|------|----------|------|------|---------|
| **1** | Google Trends 공식 API (alpha) | API 접근 권한 획득 시 | API 비용 | 최우선 |
| **2** | SerpAPI / ValueSERP | 공식 API 미접근 시 | ~$0.01/req | 대체 |
| **3** | Naver DataLab + 기존 소스 조합 | 유료 API 미사용 시 | 무료 | 최종 fallback |

- Phase 3 구현 시 계층 3(Naver DataLab fallback)부터 구현
- 공식 API 접근 권한 확보 시 계층 1로 전환
- **pytrends는 어떤 상황에서도 사용하지 않음** (아카이브된 비공식 API)

### 7.2 Circuit Breaker 패턴 (Red Team 제안 반영)

외부 API(Google Trends, NewsData, YouTube 등) 장애 시 전체 파이프라인 중단을 방지하는 회로 차단기:

```python
class CircuitBreaker:
    """외부 API 장애 시 자동 차단 및 복구 (v1.3: 점증 backoff + Half-Open 제한)"""

    # v1.3: 점증 backoff (Consultant 권고)
    _BACKOFF_DURATIONS = [60, 300, 900]  # 1분 → 5분 → 15분

    def __init__(self, failure_threshold: int = 3, half_open_max_requests: int = 2):
        self._failure_counts: dict[str, int] = {}
        self._open_since: dict[str, float] = {}
        self._backoff_level: dict[str, int] = {}  # v1.3: 점증 레벨
        self._half_open_requests: dict[str, int] = {}  # v1.3: Half-Open 시험 요청 수
        self._threshold = failure_threshold
        self._half_open_max = half_open_max_requests

    def is_open(self, service: str) -> bool:
        """해당 서비스의 회로가 열려있는지 (차단 상태) 확인"""
        if service not in self._open_since:
            return False
        level = self._backoff_level.get(service, 0)
        timeout = self._BACKOFF_DURATIONS[min(level, len(self._BACKOFF_DURATIONS) - 1)]
        elapsed = time.time() - self._open_since[service]
        if elapsed > timeout:
            # Half-open: 시험 요청 제한 (v1.3: 최대 2건)
            self._half_open_requests[service] = 0
            del self._open_since[service]
            return False
        return True

    def is_half_open_allowed(self, service: str) -> bool:
        """Half-Open 상태에서 시험 요청 허용 여부 (v1.3)"""
        count = self._half_open_requests.get(service, 0)
        if count < self._half_open_max:
            self._half_open_requests[service] = count + 1
            return True
        return False

    def record_failure(self, service: str) -> None:
        """실패 기록 — threshold 초과 시 회로 열기 + backoff 레벨 증가"""
        self._failure_counts[service] = self._failure_counts.get(service, 0) + 1
        if self._failure_counts[service] >= self._threshold:
            self._open_since[service] = time.time()
            self._backoff_level[service] = self._backoff_level.get(service, 0) + 1

    def record_success(self, service: str) -> None:
        """성공 기록 — 카운터 및 backoff 레벨 리셋"""
        self._failure_counts[service] = 0
        self._backoff_level[service] = 0
        self._open_since.pop(service, None)
        self._half_open_requests.pop(service, None)
```

- 모든 외부 소스(BaseTrendSource 하위 클래스)에 Circuit Breaker 적용
- 3회 연속 실패 시 해당 소스 5분간 차단 (나머지 소스로 계속)
- 기존 `safe_fetch_with_status()` graceful degradation과 보완적으로 동작

---

## 8. 구현 로드맵 (우선순위)

### Phase 1: Critical 결함 수정 (예상 3-4일)

> **목표**: 고정 쿼리 제거 + 카테고리 관통 전달

| 작업 | 변경 대상 | 결함 # | 설명 |
|------|----------|--------|------|
| 1-1 | `collector.py` | #1, #2 | `_build_keyword_query(category)` 동적 쿼리 빌더 구현 |
| 1-2 | `schema/__init__.py` | #2 | `KeywordCollectRequest.category` 필드 추가 |
| 1-3 | `content_marketing_service.py` | #2 | category를 `collect_community_keywords_with_status()`에 전달 |
| 1-4 | `collector.py` | #6 | `time_range` 파라미터를 수집/검색 전 과정에 관통 전달 |
| 1-5 | `newsdata_source.py` | #6 | `_TIMEFRAME_MAP` 한계 명시 + 48h 초과 시 로깅 |
| 1-6 | Frontend | #2 | `KeywordCollector` 컴포넌트에 카테고리 선택 UI 추가 |

### Phase 2: 핵심 개선 (예상 5-6일)

> **목표**: YouTube 통계 + RRF + Engagement + legal_score fallback + Query Expansion + Shorts 보정

| 작업 | 변경 대상 | 결함 # | 설명 |
|------|----------|--------|------|
| 2-1 | `youtube_source.py` | #3, #9 | 2단계 API (search → videos.list statistics) |
| 2-2 | `models.py` | #3 | `RawTrendItem`에 `view_count`, `comment_count`, `engagement_velocity` 추가 |
| 2-3 | `collector.py` | #10 | Weighted RRF 소스 통합 구현 |
| 2-4 | `article_scorer.py` | #7 | `_calculate_legal_v2()` 3단계 fallback |
| 2-5 | `article_scorer.py` | #3 | `_calculate_engagement()` 구현 |
| 2-6 | `schema/__init__.py` | #3 | `NewsArticle`에 `engagement_score`, `view_count` 필드 추가 |
| 2-7 | Frontend | #3 | 뉴스 기사 카드에 조회수/참여도 표시 |
| 2-8 | `collector.py` | - | **Query Expansion (LLM 기반)**: 키워드 → 법률 전문 용어 불리언 쿼리 확장 (선택적 "심층 검색" 옵션). Red Team 제안. |
| 2-9 | `youtube_source.py` | #3 | **Shorts viewCount 보정**: `contentDetails` part 추가, Shorts 판별 + 정규화 계수 적용. Consultant 제안. |

### Phase 3: 고도화 (예상 6-7일)

> **목표**: Google Trends (Provider Abstraction) + KR-WordRank + Semantic Dedup + 수렴 탐지 + 보안 강화

| 작업 | 변경 대상 | 결함 # | 설명 |
|------|----------|--------|------|
| 3-1 | `sources/google_trends_source.py` (NEW) | #8 | Provider Abstraction 3계층 Google Trends 소스 (§7.1). **pytrends 미사용**. DataLab fallback 우선 구현. |
| 3-2 | `sources/naver_datalab_source.py` (NEW) | - | 네이버 DataLab 트렌드 소스 구현 (Google Trends fallback 계층 3의 핵심) |
| 3-3 | `keyword_extractor.py` | - | `HybridKeywordExtractor` (KR-WordRank + LLM) |
| 3-4 | `collector.py` | #4 | `SemanticDeduplicator` 구현 (KURE-v1 임베딩 + URL fallback) |
| 3-5 | `collector.py` | #5 | 크로스플랫폼 수렴 탐지 + `convergence_score` 계산 |
| 3-6 | `article_scorer.py` | #5 | 5차원 스코어링 (convergence 포함) 반영 |
| 3-7 | `scorer.py` | - | `SemanticCategoryMatcher` 임베딩 카테고리 매칭 |
| 3-8 | Frontend | - | 키워드 수렴 점수 시각화 (뱃지/태그) |
| 3-9 | `collector.py` | - | Circuit Breaker 패턴 적용 (§7.2) — 모든 외부 소스에 적용 |
| 3-10 | `collector.py`, `content_marketing_service.py` | - | **입력 세니타이징**: 카테고리/페르소나 화이트리스트 검증, 검색 쿼리 특수문자 필터링 (Red Team 제안) |
| 3-11 | `content_marketing_service.py` | - | **수집 텍스트 세니타이징**: LLM 스코어링 전 프롬프트 인젝션 패턴 제거 (Red Team 제안) |

---

## 9. 리스크 분석

| 리스크 | 영향 | 확률 | 심각도 | 완화 전략 |
|--------|------|------|--------|----------|
| **[CRITICAL] pytrends 아카이브** | Google Trends 소스 사용 불가 | 확정 | Critical | ~~pytrends 사용 금지~~. Provider Abstraction 3계층으로 전환 (§7.1). Phase 3에서 DataLab fallback 우선 구현. |
| **[HIGH] Dynamic Query Injection** | 검색 API 특수 연산자 주입으로 데이터 노출/할당량 소모 | 중간 | High | `_build_keyword_query()`에 입력 세니타이징 적용. 카테고리/페르소나 값을 화이트리스트 방식으로 검증. 특수문자(`"`, `OR`, `AND` 등) 필터링. (Red Team 제안) |
| **[HIGH] API Key/Credential 노출** | 다수 외부 소스 연동 시 대규모 과금 피해 | 낮음 | High | Secret Rotation 정책 수립 (90일 주기), 환경변수 관리 강화, API 키별 사용량 알림 설정. (Red Team 제안) |
| **[MEDIUM] Prompt Injection via Content** | 수집된 뉴스/커뮤니티 본문에 프롬프트 인젝션 유도 텍스트 포함 | 중간 | Medium | LLM 스코어링 전 수집 텍스트 세니타이징: `<`, `>`, 시스템 프롬프트 패턴 제거. 인젝션 탐지 필터 추가. (Red Team 제안) |
| **YouTube API 할당량 초과** | 통계 수집 불가 | 중간 | Medium | 2단계 API 호출: search.list(100) + videos.list(1) = 101 유닛/호출, 기본 10,000 유닛/일 → ~99회/일 가능. 일일 할당량 모니터링 + 캐싱. (Backend 분석 반영) |
| **외부 소스 장애** | 수집 결과 감소 | 중간 | Medium | Circuit Breaker (§7.2) + 기존 `safe_fetch_with_status()` 이중 보호. 최소 2소스 성공 시 정상 처리. |
| **LLM 비용 증가** | 운영비 상승 | 낮음 | Low | KR-WordRank로 1차 추출 → LLM은 스코어링에만 사용 (호출 횟수 유지 또는 감소) |
| **임베딩 계산 지연** | Semantic Dedup 속도 저하 | 중간 | Medium | 30건 이상일 때만 활성화 + 배치 처리(10건 단위). KURE-v1 미로드 시 URL dedup fallback. |
| **법률 도메인 특수성** | 일반 트렌드와 법률 트렌드 괴리 | 중간 | Medium | `legal_relevance` 가중치 0.30 (최고), Legal Gate(0.3 임계값) 유지 |
| **네이버 DataLab API 변경** | 트렌드 추이 수집 불가 | 낮음 | Low | DataLab은 보조 소스, 핵심은 Naver News + 기타 소스 |
| **KR-WordRank 한국어 품질** | 키워드 추출 정확도 | 낮음 | Low | LLM 스코어링으로 2차 필터 (legal_relevance < 0.3 제거) |
| **Rate Limit 집중** | 병렬 수집 시 IP 차단/Quota 소모 | 중간 | Medium | 소스별 요청 간격 조절 + Circuit Breaker 자동 차단. (Red Team 제안) |
| **YouTube Shorts viewCount 정의 변경** | 조회수 팽창으로 engagement 편향 | 확정 | Medium | Shorts 판별 + 정규화 축소 계수(0.3) 적용. `contentDetails.duration` 기반 분류. (Consultant 제안) |

---

## 10. KPI

### 10.1 정량 KPI

| 지표 | 현재 (추정) | Phase 1 목표 | Phase 2 목표 | Phase 3 목표 | 측정 방법 |
|------|-----------|-------------|-------------|-------------|----------|
| **카테고리 일치율** | 40~50% | 70% | 80% | 85%+ | 수동 평가 100건 |
| **뉴스 관련도 평균** | 30~40점 | 45+ | 55+ | 65+ | `total_score` 평균 |
| **legal_score > 0 비율** | ~60% (RAG 의존) | 80% | 90% | 95%+ | RAG 실패 시에도 키워드 매칭 보정 |
| **크로스플랫폼 수렴 탐지** | 0% (미지원) | - | - | 30%+ 키워드에 convergence > 0.5 | convergence_score 분포 |
| **키워드 수집 LLM 비용** | ~$0.03/수집 | 유지 | 유지 | ~$0.01/수집 (KR-WordRank) | LLM 호출 횟수 |
| **YouTube 통계 수집률** | 0% | - | 90%+ | 95%+ | videos.list 성공률 |
| **중복 기사 비율** | ~20% (URL만 제거) | ~15% | ~10% | ~5% (Semantic Dedup) | 수동 확인 50건 |

### 10.2 정성 KPI

| 지표 | 현재 | 목표 |
|------|------|------|
| 네이버 뉴스 베스트 대비 | 열위 (고정 쿼리, 카테고리 무시) | Phase 3 완료 후 동등 이상 |
| 사용자 키워드 만족도 | 측정 불가 | 피드백 시스템 도입 후 4.0/5.0+ |
| 법률 전문가 관점 품질 | 법률 무관 키워드 혼재 | legal_relevance 0.3+ 게이트로 품질 보장 |

### 10.3 실험 체계 (Consultant 제안 반영)

> "정확도 개선"이 지속 가능하게 증명되려면 오프라인/온라인 이중 평가 체계가 필요.

#### 오프라인 평가 (Phase 2부터)

| 지표 | 설명 | 목표 |
|------|------|------|
| **NDCG@10** | 검색 결과 순위 품질 | 0.65+ |
| **MRR** | 첫 관련 결과 순위 역수 | 0.55+ |
| **Precision@5** | 상위 5건 중 관련 기사 비율 | 0.70+ |

- 평가 데이터셋: 카테고리별 20개 키워드 × 10건 기사 = 200건 수동 라벨링
- Phase별 비교: Phase N vs Phase N-1 지표 비교 자동화

#### 온라인 평가 (Phase 3 이후)

| 지표 | 설명 | 측정 방법 |
|------|------|----------|
| **클릭률 (CTR)** | 키워드/기사 선택 비율 | 프론트엔드 이벤트 로깅 |
| **제작 전환율** | 키워드 → 대본 생성 전환 | 기존 스크립트 생성 데이터 |
| **세션 시간** | 키워드 탐색 평균 시간 | 프론트엔드 세션 추적 |

- A/B 테스트: 동일 카테고리에서 기존 vs 개선 알고리즘 비교 (Phase 3 이후)

---

## 11. 데이터 모델 변경 요약

### 11.1 RawTrendItem 확장

```python
@dataclass
class RawTrendItem:
    """소스별 수집된 원시 트렌드 항목 (v3.0 확장)"""

    title: str
    url: str
    snippet: str
    source: TrendSource
    published_at: datetime | None = None
    mention_count: int = 0
    raw_data: dict[str, object] = field(default_factory=dict)
    # v3.0 NEW
    view_count: int | None = None          # YouTube 조회수
    comment_count: int | None = None       # YouTube 댓글수
    like_count: int | None = None          # YouTube 좋아요수
    engagement_velocity: float | None = None  # views / hours
    merged_sources: list[TrendSource] = field(default_factory=list)  # Semantic Dedup 병합 소스
```

### 11.2 NewsArticle 확장

```python
class NewsArticle(BaseModel):
    # 기존 필드 유지
    title: str
    url: str
    source: str
    published_at: datetime | None = None
    snippet: str = ""
    # 기존 점수
    relevance_score: float = 0.0
    legal_score: float = 0.0
    recency_score: float = 0.0
    total_score: float = 0.0
    # v3.0 NEW
    engagement_score: float = 0.0         # Engagement velocity 점수
    convergence_score: float = 0.0        # 크로스플랫폼 수렴 점수
    view_count: int | None = None         # YouTube 조회수
    comment_count: int | None = None      # YouTube 댓글수
    source_weight: float = 0.5            # RRF 소스 권위 가중치 (Frontend에서 표시용)
```

### 11.3 KeywordCollectRequest 확장

```python
class KeywordCollectRequest(BaseModel):
    max_keywords: int = Field(default=10, ge=1, le=30)
    time_range: TimeRange = TimeRange.HOURS_48
    category: TrendCategory = TrendCategory.ALL  # NEW
    community_domains: list[str] | None = None
```

---

## 12. API 변경 요약

### 12.1 기존 API 변경

| 엔드포인트 | 변경 사항 |
|-----------|----------|
| `POST /keywords/collect` | Request Body에 `category` 필드 추가 |
| `GET /keywords/collect/stream` | Query Parameter에 `category` 추가 |
| `POST /keywords/{keyword_id}/news` | Response에 `engagement_score`, `convergence_score`, `source_weight` 추가 |

### 12.2 신규 API (Phase 3 이후 고려)

| 엔드포인트 | 설명 |
|-----------|------|
| `GET /keywords/trending` | Google Trends 실시간 급상승 키워드 조회 (Phase 3) |

---

## 13. 프론트엔드 변경 요약

### Phase 1

- `KeywordCollector`: 카테고리 선택 드롭다운 추가 (TrendCategory enum 기반)
- API 호출 시 `category` 파라미터 전달

### Phase 2

- `KeywordNewsList`: 기사 카드에 조회수/참여도 뱃지 표시
- 정렬 옵션 추가: `total_score`, `engagement_score`, `recency_score`
- `KeywordNewsList`: 소스 권위 가중치(`source_weight`) 표시 — Backend 응답 값 사용 (프론트엔드 하드코딩 금지, Frontend 피드백 반영)

### Phase 3

- 키워드 카드에 `convergence_score` 시각화 (수렴 뱃지)
- 트렌딩 키워드 실시간 배너 (Google Trends)

---

## 변경 이력

| 날짜 | 버전 | 변경 내용 |
|------|------|----------|
| 2026-03-01 | 1.0 | 초안 작성 |
| 2026-03-01 | 1.1 | **Iteration 1**: Red Team + External Consultant + Backend/Frontend 피드백 반영. 주요 변경: (1) pytrends 제거 → Provider Abstraction 3계층 전환 [CRITICAL], (2) Circuit Breaker 패턴 추가, (3) 보안 강화 — Dynamic Query Injection/Prompt Injection 대응, (4) RRF Naver 가중치 1.0→1.2, (5) YouTube 할당량 상세 (101유닛/호출, ~99회/일), (6) Z-score velocity 추가, (7) KURE-v1 fallback 전략, (8) 실험 체계 (NDCG/MRR/Precision), (9) source_weight API 필드 추가 |
| 2026-03-01 | 1.2 | **Iteration 1 보완**: team-lead 추가 수정 요청 반영. (1) LLM 기반 Query Expansion — Phase 2 작업 2-8 배치, (2) YouTube Shorts viewCount 정의 변경(2025-03-31) 대응 — Shorts 판별 + 정규화 축소 계수, Phase 2 작업 2-9, (3) Early Signal Capture — 네이버보다 30분~1시간 빠른 이슈 선점 전략 비전에 추가 |
| 2026-03-01 | 1.3 | **Iteration 2**: Red Team + External Consultant 재검증 피드백 반영. (1) Circuit Breaker 점증 backoff (1분→5분→15분) + Half-Open 시험 요청 제한(2건), (2) Semantic Dedup 2단계 (>0.90 자동, 0.82~0.90 검증) + 활성화 임계 30건→15건 하향, (3) legal_score max→가중 결합 (keyword 0.2, RAG 0.4, LLM 0.4) + confidence 메타데이터 보존, (4) Shorts 계수 고정 0.3→engaged_views/views 동적 보정 (0.15~0.60), (5) 전체 등급: B+ (Consultant 평가) |
