# Content Marketing Keyword Flow 기획 보고서

> **Summary**: 트렌드 수집 파이프라인을 "키워드 수집 -> 사용자 선택 -> 뉴스 검색" 3단계 인터랙티브 플로우로 리팩토링
>
> **Project**: 법률 서비스 플랫폼
> **Author**: CTO Lead
> **Date**: 2026-02-25
> **Status**: Final (v1.0)

---

## 1. 개요

### 1.1 목적

현재 콘텐츠 마케팅 트렌드 파이프라인은 5단계를 한 번에 실행하여 사용자에게 최종 결과만 보여준다. 이 방식은 다음과 같은 문제를 가진다.

1. **API 비용 낭비**: 모든 키워드에 대해 6개 소스를 병렬 호출하므로, 사용자가 관심 없는 키워드에도 API 호출이 발생한다.
2. **대기 시간 과다**: Tavily 수집 + LLM 키워드 추출 + 6개 소스 병렬 검색 + LLM 스코어링 + LLM 요약 + RAG 매칭이 순차/병렬로 실행되어 전체 응답 시간이 길다.
3. **사용자 제어 부재**: 어떤 키워드가 수집되었는지 확인할 수 없고, 원하는 키워드를 선택할 수 없다.

본 기획은 파이프라인을 사용자 인터랙션이 포함된 3단계로 분리하여 위 문제를 해결한다.

### 1.2 배경

콘텐츠 마케팅 모듈은 법률 유튜브 채널 운영 변호사를 위한 트렌드 분석 도구이다. 핵심 가치는 "지금 사람들이 관심을 갖는 법률 관련 사건/사고를 빠르게 파악하고, 그에 대한 뉴스 기사를 수집하여 콘텐츠 제작에 활용"하는 것이다.

대한민국 마이너 커뮤니티(DC갤러리, 에펨코리아, 더쿠, 보배드림, 인벤 등)에서 화제가 되는 사건/사고를 키워드로 추출하고, 사용자가 그 중 관심 있는 키워드를 선택한 뒤에야 뉴스 기사를 검색하는 온디맨드 방식으로 전환한다.

### 1.3 관련 문서

- 기존 기획: `docs/01-plan/features/content-marketing.plan.md`
- v2.0 기획: `docs/01-plan/features/content-marketing-v2.plan.md`
- v2.0 설계: `docs/02-design/features/content-marketing-v2.design.md`

---

## 2. AS-IS / TO-BE 비교

### 2.1 AS-IS (현재 구현)

```
[POST /api/content-marketing/trends] ─── 단일 호출, 전체 파이프라인 실행 ───

Stage 1: Tavily → 커뮤니티 인기글 수집 (TavilySource.fetch)
    |
Stage 2: LLM → 키워드 3~5개 추출 (CommunityKeywordExtractor.extract)
    |
Stage 3: Naver/YouTube/Perplexity/Google/NewsData/NewsAPI → 키워드 기반 병렬 검색
    |     (TrendCollector._collect_news → 6개 소스 동시 호출)
    |
Stage 4: LegalGateScorer → 5차원 + Legal Gate 스코어링
    |     (LLM 통합 분석 → 그룹화 + 가중합 계산)
    |
Stage 5: IssueSummarizer → LLM 요약 + RAG 법령/판례 매칭
    |     (병렬: LLM key_points + LLM summary + RAG 법령/판례)
    v
TrendResponse → 트렌드 카드 그리드 표시
```

**문제점:**
- 1회 호출에 Tavily 1회 + LLM 2회(키워드 추출 + 스코어링) + 뉴스 소스 6회 + LLM N회(요약) + RAG N회 = 매우 많은 API 호출
- 전체 파이프라인 응답 시간: 약 30~120초
- 사용자가 키워드를 제어할 수 없음

### 2.2 TO-BE (변경 후)

```
[Step 1] POST /api/content-marketing/keywords/collect
    |
    | Tavily → 마이너 커뮤니티 화제 사건/사고 수집
    | LLM → 키워드 정제 + 4차원 점수 산출
    |
    v
KeywordCollectResponse → 키워드 카드 목록 표시 (사용자에게 먼저 보여줌)
    |
    | *** 사용자가 키워드 1개를 선택 ***
    |
    v
[Step 2] POST /api/content-marketing/keywords/{keyword_id}/news
    |
    | 설정된 검색 API(Naver/Google/NewsData/NewsAPI 등) → 선택 키워드 뉴스 10개 검색
    |
    v
KeywordNewsResponse → 뉴스 기사 10개 목록 표시
```

**개선점:**
- Step 1은 Tavily 1회 + LLM 1회만 호출 (빠른 응답)
- Step 2는 사용자가 선택한 키워드에 대해서만 뉴스 검색 (API 절감)
- 사용자가 키워드를 보고 선택할 수 있음 (제어권 확보)
- 기존 LegalGateScorer(Stage 4)와 IssueSummarizer(Stage 5)는 제거

### 2.3 변경 요약 매트릭스

| 항목 | AS-IS | TO-BE | 변경 사유 |
|------|-------|-------|----------|
| 파이프라인 단계 | 5단계 일괄 실행 | 2단계 + 사용자 인터랙션 | 사용자 제어권 + API 절감 |
| API 엔드포인트 | POST /trends 1개 | POST /keywords/collect + POST /keywords/{id}/news 2개 | 단계 분리 |
| Tavily 수집 대상 | 일반 커뮤니티 | 마이너 커뮤니티 (DC, 에펨, 더쿠, 보배, 인벤) | 화제 사건/사고 특화 |
| 키워드 추출 | 3~5개, 점수 없음 | 키워드별 4차원 점수 (화제성, 사회적영향도, 법적연관성, 콘텐츠적합도) | 사용자 판단 지원 |
| 뉴스 검색 | 모든 키워드 일괄 | 선택 키워드만 온디맨드 | API 비용 절감 |
| 스코어링 (LegalGateScorer) | 5차원 + Legal Gate | 제거 (키워드 점수로 대체) | 파이프라인 단순화 |
| 요약 (IssueSummarizer) | LLM 요약 + RAG 매칭 | 제거 | 뉴스 기사 직접 표시로 대체 |
| 응답 시간 (Step 1) | 30~120초 | 5~15초 | Tavily + LLM 1회만 |
| 뉴스 소스 호출 횟수 | 키워드 수 x 소스 수 | 1 키워드 x 소스 수 | 최대 80% 절감 |

---

## 3. 변경 범위

### 3.1 Backend API

| 구분 | 파일 | 변경 유형 | 설명 |
|------|------|----------|------|
| 라우터 | `backend/app/modules/content_marketing/router/__init__.py` | 수정 | 신규 엔드포인트 2개 추가 |
| 스키마 | `backend/app/modules/content_marketing/schema/__init__.py` | 수정 | 키워드 관련 스키마 추가 |
| 서비스 | `backend/app/services/service_function/content_marketing_service.py` | 수정 | 키워드 수집/뉴스 검색 서비스 함수 추가 |
| 키워드 추출 | `backend/app/tools/trend/keyword_extractor.py` | 수정 | 키워드 점수 산출 기능 추가 |
| Tavily 소스 | `backend/app/tools/trend/sources/tavily_source.py` | 수정 | 마이너 커뮤니티 특화 쿼리 |
| 수집기 | `backend/app/tools/trend/collector.py` | 수정 | 키워드 수집 전용 메서드 + 뉴스 검색 전용 메서드 분리 |
| 내부 모델 | `backend/app/tools/trend/models.py` | 수정 | ScoredKeyword 데이터클래스 추가 |
| Blacklist | `backend/app/tools/trend/keyword_blacklist.py` | 신규 | 키워드 Sanitize + Blacklist 필터링 |
| Blacklist 설정 | `backend/app/tools/trend/blacklist.json` | 신규 | 차단 키워드 목록 (exact/contains/regex) |
| Rate Limiter | `backend/app/tools/trend/rate_limiter.py` | 신규 | 인메모리 Rate Limiting |

### 3.2 Frontend

| 구분 | 파일 | 변경 유형 | 설명 |
|------|------|----------|------|
| 타입 | `frontend/src/features/content-marketing/types/index.ts` | 수정 | 키워드/뉴스 타입 추가 |
| 서비스 | `frontend/src/features/content-marketing/services/index.ts` | 수정 | 키워드/뉴스 API 호출 함수 추가 |
| 훅 | `frontend/src/features/content-marketing/hooks/useKeywordFlow.ts` | 신규 | 키워드 플로우 상태 관리 훅 |
| 컴포넌트 | `frontend/src/features/content-marketing/components/KeywordCollector.tsx` | 신규 | 키워드 카드 목록 + 선택 UI |
| 컴포넌트 | `frontend/src/features/content-marketing/components/KeywordCard.tsx` | 신규 | 개별 키워드 카드 (점수 표시) |
| 컴포넌트 | `frontend/src/features/content-marketing/components/KeywordNewsList.tsx` | 신규 | 뉴스 기사 10개 목록 |
| 대시보드 | `frontend/src/features/content-marketing/components/TrendDashboard.tsx` | 수정 | 키워드 플로우 탭/뷰 통합 |

### 3.3 변경하지 않는 파일 (기존 유지)

| 파일 | 사유 |
|------|------|
| `backend/app/tools/trend/scorer.py` | 기존 TrendScorer, LegalGateScorer는 그대로 유지 (기존 /trends API 하위호환) |
| `backend/app/tools/trend/summarizer.py` | 기존 IssueSummarizer는 그대로 유지 (기존 /trends API 하위호환) |
| `backend/app/tools/trend/sources/naver_source.py` 외 5개 | 뉴스 소스는 기존 인터페이스 그대로 재사용 |
| `frontend/src/features/content-marketing/components/TrendCard.tsx` | 기존 트렌드 카드는 그대로 유지 |
| 페르소나 관련 전체 | 페르소나 시스템은 변경 없음 |

---

## 4. 신규/수정 API 엔드포인트

### 4.1 [NEW] POST /api/content-marketing/keywords/collect

키워드 수집 + 정제 + 점수 산출.

**Rate Limiting:** 사용자별 1시간 5회 제한. 초과 시 HTTP 429 + `Retry-After` 헤더 반환.

**Request:**
```json
{
  "time_range": "48h",
  "category": "all",
  "max_keywords": 10,
  "persona_id": null
}
```

**Response (KeywordCollectResponse):**
```json
{
  "keywords": [
    {
      "id": "kw-uuid-1",
      "keyword": "음주운전 역주행 사망사고",
      "context": "DC갤러리, 보배드림에서 화제. 12시간 내 글 50+ 작성",
      "score_reason": "커뮤니티 3곳에서 동시 화제, 위험운전치사 법적 쟁점 명확, 유튜브 해설 수요 높음",
      "source_posts": [
        {
          "title": "역주행 음주운전 사건 관련 커뮤니티 반응",
          "url": "https://...",
          "source": "tavily",
          "snippet": "..."
        }
      ],
      "scores": {
        "virality": 0.92,
        "social_impact": 0.85,
        "legal_relevance": 0.78,
        "content_fitness": 0.88
      },
      "total_score": 86.2,
      "confidence": 0.85,
      "rank": 1
    },
    {
      "id": "kw-uuid-2",
      "keyword": "전세사기 피해자 구제법",
      "context": "에펨코리아, 더쿠에서 논의 활발",
      "score_reason": "전세사기특별법 관련 법적 쟁점 풍부, 피해자 구제 관심도 높음",
      "source_posts": [...],
      "scores": {
        "virality": 0.75,
        "social_impact": 0.90,
        "legal_relevance": 0.95,
        "content_fitness": 0.82
      },
      "total_score": 85.5,
      "confidence": 0.90,
      "rank": 2
    }
  ],
  "total_count": 8,
  "collected_at": "2026-02-25T10:00:00Z",
  "community_sources_scanned": ["dcinside", "fmkorea", "theqoo", "bobaedream", "inven"],
  "prompt_version": "keyword-extract-v1.0",
  "model_version": "solar-pro2-preview"
}
```

**처리 흐름:**
1. Rate Limit 확인 (사용자별 `collect` 버킷: 1시간 5회)
2. **Tavily + Naver 뉴스 하이브리드 수집**
   - Tavily로 마이너 커뮤니티 인기글 수집 (커뮤니티 사이트 도메인 포함 쿼리)
   - Naver 뉴스에서 "커뮤니티 화제" 관련 기사를 추가 수집 (커버리지 보완)
   - 두 소스 결과를 병합하여 LLM 입력으로 전달
3. LLM이 수집된 글에서 사건/사고 키워드를 추출하고 4차원 점수 + 점수 근거(`score_reason`)를 산출
4. **Query Sanitization**: LLM 추출 키워드에 Blacklist 필터링 + 특수문자 제거 적용
5. 점수 기준 정렬 후 응답 (`prompt_version`, `model_version`, `confidence` 포함)

**소요 시간 예상:** 5~15초 (Tavily 1회 + LLM 1회)

### 4.2 [NEW] POST /api/content-marketing/keywords/{keyword_id}/news

선택된 키워드에 대한 뉴스 기사 검색.

**Rate Limiting:** 사용자별 `news` 버킷: 1시간 30회 제한 (collect 버킷과 분리). 초과 시 HTTP 429.
**Query Sanitization:** 캐시에서 조회한 키워드 문자열을 뉴스 검색 API에 전달하기 전 Sanitize 필터 적용.

**Request:**
```json
{
  "search_sources": ["naver", "google_trends", "newsdata", "newsapi"],
  "max_results": 10
}
```

**Response (KeywordNewsResponse):**
```json
{
  "keyword_id": "kw-uuid-1",
  "keyword": "음주운전 역주행 사망사고",
  "news_articles": [
    {
      "title": "음주 역주행 사고, 위험운전치사 혐의 적용 여부 논란",
      "url": "https://...",
      "source": "naver",
      "published_at": "2026-02-25T08:30:00Z",
      "snippet": "경찰은 음주 상태에서 역주행하여..."
    }
  ],
  "related_laws": [
    {
      "law_name": "도로교통법 제148조의2",
      "issue_label": "음주운전 처벌 강화 (윤창호법)"
    },
    {
      "law_name": "특정범죄가중처벌법 제5조의11",
      "issue_label": "위험운전치사상"
    }
  ],
  "total_count": 10,
  "sources_used": ["naver", "google_trends", "newsdata"],
  "searched_at": "2026-02-25T10:05:00Z"
}
```

**처리 흐름:**
1. Rate Limit 확인 (사용자별 `news` 버킷: 1시간 30회)
2. 캐시에서 keyword_id로 원본 키워드 문자열 조회 (user_id 기반 세션 격리)
3. **Query Sanitization**: 키워드 문자열에 Sanitize 필터 적용 (특수문자/인젝션 패턴 제거)
4. 설정된 검색 소스로 병렬 뉴스 검색 (소스별 차등 타임아웃 적용)
5. **경량 법률 enrichment**: 기존 RAGPipeline을 활용하여 키워드 관련 법령 1~2개 + 쟁점 라벨을 조회 (응답에 `related_laws` 포함)
6. 중복 제거 + 최신순 정렬 후 상위 10개 반환

**소요 시간 예상:** 3~8초 (뉴스 소스 병렬 호출)

### 4.3 기존 API 유지 (하위호환)

| 엔드포인트 | 상태 | 설명 |
|-----------|------|------|
| `POST /api/content-marketing/trends` | 유지 | 기존 5단계 파이프라인 그대로 작동 |
| `GET /api/content-marketing/trends/{trend_id}` | 유지 | 기존 상세 조회 |
| Persona API 5개 | 유지 | 변경 없음 |
| Script API 2개 | 유지 | 변경 없음 |

---

## 5. 키워드 스코어링 기준

### 5.1 4차원 점수 체계

LLM이 커뮤니티 인기글 맥락에서 추출한 각 키워드에 대해 4가지 차원의 점수를 산출한다.

| 차원 | 필드명 | 범위 | 설명 | 가중치 |
|------|--------|------|------|--------|
| 화제성 | `virality` | 0.0~1.0 | 커뮤니티에서의 관심도, 게시글/댓글 활성도 | 0.25 |
| 사회적 영향도 | `social_impact` | 0.0~1.0 | 해당 이슈가 사회에 미치는 영향 규모 | 0.25 |
| 법적 연관성 | `legal_relevance` | 0.0~1.0 | 법률적 분석/해석이 가능한 정도 | 0.30 |
| 콘텐츠 적합도 | `content_fitness` | 0.0~1.0 | 법률 유튜브 콘텐츠로 제작하기 적합한 정도 | 0.20 |

### 5.2 종합 점수 산출

```
total_score = (0.25 * virality + 0.25 * social_impact + 0.30 * legal_relevance + 0.20 * content_fitness) * 100
```

- 범위: 0.0 ~ 100.0
- 정렬: total_score 내림차순

### 5.3 LLM 프롬프트 설계 (개요)

**LLM 호출 파라미터:**
- `temperature`: **0** (점수 재현성 확보)
- `response_format`: Upstage Solar Pro2가 Structured Outputs(`response_format`)을 지원하면 JSON Schema 강제 적용. 미지원 시 기존 3단계 폴백(raw JSON 파싱 -> 정규식 추출 -> 폴백 균일값) 유지
- 응답에 `prompt_version`, `model_version`을 기록하여 감사 추적 가능

```
당신은 한국 법률 콘텐츠 전략가입니다.
아래 온라인 커뮤니티 인기글 목록에서 사회적으로 화제가 된 사건/사고 키워드를 추출하세요.

[커뮤니티 인기글 목록]
{context}

각 키워드에 대해 JSON 배열로 응답하세요:
[
  {
    "keyword": "키워드 (2~8단어)",
    "context": "커뮤니티에서 이 키워드가 화제인 이유 (1문장)",
    "score_reason": "이 점수를 부여한 근거 (1~2문장, 구체적 사실 기반)",
    "virality": 0.0~1.0,
    "social_impact": 0.0~1.0,
    "legal_relevance": 0.0~1.0,
    "content_fitness": 0.0~1.0
  }
]

규칙:
- 사건/사고/사회적 논란 중심으로 5~10개 키워드 추출
- 게임, 유머, 엔터테인먼트, 스포츠 주제는 제외
- legal_relevance: 관련 법령/판례가 존재하면 0.5+, 법적 쟁점이 명확하면 0.7+
- content_fitness: 법률 유튜브 영상 주제로 적합하면 0.7+
- score_reason: 점수의 구체적 근거를 사실 기반으로 1~2문장 작성 (예: "커뮤니티 3곳에서 동시 화제, 위험운전치사 법적 쟁점 명확")
- **유사 이슈 통합**: 동일 사건의 다른 표현(예: "음주 역주행", "음주운전 역주행 사고")은 하나의 키워드로 통합하세요
- **제외 대상**: 혐오 표현, 특정 정치인/정당 편향, 선정적 표현이 포함된 키워드는 제외하세요
```

### 5.4 폴백 전략

LLM 호출 실패 시:
1. 기존 `CommunityKeywordExtractor._fallback_keywords()` 방식으로 제목 단어 빈도 기반 키워드 추출
2. 점수는 균일값(각 0.5) 할당
3. 사용자에게 "자동 분석 실패, 빈도 기반 추출 결과입니다" 안내

### 5.5 Keyword Blacklist 필터링

LLM 추출 결과와 폴백 결과 모두에 적용하는 사후 필터링 규칙.

**필터링 대상:**

| 카테고리 | 설명 | 예시 |
|---------|------|------|
| 혐오 표현 | 인종/성별/장애 등 차별적 표현 | 설정 파일에서 관리 |
| 정치 편향 | 특정 정당/정치인을 직접 겨냥한 편향적 키워드 | 설정 파일에서 관리 |
| 선정적 표현 | 음란/폭력적 표현이 주가 되는 키워드 | 설정 파일에서 관리 |
| 인젝션 패턴 | SQL/XSS/프롬프트 인젝션 시도 | `'; DROP`, `<script>`, `{{`, `{%` |

**구현 방식:**

```python
# backend/app/tools/trend/keyword_blacklist.py

BLACKLIST_PATTERNS: list[str] = []  # 설정 파일 또는 환경변수에서 로드
INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"['\";].*(?:DROP|DELETE|UPDATE|INSERT)", re.IGNORECASE),
    re.compile(r"<script", re.IGNORECASE),
    re.compile(r"\{\{|\{%"),
]

def sanitize_keyword(keyword: str) -> str | None:
    """키워드를 Sanitize하고, Blacklist에 해당하면 None 반환."""
    # 1. 특수문자 제거 (한글, 영문, 숫자, 공백만 허용)
    # 2. Blacklist 패턴 매칭 검사
    # 3. 인젝션 패턴 검사
    # 4. 길이 제한 (2~50자)
    ...
```

**적용 시점:**
1. LLM 키워드 추출 직후 (Step 1)
2. 뉴스 검색 API 호출 직전 (Step 2)

---

## 6. 데이터 플로우

### 6.1 Step 1: 키워드 수집 플로우

```
[사용자] POST /api/content-marketing/keywords/collect
    |
    v
[Router] → Rate Limit 확인 (user_id 기준, collect 버킷: 1시간 5회)
    |
    v
[Router] → content_marketing_service.collect_keywords(user_id)
    |
    v
[TrendCollector] collect_community_keywords()
    |
    +--→ [병렬 수집: Tavily + Naver 뉴스 하이브리드]
    |      |
    |      +--→ TavilySource.fetch()
    |      |      query: "대한민국 커뮤니티 화제 사건 사고 site:dcinside.com OR site:fmkorea.com OR ..."
    |      |      → RawTrendItem[] (커뮤니티 인기글)
    |      |      ※ dcinside.com 등 robots.txt Disallow:/ 사이트는 Tavily 직접 크롤링 불가 가능성 있음
    |      |
    |      +--→ NaverSource.fetch()
    |             query: "커뮤니티 화제 사건 사고"
    |             → RawTrendItem[] (커뮤니티 화제 관련 뉴스 기사)
    |             ※ Tavily 커버리지 부족분을 Naver 뉴스로 보완
    |
    +--→ 두 소스 결과 병합 + 중복 제거
    |
    +--→ CommunityKeywordExtractor.extract_with_scores()
    |      input: RawTrendItem[] (병합된 결과)
    |      → LLM 호출 (temperature=0, 키워드 + 4차원 점수 + score_reason JSON)
    |      → ScoredKeyword[]
    |      → prompt_version, model_version 기록
    |
    v
[Sanitize] keyword_blacklist.sanitize_keyword() 적용
    +--→ Blacklist 필터링 (혐오/편향/선정적 키워드 제거)
    +--→ 특수문자/인젝션 패턴 제거
    |
    v
[Service]
    +--→ ScoredKeyword[] → 점수 정렬 + UUID 부여
    +--→ 캐시 저장 ({user_id}:{keyword_id} → ScoredKeyword 매핑, 세션 격리)
    |
    v
[Response] KeywordCollectResponse (prompt_version, model_version, confidence 포함)
```

### 6.2 Step 2: 뉴스 검색 플로우

```
[사용자] keyword 카드 클릭 → POST /api/content-marketing/keywords/{keyword_id}/news
    |
    v
[Router] → Rate Limit 확인 (user_id 기준, news 버킷: 1시간 30회)
    |
    v
[Router] → content_marketing_service.search_keyword_news(user_id, keyword_id)
    |
    v
[Service]
    +--→ 캐시에서 {user_id}:{keyword_id} → keyword 문자열 조회 (세션 격리)
    |
    +--→ [Sanitize] 키워드 문자열에 sanitize_keyword() 적용
    |      → 특수문자/인젝션 패턴 제거 후 검색 쿼리로 사용
    |
    +--→ [병렬 실행]
    |      |
    |      +--→ TrendCollector.search_news_for_keyword()
    |      |      keyword: "음주운전 역주행 사망사고"
    |      |      sources: [NaverSource, GoogleSource, ...]
    |      |      → asyncio.wait_for(source.safe_fetch(...), timeout=소스별_타임아웃)
    |      |      → RawTrendItem[]
    |      |
    |      +--→ [경량 법률 enrichment] RAGPipeline.search_related_laws()
    |             keyword: "음주운전 역주행 사망사고"
    |             → 기존 LanceDB legal_chunks에서 관련 법령 검색
    |             → 상위 1~2개 법령명 + 쟁점 라벨 반환
    |             → RelatedLaw[] (예: {"law_name": "도로교통법 제148조의2", "issue_label": "음주운전 처벌 강화"})
    |             ※ 기존 RAGPipeline 활용, 추가 LLM 호출 없음 (벡터 검색만)
    |
    +--→ 중복 제거 + 최신순 정렬 + 상위 10개 선택
    +--→ related_laws 결과 병합
    |
    v
[Response] KeywordNewsResponse (related_laws 포함)
```

### 6.3 캐시 전략

| 캐시 대상 | 키 | TTL | 목적 |
|----------|-----|-----|------|
| 키워드 수집 결과 | `{user_id}:kw-collect:{time_range}:{category}` | 1시간 | 동일 조건 재요청 방지 + **세션 격리** |
| 키워드 → 뉴스 매핑 | `{user_id}:kw-news:{keyword_id}` | 30분 | 같은 키워드 뉴스 재요청 방지 + **세션 격리** |
| keyword_id → ScoredKeyword | `{user_id}:kw-meta:{keyword_id}` | 1시간 | Step 2에서 키워드 문자열 조회 + **세션 격리** |

캐시는 기존 `_cache` 딕셔너리 방식을 그대로 활용한다 (인메모리, 서버 재시작 시 초기화).

**IDOR 방지**: 모든 캐시 키에 `user_id`를 포함하여 다른 사용자의 키워드 데이터에 접근하는 것을 방지한다. `keyword_id`만으로는 캐시 조회가 불가능하며, 반드시 `{user_id}:{keyword_id}` 복합 키로만 조회한다.

> **향후 스케일 아웃 시 Redis 전환 로드맵**: 현재 단일 서버 환경에서는 인메모리 딕셔너리 + user_id 격리로 충분하다. 다중 서버 환경으로 전환 시 Redis 기반 분산 캐시로 마이그레이션한다.

---

## 7. UI/UX 변경

### 7.1 새로운 사용자 플로우

```
[트렌드 대시보드]
    |
    +--- [탭: 키워드 탐색] (NEW - 기본 탭)
    |       |
    |       +--- [수집 버튼] "커뮤니티 화제 키워드 수집"
    |       |       → 로딩 (5~15초)
    |       |       → 키워드 카드 목록 표시
    |       |
    |       +--- [키워드 카드 그리드] (2열)
    |       |       카드 내용:
    |       |       - 순위 번호
    |       |       - 키워드 텍스트 (굵게)
    |       |       - 맥락 설명 (1줄)
    |       |       - 4차원 스코어 바 (화제성/사회적영향/법적연관/콘텐츠적합)
    |       |       - 종합 점수
    |       |       - [뉴스 검색] 버튼
    |       |
    |       +--- [뉴스 목록 패널] (키워드 선택 후 표시)
    |               - 뉴스 기사 10개 리스트
    |               - 각 기사: 제목(링크), 소스 태그, 발행일, 스니펫
    |               - [대본 생성] 버튼 (뉴스 기사 기반)
    |
    +--- [탭: 트렌드 분석] (기존 - 하위호환)
            → 기존 TrendCard 그리드 (변경 없음)
```

### 7.2 키워드 카드 UI 명세

```
+--------------------------------------------+
| 1위                            종합 86.2점  |
|                                             |
| 음주운전 역주행 사망사고                       |
|                                             |
| DC갤러리, 보배드림에서 화제                    |
|                                             |
| 화제성      ████████████░░ 0.92             |
| 사회적영향   ████████████░░ 0.85             |
| 법적연관성   ██████████░░░░ 0.78             |
| 콘텐츠적합   ████████████░░ 0.88             |
|                                             |
| 💡 커뮤니티 3곳에서 동시 화제, 위험운전치사    |
|    법적 쟁점 명확, 유튜브 해설 수요 높음       |
|                                             |
| [뉴스 검색]                                  |
+--------------------------------------------+
```

**점수 근거(score_reason) 표시 규칙:**
- 키워드 카드 하단의 스코어 바 아래에 LLM이 생성한 점수 근거 1~2문장을 표시
- 텍스트 색상: muted (회색 계열), 최대 2줄 (넘치면 truncate + 툴팁)

### 7.3 뉴스 목록 UI 명세

키워드 카드의 [뉴스 검색] 버튼을 클릭하면, 키워드 카드 목록 아래(또는 오른쪽 패널)에 뉴스 기사 목록이 표시된다.

```
+--------------------------------------------+
| "음주운전 역주행 사망사고" 관련 뉴스 10건       |
+--------------------------------------------+
| 관련 법령:                                   |
|   [도로교통법 제148조의2] 음주운전 처벌 강화   |
|   [특정범죄가중처벌법 제5조의11] 위험운전치사상  |
+--------------------------------------------+
| 1. 음주 역주행 사고, 위험운전치사 혐의 적용...  |
|    Naver  |  2026-02-25 08:30                |
|    경찰은 음주 상태에서 역주행하여...           |
|                                             |
| 2. 역주행 음주운전 피해자 유족 "엄벌 촉구"      |
|    Google |  2026-02-25 07:15                |
|    피해자 유족이 국민청원을 통해...              |
|                                             |
| ...                                         |
+--------------------------------------------+
| [대본 생성하기]                               |
+--------------------------------------------+
```

**관련 법령(related_laws) 표시 규칙:**
- 뉴스 목록 상단에 RAG 검색으로 조회된 관련 법령 1~2개를 태그/뱃지 형태로 표시
- 각 법령명은 클릭 시 법령 상세 페이지로 이동 (기존 case-precedent 모듈 연동)

**뉴스 -> 대본 생성 연결:**
- [대본 생성하기] 버튼 클릭 시 기존 `ScriptGenerator`로 연결
- 전달 데이터: 키워드, 선택된 뉴스 기사 목록(제목+snippet), related_laws
- 연결 방식: `POST /api/content-marketing/scripts/generate` 엔드포인트에 뉴스 기사 컨텍스트를 포함하여 호출
- 기존 ScriptGenerator의 `persona_id` 파라미터도 함께 전달 가능

### 7.4 SSE 기반 단계별 진행 표시

Step 1(키워드 수집)은 5~15초가 소요되므로, SSE(Server-Sent Events) 스트리밍으로 사용자에게 단계별 진행 상태를 실시간으로 표시한다.

**진행 단계별 메시지:**

| 진행률 | SSE 이벤트 | 사용자 표시 메시지 |
|--------|-----------|-----------------|
| 10% | `{"step": "tavily_start"}` | "커뮤니티 인기글 수집 중..." |
| 40% | `{"step": "tavily_done", "count": 20}` | "20개 게시글 수집 완료" |
| 50% | `{"step": "llm_start"}` | "키워드 분석 중..." |
| 80% | `{"step": "llm_done", "count": 8}` | "8개 키워드 추출 완료" |
| 90% | `{"step": "scoring"}` | "키워드 점수 산출 중..." |
| 100% | `{"step": "done"}` | "완료" |

**구현 방식:**
- Backend: `StreamingResponse`(SSE)로 단계별 이벤트 전송
- Frontend: `EventSource` API로 수신, 프로그레스 바 + 메시지 표시
- 기존 `POST /keywords/collect` 응답은 그대로 유지하되, 별도 SSE 엔드포인트 제공
  - `GET /api/content-marketing/keywords/collect/stream` (SSE 스트리밍 버전)
  - Frontend에서는 SSE 버전을 기본 사용, 미지원 브라우저는 일반 POST 폴백

**프론트엔드 UI:**

```
+--------------------------------------------+
| 커뮤니티 화제 키워드 수집                      |
|                                             |
| [===========================-------] 80%   |
| 8개 키워드 추출 완료                          |
+--------------------------------------------+
```

### 7.5 TrendDashboard 통합 방식

기존 `TrendDashboard` 컴포넌트에 탭 네비게이션을 추가하여 "키워드 탐색"과 "트렌드 분석"을 전환할 수 있게 한다.

- **키워드 탐색** 탭 (기본): 새로운 키워드 수집 → 선택 → 뉴스 검색 플로우
- **트렌드 분석** 탭: 기존 `POST /trends` 기반 TrendCard 그리드 (변경 없음)

---

## 8. 기술적 고려사항

### 8.1 기존 코드 재사용 전략

| 기존 컴포넌트 | 재사용 방법 |
|-------------|-----------|
| `TavilySource` | 쿼리 문자열만 변경 (마이너 커뮤니티 도메인 포함). `fetch()` 메서드는 그대로 사용 |
| `CommunityKeywordExtractor` | `extract_with_scores()` 메서드를 새로 추가. 기존 `extract()`는 유지 (하위호환) |
| `BaseTrendSource` + 6개 소스 | `TrendCollector.search_news_for_keyword()`에서 기존 소스의 `safe_fetch()` 그대로 호출 |
| `SourceConfig` | 그대로 사용 |
| `_deduplicate()` | 뉴스 중복 제거에 그대로 활용 |
| `ScoreBar` 컴포넌트 | 키워드 카드의 4차원 점수 표시에 재사용 |

### 8.2 Tavily 쿼리 최적화

마이너 커뮤니티에서 화제 사건/사고를 수집하기 위한 Tavily 쿼리 전략:

```python
# 옵션 A: 사이트 제한 쿼리
query = (
    "화제 사건 사고 논란 "
    "site:dcinside.com OR site:fmkorea.com OR site:theqoo.net "
    "OR site:bobaedream.co.kr OR site:inven.co.kr"
)

# 옵션 B: include_domains 파라미터 활용 (Tavily API 지원 시)
response = await client.search(
    query="화제 사건 사고 논란",
    include_domains=["dcinside.com", "fmkorea.com", "theqoo.net", "bobaedream.co.kr", "inven.co.kr"],
    search_depth="advanced",
    max_results=20,
)
```

Tavily API의 `include_domains` 파라미터 지원 여부를 확인하고, 지원하면 옵션 B를 사용한다. 지원하지 않으면 옵션 A의 site: 쿼리를 사용한다.

### 8.3 키워드 ID 관리

- `keyword_id`는 서버 측에서 UUID v4로 생성
- 인메모리 딕셔너리에 `keyword_id → ScoredKeyword` 매핑 저장
- TTL 1시간 후 자동 만료
- 서버 재시작 시 캐시 초기화 (사용자에게 재수집 요청)

### 8.4 에러 처리

| 시나리오 | 처리 방식 |
|---------|----------|
| Tavily API 실패 | HTTP 502 + "커뮤니티 수집에 실패했습니다. 잠시 후 다시 시도해주세요." |
| LLM 키워드 추출 실패 | 폴백: 제목 빈도 기반 추출 (점수 균일값) + 응답에 `fallback: true` 표시 |
| keyword_id 캐시 미스 | HTTP 404 + "키워드 정보가 만료되었습니다. 키워드를 다시 수집해주세요." |
| 뉴스 소스 전체 실패 | HTTP 502 + "뉴스 검색에 실패했습니다." |
| 뉴스 소스 부분 실패 | 성공한 소스 결과만 반환 (graceful degradation, 기존 `safe_fetch` 패턴) |

### 8.5 설정 (config.py 추가 항목)

| 환경변수 | 기본값 | 설명 |
|---------|-------|------|
| `KEYWORD_COLLECT_CACHE_TTL` | `3600` | 키워드 수집 캐시 TTL (초) |
| `KEYWORD_NEWS_CACHE_TTL` | `1800` | 키워드 뉴스 캐시 TTL (초) |
| `KEYWORD_MAX_RESULTS` | `10` | 최대 키워드 추출 개수 |
| `KEYWORD_NEWS_MAX_RESULTS` | `10` | 키워드당 최대 뉴스 개수 |
| `KEYWORD_COMMUNITY_DOMAINS` | `["dcinside.com", "fmkorea.com", "theqoo.net", "bobaedream.co.kr", "inven.co.kr"]` | 수집 대상 커뮤니티 도메인 |

### 8.6 소스별 차등 타임아웃

뉴스 소스마다 응답 속도가 다르므로, `asyncio.wait_for`로 개별 타임아웃을 적용하여 느린 소스가 전체 응답을 지연시키는 것을 방지한다.

| 소스 | 타임아웃 | 근거 |
|------|---------|------|
| Naver News | 5초 | 국내 API, 빠른 응답 |
| Google Trends | 8초 | 해외 API, 중간 속도 |
| NewsData | 10초 | 해외 API, 상대적으로 느림 |
| NewsAPI | 8초 | 해외 API, 중간 속도 |
| Tavily | 15초 | 크롤링 기반, 가장 느림 |

**구현:**

```python
SOURCE_TIMEOUTS: dict[str, int] = {
    "naver": 5,
    "google_trends": 8,
    "newsdata": 10,
    "newsapi": 8,
    "tavily": 15,
}

async def _fetch_with_timeout(source: BaseTrendSource, keyword: str, config: SourceConfig) -> list[RawTrendItem]:
    timeout = SOURCE_TIMEOUTS.get(source.name, 10)
    try:
        return await asyncio.wait_for(source.safe_fetch(keyword, config), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"소스 {source.name} 타임아웃 ({timeout}초)")
        return []
```

### 8.7 Rate Limiting 구현

사용자별 수집 요청 횟수를 제한하여 API 비용 폭증과 남용을 방지한다.

**정책 (버킷 분리):**

| 버킷 | 엔드포인트 | 제한 | 윈도우 | 초과 시 | 분리 사유 |
|------|-----------|------|--------|---------|----------|
| `collect` | `POST /keywords/collect` | 5회 | 1시간 | HTTP 429 + `Retry-After` | Tavily + LLM 호출 포함, 고비용 |
| `news` | `POST /keywords/{id}/news` | 30회 | 1시간 | HTTP 429 + `Retry-After` | 뉴스 검색만, 저비용 |

**구현 방식:**

```python
# backend/app/tools/trend/rate_limiter.py
from collections import defaultdict
from time import time

class InMemoryRateLimiter:
    """인메모리 Rate Limiter (단일 서버 환경, 버킷별 분리)."""

    def __init__(self, max_requests: int = 5, window_seconds: int = 3600) -> None:
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, user_id: str, bucket: str = "default") -> bool:
        """요청 허용 여부 확인. 허용 시 요청 기록. bucket으로 제한 분리."""
        key = f"{user_id}:{bucket}"
        now = time()
        window_start = now - self._window_seconds
        # 윈도우 밖의 오래된 기록 제거
        self._requests[key] = [
            t for t in self._requests[key] if t > window_start
        ]
        if len(self._requests[key]) >= self._max_requests:
            return False
        self._requests[key].append(now)
        return True

    def retry_after(self, user_id: str, bucket: str = "default") -> int:
        """다음 요청까지 대기해야 할 초."""
        key = f"{user_id}:{bucket}"
        if not self._requests[key]:
            return 0
        oldest = min(self._requests[key])
        return max(0, int(self._window_seconds - (time() - oldest)))

# 버킷별 인스턴스 생성
collect_limiter = InMemoryRateLimiter(max_requests=5, window_seconds=3600)
news_limiter = InMemoryRateLimiter(max_requests=30, window_seconds=3600)
```

**user_id 식별:**
- 인증된 사용자: JWT 토큰에서 user_id 추출
- 미인증 사용자: 클라이언트 IP 기반 (X-Forwarded-For 헤더)

### 8.8 Query Sanitization 구현

LLM 추출 키워드를 외부 뉴스 검색 API에 전달하기 전, 보안 필터를 적용하여 인젝션 공격을 방지한다.

**Sanitize 규칙:**

| 규칙 | 설명 | 예시 |
|------|------|------|
| 허용 문자 제한 | 한글, 영문, 숫자, 공백만 허용 | `음주운전 역주행` (허용) |
| 인젝션 패턴 차단 | SQL/XSS/프롬프트 인젝션 패턴 제거 | `'; DROP TABLE` -> 차단 |
| 길이 제한 | 2~50자 | 초과 시 잘라내기 |
| 연속 공백 제거 | 다중 공백을 단일 공백으로 | `음주  운전` -> `음주 운전` |

**구현:**

```python
import re

ALLOWED_PATTERN = re.compile(r"[^가-힣a-zA-Z0-9\s]")
INJECTION_PATTERNS = [
    re.compile(r"['\";].*(?:DROP|DELETE|UPDATE|INSERT)", re.IGNORECASE),
    re.compile(r"<script", re.IGNORECASE),
    re.compile(r"\{\{|\{%"),
    re.compile(r"\\x[0-9a-fA-F]{2}"),
]

def sanitize_search_query(keyword: str) -> str:
    """뉴스 검색 API에 전달할 키워드를 Sanitize."""
    # 1. 인젝션 패턴 검사
    for pattern in INJECTION_PATTERNS:
        if pattern.search(keyword):
            raise ValueError(f"의심스러운 키워드 패턴 감지: {keyword[:20]}...")
    # 2. 허용 문자만 남김
    cleaned = ALLOWED_PATTERN.sub("", keyword)
    # 3. 연속 공백 제거 + 양끝 공백 제거
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # 4. 길이 검증
    if len(cleaned) < 2 or len(cleaned) > 50:
        raise ValueError(f"키워드 길이 부적합: {len(cleaned)}자")
    return cleaned
```

### 8.9 Keyword Blacklist 시스템

혐오, 정치 편향, 선정적 키워드를 필터링하는 시스템.

**아키텍처:**

```
[설정 파일] backend/app/tools/trend/blacklist.json
    |
    v
[KeywordBlacklistFilter] 서버 시작 시 로드, 핫 리로드 지원
    |
    +--→ exact_match: 정확히 일치하는 키워드 차단
    +--→ contains_match: 포함 키워드 차단
    +--→ regex_match: 정규식 패턴 차단
```

**설정 파일 구조:**

```json
{
  "exact": ["비속어1", "비속어2"],
  "contains": ["혐오표현", "차별용어"],
  "regex": ["특정패턴\\d+"]
}
```

**적용 시점:**
1. LLM 키워드 추출 직후 (Step 1에서 Blacklist에 해당하는 키워드 자동 제거)
2. 폴백 키워드 추출 후에도 동일 필터 적용
3. Blacklist로 제거된 키워드 수를 로그에 기록 (모니터링용)

### 8.10 성능 예측

| 단계 | 호출 내역 | 예상 소요 시간 |
|------|----------|-------------|
| Step 1 키워드 수집 | Tavily 1회 + LLM 1회 + Sanitize + Blacklist | 5~15초 |
| Step 2 뉴스 검색 | 뉴스 소스 N개 병렬 (소스별 차등 타임아웃) | 3~8초 |
| **총합 (사용자 체감)** | Step 1 대기 (SSE 진행 표시) + 선택 + Step 2 대기 | **8~23초** (기존 30~120초 대비 대폭 개선) |

### 8.11 Red Team 피드백 반영 사항

v0.2에서 Red Team (Gemini CLI) 보안/성능 검증 결과를 반영한 항목 요약.

#### 채택 반영 항목

| 우선순위 | 항목 | 반영 섹션 | 설명 |
|---------|------|----------|------|
| High | IDOR 방지 (캐시 세션 격리) | 6.3 | 캐시 키에 `{user_id}` 포함, 복합 키로만 조회 |
| High | Search Query Sanitization | 4.1, 4.2, 6.1, 6.2, 8.8 | LLM 추출 키워드를 외부 API 전달 전 Sanitize 필터 적용 |
| Medium | Rate Limiting | 4.1, 4.2, 6.1, 6.2, 8.7 | 사용자별 1시간 5회 제한, InMemoryRateLimiter |
| 성능 | 소스별 차등 타임아웃 | 6.2, 8.6 | asyncio.wait_for 개별 타임아웃 |
| UX | SSE 단계별 진행 표시 | 7.4 | StreamingResponse + EventSource |
| UX | Keyword Blacklist | 5.3, 5.5, 8.9 | 혐오/정치 편향 키워드 필터링 |

#### 검토 후 부분 채택 항목

| 항목 | 결정 | 사유 |
|------|------|------|
| Redis 분산 캐시 | 로드맵 기록 | 현재 단일 서버 환경, 인메모리 + user_id 격리로 충분. 스케일 아웃 시 전환 |
| Legal-Specific Scoring (RAG 밀도 측정) | Phase 2 로드맵 | 응답 시간 유지 우선. 향후 RAG enrichment 옵션으로 제공 |
| Semantic Clustering | v3.0 로드맵 | LLM 프롬프트 "유사 이슈 통합" 지시로 간소화 (5.3 반영) |

#### 보류/미채택 항목

| 항목 | 사유 |
|------|------|
| Async Task Queue (Celery) | 프로젝트에 Celery 미도입. 5~15초는 HTTP 타임아웃 범위 내. SSE로 UX 개선이 더 적합 |
| Grafana 대시보드 | 인프라 미구축. 로깅으로 대체 |

### 8.12 Consultant 피드백 반영 사항

v1.0에서 External Consultant (Codex CLI) 검증 결과를 반영한 항목 요약.

#### 즉시 채택 (P0 - v1.0 반영)

| 항목 | 반영 섹션 | 설명 |
|------|----------|------|
| Structured Outputs 전환 | 5.3 | Upstage Solar Pro2의 `response_format` 지원 여부 확인 후 JSON Schema 강제 적용. 미지원 시 기존 3단계 폴백 유지 + temperature=0 |
| 스코어 재현성 (감사 추적) | 4.1, 5.3 | 응답에 `prompt_version`, `model_version` 포함. 어떤 프롬프트/모델로 산출된 점수인지 추적 가능 |
| Rate Limit 버킷 분리 | 4.1, 4.2, 8.7 | collect 5회/h, news 30회/h로 분리. 뉴스 검색은 저비용이므로 넉넉하게 설정 |

#### 단기 채택 (P1 - v1.0 반영)

| 항목 | 반영 섹션 | 설명 |
|------|----------|------|
| Tavily + Naver 뉴스 하이브리드 수집 | 4.1, 6.1 | Step 1에서 Tavily와 Naver 뉴스를 병렬 수집 후 병합. dcinside.com robots.txt Disallow:/ 대응 |
| 점수 근거 노출 (score_reason) | 4.1, 5.3, 7.2 | 키워드 카드에 LLM이 생성한 점수 근거 1~2문장 표시 |
| 경량 법률 enrichment 복원 | 4.2, 6.2, 7.3 | 뉴스 검색 결과에 RAGPipeline 기반 관련 법령 1~2개 + 쟁점 라벨 추가 |
| 뉴스 -> 대본 생성 연결 | 7.3 | Step 2 완료 후 [대본 생성하기] 버튼으로 기존 ScriptGenerator 연결 |
| confidence 필드 추가 | 4.1 | LLM 응답의 신뢰도 점수를 키워드 응답에 포함 |

#### 로드맵 기록 (v2.0+)

| 항목 | 목표 버전 | 설명 |
|------|----------|------|
| REST 정합성 (GET /keywords/{id}/news) | v2.0 | 현재 POST 유지, v2.0에서 GET 전환 리팩토링 |
| RFC 9457 에러 표준화 | v2.0 | 프로젝트 전체 에러 핸들링 리팩토링 시 적용 |
| 피드백 루프 (채택률 기반 재랭킹) | v2.0 | 사용자가 선택한 키워드의 채택률을 축적하여 점수 가중치 자동 조정 |
| 요금제/화이트라벨 | 별도 | 비즈니스 전략으로 별도 관리 |

---

## 9. 구현 단계별 계획

### Phase 1: Backend 스키마 + 내부 모델 (1일)

**목표:** 새로운 요청/응답 스키마와 내부 데이터 모델 정의

1. `backend/app/tools/trend/models.py`에 `ScoredKeyword` 데이터클래스 추가
   ```python
   @dataclass
   class KeywordScore:
       virality: float
       social_impact: float
       legal_relevance: float
       content_fitness: float

   @dataclass
   class ScoredKeyword:
       id: str
       keyword: str
       context: str
       source_posts: list[RawTrendItem]
       scores: KeywordScore
       total_score: float
       rank: int
   ```

2. `backend/app/modules/content_marketing/schema/__init__.py`에 스키마 추가
   - `KeywordCollectRequest`
   - `KeywordScoreSchema`
   - `KeywordItem`
   - `KeywordCollectResponse`
   - `KeywordNewsRequest`
   - `NewsArticle`
   - `KeywordNewsResponse`

### Phase 2: Backend 키워드 추출기 확장 (1일)

**목표:** LLM 기반 키워드 + 점수 추출 기능 추가

1. `backend/app/tools/trend/keyword_extractor.py`에 `extract_with_scores()` 메서드 추가
   - 새 프롬프트: 키워드 + 4차원 점수 JSON 배열 출력 요구
   - JSON 파싱 + 폴백 (기존 3단계 폴백 패턴 참고)
   - 기존 `extract()` 메서드는 그대로 유지

2. Tavily 커뮤니티 특화 쿼리 구성
   - `backend/app/tools/trend/sources/tavily_source.py` 또는 collector에서 쿼리 구성
   - `include_domains` 파라미터 또는 `site:` 쿼리 활용

### Phase 3: Backend 서비스 + 라우터 (1일)

**목표:** 새 API 엔드포인트 2개 구현

1. `TrendCollector`에 메서드 추가
   - `collect_community_keywords()`: Stage 1+2만 실행하여 `ScoredKeyword[]` 반환
   - `search_news_for_keyword()`: 특정 키워드로 뉴스 소스 병렬 검색

2. `content_marketing_service.py`에 서비스 함수 추가
   - `collect_keywords()`: 키워드 수집 + 캐시 저장
   - `search_keyword_news()`: 캐시에서 키워드 조회 + 뉴스 검색

3. `router/__init__.py`에 엔드포인트 추가
   - `POST /keywords/collect`
   - `POST /keywords/{keyword_id}/news`

4. `config.py`에 환경변수 추가

### Phase 4: Frontend 타입 + 서비스 (0.5일)

**목표:** 프론트엔드 API 연동 계층 구현

1. `types/index.ts`에 새 타입 추가
   - `KeywordCollectRequest`, `KeywordItem`, `KeywordCollectResponse`
   - `KeywordNewsRequest`, `NewsArticle`, `KeywordNewsResponse`

2. `services/index.ts`에 API 함수 추가
   - `collectKeywords()`
   - `searchKeywordNews()`

### Phase 5: Frontend 컴포넌트 (1.5일)

**목표:** 키워드 플로우 UI 구현

1. `useKeywordFlow.ts` 훅 구현
   - 상태: keywords, selectedKeyword, newsArticles, loading 단계별
   - 액션: collectKeywords, selectKeyword, searchNews

2. `KeywordCard.tsx` 컴포넌트 구현
   - 4차원 ScoreBar 표시
   - [뉴스 검색] 버튼

3. `KeywordCollector.tsx` 컴포넌트 구현
   - 수집 버튼 + 키워드 카드 그리드

4. `KeywordNewsList.tsx` 컴포넌트 구현
   - 뉴스 기사 10개 리스트
   - [대본 생성] 버튼

5. `TrendDashboard.tsx` 수정
   - 탭 네비게이션 추가 (키워드 탐색 / 트렌드 분석)

### Phase 6: 통합 테스트 + 검증 (0.5일)

**목표:** 전체 플로우 동작 확인

1. Backend 린트/타입 검증: `uv run ruff check`, `uv run mypy`
2. Frontend 빌드 검증: `npm run build`
3. 수동 E2E 테스트: 키워드 수집 -> 선택 -> 뉴스 검색 플로우
4. API 응답 확인: `curl`로 엔드포인트별 응답 검증

### 총 예상 소요: 5.5일

| Phase | 소요 | 누적 |
|-------|------|------|
| Phase 1: 스키마/모델 | 1일 | 1일 |
| Phase 2: 키워드 추출기 | 1일 | 2일 |
| Phase 3: 서비스/라우터 | 1일 | 3일 |
| Phase 4: FE 타입/서비스 | 0.5일 | 3.5일 |
| Phase 5: FE 컴포넌트 | 1.5일 | 5일 |
| Phase 6: 통합 테스트 | 0.5일 | 5.5일 |

---

## 리스크 및 완화 방안

| 리스크 | 영향 | 가능성 | 완화 방안 |
|--------|------|--------|----------|
| Tavily가 마이너 커뮤니티 인덱싱을 충분히 하지 않을 수 있음 | 높음 | 중간 | include_domains 대신 일반 쿼리 + 키워드 필터로 폴백. 필요시 직접 커뮤니티 크롤링 검토 |
| LLM이 키워드 + 점수를 안정적으로 JSON으로 반환하지 않을 수 있음 | 중간 | 낮음 | 기존 3단계 JSON 폴백 패턴 적용. 최종 폴백은 빈도 기반 추출 |
| keyword_id 캐시 만료로 Step 2 실패 | 낮음 | 중간 | 404 응답 + "키워드를 다시 수집해주세요" 안내. TTL을 넉넉히 설정 (1시간) |
| 기존 /trends API 사용자 혼란 | 낮음 | 낮음 | 기존 API 완전 유지, 프론트엔드에서 탭으로 분리 |

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-02-25 | 초안 작성 | CTO Lead |
| 0.2 | 2026-02-25 | Red Team 피드백 반영 - IDOR 방지(캐시 세션 격리), Query Sanitization, Rate Limiting, 소스별 차등 타임아웃, SSE 진행 표시, Keyword Blacklist | CTO Lead |
| 1.0 | 2026-02-25 | Consultant 피드백 반영 - Structured Outputs 전환, 스코어 재현성(prompt/model_version), Rate Limit 버킷 분리(collect 5회/h + news 30회/h), Tavily+Naver 하이브리드 수집, 점수 근거(score_reason) 노출, 경량 법률 enrichment(RAG) 복원, 뉴스->대본 생성 연결, v2.0+ 로드맵 기록 | CTO Lead |
