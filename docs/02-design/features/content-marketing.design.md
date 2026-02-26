# Content Marketing (콘텐츠 마케팅 자동화) Design Document

> **Summary**: 실시간 법률 트렌드 분석(모듈 A) + AI 유튜브 대본 생성(모듈 B) 상세 설계
>
> **Project**: law-3 (Legal President / 법률 대통령)
> **Version**: 0.1.0
> **Author**: Claude
> **Date**: 2026-02-20
> **Status**: Draft
> **Planning Doc**: [content-marketing.plan.md](../01-plan/features/content-marketing.plan.md)

### Pipeline References

| Phase | Document | Status |
|-------|----------|--------|
| Phase 1 | [Schema Definition](../01-plan/schema.md) | N/A |
| Phase 2 | [Coding Conventions](../01-plan/conventions.md) | N/A |
| Phase 3 | Mockup (본 문서 Section 5 UI/UX) | ✅ |
| Phase 4 | API Spec (본 문서 Section 4) | ✅ |

---

## 1. Overview

### 1.1 Design Goals

1. **기존 아키텍처 일관성**: ModuleRegistry 자동 등록, BaseChatAgent 상속, AGENT_NODE_MAP 패턴을 100% 준수하여 content_marketing 모듈을 추가한다.
2. **트렌드 소스 확장성**: Strategy 패턴으로 개별 데이터 소스(Tavily, Naver, Perplexity 등)를 어댑터로 추상화하여 소스 추가/교체가 용이한 구조를 설계한다.
3. **기존 RAG 파이프라인 재사용**: 판례/법령 검색에 `RAGPipeline`을 그대로 활용하여 중복 구현을 방지한다.
4. **SSE 스트리밍 대본 생성**: 대본 생성 시 LLM 토큰을 실시간 스트리밍하여 사용자 경험을 개선한다.

### 1.2 Design Principles

- **Single Responsibility**: 트렌드 수집(Collector), 스코어링(Scorer), 요약(Summarizer), 대본 생성(Generator)을 각각 독립 클래스로 분리
- **Open/Closed**: `BaseTrendSource` 추상 클래스를 상속하여 새 데이터 소스 추가 시 기존 코드 수정 불필요
- **Dependency Inversion**: 외부 API 클라이언트는 인터페이스(Protocol)로 추상화하여 테스트 시 목(mock) 교체 가능
- **기존 패턴 준수**: BaseChatAgent, ModuleRegistry, AGENT_NODE_MAP, snake_case API 계약 100% 유지

---

## 2. Architecture

### 2.1 Component Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     Next.js App (/content-marketing)                      │
│                                                                           │
│  ┌──────────────────────────────┐  ┌──────────────────────────────────┐  │
│  │    Tab 1: 트렌드 대시보드      │  │    Tab 2: 대본 생성기              │  │
│  │  ┌──────────────────────────┐│  │  ┌──────────────────────────────┐│  │
│  │  │ TrendDashboard           ││  │  │ ScriptGenerator              ││  │
│  │  │  ├── TrendFilters        ││  │  │  ├── PersonaSelector         ││  │
│  │  │  ├── TrendCard[]         ││  │  │  ├── ScriptEditor            ││  │
│  │  │  └── TrendDetailView     ││  │  │  ├── ScriptPreview           ││  │
│  │  └──────────────────────────┘│  │  │  ├── MetadataPanel           ││  │
│  └──────────────────────────────┘  │  │  └── ExportButton            ││  │
│                                     │  └──────────────────────────────┘│  │
│                                     └──────────────────────────────────┘  │
│                           │                           │                   │
│                    ┌──────┴───────────────────────────┴──────┐            │
│                    │          API Service (fetch/SSE)         │            │
│                    └──────────────────┬──────────────────────┘            │
└───────────────────────────────────────┼──────────────────────────────────┘
                                        │ HTTP
┌───────────────────────────────────────┼──────────────────────────────────┐
│                           FastAPI Backend                                  │
│                                        │                                  │
│  ┌─────────────────────────────────────┴─────────────────────────────┐   │
│  │            /api/content-marketing (모듈 라우터)                      │   │
│  │  POST /trends          → TrendCollector → Scorer → Summarizer     │   │
│  │  GET  /trends/{id}     → 캐시 조회 / 상세 데이터                    │   │
│  │  POST /script/generate → ScriptGenerator (SSE 스트리밍)            │   │
│  │  POST /script/metadata → MetadataGenerator                        │   │
│  └─────────────────────────────────────┬─────────────────────────────┘   │
│                                        │                                  │
│  ┌─────────────────────────────────────┴─────────────────────────────┐   │
│  │                   Service Layer                                     │   │
│  │                                                                     │   │
│  │  ┌─────────────────────────┐  ┌──────────────────────────────┐    │   │
│  │  │  TrendCollector          │  │  ScriptGenerator              │    │   │
│  │  │  ├── TavilySource       │  │  ├── TemplateManager          │    │   │
│  │  │  ├── NaverSource        │  │  ├── RAGPipeline (기존)       │    │   │
│  │  │  └── (확장 소스...)      │  │  └── MetadataGenerator       │    │   │
│  │  └────────────┬────────────┘  └──────────────┬───────────────┘    │   │
│  │               │                               │                    │   │
│  │  ┌────────────┴────────────┐  ┌──────────────┴───────────────┐    │   │
│  │  │  TrendScorer             │  │  LLM Client (Solar/OpenAI)   │    │   │
│  │  │  ├── MentionScorer      │  └──────────────────────────────┘    │   │
│  │  │  └── LegalRelevanceJudge│                                      │   │
│  │  └────────────┬────────────┘                                      │   │
│  │               │                                                    │   │
│  │  ┌────────────┴────────────┐                                      │   │
│  │  │  IssueSummarizer         │                                      │   │
│  │  │  ├── LLM 3줄 요약       │                                      │   │
│  │  │  └── RAGPipeline (기존) │                                      │   │
│  │  └─────────────────────────┘                                      │   │
│  └───────────────────────────────────────────────────────────────────┘   │
│                                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────────┐   │
│  │  LLM (Solar)  │  │  LanceDB RAG │  │  in-memory Cache (TTL 24h)  │   │
│  │  (스코어링+생성)│  │  (판례/법령)  │  │  (트렌드 결과 캐시)          │   │
│  └──────────────┘  └──────────────┘  └──────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                 채팅 위젯 연동 (LangGraph Multi-Agent)                      │
│                                                                           │
│  router_node ──→ content_marketing_node ──→ END                          │
│                  ├── TrendAnalysisAgent.process()                         │
│                  └── ScriptGeneratorAgent.process()                       │
│                                                                           │
│  AgentType.CONTENT_MARKETING → AGENT_NODE_MAP["content_marketing"]       │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

#### 트렌드 수집 흐름

```
사용자 요청 (POST /trends)
    │
    ▼
TrendCollector.collect(config)
    │
    ├── 캐시 확인 (TTL 24h) ─── 캐시 히트 → 즉시 반환
    │
    ├── asyncio.gather() (병렬 수집)
    │   ├── TavilySource.fetch(query, time_range)
    │   ├── NaverSource.fetch(query, time_range)
    │   └── (Phase 2: PerplexitySource, GoogleTrendsSource, YouTubeSource)
    │
    ├── 중복 제거 (URL 기반 + 제목 유사도 0.8 이상)
    │
    ▼
TrendScorer.score(raw_issues)
    │
    ├── MentionScorer: 언급량 정규화 (0~1, min-max scaling)
    │
    ├── LegalRelevanceJudge: LLM 판단 (0~1)
    │   ├── 법률 키워드 사전 매칭 (보정 +0.1)
    │   └── 프롬프트: "이 이슈가 법적 분석/해석이 가능한지 0~1 점수"
    │
    ├── 종합 점수 = α × mention + β × legal_relevance
    │   (α=0.4, β=0.6, 환경변수로 조정 가능)
    │
    ▼
IssueSummarizer.summarize(scored_issues)
    │
    ├── LLM: 각 이슈별 핵심 쟁점 3줄 요약
    │
    ├── RAGPipeline: 관련 법령/판례 검색 (이슈 제목+요약 기반)
    │   ├── PipelineConfig(n_results=5, enable_rerank=True, rerank_top_k=3)
    │   └── 법령/판례 ID + 제목 반환
    │
    ├── 점수 순 정렬 + limit 적용
    │
    ▼
TrendResponse (JSON) → Frontend 대시보드
```

#### 대본 생성 흐름

```
사용자 요청 (POST /script/generate)
    │
    ├── trend_id → 트렌드 캐시에서 이슈 정보 조회
    │   또는 topic → 직접 입력 주제 사용
    │
    ▼
ScriptGenerator.generate(request) [SSE 스트리밍]
    │
    ├── 1. RAG 심화 검색
    │   ├── RAGPipeline: 판례 검색 (n_results=10, enable_rerank=True)
    │   ├── RAGPipeline: 법령 검색 (n_results=10, enable_rerank=True)
    │   └── 인용 후보 목록 구성
    │
    ├── 2. 도입부 생성 (Hooking)
    │   ├── 프롬프트: 시청자 관심을 끄는 사례/질문 제시
    │   ├── persona 적용 (professional / casual)
    │   └── SSE 스트리밍 → {"section": "hooking", "content": "..."}
    │
    ├── 3. 본론 생성 (Legal Analysis)
    │   ├── 프롬프트: 법 조항 + 판례 기반 전문 분석
    │   ├── RAG 검색 결과를 context로 주입
    │   ├── 인용 출처 마크업: [📋 인용: 대법원 2024다12345]
    │   └── SSE 스트리밍 → {"section": "analysis", "content": "..."}
    │
    ├── 4. 결론 생성 (Advice & CTA)
    │   ├── 프롬프트: 실질적 조언 + 상담 유도
    │   └── SSE 스트리밍 → {"section": "advice_cta", "content": "..."}
    │
    ▼
ScriptResult (전체 대본 + 인용 목록)
    │
    ▼
MetadataGenerator.generate(script_result)
    │
    ├── 영상 설명문 생성 (200~300자)
    ├── SEO 태그 생성 (10~15개)
    └── CTA 문구 생성 ("무료 법률 상담 신청: ...")
```

### 2.3 Dependencies

| Component | Depends On | Purpose |
|-----------|-----------|---------|
| TrendCollector | TavilySource, NaverSource | 멀티소스 트렌드 수집 |
| TavilySource | `tavily-python`, TAVILY_API_KEY | Tavily Search API 클라이언트 |
| NaverSource | `httpx`, NAVER_CLIENT_ID/SECRET | Naver Search API 호출 |
| TrendScorer | LLM Client, 법률 키워드 사전 | 스코어링 |
| IssueSummarizer | LLM Client, RAGPipeline | 3줄 요약 + 관련 법령/판례 |
| ScriptGenerator | LLM Client, RAGPipeline | 3단 구조 대본 생성 |
| MetadataGenerator | LLM Client | 영상 메타데이터 생성 |
| ContentMarketingRouter | ContentMarketingService | FastAPI 라우터 |
| TrendAnalysisAgent | BaseChatAgent, TrendCollector | 채팅 위젯 연동 |
| ScriptGeneratorAgent | BaseChatAgent, ScriptGenerator | 채팅 위젯 연동 |

---

## 3. Data Model

### 3.1 Pydantic Schema — Trend (트렌드 관련)

```python
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


# ── Enum ──

class TrendSource(str, Enum):
    """트렌드 데이터 소스"""
    TAVILY = "tavily"
    NAVER = "naver"
    PERPLEXITY = "perplexity"          # Phase 2
    GOOGLE_TRENDS = "google_trends"    # Phase 2
    YOUTUBE = "youtube"                # Phase 2


class TrendCategory(str, Enum):
    """트렌드 카테고리 (법률 분류)"""
    ALL = "all"
    CRIMINAL = "criminal"       # 형사
    CIVIL = "civil"             # 민사
    LABOR = "labor"             # 노동
    FAMILY = "family"           # 가사
    ADMINISTRATIVE = "administrative"  # 행정
    CORPORATE = "corporate"     # 기업/상사
    IP = "ip"                   # 지식재산


class TimeRange(str, Enum):
    """트렌드 조회 기간"""
    HOURS_24 = "24h"
    HOURS_48 = "48h"
    DAYS_7 = "7d"


# ── Request ──

class TrendRequest(BaseModel):
    """트렌드 조회 요청"""
    time_range: TimeRange = TimeRange.HOURS_24
    category: TrendCategory = TrendCategory.ALL
    limit: int = Field(default=10, ge=1, le=30)
    query: str | None = Field(
        default=None,
        description="키워드 필터 (없으면 전체 트렌드)",
    )


# ── Response ──

class SourceArticle(BaseModel):
    """개별 소스 기사/게시글"""
    title: str
    url: str
    source: TrendSource
    published_at: datetime | None = None
    snippet: str = ""


class RelatedLaw(BaseModel):
    """관련 법령 정보"""
    law_id: str
    law_name: str
    relevance_score: float = Field(ge=0, le=1)


class RelatedCase(BaseModel):
    """관련 판례 정보"""
    case_id: str
    case_number: str
    case_name: str
    relevance_score: float = Field(ge=0, le=1)


class TrendIssue(BaseModel):
    """트렌드 이슈 (대시보드 카드)"""
    id: str                                    # UUID v4
    title: str
    summary: str                               # LLM 생성 요약 (1~2문장)
    key_points: list[str] = Field(             # 핵심 쟁점 3줄
        min_length=1, max_length=5,
    )
    score: float = Field(ge=0, le=100)         # 종합 점수
    mention_score: float = Field(ge=0, le=1)   # 정규화된 언급량
    legal_relevance_score: float = Field(ge=0, le=1)  # 법적 해석 가능성
    category: TrendCategory
    sources: list[TrendSource]
    source_articles: list[SourceArticle]
    related_laws: list[RelatedLaw]
    related_cases: list[RelatedCase]
    collected_at: datetime


class TrendResponse(BaseModel):
    """트렌드 조회 응답"""
    trends: list[TrendIssue]
    total_count: int
    collected_at: datetime
    sources_used: list[TrendSource]
    cache_hit: bool = False


class TrendDetailResponse(BaseModel):
    """트렌드 상세 조회 응답"""
    issue: TrendIssue
    source_articles: list[SourceArticle]       # 전체 원본 기사
    related_laws_detail: list[dict]            # RAG 법령 상세 (content 포함)
    related_cases_detail: list[dict]           # RAG 판례 상세 (요지 포함)
```

### 3.2 Pydantic Schema — Script (대본 관련)

```python
from pydantic import BaseModel, Field


# ── Enum ──

class PersonaType(str, Enum):
    """대본 페르소나 타입"""
    PROFESSIONAL = "professional"  # 전문가 톤
    CASUAL = "casual"              # 유튜브 구어체


class ScriptDuration(int, Enum):
    """대본 길이 (분)"""
    SHORT = 5
    MEDIUM = 10
    LONG = 15


class SectionType(str, Enum):
    """대본 섹션 타입"""
    HOOKING = "hooking"
    ANALYSIS = "analysis"
    ADVICE_CTA = "advice_cta"


# ── Request ──

class ScriptRequest(BaseModel):
    """대본 생성 요청"""
    topic: str = Field(
        min_length=5, max_length=500,
        description="주제 (직접 입력 또는 트렌드 요약)",
    )
    trend_id: str | None = Field(
        default=None,
        description="연결된 트렌드 ID (트렌드에서 선택 시)",
    )
    persona: PersonaType = PersonaType.PROFESSIONAL
    duration: ScriptDuration = ScriptDuration.MEDIUM
    related_laws: list[str] = Field(
        default_factory=list,
        description="사전 매칭된 법령 ID 목록",
    )
    related_cases: list[str] = Field(
        default_factory=list,
        description="사전 매칭된 판례 ID 목록",
    )


class MetadataRequest(BaseModel):
    """메타데이터 생성 요청"""
    script_content: str = Field(
        min_length=100,
        description="대본 전체 내용",
    )
    topic: str
    persona: PersonaType = PersonaType.PROFESSIONAL


# ── Response ──

class Citation(BaseModel):
    """인용 출처"""
    source_type: str            # "precedent" | "law"
    source_id: str
    source_name: str            # "대법원 2024다12345" 또는 "민법 제750조"
    relevant_text: str = ""     # 인용된 원문 발췌


class ScriptSection(BaseModel):
    """대본 섹션"""
    section_type: SectionType
    title: str                  # "도입 (Hooking)" 등
    content: str
    citations: list[Citation]
    word_count: int


class ScriptMetadata(BaseModel):
    """영상 메타데이터"""
    description: str            # 영상 설명문 (200~300자)
    tags: list[str]             # SEO 태그 (10~15개)
    cta_text: str               # 상담 유도 문구
    hashtags: list[str]         # 유튜브 해시태그 (5개)


class ScriptResult(BaseModel):
    """대본 생성 결과"""
    id: str                     # UUID v4
    topic: str
    persona: PersonaType
    sections: list[ScriptSection]
    metadata: ScriptMetadata
    all_citations: list[Citation]
    word_count: int
    estimated_duration: int     # 분 단위
    disclaimer: str = "본 콘텐츠는 AI가 생성한 것으로, 법률 자문이 아닙니다."
    created_at: datetime


# ── SSE Event Types ──

class ScriptStreamEvent(BaseModel):
    """SSE 스트리밍 이벤트"""
    event: str                  # "section_start" | "content" | "section_end" | "metadata" | "done" | "error"
    section: SectionType | None = None
    content: str = ""
    metadata: ScriptMetadata | None = None
    error: str | None = None
```

### 3.3 보조 타입 — 내부 모델

```python
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class RawTrendItem:
    """소스별 수집된 원시 트렌드 항목"""
    title: str
    url: str
    snippet: str
    source: TrendSource
    published_at: datetime | None = None
    mention_count: int = 0          # 조회수/공유수 등
    raw_data: dict = field(default_factory=dict)


@dataclass
class ScoredIssue:
    """스코어링 완료된 이슈 (내부 처리용)"""
    id: str
    title: str
    raw_items: list[RawTrendItem]
    mention_score: float            # 0~1
    legal_relevance_score: float    # 0~1
    combined_score: float           # 0~100
    category: TrendCategory = TrendCategory.ALL


@dataclass
class TrendCacheEntry:
    """트렌드 캐시 엔트리"""
    key: str                        # f"{time_range}:{category}:{query}"
    response: TrendResponse
    created_at: datetime
    ttl_seconds: int = 86400        # 24시간
```

---

## 4. API Specification

### 4.1 트렌드 API

#### POST /api/content-marketing/trends

트렌드 이슈를 수집하고 스코어링된 결과를 반환한다.

**Request:**
```json
{
  "time_range": "24h",
  "category": "all",
  "limit": 10,
  "query": null
}
```

**Response (200):**
```json
{
  "trends": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "title": "XX 사건 손해배상 소송 — 대법원 판결 주목",
      "summary": "XX 기업의 제품 결함으로 인한 대규모 손해배상 소송에서 대법원이 원고 승소 판결을 내려 기업 책임 범위에 대한 새로운 기준을 제시했다.",
      "key_points": [
        "제조물 책임법상 결함 입증 책임 전환 여부",
        "손해배상 범위의 확대 해석 가능성",
        "기업의 사전 예방 의무 강화 추세"
      ],
      "score": 92.5,
      "mention_score": 0.87,
      "legal_relevance_score": 0.96,
      "category": "civil",
      "sources": ["tavily", "naver"],
      "source_articles": [
        {
          "title": "XX 사건 대법원 판결...",
          "url": "https://...",
          "source": "tavily",
          "published_at": "2026-02-20T10:00:00Z",
          "snippet": "..."
        }
      ],
      "related_laws": [
        {
          "law_id": "law_1234",
          "law_name": "제조물 책임법",
          "relevance_score": 0.95
        }
      ],
      "related_cases": [
        {
          "case_id": "76396",
          "case_number": "2024다12345",
          "case_name": "손해배상(기)",
          "relevance_score": 0.88
        }
      ],
      "collected_at": "2026-02-20T10:30:00Z"
    }
  ],
  "total_count": 10,
  "collected_at": "2026-02-20T10:30:00Z",
  "sources_used": ["tavily", "naver"],
  "cache_hit": false
}
```

**Error (503):** 외부 API 전체 실패 시
```json
{
  "detail": "트렌드 데이터 소스에 연결할 수 없습니다. 잠시 후 다시 시도해주세요."
}
```

#### GET /api/content-marketing/trends/{trend_id}

캐시된 트렌드 이슈의 상세 정보를 반환한다.

**Response (200):**
```json
{
  "issue": { "...TrendIssue..." },
  "source_articles": [ "...전체 기사 목록..." ],
  "related_laws_detail": [
    {
      "law_id": "law_1234",
      "law_name": "제조물 책임법",
      "content": "제3조(제조물 책임) ① 제조업자는...",
      "relevance_score": 0.95
    }
  ],
  "related_cases_detail": [
    {
      "case_id": "76396",
      "case_number": "2024다12345",
      "case_name": "손해배상(기)",
      "ruling": "원고 승소 판결...",
      "reasoning_summary": "...",
      "relevance_score": 0.88
    }
  ]
}
```

**Error (404):** 캐시 만료 또는 존재하지 않는 ID
```json
{
  "detail": "해당 트렌드 이슈를 찾을 수 없습니다. 트렌드를 다시 조회해주세요."
}
```

### 4.2 대본 API

#### POST /api/content-marketing/script/generate

대본을 SSE 스트리밍으로 생성한다.

**Request:**
```json
{
  "topic": "제조물 책임법 개정과 소비자 보호",
  "trend_id": "550e8400-e29b-41d4-a716-446655440000",
  "persona": "professional",
  "duration": 10,
  "related_laws": ["law_1234"],
  "related_cases": ["76396"]
}
```

**Response (200, SSE stream):**
```
event: section_start
data: {"event": "section_start", "section": "hooking"}

event: content
data: {"event": "content", "section": "hooking", "content": "안녕하세요, 변호사 "}

event: content
data: {"event": "content", "section": "hooking", "content": "김법률입니다. "}

event: section_end
data: {"event": "section_end", "section": "hooking"}

event: section_start
data: {"event": "section_start", "section": "analysis"}

event: content
data: {"event": "content", "section": "analysis", "content": "제조물 책임법 제3조에 따르면..."}

event: section_end
data: {"event": "section_end", "section": "analysis"}

event: section_start
data: {"event": "section_start", "section": "advice_cta"}

event: content
data: {"event": "content", "section": "advice_cta", "content": "이런 상황에서는..."}

event: section_end
data: {"event": "section_end", "section": "advice_cta"}

event: metadata
data: {"event": "metadata", "metadata": {"description": "...", "tags": [...], "cta_text": "...", "hashtags": [...]}}

event: done
data: {"event": "done"}
```

**Error (422):** 유효성 검증 실패
**Error (503):** LLM 서비스 불가

#### POST /api/content-marketing/script/metadata

기존 대본 내용으로 메타데이터만 재생성한다.

**Request:**
```json
{
  "script_content": "## 도입 (Hooking)\n안녕하세요...",
  "topic": "제조물 책임법 개정과 소비자 보호",
  "persona": "professional"
}
```

**Response (200):**
```json
{
  "description": "최근 대법원이 제조물 책임법에 대한 중요한 판결을 내렸습니다...",
  "tags": ["제조물책임법", "손해배상", "대법원판결", "소비자보호", "변호사", "법률상식", ...],
  "cta_text": "제조물 책임 관련 법률 상담이 필요하시면 아래 링크로 무료 상담을 신청하세요.",
  "hashtags": ["#법률", "#변호사", "#제조물책임", "#손해배상", "#법률상식"]
}
```

### 4.3 채팅 API (기존 /api/chat 확장)

채팅 위젯에서 트렌드/대본 관련 질의 시 기존 `/api/chat` 엔드포인트를 통해 라우팅된다.

**라우팅 규칙 (router_node 확장):**

| 사용자 메시지 | 라우팅 | 에이전트 |
|-------------|--------|---------|
| "요즘 핫한 법률 이슈 알려줘" | `content_marketing` | TrendAnalysisAgent |
| "이 주제로 유튜브 대본 만들어줘" | `content_marketing` | ScriptGeneratorAgent |
| "법률 트렌드 분석해줘" | `content_marketing` | TrendAnalysisAgent |

---

## 5. UI/UX Design

### 5.1 페이지 레이아웃

```
┌─────────────────────────────────────────────────────────────┐
│  🏛 법률 대통령                                [프로필] [로그인] │
├─────────────────────────────────────────────────────────────┤
│  콘텐츠 마케팅 자동화                                          │
│                                                              │
│  ┌──────────────────┐ ┌────────────────────┐                │
│  │ 📊 트렌드 분석    │ │ 📝 대본 생성        │  ← 탭 전환     │
│  └──────────────────┘ └────────────────────┘                │
│                                                              │
│  ═══════════════════════════════════════════════════════════  │
│                                                              │
│                     [탭 콘텐츠 영역]                           │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│  ⚠️ 본 콘텐츠는 AI가 생성한 것으로, 법률 자문이 아닙니다.        │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 트렌드 대시보드 (Tab 1)

```
┌──────────────────────────────────────────────────────────────┐
│  필터 바                                                       │
│  카테고리: [전체 ▼]  기간: [24시간 ▼]  [🔄 새로고침]           │
│                                                               │
│  ┌─────────────────────────────┐ ┌─────────────────────────┐ │
│  │ 🔥 1위                      │ │ 📊 2위                   │ │
│  │ ┌─────────────────────────┐ │ │ ┌───────────────────────┐│ │
│  │ │ XX 사건 손해배상 소송     │ │ │ │ YY법 개정 논란         ││ │
│  │ └─────────────────────────┘ │ │ └───────────────────────┘│ │
│  │                              │ │                          │ │
│  │ 종합점수: 92.5 / 100        │ │ 종합점수: 87.3 / 100     │ │
│  │                              │ │                          │ │
│  │ 언급량   ████████░░ 0.87    │ │ 언급량   ██████░░░ 0.72  │ │
│  │ 법적해석 █████████░ 0.96    │ │ 법적해석 ████████░ 0.93  │ │
│  │                              │ │                          │ │
│  │ 핵심 쟁점:                   │ │ 핵심 쟁점:               │ │
│  │ 1. 제조물 책임법상 결함...   │ │ 1. 소급적용 가능성...    │ │
│  │ 2. 손해배상 범위 확대...     │ │ 2. 기본권 제한 여부...   │ │
│  │ 3. 기업 사전 예방 의무...    │ │ 3. 입법 취지 검토...     │ │
│  │                              │ │                          │ │
│  │ 소스: [Tavily] [Naver]      │ │ 소스: [Tavily] [Naver]  │ │
│  │                              │ │                          │ │
│  │ [상세보기] [📝 대본 생성]    │ │ [상세보기] [📝 대본 생성] │ │
│  └──────────────────────────────┘ └──────────────────────────┘ │
│                                                               │
│  ┌─────────────────────────────┐ ┌─────────────────────────┐ │
│  │ 3위 ...                      │ │ 4위 ...                  │ │
│  └──────────────────────────────┘ └──────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

### 5.3 이슈 상세 뷰 (모달 또는 패널)

```
┌──────────────────────────────────────────────────────────────┐
│  ← 뒤로                               XX 사건 손해배상 소송   │
│                                                               │
│  ┌─── 요약 ──────────────────────────────────────────────┐   │
│  │ XX 기업의 제품 결함으로 인한 대규모 손해배상 소송에서...  │   │
│  └───────────────────────────────────────────────────────┘   │
│                                                               │
│  ┌─── 핵심 쟁점 ─────────────────────────────────────────┐   │
│  │ 1. 제조물 책임법상 결함 입증 책임 전환 여부              │   │
│  │ 2. 손해배상 범위의 확대 해석 가능성                     │   │
│  │ 3. 기업의 사전 예방 의무 강화 추세                      │   │
│  └───────────────────────────────────────────────────────┘   │
│                                                               │
│  ┌─── 관련 법령 ─────────────────────────────────────────┐   │
│  │ 📜 제조물 책임법 제3조                  관련도: 95%     │   │
│  │ 📜 민법 제750조 (불법행위)              관련도: 82%     │   │
│  └───────────────────────────────────────────────────────┘   │
│                                                               │
│  ┌─── 관련 판례 ─────────────────────────────────────────┐   │
│  │ ⚖️ 대법원 2024다12345 (손해배상)        관련도: 88%     │   │
│  │ ⚖️ 대법원 2023다67890 (제조물책임)      관련도: 76%     │   │
│  └───────────────────────────────────────────────────────┘   │
│                                                               │
│  ┌─── 원본 기사 (5건) ───────────────────────────────────┐   │
│  │ [Tavily] XX 사건 대법원 판결 주목...  🔗               │   │
│  │ [Naver]  XX 사건 1심 판결 뒤집어...    🔗               │   │
│  │ ...                                                    │   │
│  └───────────────────────────────────────────────────────┘   │
│                                                               │
│                          [📝 이 주제로 대본 생성하기]          │
└──────────────────────────────────────────────────────────────┘
```

### 5.4 대본 생성기 (Tab 2)

```
┌──────────────────────────────────────────────────────────────┐
│  대본 생성 설정                                                │
│                                                               │
│  주제: ┌──────────────────────────────────────────────────┐  │
│        │ 제조물 책임법 개정과 소비자 보호                    │  │
│        └──────────────────────────────────────────────────┘  │
│        (트렌드에서 선택됨: XX 사건 손해배상 소송)              │
│                                                               │
│  설정:                                                        │
│  톤:    ● 전문가  ○ 구어체      길이: ○ 5분  ● 10분  ○ 15분  │
│                                                               │
│                    [▶ 대본 생성하기]                           │
│                                                               │
│  ┌─── 대본 미리보기 ─────────────────────────── [편집 모드] ──┐│
│  │                                                           ││
│  │  ## 1. 도입 (Hooking)                                     ││
│  │  ─────────────────────────────────────────                ││
│  │  안녕하세요, 변호사 OOO입니다. 최근 XX 사건이 큰 논란이    ││
│  │  되고 있는데요. 이 사건에서 대법원이 내린 판결은 앞으로    ││
│  │  소비자 보호에 어떤 영향을 미칠까요?                       ││
│  │                                                           ││
│  │  ## 2. 본론 (Legal Analysis)                              ││
│  │  ─────────────────────────────────────────                ││
│  │  제조물 책임법 제3조에 따르면, 제조업자는 제품의 결함으로  ││
│  │  인해 발생한 손해에 대해 배상 책임을 집니다.               ││
│  │  [📋 인용: 제조물 책임법 제3조]                           ││
│  │                                                           ││
│  │  이번 대법원 판결(2024다12345)에서는 특히...              ││
│  │  [📋 인용: 대법원 2024다12345]                            ││
│  │                                                           ││
│  │  ## 3. 결론 (Advice & CTA)                                ││
│  │  ─────────────────────────────────────────                ││
│  │  제품 관련 피해를 입으셨다면, 증거 보존이 가장 중요합니다. ││
│  │  법률 전문가의 상담을 통해 정확한 권리를 확인하세요.       ││
│  │                                                           ││
│  └───────────────────────────────────────────────────────────┘│
│                                                               │
│  ┌─── 메타데이터 ────────────────────────────────────────┐   │
│  │ 📄 영상 설명문:                                        │   │
│  │ "최근 대법원의 제조물 책임법 관련 판결을 분석합니다..."  │   │
│  │                                                        │   │
│  │ 🏷️ SEO 태그:                                           │   │
│  │ #제조물책임법 #손해배상 #대법원판결 #소비자보호 ...       │   │
│  │                                                        │   │
│  │ 📢 CTA:                                                │   │
│  │ "법률 상담이 필요하시면: [상담 신청 링크]"               │   │
│  └────────────────────────────────────────────────────────┘   │
│                                                               │
│  ┌─── 인용 출처 ─────────────────────────────────────────┐   │
│  │ 📋 제조물 책임법 제3조                                  │   │
│  │ 📋 대법원 2024다12345 (손해배상)                        │   │
│  │ 📋 민법 제750조                                        │   │
│  └────────────────────────────────────────────────────────┘   │
│                                                               │
│              [📋 전체 복사] [⬇ TXT 다운로드] [⬇ MD 다운로드]  │
└──────────────────────────────────────────────────────────────┘
```

### 5.5 User Flow

```
[/content-marketing 접속]
        │
        ├── Tab 1: 트렌드 대시보드
        │   ├── 자동 트렌드 로딩 (최근 24h)
        │   ├── 필터 변경 → 재조회
        │   ├── 카드 클릭 → 상세 뷰 (모달)
        │   │   ├── 관련 법령/판례 확인
        │   │   └── [대본 생성] → Tab 2로 이동 (주제 자동 입력)
        │   └── [새로고침] → 캐시 무효화 + 재수집
        │
        └── Tab 2: 대본 생성
            ├── 주제 입력 (직접 또는 트렌드에서 전달)
            ├── 옵션 설정 (톤, 길이)
            ├── [대본 생성] → SSE 스트리밍
            │   ├── 섹션별 순차 표시 (도입 → 본론 → 결론)
            │   └── 스피너 + 프로그레스 표시
            ├── 대본 편집 (인라인 에디터)
            ├── 메타데이터 확인/수정
            └── 내보내기 (복사/다운로드)
```

### 5.6 컴포넌트 목록

| Component | 위치 | 역할 | Props 주요 타입 |
|-----------|------|------|----------------|
| `ContentMarketingPage` | `app/content-marketing/page.tsx` | 페이지 엔트리, 탭 관리 | - |
| `TrendDashboard` | `features/.../components/` | 트렌드 대시보드 메인 | `trends: TrendIssue[]` |
| `TrendCard` | `features/.../components/` | 이슈 카드 UI | `issue: TrendIssue, onSelect, onGenerateScript` |
| `TrendDetailView` | `features/.../components/` | 이슈 상세 뷰 (모달) | `issue: TrendIssue, details: TrendDetailResponse` |
| `TrendFilters` | `features/.../components/` | 필터 바 | `onFilterChange: (filters) => void` |
| `ScoreBar` | `features/.../components/` | 점수 바 시각화 | `score: number, label: string` |
| `ScriptGenerator` | `features/.../components/` | 대본 생성 메인 | `initialTopic?, initialTrendId?` |
| `PersonaSelector` | `features/.../components/` | 톤/길이 선택 UI | `persona, duration, onChange` |
| `ScriptEditor` | `features/.../components/` | 마크다운 에디터 | `sections: ScriptSection[], onEdit` |
| `ScriptPreview` | `features/.../components/` | 대본 읽기 전용 뷰 | `sections: ScriptSection[]` |
| `MetadataPanel` | `features/.../components/` | 메타데이터 표시/편집 | `metadata: ScriptMetadata` |
| `CitationList` | `features/.../components/` | 인용 출처 목록 | `citations: Citation[]` |
| `ExportButton` | `features/.../components/` | 복사/다운로드 버튼 | `content: string, format: "txt" \| "md"` |
| `DisclaimerBanner` | `features/.../components/` | 면책 고지 배너 | - |

---

## 6. Backend 상세 설계

### 6.1 트렌드 소스 어댑터 (Strategy 패턴)

```python
# backend/app/tools/trend/sources/__init__.py

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SourceConfig:
    """소스별 설정"""
    time_range: str = "24h"
    max_results: int = 20
    language: str = "ko"


class BaseTrendSource(ABC):
    """트렌드 데이터 소스 추상 클래스"""

    @property
    @abstractmethod
    def name(self) -> TrendSource:
        """소스 식별자"""
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """API 키 설정 여부로 사용 가능 판단"""
        ...

    @abstractmethod
    async def fetch(
        self,
        query: str | None,
        config: SourceConfig,
    ) -> list[RawTrendItem]:
        """트렌드 데이터 수집

        Args:
            query: 키워드 필터 (None이면 전체 트렌드)
            config: 소스 설정

        Returns:
            수집된 원시 트렌드 항목 목록

        Raises:
            TrendSourceError: 소스 API 호출 실패 시
        """
        ...

    async def safe_fetch(
        self,
        query: str | None,
        config: SourceConfig,
    ) -> list[RawTrendItem]:
        """실패 시 빈 목록 반환 (graceful degradation)"""
        try:
            return await self.fetch(query, config)
        except Exception:
            logger.warning("소스 %s 수집 실패, 건너뜀", self.name)
            return []
```

### 6.2 TavilySource 구현

```python
# backend/app/tools/trend/sources/tavily_source.py

from tavily import AsyncTavilyClient

from app.core.config import settings


class TavilySource(BaseTrendSource):
    """Tavily Search API 데이터 소스"""

    @property
    def name(self) -> TrendSource:
        return TrendSource.TAVILY

    @property
    def is_available(self) -> bool:
        return bool(settings.TAVILY_API_KEY)

    def __init__(self) -> None:
        self._client: AsyncTavilyClient | None = None

    def _get_client(self) -> AsyncTavilyClient:
        if self._client is None:
            self._client = AsyncTavilyClient(api_key=settings.TAVILY_API_KEY)
        return self._client

    async def fetch(
        self,
        query: str | None,
        config: SourceConfig,
    ) -> list[RawTrendItem]:
        client = self._get_client()
        search_query = query or "최신 법률 이슈 한국"

        # Tavily search_context로 최근 뉴스 수집
        response = await client.search(
            query=search_query,
            search_depth="advanced",
            max_results=config.max_results,
            include_domains=["news.naver.com", "law.go.kr", "courts.go.kr"],
            topic="news",
        )

        items: list[RawTrendItem] = []
        for result in response.get("results", []):
            items.append(
                RawTrendItem(
                    title=result["title"],
                    url=result["url"],
                    snippet=result.get("content", "")[:300],
                    source=TrendSource.TAVILY,
                    published_at=_parse_date(result.get("published_date")),
                    raw_data=result,
                )
            )
        return items
```

### 6.3 NaverSource 구현

```python
# backend/app/tools/trend/sources/naver_source.py

import httpx

from app.core.config import settings

NAVER_SEARCH_URL = "https://openapi.naver.com/v1/search/news.json"


class NaverSource(BaseTrendSource):
    """Naver Search API 데이터 소스"""

    @property
    def name(self) -> TrendSource:
        return TrendSource.NAVER

    @property
    def is_available(self) -> bool:
        return bool(settings.NAVER_CLIENT_ID and settings.NAVER_CLIENT_SECRET)

    async def fetch(
        self,
        query: str | None,
        config: SourceConfig,
    ) -> list[RawTrendItem]:
        search_query = query or "법률 이슈"

        headers = {
            "X-Naver-Client-Id": settings.NAVER_CLIENT_ID,
            "X-Naver-Client-Secret": settings.NAVER_CLIENT_SECRET,
        }
        params = {
            "query": search_query,
            "display": config.max_results,
            "sort": "date",  # 최신순
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                NAVER_SEARCH_URL,
                headers=headers,
                params=params,
            )
            response.raise_for_status()
            data = response.json()

        items: list[RawTrendItem] = []
        for item in data.get("items", []):
            items.append(
                RawTrendItem(
                    title=_strip_html(item["title"]),
                    url=item["originallink"] or item["link"],
                    snippet=_strip_html(item.get("description", ""))[:300],
                    source=TrendSource.NAVER,
                    published_at=_parse_naver_date(item.get("pubDate")),
                    raw_data=item,
                )
            )
        return items
```

### 6.4 TrendCollector (수집 + 중복 제거)

```python
# backend/app/tools/trend/collector.py

import asyncio
import uuid
from datetime import datetime

from app.tools.trend.sources import BaseTrendSource, SourceConfig
from app.tools.trend.sources.tavily_source import TavilySource
from app.tools.trend.sources.naver_source import NaverSource


class TrendCollector:
    """멀티소스 트렌드 수집기"""

    def __init__(self) -> None:
        self._sources: list[BaseTrendSource] = [
            TavilySource(),
            NaverSource(),
            # Phase 2: PerplexitySource(), GoogleTrendsSource(), YouTubeSource()
        ]
        self._cache: dict[str, TrendCacheEntry] = {}

    def _get_available_sources(self) -> list[BaseTrendSource]:
        """API 키가 설정된 소스만 반환"""
        return [s for s in self._sources if s.is_available]

    def _make_cache_key(self, request: TrendRequest) -> str:
        return f"{request.time_range}:{request.category}:{request.query or ''}"

    def _get_cached(self, key: str) -> TrendResponse | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        elapsed = (datetime.utcnow() - entry.created_at).total_seconds()
        if elapsed > entry.ttl_seconds:
            del self._cache[key]
            return None
        response = entry.response
        response.cache_hit = True
        return response

    async def collect(self, request: TrendRequest) -> list[RawTrendItem]:
        """모든 소스에서 병렬 수집 + 중복 제거"""
        sources = self._get_available_sources()
        if not sources:
            raise TrendSourceError("사용 가능한 트렌드 소스가 없습니다.")

        config = SourceConfig(
            time_range=request.time_range.value,
            max_results=request.limit * 2,  # 중복 제거 여유분
        )

        # 병렬 수집
        results = await asyncio.gather(
            *[source.safe_fetch(request.query, config) for source in sources]
        )

        # 평탄화 + 중복 제거
        all_items: list[RawTrendItem] = []
        for items in results:
            all_items.extend(items)

        return self._deduplicate(all_items)

    def _deduplicate(self, items: list[RawTrendItem]) -> list[RawTrendItem]:
        """URL 기반 중복 제거"""
        seen_urls: set[str] = set()
        unique: list[RawTrendItem] = []
        for item in items:
            normalized_url = item.url.rstrip("/").lower()
            if normalized_url not in seen_urls:
                seen_urls.add(normalized_url)
                unique.append(item)
        return unique
```

### 6.5 TrendScorer (스코어링 엔진)

```python
# backend/app/tools/trend/scorer.py

import logging

from app.tools.llm.client import get_llm_client

logger = logging.getLogger(__name__)

# 법률 키워드 사전 (TF 기반 보정용)
LEGAL_KEYWORDS: frozenset[str] = frozenset({
    "판결", "소송", "법원", "변호사", "검찰", "형사", "민사",
    "손해배상", "기소", "무죄", "유죄", "항소", "상고", "헌법",
    "법률", "법령", "조례", "판례", "재판", "공판", "고소", "고발",
    "구속", "보석", "가처분", "압류", "경매", "파산", "회생",
})

MENTION_WEIGHT = 0.4   # settings.TREND_MENTION_WEIGHT
LEGAL_WEIGHT = 0.6     # settings.TREND_LEGAL_WEIGHT


class TrendScorer:
    """트렌드 스코어링 엔진 (하이브리드: 수치 + LLM)"""

    async def score(
        self,
        items: list[RawTrendItem],
    ) -> list[ScoredIssue]:
        """원시 항목을 그룹화 + 스코어링"""
        # 1. 제목 기반 그룹화 (유사 기사 병합)
        groups = self._group_by_topic(items)

        # 2. 각 그룹 스코어링
        scored: list[ScoredIssue] = []
        for group_title, group_items in groups.items():
            mention = self._calculate_mention_score(group_items, len(items))
            legal = await self._judge_legal_relevance(group_title, group_items)

            combined = (
                MENTION_WEIGHT * mention + LEGAL_WEIGHT * legal
            ) * 100  # 0~100 스케일

            scored.append(
                ScoredIssue(
                    id=str(uuid.uuid4()),
                    title=group_title,
                    raw_items=group_items,
                    mention_score=mention,
                    legal_relevance_score=legal,
                    combined_score=round(combined, 1),
                )
            )

        # 점수 내림차순 정렬
        scored.sort(key=lambda x: x.combined_score, reverse=True)
        return scored

    def _group_by_topic(
        self, items: list[RawTrendItem],
    ) -> dict[str, list[RawTrendItem]]:
        """제목 유사도 기반 그룹화 (간단한 단어 겹침 방식)"""
        # 구현: 단어 집합 교집합 비율 >= 0.5이면 같은 그룹
        ...

    def _calculate_mention_score(
        self,
        group_items: list[RawTrendItem],
        total_items: int,
    ) -> float:
        """언급 빈도 기반 점수 (0~1)"""
        return min(len(group_items) / max(total_items * 0.3, 1), 1.0)

    async def _judge_legal_relevance(
        self,
        title: str,
        items: list[RawTrendItem],
    ) -> float:
        """LLM 기반 법적 해석 가능성 판단 + 키워드 보정"""
        # 키워드 매칭 보정
        text = f"{title} {' '.join(i.snippet for i in items[:3])}"
        keyword_hits = sum(1 for kw in LEGAL_KEYWORDS if kw in text)
        keyword_bonus = min(keyword_hits * 0.05, 0.2)

        # LLM 판단
        llm_client = get_llm_client()
        prompt = (
            "다음 뉴스 이슈가 법적 분석/해석이 가능한지 0.0~1.0 사이 "
            "숫자 하나만 응답하세요. 법률 관련성이 전혀 없으면 0.0, "
            "법적 쟁점이 명확하면 1.0입니다.\n\n"
            f"이슈: {title}\n"
            f"내용: {text[:500]}"
        )
        llm_score = await llm_client.agenerate_score(prompt)

        return min(llm_score + keyword_bonus, 1.0)
```

### 6.6 IssueSummarizer (요약 + RAG 매칭)

```python
# backend/app/tools/trend/summarizer.py

from app.services.rag.pipeline import RAGPipeline, PipelineConfig


class IssueSummarizer:
    """이슈 요약 + 관련 법령/판례 매칭"""

    def __init__(self) -> None:
        self._rag = RAGPipeline()

    async def summarize(
        self,
        scored_issues: list[ScoredIssue],
        limit: int = 10,
    ) -> list[TrendIssue]:
        """스코어링된 이슈를 최종 TrendIssue로 변환"""
        issues: list[TrendIssue] = []

        for issue in scored_issues[:limit]:
            # LLM 3줄 요약
            key_points = await self._generate_key_points(issue)

            # RAG 관련 법령/판례 검색
            related_laws, related_cases = await self._find_related_legal(issue)

            # LLM 1~2문장 요약
            summary = await self._generate_summary(issue)

            # 카테고리 판단
            category = await self._classify_category(issue)

            issues.append(
                TrendIssue(
                    id=issue.id,
                    title=issue.title,
                    summary=summary,
                    key_points=key_points,
                    score=issue.combined_score,
                    mention_score=issue.mention_score,
                    legal_relevance_score=issue.legal_relevance_score,
                    category=category,
                    sources=[item.source for item in issue.raw_items],
                    source_articles=[
                        SourceArticle(
                            title=item.title,
                            url=item.url,
                            source=item.source,
                            published_at=item.published_at,
                            snippet=item.snippet,
                        )
                        for item in issue.raw_items
                    ],
                    related_laws=related_laws,
                    related_cases=related_cases,
                    collected_at=datetime.utcnow(),
                )
            )

        return issues

    async def _generate_key_points(self, issue: ScoredIssue) -> list[str]:
        """LLM으로 핵심 쟁점 3줄 생성"""
        context = "\n".join(
            f"- {item.title}: {item.snippet}" for item in issue.raw_items[:5]
        )
        prompt = (
            "다음 뉴스 이슈의 법적 핵심 쟁점을 3줄로 요약하세요.\n"
            "각 줄은 30자 이내로, 법적 관점에서 핵심 논점만 작성합니다.\n\n"
            f"이슈: {issue.title}\n"
            f"관련 기사:\n{context}"
        )
        response = await self._llm_generate(prompt)
        return [line.strip() for line in response.strip().split("\n") if line.strip()][:3]

    async def _find_related_legal(
        self, issue: ScoredIssue,
    ) -> tuple[list[RelatedLaw], list[RelatedCase]]:
        """RAG 파이프라인으로 관련 법령/판례 검색"""
        query = f"{issue.title} {issue.raw_items[0].snippet[:200]}"

        # 법령 검색
        law_config = PipelineConfig(
            n_results=5, doc_type="law",
            enable_rerank=True, rerank_top_k=3,
        )
        law_result = self._rag.execute(query, law_config)
        related_laws = [
            RelatedLaw(
                law_id=doc.get("source_id", ""),
                law_name=doc.get("title", ""),
                relevance_score=doc.get("rerank_score", 0.5),
            )
            for doc in law_result.documents
        ]

        # 판례 검색
        case_config = PipelineConfig(
            n_results=5, doc_type="precedent",
            enable_rerank=True, rerank_top_k=3,
        )
        case_result = self._rag.execute(query, case_config)
        related_cases = [
            RelatedCase(
                case_id=doc.get("source_id", ""),
                case_number=doc.get("case_number", ""),
                case_name=doc.get("title", ""),
                relevance_score=doc.get("rerank_score", 0.5),
            )
            for doc in case_result.documents
        ]

        return related_laws, related_cases
```

### 6.7 ScriptGenerator (대본 생성기)

```python
# backend/app/tools/script/generator.py

from collections.abc import AsyncGenerator

from app.services.rag.pipeline import RAGPipeline, PipelineConfig
from app.tools.script.templates import get_section_prompt


# 페르소나별 분당 글자 수 (기본값)
WORDS_PER_MINUTE = {
    PersonaType.PROFESSIONAL: 250,
    PersonaType.CASUAL: 300,
}


class ScriptGenerator:
    """3단 구조 대본 생성기 (SSE 스트리밍)"""

    def __init__(self) -> None:
        self._rag = RAGPipeline()

    async def generate_stream(
        self,
        request: ScriptRequest,
    ) -> AsyncGenerator[ScriptStreamEvent, None]:
        """대본을 섹션별로 SSE 스트리밍 생성"""

        # 1. RAG 심화 검색
        rag_context = await self._search_legal_context(request)

        # 2. 목표 글자 수 계산
        target_words = WORDS_PER_MINUTE[request.persona] * request.duration

        sections_config = [
            (SectionType.HOOKING, "도입 (Hooking)", 0.15),
            (SectionType.ANALYSIS, "본론 (Legal Analysis)", 0.65),
            (SectionType.ADVICE_CTA, "결론 (Advice & CTA)", 0.20),
        ]

        all_citations: list[Citation] = []

        for section_type, title, ratio in sections_config:
            section_words = int(target_words * ratio)

            # 섹션 시작 이벤트
            yield ScriptStreamEvent(
                event="section_start",
                section=section_type,
            )

            # LLM 스트리밍 생성
            prompt = get_section_prompt(
                section_type=section_type,
                topic=request.topic,
                persona=request.persona,
                target_words=section_words,
                rag_context=rag_context,
            )

            async for chunk in self._llm_stream(prompt):
                yield ScriptStreamEvent(
                    event="content",
                    section=section_type,
                    content=chunk,
                )

            # 섹션 종료 이벤트
            yield ScriptStreamEvent(
                event="section_end",
                section=section_type,
            )

        # 3. 메타데이터 생성
        metadata = await self._generate_metadata(request)
        yield ScriptStreamEvent(
            event="metadata",
            metadata=metadata,
        )

        # 4. 완료
        yield ScriptStreamEvent(event="done")

    async def _search_legal_context(
        self,
        request: ScriptRequest,
    ) -> dict:
        """RAG로 법령/판례 검색하여 컨텍스트 구성"""
        law_config = PipelineConfig(
            n_results=10, doc_type="law",
            enable_rerank=True, rerank_top_k=5,
        )
        case_config = PipelineConfig(
            n_results=10, doc_type="precedent",
            enable_rerank=True, rerank_top_k=5,
        )

        law_result = self._rag.execute(request.topic, law_config)
        case_result = self._rag.execute(request.topic, case_config)

        return {
            "laws": law_result.documents,
            "cases": case_result.documents,
        }
```

### 6.8 대본 프롬프트 템플릿

```python
# backend/app/tools/script/templates.py

HOOKING_PROMPT = """당신은 법률 유튜브 채널의 {persona_desc}입니다.
다음 주제에 대한 도입부(Hooking)를 작성하세요.

## 요구사항
- 시청자의 관심을 끄는 강렬한 사례/질문으로 시작
- 이 주제가 왜 중요한지 간략히 설명
- {target_words}자 내외로 작성
- {tone_guide}

## 주제
{topic}

## 관련 법령/판례 (참고용)
{rag_context}
"""

ANALYSIS_PROMPT = """당신은 법률 유튜브 채널의 {persona_desc}입니다.
다음 주제에 대한 본론(Legal Analysis)을 작성하세요.

## 요구사항
- 관련 법 조항을 정확히 인용 (예: "민법 제750조에 따르면...")
- 관련 판례를 구체적으로 언급 (예: "대법원 2024다12345 판결에서...")
- 인용 시 [📋 인용: 출처명] 형식으로 마크업
- 법적 쟁점을 명확히 분석
- {target_words}자 내외로 작성
- {tone_guide}
- **중요**: 아래 제공된 법령/판례만 인용하세요. 제공되지 않은 판례를 만들어내지 마세요.

## 주제
{topic}

## 관련 법령 (인용 대상)
{law_context}

## 관련 판례 (인용 대상)
{case_context}
"""

ADVICE_CTA_PROMPT = """당신은 법률 유튜브 채널의 {persona_desc}입니다.
다음 주제에 대한 결론(Advice & CTA)을 작성하세요.

## 요구사항
- 시청자에게 실질적인 조언 제공
- 법률 상담 유도 CTA 문구 포함
- 면책 고지 자연스럽게 포함 ("다만, 개별 사안에 따라 다를 수 있으므로...")
- {target_words}자 내외로 작성
- {tone_guide}

## 주제
{topic}
"""

PERSONA_DESC = {
    PersonaType.PROFESSIONAL: "전문 변호사",
    PersonaType.CASUAL: "친근한 법률 유튜버",
}

TONE_GUIDE = {
    PersonaType.PROFESSIONAL: "경어체, 전문 용어 사용, 신뢰감 있는 톤",
    PersonaType.CASUAL: "반말+존댓말 혼합, 쉬운 비유 활용, 친근하고 편안한 톤",
}
```

### 6.9 에이전트 구현 (채팅 위젯 연동)

```python
# backend/app/multi_agent/agents/trend_analysis_agent.py

from app.multi_agent.agents.base_chat import BaseChatAgent
from app.multi_agent.schemas.plan import AgentResult
from app.tools.trend.collector import TrendCollector
from app.tools.trend.scorer import TrendScorer
from app.tools.trend.summarizer import IssueSummarizer


class TrendAnalysisAgent(BaseChatAgent):
    """채팅 위젯용 트렌드 분석 에이전트"""

    @property
    def name(self) -> str:
        return "trend_analysis"

    @property
    def description(self) -> str:
        return "실시간 법률 트렌드를 분석하여 핵심 이슈와 법적 쟁점을 안내합니다."

    async def process(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        session_data: dict[str, Any] | None = None,
        user_location: dict[str, float] | None = None,
    ) -> AgentResult:
        collector = TrendCollector()
        scorer = TrendScorer()
        summarizer = IssueSummarizer()

        # 트렌드 수집 + 스코어링 + 요약
        request = TrendRequest(limit=5)
        raw_items = await collector.collect(request)
        scored = await scorer.score(raw_items)
        issues = await summarizer.summarize(scored, limit=5)

        # 응답 포맷팅
        response_parts: list[str] = [
            "## 최근 법률 트렌드 이슈 TOP 5\n"
        ]
        for i, issue in enumerate(issues, 1):
            response_parts.append(
                f"### {i}위. {issue.title} (점수: {issue.score}/100)\n"
                f"{issue.summary}\n\n"
                f"**핵심 쟁점:**\n"
                + "\n".join(f"- {p}" for p in issue.key_points)
                + "\n"
            )

        return AgentResult(
            response="\n".join(response_parts),
            agent_used="trend_analysis",
        )
```

```python
# backend/app/multi_agent/agents/script_generator_agent.py

class ScriptGeneratorAgent(BaseChatAgent):
    """채팅 위젯용 대본 생성 에이전트"""

    @property
    def name(self) -> str:
        return "script_generator"

    @property
    def description(self) -> str:
        return "법률 주제에 대한 유튜브 대본을 3단 구조로 생성합니다."

    async def process(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        session_data: dict[str, Any] | None = None,
        user_location: dict[str, float] | None = None,
    ) -> AgentResult:
        generator = ScriptGenerator()

        request = ScriptRequest(
            topic=message,
            persona=PersonaType.PROFESSIONAL,
            duration=ScriptDuration.MEDIUM,
        )

        # 채팅에서는 비스트리밍 (전체 대본 한 번에)
        sections: list[str] = []
        async for event in generator.generate_stream(request):
            if event.event == "content":
                sections.append(event.content)

        return AgentResult(
            response="".join(sections),
            agent_used="script_generator",
        )
```

### 6.10 시스템 통합 변경사항

#### router.py — AgentType 추가

```python
class AgentType(str, Enum):
    # ... 기존 ...
    MOCK_TRIAL = "mock_trial"
    # 신규 추가
    CONTENT_MARKETING = "content_marketing"
    # 폴백
    GENERAL = "general"
```

#### nodes.py — AGENT_NODE_MAP 추가

```python
AGENT_NODE_MAP: dict[str, str] = {
    # ... 기존 ...
    "mock_trial": "mock_trial_subgraph",
    # 신규 추가
    "content_marketing": "content_marketing_node",
    # 폴백
    "general": "simple_chat_node",
}
```

#### graph.py — 노드 등록

```python
# content_marketing_node 함수 정의 후 그래프에 추가
graph.add_node("content_marketing_node", content_marketing_node)
graph.add_edge("content_marketing_node", END)
```

#### config.py — 환경 변수 추가

```python
class Settings(BaseSettings):
    # ... 기존 ...

    # Content Marketing
    TAVILY_API_KEY: str = ""
    NAVER_CLIENT_ID: str = ""
    NAVER_CLIENT_SECRET: str = ""
    PERPLEXITY_API_KEY: str = ""          # Phase 2
    YOUTUBE_API_KEY: str = ""              # Phase 2
    TREND_MENTION_WEIGHT: float = 0.4
    TREND_LEGAL_WEIGHT: float = 0.6
    CONTENT_MARKETING_CACHE_TTL: int = 86400  # 24시간
```

#### 모듈 라우터 (modules/content_marketing/router/__init__.py)

```python
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.modules.content_marketing.schema import (
    TrendRequest,
    TrendResponse,
    TrendDetailResponse,
    ScriptRequest,
    MetadataRequest,
    ScriptMetadata,
)
from app.services.service_function.content_marketing_service import (
    collect_trends,
    get_trend_detail,
    generate_script_stream,
    generate_metadata,
)

router = APIRouter()


@router.post("/trends", response_model=TrendResponse)
async def get_trends(request: TrendRequest) -> TrendResponse:
    """트렌드 이슈 수집 및 스코어링"""
    return await collect_trends(request)


@router.get("/trends/{trend_id}", response_model=TrendDetailResponse)
async def get_trend_detail_view(trend_id: str) -> TrendDetailResponse:
    """트렌드 이슈 상세 조회"""
    return await get_trend_detail(trend_id)


@router.post("/script/generate")
async def generate_script(request: ScriptRequest) -> StreamingResponse:
    """대본 SSE 스트리밍 생성"""
    return StreamingResponse(
        generate_script_stream(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.post("/script/metadata", response_model=ScriptMetadata)
async def generate_script_metadata(
    request: MetadataRequest,
) -> ScriptMetadata:
    """대본 메타데이터 재생성"""
    return await generate_metadata(request)
```

---

## 7. Frontend 상세 설계

### 7.1 모듈 등록

#### modules.ts

```typescript
// frontend/src/lib/modules.ts

{
  id: "content-marketing",
  name: "콘텐츠 마케팅",
  description: "실시간 트렌드 분석 + AI 유튜브 대본 생성",
  icon: "📊",
  path: "/content-marketing",
  enabled: true,
  category: "lawyer",  // 변호사 전용
}
```

#### api.ts

```typescript
// frontend/src/lib/api.ts

contentMarketing: {
  trends: "/api/content-marketing/trends",
  trendDetail: (id: string) => `/api/content-marketing/trends/${id}`,
  scriptGenerate: "/api/content-marketing/script/generate",
  scriptMetadata: "/api/content-marketing/script/metadata",
}
```

#### next.config.js

```javascript
// frontend/next.config.js rewrites 추가

{
  source: "/api/content-marketing/:path*",
  destination: "http://localhost:8000/api/content-marketing/:path*",
}
```

### 7.2 TypeScript 타입 정의

```typescript
// frontend/src/features/content-marketing/types/index.ts

// ── Enum ──

export type TrendSourceType =
  | "tavily"
  | "naver"
  | "perplexity"
  | "google_trends"
  | "youtube";

export type TrendCategoryType =
  | "all"
  | "criminal"
  | "civil"
  | "labor"
  | "family"
  | "administrative"
  | "corporate"
  | "ip";

export type TimeRangeType = "24h" | "48h" | "7d";

export type PersonaType = "professional" | "casual";

export type ScriptDurationType = 5 | 10 | 15;

export type SectionType = "hooking" | "analysis" | "advice_cta";

// ── Trend Types ──

export interface SourceArticle {
  title: string;
  url: string;
  source: TrendSourceType;
  published_at: string | null;
  snippet: string;
}

export interface RelatedLaw {
  law_id: string;
  law_name: string;
  relevance_score: number;
}

export interface RelatedCase {
  case_id: string;
  case_number: string;
  case_name: string;
  relevance_score: number;
}

export interface TrendIssue {
  id: string;
  title: string;
  summary: string;
  key_points: string[];
  score: number;
  mention_score: number;
  legal_relevance_score: number;
  category: TrendCategoryType;
  sources: TrendSourceType[];
  source_articles: SourceArticle[];
  related_laws: RelatedLaw[];
  related_cases: RelatedCase[];
  collected_at: string;
}

export interface TrendResponse {
  trends: TrendIssue[];
  total_count: number;
  collected_at: string;
  sources_used: TrendSourceType[];
  cache_hit: boolean;
}

export interface TrendDetailResponse {
  issue: TrendIssue;
  source_articles: SourceArticle[];
  related_laws_detail: Record<string, unknown>[];
  related_cases_detail: Record<string, unknown>[];
}

export interface TrendFilters {
  time_range: TimeRangeType;
  category: TrendCategoryType;
  limit: number;
  query: string | null;
}

// ── Script Types ──

export interface Citation {
  source_type: "precedent" | "law";
  source_id: string;
  source_name: string;
  relevant_text: string;
}

export interface ScriptSection {
  section_type: SectionType;
  title: string;
  content: string;
  citations: Citation[];
  word_count: number;
}

export interface ScriptMetadata {
  description: string;
  tags: string[];
  cta_text: string;
  hashtags: string[];
}

export interface ScriptResult {
  id: string;
  topic: string;
  persona: PersonaType;
  sections: ScriptSection[];
  metadata: ScriptMetadata;
  all_citations: Citation[];
  word_count: number;
  estimated_duration: number;
  disclaimer: string;
  created_at: string;
}

export interface ScriptRequest {
  topic: string;
  trend_id?: string;
  persona: PersonaType;
  duration: ScriptDurationType;
  related_laws: string[];
  related_cases: string[];
}

// ── SSE Event ──

export interface ScriptStreamEvent {
  event:
    | "section_start"
    | "content"
    | "section_end"
    | "metadata"
    | "done"
    | "error";
  section?: SectionType;
  content?: string;
  metadata?: ScriptMetadata;
  error?: string;
}
```

### 7.3 Custom Hooks

```typescript
// frontend/src/features/content-marketing/hooks/useTrends.ts

import { useQuery, useMutation } from "@tanstack/react-query";

export function useTrends(filters: TrendFilters) {
  return useQuery({
    queryKey: ["trends", filters],
    queryFn: () => fetchTrends(filters),
    staleTime: 5 * 60 * 1000,  // 5분
    refetchOnWindowFocus: false,
  });
}

export function useTrendDetail(trendId: string | null) {
  return useQuery({
    queryKey: ["trendDetail", trendId],
    queryFn: () => fetchTrendDetail(trendId!),
    enabled: !!trendId,
  });
}
```

```typescript
// frontend/src/features/content-marketing/hooks/useScriptGeneration.ts

import { useState, useCallback, useRef } from "react";

export function useScriptGeneration() {
  const [sections, setSections] = useState<ScriptSection[]>([]);
  const [metadata, setMetadata] = useState<ScriptMetadata | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [currentSection, setCurrentSection] = useState<SectionType | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const generate = useCallback(async (request: ScriptRequest) => {
    setIsGenerating(true);
    setSections([]);
    setMetadata(null);

    abortRef.current = new AbortController();

    try {
      const response = await fetch(endpoints.contentMarketing.scriptGenerate, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
        signal: abortRef.current.signal,
      });

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      // ... SSE 파싱 로직 ...

    } finally {
      setIsGenerating(false);
      setCurrentSection(null);
    }
  }, []);

  const cancel = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  return { sections, metadata, isGenerating, currentSection, generate, cancel };
}
```

### 7.4 API Service

```typescript
// frontend/src/features/content-marketing/services/index.ts

import { api, endpoints } from "@/lib/api";

export async function fetchTrends(
  filters: TrendFilters,
): Promise<TrendResponse> {
  const response = await api.post(endpoints.contentMarketing.trends, {
    time_range: filters.time_range,
    category: filters.category,
    limit: filters.limit,
    query: filters.query,
  });
  return response.data;
}

export async function fetchTrendDetail(
  trendId: string,
): Promise<TrendDetailResponse> {
  const response = await api.get(
    endpoints.contentMarketing.trendDetail(trendId),
  );
  return response.data;
}

export async function generateMetadata(
  scriptContent: string,
  topic: string,
  persona: PersonaType,
): Promise<ScriptMetadata> {
  const response = await api.post(endpoints.contentMarketing.scriptMetadata, {
    script_content: scriptContent,
    topic,
    persona,
  });
  return response.data;
}
```

---

## 8. Error Handling

### 8.1 Backend 에러 전략

| 시나리오 | HTTP Code | 처리 |
|---------|-----------|------|
| 외부 API 일부 실패 | 200 | Graceful degradation — 성공한 소스만으로 결과 반환 |
| 외부 API 전체 실패 | 503 | `TrendSourceError` → "트렌드 소스 연결 불가" |
| 캐시 미스 (trend_id) | 404 | "트렌드 이슈를 찾을 수 없음" |
| LLM 서비스 불가 | 503 | "AI 서비스 일시 불가" |
| API 키 미설정 | 503 | 해당 소스 건너뛰기 (최소 1개 필수) |
| 요청 유효성 실패 | 422 | Pydantic ValidationError (FastAPI 자동 처리) |
| SSE 스트리밍 중 에러 | SSE error 이벤트 | `{"event": "error", "error": "..."}` |

### 8.2 Frontend 에러 전략

| 시나리오 | UI 처리 |
|---------|---------|
| 트렌드 로딩 실패 | 재시도 버튼 + 에러 메시지 토스트 |
| 대본 생성 중 에러 | 생성 중단 + 에러 메시지, 지금까지 생성된 내용 보존 |
| 네트워크 오류 | "서버 연결을 확인해주세요" 토스트 |
| SSE 연결 끊김 | 자동 재연결 시도 (최대 3회) |

### 8.3 커스텀 예외 클래스

```python
# backend/app/tools/trend/exceptions.py

class ContentMarketingError(Exception):
    """콘텐츠 마케팅 모듈 기본 예외"""
    pass

class TrendSourceError(ContentMarketingError):
    """트렌드 소스 수집 실패"""
    pass

class ScriptGenerationError(ContentMarketingError):
    """대본 생성 실패"""
    pass
```

---

## 9. Security Considerations

| 항목 | 대책 |
|------|------|
| API 키 보안 | 환경변수로 관리, 코드에 하드코딩 금지 |
| 외부 API 과다 호출 | 24시간 TTL 캐싱, 분당 요청 제한 (rate limit) |
| LLM 환각 방지 | RAG 검색 결과만 인용하도록 프롬프트 제약 |
| XSS 방지 | 외부 소스 HTML 태그 strip 처리, 프론트엔드 sanitize |
| 개인정보 | 트렌드 데이터에서 개인정보 필터링 (이름, 전화번호 등) |
| 면책 고지 | 모든 AI 생성 콘텐츠에 "법률 자문이 아닙니다" 표시 |
| 저작권 | 뉴스 원문 사용 금지 — 요약만 사용, 출처 링크 제공 |
| CORS | 기존 next.config.js rewrites 프록시 패턴 사용 |

---

## 10. Test Plan

### 10.1 Unit Tests

| 테스트 | 대상 | 파일 |
|--------|------|------|
| TavilySource 정상 응답 파싱 | `tools/trend/sources/tavily_source.py` | `tests/unit/test_tavily_source.py` |
| NaverSource 정상 응답 파싱 | `tools/trend/sources/naver_source.py` | `tests/unit/test_naver_source.py` |
| TrendCollector 중복 제거 | `tools/trend/collector.py` | `tests/unit/test_trend_collector.py` |
| TrendScorer 점수 계산 | `tools/trend/scorer.py` | `tests/unit/test_trend_scorer.py` |
| ScriptGenerator 프롬프트 | `tools/script/templates.py` | `tests/unit/test_script_templates.py` |
| Pydantic 스키마 유효성 | `modules/.../schema/` | `tests/unit/test_content_marketing_schema.py` |

### 10.2 Integration Tests

| 테스트 | 대상 | 파일 |
|--------|------|------|
| POST /trends 전체 흐름 | 라우터 → 서비스 → 소스 | `tests/integration/test_content_marketing_trends.py` |
| POST /script/generate SSE | 라우터 → SSE 스트리밍 | `tests/integration/test_script_generation.py` |
| RAG 연동 법령/판례 매칭 | IssueSummarizer → RAGPipeline | `tests/integration/test_trend_rag.py` |
| 에이전트 라우팅 | router_node → content_marketing_node | `tests/integration/test_content_marketing_agent.py` |

### 10.3 Frontend Tests

| 테스트 | 대상 |
|--------|------|
| TrendCard 렌더링 | 점수 바, 핵심 쟁점 표시 |
| ScriptPreview 마크다운 렌더링 | 섹션별 구분, 인용 마크업 |
| SSE 파싱 | useScriptGeneration 훅의 이벤트 처리 |
| 필터 변경 → 재조회 | useTrends 훅의 queryKey 변경 |

---

## 11. Clean Architecture

### 11.1 레이어 구조

```
┌─────────────────────────────────────────────────────────┐
│  Presentation Layer                                       │
│  ├── modules/content_marketing/router/    (FastAPI 라우터) │
│  └── multi_agent/agents/                  (채팅 에이전트)  │
├─────────────────────────────────────────────────────────┤
│  Service Layer                                            │
│  └── services/service_function/                           │
│      content_marketing_service.py         (비즈니스 로직)  │
├─────────────────────────────────────────────────────────┤
│  Tool Layer                                               │
│  ├── tools/trend/                         (트렌드 도구)   │
│  │   ├── collector.py                                     │
│  │   ├── scorer.py                                        │
│  │   ├── summarizer.py                                    │
│  │   └── sources/ (어댑터)                                │
│  └── tools/script/                        (대본 도구)     │
│      ├── generator.py                                     │
│      ├── templates.py                                     │
│      └── metadata.py                                      │
├─────────────────────────────────────────────────────────┤
│  Infrastructure Layer                                     │
│  ├── tools/llm/           (LLM 클라이언트)                │
│  ├── services/rag/        (RAG 파이프라인)                │
│  └── core/config.py       (환경변수)                      │
└─────────────────────────────────────────────────────────┘
```

### 11.2 의존성 규칙

- Presentation → Service → Tool → Infrastructure
- 상위 레이어는 하위만 의존 (역방향 금지)
- Tool 레이어 간 직접 의존 허용 (같은 수준)
- 외부 API 클라이언트는 Tool 레이어 내 sources/ 어댑터로 캡슐화

---

## 12. Implementation Guide

### 12.1 구현 순서 (Phase별)

| Phase | Step | 작업 | 핵심 파일 | 의존성 |
|-------|------|------|----------|--------|
| **1. 스캐폴딩** | 1 | 모듈 디렉토리 생성 | `modules/content_marketing/` | - |
| | 2 | Pydantic 스키마 정의 | `modules/.../schema/__init__.py` | Step 1 |
| | 3 | 환경변수 추가 | `core/config.py`, `.env.example` | - |
| **2. 트렌드 백엔드** | 4 | BaseTrendSource 인터페이스 | `tools/trend/sources/__init__.py` | Step 2 |
| | 5 | TavilySource | `tools/trend/sources/tavily_source.py` | Step 4 |
| | 6 | NaverSource | `tools/trend/sources/naver_source.py` | Step 4 |
| | 7 | TrendCollector | `tools/trend/collector.py` | Step 5-6 |
| | 8 | TrendScorer | `tools/trend/scorer.py` | Step 7 |
| | 9 | IssueSummarizer | `tools/trend/summarizer.py` | Step 8 |
| | 10 | 서비스 함수 + API 라우터 | `service_function/`, `router/` | Step 9 |
| **3. 대본 백엔드** | 11 | 프롬프트 템플릿 정의 | `tools/script/templates.py` | Step 2 |
| | 12 | ScriptGenerator (SSE) | `tools/script/generator.py` | Step 11 |
| | 13 | MetadataGenerator | `tools/script/metadata.py` | Step 12 |
| | 14 | API 엔드포인트 (SSE) | `router/__init__.py` | Step 12-13 |
| **4. 에이전트** | 15 | TrendAnalysisAgent | `multi_agent/agents/` | Step 10 |
| | 16 | ScriptGeneratorAgent | `multi_agent/agents/` | Step 14 |
| | 17 | 라우터/노드/그래프 통합 | `router.py`, `nodes.py`, `graph.py` | Step 15-16 |
| **5. 프론트엔드** | 18 | 모듈 등록 (3곳) | `modules.ts`, `api.ts`, `next.config.js` | - |
| | 19 | TypeScript 타입 정의 | `features/.../types/index.ts` | Step 18 |
| | 20 | API 서비스 + 훅 | `features/.../services/`, `hooks/` | Step 19 |
| | 21 | 트렌드 대시보드 UI | `TrendDashboard`, `TrendCard`, `TrendFilters` | Step 20 |
| | 22 | 이슈 상세 뷰 | `TrendDetailView` | Step 21 |
| | 23 | 대본 생성 UI | `ScriptGenerator`, `PersonaSelector` | Step 20 |
| | 24 | 대본 에디터/미리보기 | `ScriptEditor`, `ScriptPreview` | Step 23 |
| | 25 | 메타데이터 + 내보내기 | `MetadataPanel`, `ExportButton` | Step 24 |
| **6. 통합** | 26 | Frontend ↔ Backend 연동 | 전체 | Step 14, 25 |
| | 27 | 정적 검증 | `ruff`, `mypy`, `npm run build` | Step 26 |
| | 28 | E2E 흐름 테스트 | 전체 | Step 27 |

### 12.2 검증 체크리스트

- [ ] `uv run ruff check backend/app/` — 린트 통과
- [ ] `uv run mypy backend/app/` — 타입 체크 통과
- [ ] `npm run build` (frontend/) — 프론트엔드 빌드 통과
- [ ] POST /api/content-marketing/trends — 200 응답 + TrendResponse 구조 확인
- [ ] GET /api/content-marketing/trends/{id} — 200 응답 확인
- [ ] POST /api/content-marketing/script/generate — SSE 스트리밍 정상
- [ ] POST /api/content-marketing/script/metadata — 200 응답 확인
- [ ] Frontend → Backend API 프록시 정상 동작 (next.config.js)
- [ ] modules.ts 등록 + 사이드바 노출 확인
- [ ] 채팅 위젯에서 "법률 트렌드 알려줘" → TrendAnalysisAgent 라우팅 확인
- [ ] 면책 고지 ("AI 생성 콘텐츠") 모든 화면에 표시

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1.0 | 2026-02-20 | Initial draft — 모듈 A(트렌드 분석) + 모듈 B(대본 생성) 상세 설계. Architecture, Data Model, API Spec, UI/UX, Backend/Frontend 설계, Error Handling, Security, Test Plan, Implementation Guide 작성 | Claude |
