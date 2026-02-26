# Content Marketing v2.0 — 상세 설계 문서

> **Summary**: 변호사 개인화 페르소나 기반 유튜브 콘텐츠 자동 생성 시스템 상세 설계.
> 페르소나 초기화(Track 1/2), Legal Gate 스코어링, RAG 프롬프트 체인, 피드백 루프를 포함.
>
> **Project**: law-3 (Legal President / 리걸 프레지던트)
> **Version**: 2.0.0
> **Author**: Lead Manager TF (Opus 4.6) + Gemini CLI Red Team
> **Date**: 2026-02-22
> **Status**: Final (v2.0) — Gemini CLI Red Team 교차 검증 완료
> **Planning Doc**: [content-marketing-v2.plan.md](../../01-plan/features/content-marketing-v2.plan.md)
> **Supersedes**: [content-marketing.design.md](./content-marketing.design.md) (v0.1.0)

---

## 1. Overview

### 1.1 Design Goals

1. **페르소나 기반 개인화**: 변호사의 전문 분야, 채널 톤, 타겟 시청자를 분석하여 맞춤형 콘텐츠 생성
2. **Legal Gate 품질 보증**: 비법률 가십이 상위에 노출되지 않도록 법적 관련성 게이트 필터 적용
3. **PII 보호**: Track 1 대화 이력 분석 시 원문 텍스트 대신 **메타데이터(에이전트 유형, 법률 키워드 상위 20개, 카테고리 통계)** 기반 분석으로 PII 유출 원천 차단
4. **RAG 프롬프트 체인**: 3단계 체인(쟁점 분석 → RAG 심화 → 컨텍스트 구성)으로 할루시네이션 최소화
5. **기존 아키텍처 일관성**: ModuleRegistry, BaseChatAgent, AGENT_NODE_MAP, snake_case API 계약 100% 준수

### 1.2 Design Principles

- **Single Responsibility**: PIIMasker, PersonaAnalyzer, LegalGateScorer, PromptChainExecutor를 각각 독립 클래스로 분리
- **Open/Closed**: 기존 TrendScorer, ScriptGenerator를 상속/확장하여 v1.0 하위호환 유지
- **2-Layer Storage**: PostgreSQL(source of truth) + localStorage(빠른 캐시)로 페르소나 영속 저장
- **Fail-Safe**: Track 1 분석 실패 시 Track 2 폴백, Legal Gate 실패 시 기본 스코어링, JSON 파싱 실패 시 개별 호출 폴백
- **Auth-First**: 모든 Persona API는 `user_id`를 클라이언트에서 받지 않고 **Auth Dependency(Bearer Token)**에서 추출한 `current_user.id`만 사용 (IDOR 방지)
- **Background-First Scoring**: 트렌드 수집 + Legal Gate 스코어링은 **백그라운드 워커**에서 주기적 수행 → 글로벌 캐시 저장, API 호출 시에는 `fitness_score`만 실시간 계산
- **Metadata-over-Raw**: Track 1 분석 시 원문 텍스트를 LLM에 전달하지 않고 메타데이터(에이전트 유형, 키워드 통계, 카테고리 분포)만 전달하여 PII 유출 원천 차단

---

## 2. Architecture

### 2.1 Component Diagram (v2.0)

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                     Next.js App (/content-marketing)                           │
│                                                                                │
│  ┌──────────────────────────────────────────────────────────────────────────┐ │
│  │                    PersonaGate (진입점)                                    │ │
│  │  페르소나 존재? → Yes: 메인 대시보드                                       │ │
│  │                → No:  Track 판별 → Track 1 (자동 분석) / Track 2 (온보딩) │ │
│  └──────────────────────────────┬───────────────────────────────────────────┘ │
│                                  │                                             │
│  ┌──────────────────────────────┼──────────────────────────────────────────┐  │
│  │                    Main Dashboard                                        │  │
│  │  ┌─────────────┐                                                        │  │
│  │  │ PersonaBanner│ ← usePersona() hook                                   │  │
│  │  └─────────────┘                                                        │  │
│  │                                                                          │  │
│  │  [트렌드 분석] [대본 생성] [내 콘텐츠]  ← Tab Navigation                │  │
│  │                                                                          │  │
│  │  ┌────────────────────────┐  ┌───────────────────────────┐              │  │
│  │  │ TrendDashboard         │  │ ScriptGenerator (Enhanced)│              │  │
│  │  │  ├── TrendFilters      │  │  ├── PersonaContext       │              │  │
│  │  │  ├── TrendCard[]       │  │  ├── RAGPreview           │              │  │
│  │  │  │   ├── ScoreBar(5D)  │  │  ├── ScriptPreview       │              │  │
│  │  │  │   └── FitnessBadge  │  │  ├── FeedbackPanel (NEW) │              │  │
│  │  │  └── TrendDetailView   │  │  ├── MetadataPanel       │              │  │
│  │  └────────────────────────┘  │  └── ExportButton         │              │  │
│  │                               └───────────────────────────┘              │  │
│  └──────────────────────────────────────┬──────────────────────────────────┘  │
│                                          │                                     │
│                    ┌─────────────────────┴───────────────────────┐             │
│                    │           API Service (fetch/SSE)            │             │
│                    │  + usePersona() → GET/POST/PUT persona API  │             │
│                    └─────────────────────┬───────────────────────┘             │
└──────────────────────────────────────────┼────────────────────────────────────┘
                                           │ HTTP
┌──────────────────────────────────────────┼────────────────────────────────────┐
│                            FastAPI Backend                                      │
│                                           │                                     │
│  ┌────────────────────────────────────────┴────────────────────────────────┐   │
│  │             /api/content-marketing (모듈 라우터)                          │   │
│  │                                                                          │   │
│  │  ── Persona API (NEW) ──                                                │   │
│  │  POST /persona/analyze    → PersonaAnalyzer (Track 1)                   │   │
│  │  POST /persona/onboarding → OnboardingProcessor (Track 2)               │   │
│  │  GET  /persona/current    → PersonaDBService.get()                      │   │
│  │  PUT  /persona/update     → PersonaDBService.update()                   │   │
│  │  POST /persona/feedback   → PersonaDBService.save_feedback() (NEW)      │   │
│  │                                                                          │   │
│  │  ── Trend API (ENHANCED) ──                                             │   │
│  │  POST /trends             → LegalGateScorer + FitnessScorer             │   │
│  │  GET  /trends/{id}        → 캐시 조회 + score_detail                    │   │
│  │                                                                          │   │
│  │  ── Script API (ENHANCED) ──                                            │   │
│  │  POST /script/generate    → PromptChainExecutor + PersonaContext (SSE)  │   │
│  │  POST /script/metadata    → MetadataGenerator                           │   │
│  └────────────────────────────────────────┬────────────────────────────────┘   │
│                                            │                                    │
│  ┌─────────────────────────────────── Service Layer ───────────────────────┐   │
│  │                                                                          │   │
│  │  ┌─────────────────────┐  ┌──────────────────────┐  ┌────────────────┐ │   │
│  │  │ PersonaAnalyzer     │  │ LegalGateScorer      │  │ PromptChain    │ │   │
│  │  │  ├── PIIMasker      │  │  ├── LegalGate       │  │ Executor       │ │   │
│  │  │  ├── LLM Extractor  │  │  │   (L ≥ 0.3)      │  │  ├── Chain 1   │ │   │
│  │  │  └── RAG Validator  │  │  ├── MentionScorer   │  │  ├── Chain 2   │ │   │
│  │  └─────────┬───────────┘  │  ├── ControversySc.  │  │  └── Chain 3   │ │   │
│  │            │               │  ├── SpreadScorer    │  └────────┬───────┘ │   │
│  │  ┌─────────┴───────────┐  │  └── FitnessScorer   │           │         │   │
│  │  │ OnboardingProcessor │  └──────────┬───────────┘           │         │   │
│  │  │  └── Enum Mapper    │             │                        │         │   │
│  │  └─────────────────────┘             │                        │         │   │
│  │                                       │                        │         │   │
│  │  ┌────────────────────────────────────┴────────────────────────┴───────┐│   │
│  │  │                    PersonaDBService                                  ││   │
│  │  │  ├── get_persona(user_id)     → LawyerPersona                      ││   │
│  │  │  ├── create_persona(...)      → LawyerPersona                      ││   │
│  │  │  ├── update_persona(...)      → LawyerPersona                      ││   │
│  │  │  └── save_feedback(...)       → PersonaFeedback                    ││   │
│  │  └────────────────────────────────────────────────────────────────────┘│   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────┐  ┌──────────────────────┐   │
│  │ PostgreSQL    │  │ LanceDB RAG  │  │ LLM      │  │ in-memory Cache      │   │
│  │ lawyer_       │  │ legal_chunks │  │ (Solar/  │  │ (트렌드 + Chain 1)   │   │
│  │ personas (DB) │  │ (253K+)      │  │  OpenAI) │  │                      │   │
│  └──────────────┘  └──────────────┘  └──────────┘  └──────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 2.1.1 Architecture Notes (Red Team 반영)

| 항목 | 기존 설계 | Red Team 수정 | 근거 |
|------|----------|--------------|------|
| **Persona API 인증** | `user_id`를 Body/Query로 전달 | **Auth Dependency(Bearer Token)**에서 `current_user.id` 추출, 클라이언트에서 user_id 전달 금지 | IDOR 취약점 방지 |
| **트렌드 스코어링** | API 호출 시 매번 N건 LLM 호출 | **백그라운드 워커**가 주기적(1h) 수집+스코어링 → 글로벌 캐시, API 호출 시 `fitness_score`만 실시간 가중합 | Rate Limit/비용/지연 |
| **Track 1 PII** | 정규식 마스킹 후 원문을 LLM에 전달 | 원문 미전달, **메타데이터만 추출**(에이전트 유형, 법률 키워드 Top 20, 카테고리 통계)하여 LLM 분석 | 변호사법 위반/PII 유출 |
| **Legal Gate UI** | 미통과 이슈 하단 배치만 | 미통과 사유(`gate_rejection_reason`) 텍스트를 UI에 표시 | 서비스 신뢰도 |
| **Chain 1** | 쟁점 추출 후 바로 Chain 2 진행 | Chain 1 출력을 트렌드 `key_points`와 유사도 비교하는 **자가 검증** 추가 | 할루시네이션 방지 |
| **localStorage** | 단순 캐시 | `updated_at` 기반 **Version-based Sync** (서버 vs 로컬 비교 후 갱신) | 멀티디바이스 |

### 2.2 Data Flow — 페르소나 초기화

```
사용자가 /content-marketing 진입 (Auth: Bearer Token → current_user.id 추출)
    │
    ├── 1. localStorage 캐시 확인 (즉시 로딩)
    │   └── persona 있으면 → 즉시 메인 대시보드 표시 (stale 가능)
    │
    ├── 2. GET /api/content-marketing/persona/current (Auth Header 자동 첨부)
    │   ├── DB에 persona 있으면 → Version-based Sync
    │   │   ├── 서버 updated_at > 로컬 updated_at → 서버 데이터로 갱신
    │   │   └── 서버 updated_at ≤ 로컬 updated_at → 로컬 캐시 유지
    │   └── DB에 persona 없으면 → Track 판별로 진행
    │
    ▼
 Track 판별: 대화 이력 ≥ 30건?
    │
    ├── Yes → Track 1 (Passive)
    │   │
    │   ├── POST /api/content-marketing/persona/analyze (Auth: Bearer Token)
    │   │   ├── Step 0: MetadataExtractor.extract(chat_history) ← CRITICAL (원문 미전달)
    │   │   │   ├── 에이전트 유형별 빈도 (형사: 12건, 가사: 8건, ...)
    │   │   │   ├── 법률 키워드 상위 20개 (TF-IDF 기반)
    │   │   │   └── 카테고리 분포 통계 (%, 건수)
    │   │   ├── Step 1: 대화 이력 조회 (최근 30일, 최대 100건)
    │   │   ├── Step 2: LLM 전문 분야 추출 (메타데이터만 전달, 원문 미포함)
    │   │   ├── Step 3: LLM 관심 쟁점 추출 (메타데이터만 전달, 원문 미포함)
    │   │   └── Step 4: 4중 할루시네이션 검증
    │   │       ├── Enum 체크 → 키워드 매칭 → RAG 검증
    │   │       └── 검증 실패 시 → Track 2로 폴백
    │   │
    │   └── 분석 결과 확인 UI → 변호사 승인/수정/거부
    │       ├── 승인 → LawyerPersona 생성 → DB 저장
    │       └── 거부 → Track 2로 전환
    │
    └── No → Track 2 (Active)
        │
        ├── 온보딩 위저드 (4단계)
        │   ├── Step 1: 전문 분야 선택 (복수)
        │   ├── Step 2: 타겟 시청자 선택
        │   ├── Step 3: 영상 스타일 선택
        │   └── Step 4: 관심 쟁점 키워드 (선택)
        │
        └── POST /api/content-marketing/persona/onboarding
            └── LawyerPersona 생성 → DB 저장 + localStorage 캐시
```

### 2.3 Data Flow — 트렌드 스코어링 (Legal Gate + 백그라운드 워커)

#### 2.3.1 백그라운드 워커 (주기적 수행, 사용자 독립)

```
BackgroundTrendWorker (Cron: 매 1시간)
    │
    ▼
TrendCollector.collect(category="all")
    │
    ├── asyncio.gather() (병렬 수집: Tavily + Naver)
    └── 중복 제거 (URL 기반)
    │
    ▼
LegalGateScorer.score_global(raw_issues)  ← LLM 호출은 여기서만 발생
    │
    ├── Stage 1: 통합 LLM 프롬프트 (이슈당 1회 호출)
    │   ├── legal_score, legal_stage, controversy_ratio, category 동시 추출
    │   └── JSON 파싱 실패 시 → 3단계 폴백 (코드블록 제거 → 정규식 → 개별 호출)
    │
    ├── Stage 2: Legal Gate 필터
    │   ├── L ≥ LEGAL_THRESHOLD (0.3) → 통과
    │   │   └── gate_rejection_reason: null
    │   └── L < 0.3 → "법률 관련성 낮음" 라벨
    │       └── gate_rejection_reason: "법적 쟁점화 지표 0.12로 법률 콘텐츠 기준(0.3) 미달"
    │
    ├── Stage 3: 글로벌 기본 스코어링 (Gate 통과 이슈만, fitness 제외)
    │   ├── M = 멀티소스 가중 언급량
    │   ├── C = controversy_ratio × sentiment_divergence
    │   └── S = recent_count / past_count (시간 기반)
    │
    └── 글로벌 캐시 저장 (PostgreSQL trend_cache 테이블 또는 in-memory, TTL=1h)
```

#### 2.3.2 API 호출 (사용자별 실시간, fitness_score만 계산)

```
POST /api/content-marketing/trends (Auth: Bearer Token, persona_id 포함)
    │
    ▼
글로벌 캐시 조회
    ├── 캐시 히트 → 글로벌 스코어 결과 로드 (LLM 호출 없음)
    └── 캐시 미스 → 즉시 백그라운드 워커 트리거 + 이전 캐시 반환
    │
    ▼
FitnessScorer.score_per_user(cached_issues, persona)  ← LLM 호출 없음
    │
    ├── F = category_match + topic_overlap + audience_relevance
    └── TrendScore = L × (0.30×M + 0.25×C + 0.15×S + 0.30×F) × 100
    │
    ▼
IssueSummarizer.summarize(scored_issues, limit)
    │
    ├── LLM: 각 이슈별 핵심 쟁점 3줄 요약 (캐시 포함)
    ├── RAGPipeline: 관련 법령/판례 검색
    └── score_detail + gate_rejection_reason 포함하여 TrendResponse 반환
```

### 2.4 Data Flow — 대본 생성 (프롬프트 체인)

```
POST /api/content-marketing/script/generate (persona_id 포함)
    │
    ▼
PromptChainExecutor.execute(request, persona)
    │
    ├── Chain 1: 쟁점 분석 (캐시 확인, TTL=1h)
    │   ├── LLM: "이슈의 핵심 법적 쟁점 3개 + 예상 법령명/조문"
    │   └── 출력: [{"쟁점": "...", "예상_법령": "민법 제750조"}]
    │
    ├── Chain 2: RAG 심화 검색 (asyncio.gather 병렬)
    │   ├── 각 쟁점별 RAGPipeline.execute(query=쟁점, doc_type="law")
    │   ├── 각 쟁점별 RAGPipeline.execute(query=쟁점, doc_type="precedent")
    │   ├── PipelineConfig(n_results=10, enable_rerank=True, rerank_top_k=5)
    │   └── 교차 검증: Chain 1 예상 법령이 RAG 결과에 포함되는지 확인
    │
    ├── Chain 3: 컨텍스트 구성
    │   └── 쟁점별로 법령/판례 그룹화 → ScriptContext 생성
    │
    ▼
ScriptGenerator.generate_stream(request, persona, script_context) [SSE]
    │
    ├── 페르소나 맥락 주입 (전문 분야, 톤, 타겟, 채널 스타일)
    │
    ├── event: stage_update (Chain 진행 상황)
    │
    ├── 도입부 (Hooking) — 15%
    │   └── event: section_start → content(스트리밍) → section_end
    │
    ├── 본론 (Legal Analysis) — 65%
    │   ├── 인용 마크업: [📋 인용: 출처명]
    │   └── event: section_start → content(스트리밍) → section_end
    │
    ├── 결론 (Advice & CTA) — 20%
    │   └── event: section_start → content(스트리밍) → section_end
    │
    ├── 인용 교차 검증: 대본 내 인용이 RAG 컨텍스트에 존재하는지 확인
    │
    ├── event: metadata (제목, 설명, 태그, CTA, 해시태그)
    └── event: done
```

### 2.5 Dependencies

| Component | Depends On | Purpose |
|-----------|-----------|---------|
| MetadataExtractor | 법률 키워드 사전, Counter | 대화 이력 → PII 없는 메타데이터 추출 |
| PersonaAnalyzer | MetadataExtractor, LLM Client, RAGPipeline | Track 1 메타데이터 기반 페르소나 분석 |
| BackgroundTrendWorker | TrendCollector, LegalGateScorer, 캐시(PostgreSQL/Redis) | 주기적 트렌드 수집 + 글로벌 스코어링 |
| FitnessScorer | PersonaDBService | 사용자별 fitness_score 실시간 계산 |
| OnboardingProcessor | Enum 매핑 | Track 2 온보딩 결과 → LawyerPersona |
| PersonaDBService | SQLAlchemy, PostgreSQL | 페르소나 CRUD + 피드백 저장 |
| LegalGateScorer | LLM Client, LEGAL_KEYWORDS | Legal Gate + 5차원 스코어링 |
| PromptChainExecutor | LLM Client, RAGPipeline | 3단계 프롬프트 체인 |
| ScriptGenerator | PromptChainExecutor, PersonaContext | 페르소나 맥락 주입 대본 생성 |

---

## 3. Data Model

### 3.1 Pydantic Schema — Persona (NEW)

```python
from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class PersonaTone(str, Enum):
    """영상 톤 (v1.0 PersonaType 대체)"""
    PROFESSIONAL = "professional"    # 전문가형
    CASUAL = "casual"                # 캐주얼형
    STORYTELLING = "storytelling"    # 스토리텔링형
    EDUCATIONAL = "educational"      # 교육형


class TargetAudience(str, Enum):
    """타겟 시청자"""
    GENERAL_PUBLIC = "general_public"
    BUSINESS = "business"
    LEGAL_STUDENT = "legal_student"
    LEGAL_PROFESSIONAL = "legal_professional"


class ChannelStyle(str, Enum):
    """채널 스타일"""
    EXPERT = "expert"
    CASUAL_FRIENDLY = "casual_friendly"
    STORYTELLING = "storytelling"
    LECTURE = "lecture"


class LawyerPersona(BaseModel):
    """변호사 페르소나"""
    id: str                                          # UUID
    user_id: str                                     # 사용자 식별자
    specialty_areas: list[TrendCategory]              # 전문 분야 (1~3개)
    focus_topics: list[str] = Field(
        default_factory=list, max_length=5,
    )
    preferred_tone: PersonaTone = PersonaTone.PROFESSIONAL
    target_audience: TargetAudience = TargetAudience.GENERAL_PUBLIC
    channel_style: ChannelStyle | None = None
    source: Literal["passive", "active"]
    confidence: float = Field(default=1.0, ge=0, le=1)
    created_at: datetime
    updated_at: datetime


class PersonaAnalysisRequest(BaseModel):
    """Track 1: 자동 분석 요청 (user_id는 Auth Dependency에서 추출)"""
    max_history: int = Field(default=100, ge=10, le=500)
    days_back: int = Field(default=30, ge=7, le=90)


class PersonaOnboardingRequest(BaseModel):
    """Track 2: 온보딩 결과 (user_id는 Auth Dependency에서 추출)"""
    specialty_areas: list[TrendCategory] = Field(min_length=1, max_length=3)
    target_audience: TargetAudience
    preferred_tone: PersonaTone
    channel_style: ChannelStyle | None = None
    focus_topics: list[str] = Field(default_factory=list, max_length=5)


class PersonaUpdateRequest(BaseModel):
    """페르소나 부분 수정"""
    specialty_areas: list[TrendCategory] | None = None
    focus_topics: list[str] | None = None
    preferred_tone: PersonaTone | None = None
    target_audience: TargetAudience | None = None
    channel_style: ChannelStyle | None = None


class PersonaFeedbackRequest(BaseModel):
    """대본 생성 후 피드백"""
    persona_id: str
    script_id: str | None = None
    rating: int = Field(ge=1, le=5)
    feedback_type: Literal[
        "tone_mismatch", "specialty_mismatch",
        "audience_mismatch", "other",
    ] | None = None
    feedback_text: str | None = None
```

### 3.2 Pydantic Schema — Scoring (ENHANCED)

```python
class LegalStage(str, Enum):
    """법적 단계"""
    LITIGATION = "litigation"
    LEGISLATION = "legislation"
    PROSECUTION = "prosecution"
    DISPUTE = "dispute"
    MENTION = "mention"


class TrendScoreDetail(BaseModel):
    """트렌드 세부 점수 (v2.0)"""
    mention_score: float = Field(ge=0, le=1)
    legal_score: float = Field(ge=0, le=1)
    controversy_score: float = Field(ge=0, le=1)
    spread_score: float = Field(ge=0, le=1)
    fitness_score: float = Field(ge=0, le=1)
    legal_stage: LegalStage
    legal_gate_passed: bool                     # Legal Gate 통과 여부
    gate_rejection_reason: str | None = None    # 미통과 시 사유 (Red Team [보완 3])
    combined_score: float = Field(ge=0, le=100)
```

### 3.3 기존 스키마 변경 사항

| 스키마 | 변경 | 내용 |
|--------|------|------|
| `TrendIssue` | 필드 추가 | `score_detail: TrendScoreDetail`, `fitness_label: str` |
| `TrendRequest` | 필드 추가 | `persona_id: str \| None = None` |
| `ScriptRequest` | 필드 변경 | `persona: PersonaType` → `persona_id: str \| None = None` (하위호환: persona 필드 유지) |
| `ScriptStreamEvent` | 이벤트 추가 | `event: "stage_update"` (Chain 진행 상황) |
| `PersonaType` | deprecated | `PersonaTone`으로 대체, 매핑 유지: professional→professional, casual→casual |

### 3.4 SQLAlchemy ORM — LawyerPersona (NEW)

```python
# backend/app/models/lawyer_persona.py

from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Index,
    String,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB

from app.core.database import Base


class LawyerPersonaModel(Base):
    """변호사 페르소나 ORM 모델"""
    __tablename__ = "lawyer_personas"

    id = Column(String(36), primary_key=True)
    user_id = Column(String(255), nullable=False, unique=True)
    specialty_areas = Column(JSONB, nullable=False)       # ["criminal", "family"]
    focus_topics = Column(JSONB, default=list)             # ["이혼 재산분할", ...]
    preferred_tone = Column(String(50), nullable=False, default="professional")
    target_audience = Column(String(50), nullable=False, default="general_public")
    channel_style = Column(String(50), nullable=True)
    source = Column(String(10), nullable=False)            # "passive" | "active"
    confidence = Column(Float, default=1.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_lawyer_personas_user_id", "user_id"),
    )


class LawyerPersonaFeedbackModel(Base):
    """페르소나 피드백 ORM 모델"""
    __tablename__ = "lawyer_persona_feedback"

    id = Column(String(36), primary_key=True)
    persona_id = Column(String(36), nullable=False)
    script_id = Column(String(36), nullable=True)
    rating = Column(Float, nullable=False)
    feedback_type = Column(String(50), nullable=True)
    feedback_text = Column(String(1000), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("idx_persona_feedback_persona_id", "persona_id"),
    )
```

### 3.5 TypeScript Types — Frontend (NEW/ENHANCED)

```typescript
// frontend/src/features/content-marketing/types/index.ts — 추가분

// ── Persona Types (NEW) ──

export type PersonaTone = 'professional' | 'casual' | 'storytelling' | 'educational'
export type TargetAudience = 'general_public' | 'business' | 'legal_student' | 'legal_professional'
export type ChannelStyle = 'expert' | 'casual_friendly' | 'storytelling' | 'lecture'
export type LegalStage = 'litigation' | 'legislation' | 'prosecution' | 'dispute' | 'mention'
export type PersonaSource = 'passive' | 'active'

export interface LawyerPersona {
  id: string
  user_id: string
  specialty_areas: TrendCategory[]
  focus_topics: string[]
  preferred_tone: PersonaTone
  target_audience: TargetAudience
  channel_style: ChannelStyle | null
  source: PersonaSource
  confidence: number
  created_at: string
  updated_at: string
}

export interface PersonaAnalysisRequest {
  // user_id는 Auth Header(Bearer Token)에서 서버가 추출 — IDOR 방지
  max_history?: number
  days_back?: number
}

export interface PersonaOnboardingRequest {
  // user_id는 Auth Header(Bearer Token)에서 서버가 추출 — IDOR 방지
  specialty_areas: TrendCategory[]
  target_audience: TargetAudience
  preferred_tone: PersonaTone
  channel_style?: ChannelStyle | null
  focus_topics?: string[]
}

export interface PersonaUpdateRequest {
  specialty_areas?: TrendCategory[]
  focus_topics?: string[]
  preferred_tone?: PersonaTone
  target_audience?: TargetAudience
  channel_style?: ChannelStyle | null
}

export interface PersonaFeedbackRequest {
  persona_id: string
  script_id?: string | null
  rating: number
  feedback_type?: 'tone_mismatch' | 'specialty_mismatch' | 'audience_mismatch' | 'other' | null
  feedback_text?: string | null
}

// ── Scoring Types (ENHANCED) ──

export interface TrendScoreDetail {
  mention_score: number
  legal_score: number
  controversy_score: number
  spread_score: number
  fitness_score: number
  legal_stage: LegalStage
  legal_gate_passed: boolean
  gate_rejection_reason: string | null     // 미통과 시 사유
  combined_score: number
}

// ── Enhanced Existing Types ──

export interface TrendIssueV2 extends TrendIssue {
  score_detail: TrendScoreDetail
  fitness_label: string
}

export interface TrendRequestV2 extends TrendRequest {
  persona_id: string | null
}

export interface ScriptRequestV2 {
  topic: string
  trend_id: string | null
  persona_id: string | null
  duration: ScriptDuration
  related_laws: string[]
  related_cases: string[]
}

// ── UI State Types (NEW) ──

export type OnboardingStep = 1 | 2 | 3 | 4

export interface OnboardingState {
  step: OnboardingStep
  specialty_areas: TrendCategory[]
  target_audience: TargetAudience | null
  preferred_tone: PersonaTone | null
  channel_style: ChannelStyle | null
  focus_topics: string[]
}
```

### 3.6 보조 타입 — 내부 모델 (Python)

```python
# backend/app/tools/persona/models.py

from dataclasses import dataclass, field


@dataclass
class ChatMetadata:
    """대화 이력 메타데이터 (PII 미포함, Red Team [심각 3] 반영)"""
    total_conversations: int
    agent_type_distribution: dict[str, int]     # {"형사": 12, "가사": 8}
    top_legal_keywords: list[tuple[str, int]]   # [("이혼", 15), ("재산분할", 12)]
    category_distribution: dict[str, float]     # {"family": 0.4, "criminal": 0.3}
    avg_message_length: float
    date_range_days: int


@dataclass
class PersonaExtractionResult:
    """LLM 페르소나 추출 결과 (검증 전)"""
    specialty_areas: list[str]
    focus_topics: list[str]
    confidence: float
    raw_response: str                  # LLM 원본 응답 (디버깅용)


@dataclass
class ScoredIssueV2:
    """v2.0 스코어링 완료된 이슈"""
    id: str
    title: str
    raw_items: list  # RawTrendItem
    mention_score: float
    legal_score: float
    controversy_score: float
    spread_score: float
    fitness_score: float
    legal_stage: str
    legal_gate_passed: bool
    gate_rejection_reason: str | None = None    # 미통과 사유 (Red Team [보완 3])
    combined_score: float
    category: str = "all"


@dataclass
class ScriptContext:
    """프롬프트 체인 결과 컨텍스트"""
    issues: list[dict]                 # Chain 1 쟁점 목록
    laws_by_issue: dict[str, list]     # 쟁점별 RAG 법령
    cases_by_issue: dict[str, list]    # 쟁점별 RAG 판례
    cross_validated: bool              # 교차 검증 통과 여부
    chain_latency_ms: dict[str, int] = field(default_factory=dict)
```

---

## 4. API Specification

### 4.1 Persona API (NEW)

> **인증**: 모든 Persona API는 `Authorization: Bearer <token>` 헤더 필수.
> 서버에서 `current_user = Depends(get_current_user)` → `current_user.id`를 user_id로 사용.
> 클라이언트에서 user_id를 직접 전달하지 않음 (IDOR 방지, Red Team [심각 1]).

#### POST /api/content-marketing/persona/analyze

Track 1 — 대화 이력 기반 자동 페르소나 분석.

**Request:**
```
Authorization: Bearer <access_token>
Content-Type: application/json
```
```json
{
  "max_history": 100,
  "days_back": 30
}
```

**Response (200):**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "user_abc123",
  "specialty_areas": ["criminal", "family"],
  "focus_topics": ["이혼 재산분할", "양육권 분쟁", "위자료 산정"],
  "preferred_tone": "professional",
  "target_audience": "general_public",
  "channel_style": null,
  "source": "passive",
  "confidence": 0.82,
  "created_at": "2026-02-22T10:00:00Z",
  "updated_at": "2026-02-22T10:00:00Z"
}
```

**Error (422):** 대화 이력 30건 미만 → `{"detail": "대화 이력이 부족합니다. 온보딩을 진행해주세요.", "fallback": "track2"}`

**Error (409):** 분석 신뢰도 < 0.6 → `{"detail": "분석 신뢰도가 낮습니다. 온보딩을 진행해주세요.", "fallback": "track2"}`

#### POST /api/content-marketing/persona/onboarding

Track 2 — 온보딩 결과로 페르소나 생성.

**Request:**
```
Authorization: Bearer <access_token>
Content-Type: application/json
```
```json
{
  "specialty_areas": ["criminal", "family"],
  "target_audience": "general_public",
  "preferred_tone": "professional",
  "channel_style": "expert",
  "focus_topics": ["이혼 재산분할", "양육권"]
}
```

**Response (200):** `LawyerPersona` (위와 동일 형식, `source: "active"`, `confidence: 1.0`)

#### GET /api/content-marketing/persona/current

현재 페르소나 조회. (`Authorization: Bearer <token>` 필수, user_id는 서버에서 추출)

**Response (200):** `LawyerPersona | null`

**Response (401):** `{"detail": "인증이 필요합니다."}`

#### PUT /api/content-marketing/persona/update

페르소나 부분 수정. (`Authorization: Bearer <token>` 필수)

**Request:**
```json
{
  "preferred_tone": "casual",
  "focus_topics": ["이혼 재산분할", "부동산 사기"]
}
```

**Response (200):** 수정된 `LawyerPersona`

#### POST /api/content-marketing/persona/feedback

대본 생성 후 피드백 저장.

**Request:**
```json
{
  "persona_id": "550e8400-...",
  "script_id": "660e8400-...",
  "rating": 4,
  "feedback_type": null,
  "feedback_text": null
}
```

**Response (201):** `{"status": "saved"}`

### 4.2 Trend API (ENHANCED)

#### POST /api/content-marketing/trends

**Request (v2.0):**
```json
{
  "time_range": "24h",
  "category": "all",
  "limit": 10,
  "query": null,
  "persona_id": "550e8400-..."
}
```

**Response (200) — TrendIssue 변경분:**
```json
{
  "trends": [
    {
      "id": "...",
      "title": "XX 사건 손해배상 판결 논란",
      "summary": "...",
      "key_points": ["...", "...", "..."],
      "score": 94.0,
      "mention_score": 0.82,
      "legal_relevance_score": 0.91,
      "category": "civil",
      "score_detail": {
        "mention_score": 0.82,
        "legal_score": 0.91,
        "controversy_score": 0.78,
        "spread_score": 0.65,
        "fitness_score": 0.92,
        "legal_stage": "litigation",
        "legal_gate_passed": true,
        "combined_score": 94.0
      },
      "fitness_label": "채널 적합도 92%",
      "sources": ["tavily", "naver"],
      "source_articles": [],
      "related_laws": [],
      "related_cases": [],
      "collected_at": "2026-02-22T10:30:00Z"
    }
  ],
  "total_count": 10,
  "collected_at": "2026-02-22T10:30:00Z",
  "sources_used": ["tavily", "naver"],
  "cache_hit": false
}
```

### 4.3 Script API (ENHANCED)

#### POST /api/content-marketing/script/generate

**Request (v2.0):**
```json
{
  "topic": "XX 사건 손해배상 판결 논란",
  "trend_id": "550e8400-...",
  "persona_id": "660e8400-...",
  "duration": 10,
  "related_laws": ["law_1234"],
  "related_cases": ["76396"]
}
```

**Response (SSE stream) — 신규 이벤트 추가:**
```
event: stage_update
data: {"event": "stage_update", "stage": "chain_1", "status": "completed", "detail": "법적 쟁점 3개 추출 완료"}

event: stage_update
data: {"event": "stage_update", "stage": "chain_2", "status": "completed", "detail": "법령 5건, 판례 3건 검색 완료"}

event: stage_update
data: {"event": "stage_update", "stage": "chain_3", "status": "completed", "detail": "컨텍스트 구성 완료"}

event: section_start
data: {"event": "section_start", "section": "hooking"}

event: content
data: {"event": "content", "section": "hooking", "content": "안녕하세요, "}

... (기존 SSE 흐름과 동일)

event: done
data: {"event": "done"}
```

**하위 호환:** `persona_id`가 null이면 기존 `persona` 필드의 PersonaType 값을 PersonaTone으로 매핑하여 기본 페르소나 적용.

---

## 5. Class Design

### 5.1 MetadataExtractor (Red Team [심각 3] 반영 — PIIMasker 대체)

> **변경 근거**: 정규식 기반 PIIMasker는 한국어 법률 상담 특성상 "의뢰인 이름", "상대방 상호",
> "구체적 사건지(서초동 OO아파트)" 등을 완벽히 필터링할 수 없어 변호사법 위반 리스크가 있음.
> **해결**: 원문 텍스트를 LLM에 전달하지 않고, 메타데이터만 추출하여 전달하는 구조로 전환.

```python
# backend/app/tools/persona/metadata_extractor.py

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class ChatMetadata:
    """대화 이력에서 추출한 메타데이터 (PII 미포함)"""
    total_conversations: int
    agent_type_distribution: dict[str, int]     # {"형사": 12, "가사": 8, ...}
    top_legal_keywords: list[tuple[str, int]]   # [("이혼", 15), ("재산분할", 12), ...]
    category_distribution: dict[str, float]     # {"family": 0.4, "criminal": 0.3, ...}
    avg_message_length: float
    date_range_days: int


class MetadataExtractor:
    """대화 이력 → 메타데이터 추출 (원문 텍스트를 LLM에 전달하지 않음)"""

    # 법률 키워드 사전 (TF-IDF 가중치 계산용)
    LEGAL_KEYWORD_DICT: frozenset[str] = frozenset([
        "이혼", "양육권", "위자료", "재산분할", "상속", "유류분",
        "사기", "횡령", "배임", "폭행", "협박", "명예훼손",
        "해고", "부당해고", "임금", "퇴직금", "산재",
        "손해배상", "채무불이행", "계약해제", "부동산", "전세",
        "특허", "상표", "저작권", "회생", "파산", "조세",
    ])

    def extract(
        self,
        messages: list[dict[str, str]],   # [{"role": "user", "content": "...", "agent_type": "..."}]
        top_k: int = 20,
    ) -> ChatMetadata:
        """대화 이력에서 PII 없는 메타데이터만 추출"""
        # 1. 에이전트 유형 분포
        agent_types = Counter(m.get("agent_type", "unknown") for m in messages)

        # 2. 법률 키워드 빈도 (원문 분석은 로컬에서만, 결과만 전달)
        keyword_counter: Counter[str] = Counter()
        total_length = 0
        for msg in messages:
            content = msg.get("content", "")
            total_length += len(content)
            for keyword in self.LEGAL_KEYWORD_DICT:
                if keyword in content:
                    keyword_counter[keyword] += content.count(keyword)

        top_keywords = keyword_counter.most_common(top_k)

        # 3. 카테고리 분포 (키워드 기반 추정)
        category_map = self._estimate_categories(keyword_counter)

        return ChatMetadata(
            total_conversations=len(messages),
            agent_type_distribution=dict(agent_types),
            top_legal_keywords=top_keywords,
            category_distribution=category_map,
            avg_message_length=total_length / max(len(messages), 1),
            date_range_days=30,  # 요청 파라미터에서 전달
        )

    def _estimate_categories(
        self, keyword_counter: Counter[str],
    ) -> dict[str, float]:
        """키워드 빈도 → 법률 카테고리 분포 추정"""
        # 카테고리별 키워드 매핑 → 비율 계산
        category_keywords: dict[str, list[str]] = {
            "family": ["이혼", "양육권", "위자료", "재산분할", "상속", "유류분"],
            "criminal": ["사기", "횡령", "배임", "폭행", "협박", "명예훼손"],
            "labor": ["해고", "부당해고", "임금", "퇴직금", "산재"],
            "civil": ["손해배상", "채무불이행", "계약해제", "부동산", "전세"],
            "ip": ["특허", "상표", "저작권"],
            "corporate": ["회생", "파산", "조세"],
        }
        scores: dict[str, int] = {}
        for cat, keywords in category_keywords.items():
            scores[cat] = sum(keyword_counter.get(k, 0) for k in keywords)
        total = max(sum(scores.values()), 1)
        return {cat: round(score / total, 2) for cat, score in scores.items() if score > 0}
```

### 5.2 PersonaAnalyzer (Track 1 — 메타데이터 기반)

```python
# backend/app/tools/persona/analyzer.py

class PersonaAnalyzer:
    """Track 1: 대화 이력 메타데이터 기반 자동 페르소나 분석

    Red Team 반영:
    - [심각 1] user_id는 Auth Dependency에서 추출 (IDOR 방지)
    - [심각 3] 원문 텍스트 대신 메타데이터만 LLM에 전달 (PII 원천 차단)
    """

    MINIMUM_HISTORY = 30
    CONFIDENCE_THRESHOLD = 0.6

    def __init__(self) -> None:
        self._extractor = MetadataExtractor()

    async def analyze(
        self, user_id: str, max_history: int = 100, days_back: int = 30,
    ) -> LawyerPersona:
        """대화 이력 메타데이터에서 페르소나 추출 + 4중 검증"""
        # 1. 대화 이력 조회
        history = await self._fetch_chat_history(user_id, max_history, days_back)
        if len(history) < self.MINIMUM_HISTORY:
            raise InsufficientHistoryError(count=len(history), minimum=self.MINIMUM_HISTORY)

        # 2. 메타데이터 추출 (원문 텍스트 LLM 미전달, PII 원천 차단)
        metadata = self._extractor.extract(history)
        logger.info(
            "메타데이터 추출 완료: %d건, 키워드 %d개, 카테고리 %s",
            metadata.total_conversations,
            len(metadata.top_legal_keywords),
            list(metadata.category_distribution.keys()),
        )

        # 3. LLM 전문 분야 + 쟁점 추출 (메타데이터만 전달)
        extraction = await self._extract_persona_from_metadata(metadata)

        # 4. 4중 할루시네이션 검증
        validated = await self._validate_extraction(extraction, metadata)

        if validated.confidence < self.CONFIDENCE_THRESHOLD:
            raise LowConfidenceError(confidence=validated.confidence)

        return self._build_persona(user_id, validated)

    async def _extract_persona_from_metadata(
        self, metadata: ChatMetadata,
    ) -> PersonaExtractionResult:
        """메타데이터로 페르소나 추출 (원문 없이 분석)"""
        prompt = (
            "다음 변호사의 상담 활동 통계를 분석하여 전문 분야와 관심 쟁점을 추출하세요.\n\n"
            f"총 상담 건수: {metadata.total_conversations}건\n"
            f"에이전트 유형별 빈도: {metadata.agent_type_distribution}\n"
            f"법률 키워드 상위: {metadata.top_legal_keywords}\n"
            f"카테고리 분포: {metadata.category_distribution}\n\n"
            'JSON으로만 응답: {"specialty_areas": [...], "focus_topics": [...], "confidence": 0.0~1.0}'
        )
        # ... LLM 호출 + JSON 파싱
```

### 5.3 LegalGateScorer

```python
# backend/app/tools/trend/scorer.py (v2.0 확장)

class LegalGateScorer(TrendScorer):
    """Legal Gate + 5차원 스코어링 엔진"""

    LEGAL_THRESHOLD: float = 0.3  # 환경변수 TREND_LEGAL_THRESHOLD

    async def score(
        self,
        items: list[RawTrendItem],
        persona: LawyerPersona | None = None,
    ) -> list[ScoredIssueV2]:
        """Legal Gate 필터 + 5차원 가중합 스코어링"""
        groups = self._group_by_topic(items)
        scored: list[ScoredIssueV2] = []

        for group_title, group_items in groups.items():
            # Stage 1: 통합 LLM 분석 (1회 호출)
            analysis = await self._analyze_unified(group_title, group_items)

            # Stage 2: Legal Gate
            legal_gate_passed = analysis.legal_score >= self.LEGAL_THRESHOLD

            # Stage 3: 개별 스코어 계산
            mention = self._calculate_mention_score_v2(group_items, len(items))
            spread = self._calculate_spread_score(group_items)
            fitness = self._calculate_fitness_score(
                group_title, analysis.category, persona,
            ) if persona else 0.5

            # Stage 4: 가중합 (Legal Score = 승수)
            if legal_gate_passed:
                base = (
                    0.30 * mention
                    + 0.25 * analysis.controversy_score
                    + 0.15 * spread
                    + 0.30 * fitness
                )
                combined = analysis.legal_score * base * 100
            else:
                combined = analysis.legal_score * mention * 100  # 최소 점수

            scored.append(ScoredIssueV2(
                id=str(uuid.uuid4()),
                title=group_title,
                raw_items=group_items,
                mention_score=round(mention, 2),
                legal_score=round(analysis.legal_score, 2),
                controversy_score=round(analysis.controversy_score, 2),
                spread_score=round(spread, 2),
                fitness_score=round(fitness, 2),
                legal_stage=analysis.legal_stage,
                legal_gate_passed=legal_gate_passed,
                gate_rejection_reason=(
                    None if legal_gate_passed
                    else f"법적 쟁점화 지표 {analysis.legal_score:.2f}로 "
                         f"법률 콘텐츠 기준({self.LEGAL_THRESHOLD}) 미달"
                ),
                combined_score=round(combined, 1),
                category=analysis.category,
            ))

        scored.sort(key=lambda x: (-x.legal_gate_passed, -x.combined_score))
        return scored

    async def _analyze_unified(
        self, title: str, items: list[RawTrendItem],
    ) -> UnifiedAnalysisResult:
        """통합 LLM 프롬프트 (legal_score + legal_stage + controversy + category)"""
        text = f"{title} {' '.join(i.snippet for i in items[:3])}"
        prompt = (
            "다음 뉴스 이슈를 분석하세요. JSON으로만 응답:\n"
            '{"legal_score": 0.0~0.7, '
            '"legal_stage": "litigation|legislation|prosecution|dispute|mention", '
            '"controversy_ratio": 0.0~1.0, '
            '"category": "criminal|civil|labor|family|administrative|corporate|ip"}\n\n'
            f"이슈: {title}\n내용: {text[:500]}"
        )
        # ... JSON 파싱 + 3단계 폴백 (§3.4.1 PRD 참조)
```

### 5.4 PromptChainExecutor

```python
# backend/app/tools/script/chain_executor.py (NEW)

class PromptChainExecutor:
    """3단계 RAG 프롬프트 체인"""

    def __init__(self) -> None:
        self._rag = RAGPipeline()
        self._chain1_cache: dict[str, tuple[list[dict], float]] = {}  # TTL=1h

    async def execute(
        self, topic: str, trend_context: str = "", trend_key_points: list[str] | None = None,
    ) -> ScriptContext:
        """Chain 1 → (자가 검증) → Chain 2 → Chain 3 실행"""
        # Chain 1: 쟁점 분석 (캐시 확인)
        issues = await self._chain1_extract_issues(topic, trend_context)

        # Chain 1.5: 자가 검증 (Red Team [보완 1])
        # — Chain 1 출력을 트렌드 key_points와 유사도 비교
        if trend_key_points:
            issues = self._validate_chain1_against_key_points(issues, trend_key_points)

        # Chain 2: RAG 심화 검색 (병렬)
        laws, cases = await self._chain2_rag_search(issues)

        # Chain 3: 컨텍스트 구성 + 교차 검증
        context = self._chain3_build_context(issues, laws, cases)

        return context

    def _validate_chain1_against_key_points(
        self, issues: list[dict], key_points: list[str],
    ) -> list[dict]:
        """Chain 1 쟁점을 트렌드 key_points와 유사도 비교하여 할루시네이션 필터링

        Red Team [보완 1]: Chain 1 출력이 트렌드의 실제 key_points와 무관한
        쟁점을 생성하는 할루시네이션을 사전 차단.
        """
        validated = []
        key_points_text = " ".join(key_points)
        for issue in issues:
            # 키워드 겹침 비율로 간단 검증 (임베딩 유사도는 v2.1에서)
            issue_text = issue.get("쟁점", "")
            overlap = sum(1 for kp in key_points if kp in issue_text or issue_text in kp)
            if overlap > 0 or len(issues) <= 2:  # 최소 2개 쟁점 보장
                validated.append(issue)
        return validated if validated else issues[:2]  # 전부 탈락 시 상위 2개 유지

    async def _chain2_rag_search(
        self, issues: list[dict],
    ) -> tuple[dict[str, list], dict[str, list]]:
        """각 쟁점별 법령+판례 병렬 RAG 검색"""
        tasks = []
        for issue in issues:
            query = issue["쟁점"]
            tasks.append(self._rag.execute(query, PipelineConfig(
                n_results=10, doc_type="law", enable_rerank=True, rerank_top_k=5,
            )))
            tasks.append(self._rag.execute(query, PipelineConfig(
                n_results=10, doc_type="precedent", enable_rerank=True, rerank_top_k=5,
            )))

        results = await asyncio.gather(*tasks)
        # ... 쟁점별로 결과 매핑
```

---

## 6. Backend Structure Changes

```
backend/app/
├── modules/content_marketing/
│   ├── router/
│   │   └── __init__.py              # persona/ 엔드포인트 5개 추가
│   └── schema/
│       └── __init__.py              # PersonaTone, TargetAudience, LawyerPersona 등 추가
│
├── models/
│   ├── __init__.py                  # LawyerPersonaModel import 추가
│   └── lawyer_persona.py           # (NEW) ORM 모델 2개
│
├── services/service_function/
│   ├── content_marketing_service.py # persona 관련 함수 추가
│   └── persona_db_service.py       # (NEW) DB CRUD 서비스
│
├── tools/persona/                   # (NEW)
│   ├── __init__.py
│   ├── metadata_extractor.py       # MetadataExtractor (PIIMasker 대체, Red Team [심각 3])
│   ├── analyzer.py                 # PersonaAnalyzer (Track 1, 메타데이터 기반)
│   ├── onboarding.py              # OnboardingProcessor (Track 2)
│   └── models.py                   # 내부 dataclass
│
├── tools/trend/
│   ├── scorer.py                   # LegalGateScorer (TrendScorer 확장)
│   └── models.py                   # ScoredIssueV2 추가
│
└── tools/script/
    ├── generator.py                # 페르소나 맥락 주입, Chain 연동
    ├── chain_executor.py           # (NEW) PromptChainExecutor
    └── templates.py                # PersonaTone별 템플릿 확장
```

---

## 7. Frontend Structure Changes

```
frontend/src/features/content-marketing/
├── components/
│   ├── PersonaGate.tsx             # (NEW) 진입점 — 페르소나 유무 판별
│   ├── PersonaBanner.tsx           # (NEW) 상단 페르소나 배너
│   ├── PersonaOnboarding.tsx       # (NEW) 온보딩 위저드 (4단계)
│   ├── PersonaEditor.tsx           # (NEW) 페르소나 수정 모달
│   ├── PersonaConfirmation.tsx     # (NEW) Track 1 분석 결과 확인 UI
│   ├── FeedbackPanel.tsx           # (NEW) 대본 피드백 (별점 + 개선 요청)
│   ├── TrendDashboard.tsx          # (변경) FitnessBadge, Legal Gate 시각화
│   ├── TrendCard.tsx               # (변경) score_detail 5차원 표시
│   ├── ScoreBar.tsx                # (변경) 5차원 바 차트
│   ├── ScriptGenerator.tsx         # (변경) persona_id 주입, RAG 미리보기
│   ├── RAGPreview.tsx              # (NEW) RAG 컨텍스트 미리보기 (접기/펼치기)
│   ├── StageProgress.tsx           # (NEW) Chain 진행 상황 표시
│   └── ... (기존 컴포넌트 유지)
│
├── hooks/
│   ├── usePersona.ts               # (NEW) 페르소나 상태 관리 (2-Layer)
│   ├── useTrends.ts                # (변경) persona_id 파라미터 추가
│   └── useScript.ts                # (변경) persona_id + stage_update 처리
│
├── services/
│   └── index.ts                    # (변경) persona API 함수 5개 추가
│
└── types/
    └── index.ts                    # (변경) 신규 타입 추가 (§3.5 참조)
```

---

## 8. UI/UX Design

### 8.1 PersonaGate (진입점)

```
[변호사가 /content-marketing 진입]
        │
    ┌───┴────────────────────────────────────────────────────────┐
    │  ⏳ 페르소나 확인 중...                                       │
    │  (localStorage 즉시 확인 → 서버 동기화)                      │
    └────────────────────────────────────────────────────────────┘
        │
        ├── 페르소나 존재 → 메인 대시보드
        │
        └── 페르소나 없음 →
            ┌────────────────────────────────────────────────────┐
            │  👋 유튜브 콘텐츠 서비스에 오신 것을 환영합니다!       │
            │                                                     │
            │  맞춤형 콘텐츠를 제공하기 위해                        │
            │  선생님의 채널 프로필을 설정해 주세요.                 │
            │                                                     │
            │  [자동 분석으로 시작]  [직접 설정하기]                 │
            │   (Track 1)            (Track 2)                    │
            └────────────────────────────────────────────────────┘
```

### 8.2 Track 1 분석 결과 확인 (PersonaConfirmation)

```
┌────────────────────────────────────────────────────────────────┐
│  📊 선생님의 채널 프로필을 분석했습니다                            │
│                                                                 │
│  ┌─ 분석 결과 (신뢰도: 82%) ──────────────────────────────┐    │
│  │                                                          │    │
│  │  전문 분야: [형사법] [가사법]                              │    │
│  │  관심 쟁점: 이혼 재산분할, 양육권 분쟁, 위자료 산정         │    │
│  │  추천 톤:  전문가형                                       │    │
│  │  추천 타겟: 일반 대중                                     │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                 │
│  이 프로필이 맞습니까?                                           │
│                                                                 │
│  [맞습니다 ✓]  [일부 수정]  [직접 설정 →]                        │
└────────────────────────────────────────────────────────────────┘
```

### 8.3 트렌드 카드 (Legal Gate 반영)

```
── Legal Gate 통과 이슈 ──

┌──────────────────────────────────────┐
│ 🔥 1위  ⭐ 채널 적합도 92%            │
│                                       │
│ "XX 사건 손해배상 판결 논란"            │
│ 종합 점수: 94/100                     │
│                                       │
│ ┌─ 세부 지표 ─────────────────────┐  │
│ │ 언급량      ████████░░  0.82    │  │
│ │ 법적 쟁점화  █████████░  0.91   │  │
│ │ 논란 지수   ████████░░  0.78    │  │
│ │ 확산 속도   ██████░░░░  0.65    │  │
│ │ 채널 적합도  █████████░  0.92   │  │
│ └─────────────────────────────────┘  │
│                                       │
│  [상세 보기]  [이 주제로 대본 생성 →]   │
└──────────────────────────────────────┘

── Legal Gate 미통과 이슈 (회색 처리, 사유 표시) ──

┌──────────────────────────────────────┐
│ ░░ 법률 관련성 낮음                    │
│                                       │
│ "연예인 XX 열애설 논란"               │
│ 법적 쟁점화 점수: 0.12                │
│                                       │
│ ℹ️ 사유: 법적 쟁점화 지표 0.12로      │
│         법률 콘텐츠 기준(0.3) 미달     │
│                                       │
│ ⚠️ 법률 콘텐츠로 부적합               │
└──────────────────────────────────────┘
```

### 8.4 FeedbackPanel (대본 생성 후)

```
┌────────────────────────────────────────────────────────────┐
│  📝 생성된 대본이 선생님의 스타일에 맞나요?                    │
│                                                             │
│  ⭐⭐⭐⭐⭐  (별점 클릭)                                   │
│                                                             │
│  [선택] 개선이 필요한 부분:                                  │
│  □ 톤이 맞지 않아요                                         │
│  □ 전문 분야가 다릅니다                                     │
│  □ 타겟 시청자 수준이 안 맞아요                              │
│  □ 기타: [                          ]                       │
│                                                             │
│  [제출]  [건너뛰기]                                          │
└────────────────────────────────────────────────────────────┘
```

---

## 9. Migration Plan

### 9.1 Alembic 마이그레이션

```
alembic/versions/
└── NNN_add_lawyer_personas_and_feedback.py

-- Up (upgrade)
CREATE TABLE lawyer_personas (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL UNIQUE,
    specialty_areas JSONB NOT NULL,
    focus_topics JSONB DEFAULT '[]',
    preferred_tone VARCHAR(50) NOT NULL DEFAULT 'professional',
    target_audience VARCHAR(50) NOT NULL DEFAULT 'general_public',
    channel_style VARCHAR(50),
    source VARCHAR(10) NOT NULL,
    confidence FLOAT DEFAULT 1.0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_lawyer_personas_user_id ON lawyer_personas(user_id);

CREATE TABLE lawyer_persona_feedback (
    id VARCHAR(36) PRIMARY KEY,
    persona_id VARCHAR(36) NOT NULL,
    script_id VARCHAR(36),
    rating FLOAT NOT NULL,
    feedback_type VARCHAR(50),
    feedback_text VARCHAR(1000),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_persona_feedback_persona_id ON lawyer_persona_feedback(persona_id);

-- Down (downgrade)
DROP TABLE IF EXISTS lawyer_persona_feedback;
DROP TABLE IF EXISTS lawyer_personas;
```

### 9.2 하위 호환성

| 항목 | v1.0 동작 | v2.0 동작 | 호환 방법 |
|------|----------|----------|----------|
| `PersonaType` | `professional/casual` | `PersonaTone` (4가지) | 매핑: professional→professional, casual→casual |
| `ScriptRequest.persona` | 필수 | deprecated (유지) | `persona_id` 없으면 `persona` 필드로 기본 페르소나 생성 |
| `TrendIssue.score` | 2차원 점수 | 5차원 점수 | `score` 필드는 `combined_score`로 동일 값 유지 |
| `/api/content-marketing/trends` | `persona_id` 없음 | 선택적 | `persona_id` 없으면 `fitness_score=0.5` 기본값 |

---

## 10. Environment Variables (NEW)

```bash
# backend/.env (추가)

# Legal Gate
TREND_LEGAL_THRESHOLD=0.3       # Legal Gate 최소 임계값

# 스코어링 가중치 (Legal Gate 통과 후)
TREND_MENTION_WEIGHT=0.30
TREND_CONTROVERSY_WEIGHT=0.25
TREND_SPREAD_WEIGHT=0.15
TREND_FITNESS_WEIGHT=0.30

# 페르소나
PERSONA_MIN_HISTORY=30          # Track 1 최소 대화 이력
PERSONA_CONFIDENCE_THRESHOLD=0.6 # Track 1 최소 신뢰도
```

---

## 11. Implementation Priority

| Phase | Step | 작업 | 변경 파일 | 난이도 |
|-------|------|------|----------|--------|
| **1** | 1-1 | 스키마 확장 (Persona + Scoring) | `schema/__init__.py` | Low |
| **1** | 1-1a | DB 마이그레이션 (lawyer_personas + feedback) | `alembic/versions/`, `models/` | Medium |
| **1** | 1-2 | Track 2 온보딩 API + Frontend 위저드 | `router/`, `tools/persona/onboarding.py`, `PersonaOnboarding.tsx` | Medium |
| **1** | 1-2a | MetadataExtractor (PIIMasker 대체, Red Team [심각 3]) | `tools/persona/metadata_extractor.py` | Medium |
| **1** | 1-3 | Track 1 자동 분석 + 4중 검증 | `tools/persona/analyzer.py` | High |
| **1** | 1-4 | PersonaBanner + PersonaEditor | `PersonaBanner.tsx`, `PersonaEditor.tsx` | Medium |
| **1** | 1-5 | 피드백 루프 UI + API | `FeedbackPanel.tsx`, 라우터 | Medium |
| **1** | 1-6 | Auth Dependency 적용 (Red Team [심각 1]) | `router/__init__.py`, `deps.py` | Medium |
| **2** | 2-0 | BackgroundTrendWorker + 글로벌 캐시 (Red Team [심각 2]) | `tools/trend/worker.py`, `services/` | High |
| **2** | 2-1 | LegalGateScorer (Gate + 승수 + rejection_reason) | `tools/trend/scorer.py` | High |
| **2** | 2-2 | 통합 LLM 프롬프트 + 폴백 | `tools/trend/scorer.py` | Medium |
| **2** | 2-3 | 채널 적합도(F) 계산 | `tools/trend/scorer.py` | Medium |
| **2** | 2-4 | ScoreBar 5차원 + Legal Gate UI | `ScoreBar.tsx`, `TrendCard.tsx` | Medium |
| **3** | 3-1 | PromptChainExecutor (3단계) | `tools/script/chain_executor.py` | High |
| **3** | 3-2 | 페르소나 맥락 주입 대본 | `tools/script/generator.py`, `templates.py` | Medium |
| **3** | 3-3 | 인용 교차 검증 | `tools/script/generator.py` | Medium |
| **3** | 3-4 | RAG 미리보기 + StageProgress UI | `RAGPreview.tsx`, `StageProgress.tsx` | Low |
| **4** | 4-1 | E2E 흐름 테스트 | tests/ | Medium |
| **4** | 4-2 | 정적 검증 (ruff, mypy, npm run build) | - | Low |
| **4** | 4-3 | Gap Analysis (기획서 ↔ 구현) | - | Low |

---

## 12. Gemini CLI Red Team 검증 결과

| # | 심각도 | 항목 | 기존 설계 | Red Team 수정 | 반영 섹션 |
|---|--------|------|----------|--------------|----------|
| 1 | **[심각]** | IDOR 취약점 | user_id를 Body/Query로 전달 | Auth Dependency(Bearer Token)에서 추출 | §2.1.1, §3.1, §3.5, §4.1 |
| 2 | **[심각]** | LLM 호출 성능 병목 | API 호출마다 N건 LLM 호출 | 백그라운드 워커 + 글로벌 캐시, API에서 fitness_score만 실시간 | §2.1.1, §2.3 |
| 3 | **[심각]** | PII 마스킹 불완전 | 정규식 PIIMasker | 메타데이터 기반 MetadataExtractor (원문 미전달) | §2.1.1, §2.2, §3.6, §5.1, §5.2 |
| 4 | **[보완]** | Chain 1 자가 검증 | 쟁점 추출 후 바로 Chain 2 | key_points 유사도 비교 자가 검증 추가 | §5.4 |
| 5 | **[보완]** | localStorage 동기화 | 단순 캐시 | updated_at 기반 Version-based Sync | §2.1.1, §2.2 |
| 6 | **[보완]** | Legal Gate 미통과 사유 | 하단 배치만 | gate_rejection_reason 텍스트 제공 | §3.2, §3.5, §3.6, §5.3, §8.3 |
| 7 | **[대안]** | Persona-Aware RAG Reranking | 대본 생성에서만 페르소나 고려 | RAG Chain 2에서 specialty_areas 필터/가중치 → **v2.1 로드맵** | - |

## 13. v2.1 Roadmap (Red Team [대안] 포함)

| 항목 | 설명 | 출처 |
|------|------|------|
| **Persona-Aware RAG Reranking** | Chain 2에서 persona.specialty_areas를 LanceDB where절/rerank 가중치에 적용하여 전문 분야 밀착 검색 | Red Team [대안] |
| **Shorts/Reels 대본** | 60초 숏폼 대본 생성 + 수직형 자막 레이아웃 | PRD v2.1 |
| **LangGraph 시각화** | 멀티에이전트 처리 과정 실시간 시각화 (LangGraph Studio 연동) | PRD v2.1 |
| **Tiered Model 전략** | 고복잡도 이슈만 GPT-4o, 일반은 Solar Mini로 비용 최적화 | PRD v2.1 |

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1.0 | 2026-02-20 | v1.0 초기 설계 (트렌드 + 대본) | Claude |
| 2.0.0 | 2026-02-22 | v2.0 전면 재설계: 페르소나 시스템, Legal Gate, RAG 프롬프트 체인, 피드백 루프 | Lead Manager TF |
| 2.0.0-final | 2026-02-22 | Gemini CLI Red Team 피드백 반영: IDOR→Auth Dependency, 백그라운드 워커, MetadataExtractor, Chain 1 자가 검증, Version-based Sync, Legal Gate 사유, v2.1 로드맵 | Lead Manager TF + Gemini Red Team |
