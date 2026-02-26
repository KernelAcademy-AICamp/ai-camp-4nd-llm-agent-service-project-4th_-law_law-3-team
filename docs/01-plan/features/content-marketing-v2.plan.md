# Legal President - 유튜브 콘텐츠 자동 생성 기능 PRD v2.0

> **Summary**: 변호사 개인화 페르소나 기반 유튜브 콘텐츠 자동 생성 시스템.
> 기존 content_marketing 모듈을 **페르소나 초기화 → 지능형 트렌드 분석 → 맞춤형 대본 생성** 3단계 파이프라인으로 전면 재설계.
>
> **Project**: law-3 (Legal President / 리걸 프레지던트)
> **Author**: Lead Manager (Opus 4.6) + Agent A/B/C TF
> **Date**: 2026-02-22
> **Status**: Final (v2.0) — Gemini CLI Red Team 교차 검증 완료
> **Supersedes**: content-marketing.plan.md (v0.1, 2026-02-20)
> **Red Team Review**: Gemini CLI (Senior Manager) 교차 검증 2026-02-22 반영

---

## 1. Executive Summary

### 1.1 기존 시스템 (v1.0) 한계

| 영역 | 현재 상태 | 한계 |
|------|----------|------|
| 페르소나 | `PersonaType.PROFESSIONAL / CASUAL` 2가지 고정 | 변호사 개인의 전문 분야, 톤, 타겟 시청자를 반영하지 못함 |
| 트렌드 분석 | 언급량(0.4) × 법적해석가능성(0.6) 단순 가중합 | **논란 지수**, **확산 속도**, **법적 쟁점화 가능성**의 다차원 분석 부재 |
| 대본 생성 | 3단 구조(도입/본론/결론) 고정 | 변호사의 채널 톤, 전문 분야 맥락이 대본에 반영되지 않음 |
| 사용자 플로우 | 트렌드 탭 → 대본 탭 (직접 선택) | 처음 사용하는 변호사를 위한 온보딩 플로우 없음 |

### 1.2 v2.0 핵심 변경 사항

```
┌─────────────────────────────────────────────────────────────────┐
│                    v2.0 아키텍처 개요                              │
│                                                                  │
│  [1단계] 페르소나 초기화 (NEW)                                    │
│  ├── Track 1 (Passive): 챗봇 사용 로그 → LLM 자동 추출            │
│  └── Track 2 (Active): 3~5개 질문 인터랙티브 온보딩               │
│                   │                                              │
│                   ▼                                              │
│  [2단계] 지능형 트렌드 분석 (ENHANCED)                            │
│  ├── 멀티소스 수집 (기존 유지)                                    │
│  ├── 4차원 스코어링 (NEW): 논란×법적쟁점×확산속도×채널적합도        │
│  └── 변호사 전문 분야 기반 트렌드 필터링/랭킹 (NEW)               │
│                   │                                              │
│                   ▼                                              │
│  [3단계] 맞춤형 대본 생성 (ENHANCED)                              │
│  ├── 페르소나 맥락 주입 대본 (NEW)                                │
│  ├── RAG 심화 검색 + 프롬프트 체인 (ENHANCED)                     │
│  └── 채널 톤/스타일 일관성 보장 (NEW)                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 페르소나 초기화 시스템 (Agent C: UX/UI & Prompt Designer 기획)

### 2.1 개요

변호사가 "유튜브 콘텐츠 생성 서비스" 버튼을 클릭했을 때, **시스템이 먼저 변호사의 전문성과 채널 스타일을 파악**하는 로직.

### 2.2 Track 1: Passive (Data-Driven) — 기존 사용자 자동 프로파일링

#### 2.2.1 전제 조건

- 리걸 프레지던트 챗봇 서비스 사용 이력이 **최소 30회 이상** 존재 (Red Team [보완] 반영: 10회→30회 상향)
- 세션 데이터에 `session_id` 또는 `user_id`로 대화 이력 조회 가능

> **Gemini Red Team [보완] 반영**: 10회 미만 대화에서는 전문 분야 추출 신뢰도가 낮음. 30회 이상으로 상향하여 충분한 데이터 기반 분석 보장. 10~29회 사용자는 Track 2 (Interactive)로 안내.

#### 2.2.2 백엔드 로직

```
사용자가 /content-marketing 진입
    │
    ▼
PersonaAnalyzer.analyze_from_history(user_id)
    │
    ├── 0. PII 마스킹 전처리 (CRITICAL — Red Team [심각] 반영)
    │   ├── 대화 이력을 LLM에 전송하기 전에 반드시 PII 필터링
    │   ├── 마스킹 대상:
    │   │   ├── 의뢰인 이름 → [의뢰인A], [의뢰인B]
    │   │   ├── 사건번호 → [사건번호X]
    │   │   ├── 주소/전화번호/이메일 → [주소], [연락처]
    │   │   ├── 주민등록번호/사업자번호 → [식별번호]
    │   │   └── 금액(구체적) → [금액X원]
    │   ├── 구현: PIIMasker 유틸리티 클래스 (정규식 + 패턴 매칭)
    │   │   ├── 정규식: 주민번호(\d{6}-\d{7}), 전화번호, 이메일 등
    │   │   ├── NER 기반: 인명, 지명 (spaCy ko 또는 키워드 사전)
    │   │   └── 보수적 접근: 의심스러운 패턴은 마스킹 (과마스킹 > 누출)
    │   └── LLM에는 마스킹된 텍스트만 전달, 원본은 서버 메모리에만 유지
    │
    ├── 1. 대화 이력 조회 (최근 30일, 최대 100건)
    │   └── ChatState.messages에서 user 메시지 + agent_used 추출
    │
    ├── 2. 전문 분야 추출 (LLM 기반)
    │   ├── 입력: PII 마스킹된 대화 이력 요약 (주제, 질의 키워드, 사용 에이전트)
    │   ├── 프롬프트: "이 변호사의 주요 전문 분야를 3개 이하로 추출하세요.
    │   │            선택지: criminal, civil, labor, family, administrative,
    │   │            corporate, ip, real_estate"
    │   └── 출력: ["criminal", "family"] (법률 카테고리 Enum 매핑)
    │
    ├── 3. 관심 쟁점 패턴 추출
    │   ├── 입력: PII 마스킹된 대화에서 자주 등장한 법령/판례 키워드
    │   ├── LLM 프롬프트: "이 변호사가 반복적으로 다루는 법적 쟁점을
    │   │                5개 이하로 요약하세요."
    │   └── 출력: ["이혼 재산분할", "양육권 분쟁", "위자료 산정"]
    │
    ├── 4. 할루시네이션 방지 검증 (CRITICAL)
    │   ├── 추출된 전문 분야가 실제 대화 키워드에 1건 이상 매칭되는지 확인
    │   ├── RAG 검색으로 추출된 쟁점이 실재하는 법률 용어인지 검증
    │   └── 검증 실패 시 → Track 2 (Interactive)로 폴백
    │
    ▼
LawyerPersona (자동 생성)
    ├── specialty_areas: ["criminal", "family"]
    ├── focus_topics: ["이혼 재산분할", "양육권 분쟁", ...]
    ├── preferred_tone: "professional" (기본값, 변경 가능)
    ├── target_audience: "general_public" (기본값)
    ├── channel_style: null (Track 1에서는 미설정)
    └── confidence: 0.82 (분석 신뢰도)
```

#### 2.2.3 할루시네이션 방지 4중 검증

| 단계 | 검증 방법 | 실패 시 처리 |
|------|----------|------------|
| 1차 | LLM 출력의 전문 분야가 `TrendCategory` Enum에 존재하는지 | Enum 외 값 무시 |
| 2차 | 추출된 키워드가 원본 대화 이력에 1건 이상 존재하는지 (`in` 체크) | 해당 키워드 제거 |
| 3차 | 추출된 쟁점이 RAG 검색에서 관련 법령/판례 1건 이상 매칭되는지 | 매칭 0건이면 제거, 전체 0건이면 Track 2 폴백 |
| **4차 (Red Team 추가)** | **변호사 확인 UI**: "이 프로필이 맞습니까?" 확인 화면 제시 | 거부 시 Track 2로 전환, 부분 수정 허용 |

> **Red Team 피드백 반영**: LLM이 대화 이력에서 추출한 페르소나는 반드시 사용자(변호사)의 확인을 거쳐야 합니다. 자동 분석 결과를 무조건 신뢰하지 않고, "분석 결과 확인" 단계를 UI에 추가합니다.

### 2.3 Track 2: Active (Interactive) — 신규 사용자 온보딩

#### 2.3.1 트리거 조건

- 대화 이력 **30건 미만** **OR** (Red Team [보완] 반영: 10→30 동기화)
- Track 1 분석 신뢰도 < 0.6 **OR**
- 사용자가 "페르소나 재설정" 요청

#### 2.3.2 온보딩 질문 플로우 (3~5개 질문)

```
┌──────────────────────────────────────────────────────┐
│              페르소나 설정 온보딩 (Step 1/4)            │
│                                                       │
│  "선생님의 주요 전문 분야를 선택해 주세요."               │
│  (복수 선택 가능)                                      │
│                                                       │
│  ┌─────────┐  ┌─────────┐  ┌──────────┐             │
│  │ 형사법   │  │ 가사/가족│  │ 민사일반  │             │
│  └─────────┘  └─────────┘  └──────────┘             │
│  ┌─────────┐  ┌─────────┐  ┌──────────┐             │
│  │ 노동/고용│  │ 부동산   │  │ 기업/상사 │             │
│  └─────────┘  └─────────┘  └──────────┘             │
│  ┌─────────┐  ┌─────────┐                            │
│  │ 행정/공법│  │ 지적재산 │                            │
│  └─────────┘  └─────────┘                            │
│                                                       │
│                              [다음 →]                  │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│              페르소나 설정 온보딩 (Step 2/4)            │
│                                                       │
│  "영상의 주요 타겟 시청자는 누구인가요?"                  │
│                                                       │
│  ○ 일반 대중 (법률 비전문가)                            │
│  ○ 사업자/기업 담당자                                   │
│  ○ 법학 전공자/수험생                                   │
│  ○ 동종 변호사/법조인                                   │
│                                                       │
│                         [← 이전] [다음 →]              │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│              페르소나 설정 온보딩 (Step 3/4)            │
│                                                       │
│  "선호하는 영상 스타일은 어떤 것인가요?"                  │
│                                                       │
│  ○ 전문가형 — 신뢰감 있는 경어체, 정확한 법률 용어 사용   │
│  ○ 캐주얼형 — 편한 말투, 쉬운 비유, 시청자 친화적        │
│  ○ 스토리텔링형 — 사례 중심 서사, 몰입감 있는 구성        │
│  ○ 교육형 — 단계별 설명, 구조화된 정보 전달              │
│                                                       │
│                         [← 이전] [다음 →]              │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│              페르소나 설정 온보딩 (Step 4/4)            │
│                                                       │
│  "주로 다루는 구체적 쟁점이나 키워드가 있나요?"            │
│  (선택 사항 — 없으면 건너뛰기 가능)                      │
│                                                       │
│  ┌─────────────────────────────────────────────┐     │
│  │  예: 이혼 재산분할, 양육권, 부동산 사기         │     │
│  └─────────────────────────────────────────────┘     │
│                                                       │
│                         [← 이전] [완료 ✓]             │
└──────────────────────────────────────────────────────┘
```

#### 2.3.3 온보딩 결과 → LawyerPersona 생성

```python
class LawyerPersona(BaseModel):
    """변호사 페르소나 프로필"""
    id: str                                    # UUID
    specialty_areas: list[TrendCategory]        # 전문 분야 (1~3개)
    focus_topics: list[str]                     # 관심 쟁점 키워드 (0~5개)
    preferred_tone: PersonaTone                 # 영상 톤 (4가지)
    target_audience: TargetAudience             # 타겟 시청자
    channel_style: ChannelStyle | None          # 채널 스타일 (4가지)
    source: Literal["passive", "active"]        # 생성 경로
    confidence: float                           # 분석 신뢰도 (Track 1만)
    created_at: datetime
    updated_at: datetime
```

### 2.4 페르소나 저장 및 관리 (Red Team [심각] 반영 — DB 기반 저장)

| 항목 | 방식 |
|------|------|
| **Primary 저장소** | **PostgreSQL `lawyer_personas` 테이블** (서버 사이드, 영속) |
| **Cache 저장소** | localStorage (빠른 로딩용 캐시, 서버 응답 전 즉시 표시) |
| 수명 | DB에 영속 보관, localStorage는 캐시 역할 (불일치 시 DB 우선) |
| 수정 | 대시보드 상단 "페르소나 설정" 버튼으로 언제든 재설정 |
| 동기화 | 트렌드 필터링 + 대본 생성 양쪽에 동시 적용 |
| 멀티디바이스 | DB 기반이므로 로그인한 모든 디바이스에서 동일 페르소나 사용 |
| 초기화 | "페르소나 재설정" 버튼 → DB + localStorage 동시 삭제 |

#### 2.4.1 DB 스키마 (PostgreSQL)

```sql
CREATE TABLE lawyer_personas (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL UNIQUE,  -- 사용자 식별자
    specialty_areas JSONB NOT NULL,         -- ["criminal", "family"]
    focus_topics JSONB DEFAULT '[]',        -- ["이혼 재산분할", ...]
    preferred_tone VARCHAR(50) NOT NULL DEFAULT 'professional',
    target_audience VARCHAR(50) NOT NULL DEFAULT 'general_public',
    channel_style VARCHAR(50),
    source VARCHAR(10) NOT NULL,            -- "passive" | "active"
    confidence FLOAT DEFAULT 1.0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_lawyer_personas_user_id ON lawyer_personas(user_id);
```

#### 2.4.2 저장 전략 (2-Layer Cache)

```
[프론트엔드]                  [백엔드]
localStorage (캐시)  ←→   PostgreSQL (source of truth)

1. 페이지 진입 시:
   ├── localStorage에서 즉시 로드 (빠른 UI 표시)
   └── 동시에 GET /api/content-marketing/persona/current 호출
       └── 서버 응답이 localStorage와 다르면 → 서버 데이터로 갱신

2. 페르소나 생성/수정 시:
   ├── POST/PUT → 서버 DB 저장 (우선)
   └── 성공 응답 후 localStorage 캐시 갱신

3. 오프라인 폴백:
   └── 서버 접속 불가 시 localStorage 캐시 사용 (읽기 전용)
```

> **Gemini Red Team [심각] 반영**: localStorage만 사용하면 멀티디바이스 동기화 불가, 브라우저 초기화 시 데이터 유실. PostgreSQL을 primary storage로, localStorage를 빠른 로딩 캐시로 사용하는 2-Layer 구조로 변경.

---

## 3. 지능형 트렌드 스코어링 알고리즘 (Agent A: Data Pipeline Architect 기획)

### 3.1 기존 스코어링 (v1.0) vs 신규 스코어링 (v2.0)

| 차원 | v1.0 | v2.0 |
|------|------|------|
| 언급량 | `len(group) / (total * 0.3)` 단순 비율 | **멀티소스 가중 언급량** (소스별 가중치 차등) |
| 법적 관련성 | LLM 0~1 + 키워드 보정 | **법적 쟁점화 가능성** (소송/입법/판결 여부 세분화) |
| 논란 지수 | 없음 | **논란 지수** (찬반 비율, 감성 분석 기반) |
| 확산 속도 | 없음 | **확산 속도** (시간당 언급량 변화율) |
| 채널 적합도 | 없음 | **개인화 적합도** (변호사 전문 분야 매칭) |

### 3.2 스코어링 모델 (Red Team [심각] 반영 — Legal Gate 방식)

```
■ 핵심 변경: Legal Score를 단순 가중합 요소가 아닌 "게이트 필터"로 변경
  → 법적 관련성이 낮은 가십/연예 이슈가 높은 총점을 받는 문제 차단

■ 스코어링 공식 (2단계):

  [Stage 1: Legal Gate — 법적 관련성 필터]
  IF L < LEGAL_THRESHOLD (기본값: 0.3):
      → 해당 이슈는 결과에서 제외 (또는 "법률 관련성 낮음" 라벨 부여)
      → 프론트엔드에서 회색 처리 + 하단 배치

  [Stage 2: 가중합 스코어링 — Legal Gate 통과한 이슈만]
  TrendScore = L × (w1×M + w2×C + w3×S + w4×F)

  M = Mention Score      (멀티소스 가중 언급량)       [0~1]
  L = Legal Score        (법적 쟁점화 가능성)         [0~1]  ← 승수(multiplier)
  C = Controversy Score  (논란 지수)                 [0~1]
  S = Spread Score       (확산 속도)                 [0~1]
  F = Fitness Score      (채널 적합도, 페르소나 기반) [0~1]

  가중치 (Legal Gate 통과 후):
    w1 = 0.30  (언급량)
    w2 = 0.25  (논란 지수)
    w3 = 0.15  (확산 속도)
    w4 = 0.30  (채널 적합도)

  Legal Score는 전체 점수에 대한 승수 → L이 낮으면 총점 자체가 낮아짐

환경변수:
  TREND_LEGAL_THRESHOLD=0.3    # Legal Gate 최소 임계값
  TREND_MENTION_WEIGHT=0.30
  TREND_CONTROVERSY_WEIGHT=0.25
  TREND_SPREAD_WEIGHT=0.15
  TREND_FITNESS_WEIGHT=0.30

■ 효과:
  - 법적 관련성 없는 연예/가십 이슈: L < 0.3 → 게이트에서 차단
  - 법적 관련성 낮은 이슈: L = 0.4 → 총점에 0.4 승수 적용 → 자연스럽게 하위 랭킹
  - 법적 쟁점 이슈: L = 0.9 → 총점에 0.9 승수 → 상위 랭킹
```

> **Gemini Red Team [심각] 반영**: 기존 단순 가중합(`w1×M + w2×L + ...`)은 Legal Score가 낮아도 다른 차원(언급량, 논란)이 높으면 비법률 가십이 상위에 노출되는 치명적 결함. Legal Score를 승수(multiplier)이자 게이트(threshold)로 전환하여, 법률 콘텐츠 플랫폼의 본질적 품질을 보장.

### 3.3 각 차원 상세 로직

#### 3.3.1 M — 멀티소스 가중 언급량

```
M = Σ(source_weight[i] × normalized_count[i]) / Σ(source_weight[i])

소스별 가중치:
  Tavily (뉴스): 1.0       ← 뉴스 보도 = 가장 신뢰도 높은 지표
  Naver (뉴스/블로그): 0.8  ← 국내 트렌드 반영
  YouTube: 0.6              ← 영상 콘텐츠 관심도
  Perplexity: 0.5           ← 심층 분석 참고
  Google Trends: 0.4        ← 검색량 추이 보조

normalized_count = min(source_count / source_max_expected, 1.0)
  source_max_expected: Tavily=20, Naver=30, YouTube=10, ...
```

#### 3.3.2 L — 법적 쟁점화 가능성

```
L = LLM_legal_score + keyword_bonus + stage_bonus

1. LLM_legal_score (0~0.7):
   프롬프트: "이 이슈에 대해 법적 분석이 가능한지 판단하세요.
   판단 기준:
   - 현행법 위반 여부가 쟁점인가?
   - 법원 판결/소송이 진행 중이거나 예상되는가?
   - 법 개정/입법 논의가 있는가?
   0.0~0.7 숫자 하나만 응답."

2. keyword_bonus (0~0.15):
   법률 키워드 히트 수 × 0.03 (최대 0.15)
   키워드 사전: LEGAL_KEYWORDS (기존 28개) + 확장 키워드

3. stage_bonus (0~0.15):
   - 실제 소송/재판 진행 중: +0.15
   - 법 개정/입법 발의: +0.12
   - 검찰 수사/기소: +0.10
   - 민사 분쟁/중재: +0.08
   - 단순 법률 언급: +0.00
   (LLM이 stage를 함께 분류하도록 프롬프트에 포함)
```

#### 3.3.3 C — 논란 지수

```
C = controversy_ratio × sentiment_divergence

1. controversy_ratio (0~1):
   LLM 프롬프트: "이 이슈에 대해 찬반 양론이 얼마나 첨예한지
   0.0~1.0으로 평가하세요.
   - 0.0: 한쪽 의견만 존재 (합의 사안)
   - 1.0: 찬반이 극명하게 대립 (사회적 분열)"

2. sentiment_divergence (0~1):
   수집된 기사/게시글의 제목에서 감성 극성(긍/부) 분포 분석
   - 긍정적 제목 비율 vs 부정적 제목 비율의 편차
   - divergence = 1 - |pos_ratio - neg_ratio|
   - 균등(0.5:0.5)일수록 논란 지수 높음
```

#### 3.3.4 S — 확산 속도

```
S = min(recent_count / past_count, 3.0) / 3.0

time_range = "24h" 기준:
  recent_count = 최근 6시간 내 수집된 기사 수
  past_count = 나머지 18시간 기사 수 + 1 (0 방지)

S = min(recent_count / past_count, 3.0) / 3.0

해석:
  S ≈ 0: 확산 정체 (과거에 더 많이 언급)
  S ≈ 0.33: 균등 확산 (일정한 관심)
  S ≈ 1.0: 급속 확산 (최근 6시간 폭증)
```

#### 3.3.5 F — 채널 적합도 (페르소나 기반)

```
F = category_match + topic_overlap + audience_relevance

1. category_match (0~0.5):
   이슈 카테고리가 변호사 전문 분야에 포함되면 0.5, 아니면 0.0
   (IssueSummarizer._classify_category 결과 활용)

2. topic_overlap (0~0.3):
   이슈 핵심 쟁점 키워드와 변호사 focus_topics의 자카드 유사도
   = |교집합| / |합집합| × 0.3

3. audience_relevance (0~0.2):
   target_audience별 이슈 유형 매칭:
   - general_public: 사회적 관심 높은 이슈 선호 (+0.2 if C > 0.5)
   - business: 기업법/상사 이슈 선호
   - legal_student: 판례 변경/법리 이슈 선호
   - legal_professional: 전문적/기술적 이슈 선호
```

### 3.4 스코어링 파이프라인 최적화

```
기존 (v1.0): 이슈당 LLM 호출 2회 (법적관련성 + 카테고리 분류)
신규 (v2.0): 이슈당 LLM 호출 1회 (통합 프롬프트)

통합 프롬프트:
  "다음 뉴스 이슈를 분석하세요. JSON으로만 응답:
  {
    "legal_score": 0.0~0.7,
    "legal_stage": "litigation|legislation|prosecution|dispute|mention",
    "controversy_ratio": 0.0~1.0,
    "category": "criminal|civil|labor|family|administrative|corporate|ip"
  }
  이슈: {title}
  내용: {snippets}"

→ LLM 호출 50% 감소, 비용 절감
```

#### 3.4.1 통합 프롬프트 파싱 폴백 전략 (Red Team 보완)

```
통합 프롬프트 응답
    │
    ├── JSON 파싱 성공 → 4차원 점수 모두 사용
    │
    └── JSON 파싱 실패 (코드블록, 비정형 텍스트 등)
            │
            ├── 1차 복구: 코드블록(```) 제거 후 재파싱
            │
            ├── 2차 복구: 정규식으로 key-value 추출 시도
            │
            └── 3차 폴백: 개별 프롬프트 3회 호출 (v1.0 방식)
                ├── legal_score 개별 호출
                ├── controversy_ratio 개별 호출
                └── category 개별 호출

통합 프롬프트 성공률 목표: ≥ 95% (json_mode 사용 시)
```

### 3.5 확산 속도(S) 계산 보완 (Red Team 보완)

```
published_at 필드 처리 규칙:
  1. published_at이 null인 항목: 수집 시점(collected_at)을 대입
  2. published_at이 있는 항목이 전체의 50% 미만이면: S = 0.5 (기본값)
  3. published_at이 있는 항목만으로 S 계산 (null 제외)
  4. 모든 항목이 null이면: S = 0.5 (시간 정보 불충분)

→ 시간 정보가 불확실한 상태에서의 과적합 방지
```

### 3.6 채널 적합도(F) topic_overlap 보완 (Red Team 보완)

```
1차 매칭: 키워드 자카드 유사도
    │
    ├── score ≥ 0.3 → 그대로 사용
    │
    └── score < 0.3 → 2차 LLM 의미 매칭
            │
            LLM 프롬프트: "다음 두 주제가 법률적으로 관련 있는지
                          0.0~1.0으로 평가하세요.
                          변호사 관심 쟁점: {focus_topics}
                          뉴스 이슈: {issue_title}"
            │
            └── 결과 × 0.3 = topic_overlap

→ "이혼 재산분할"과 "재산 분할 소송"의 의미적 연관성 포착
→ LLM 호출은 자카드 유사도 < 0.3인 경우에만 발생 (비용 최소화)
```

---

## 4. RAG 연동 및 프롬프트 체인 설계 (Agent B: RAG System Engineer 기획)

### 4.1 RAG 연동 아키텍처 (기존 대비 변경)

```
v1.0 RAG 흐름:
  이슈 제목 → RAGPipeline.execute() → 법령/판례 목록 (단순 매칭)

v2.0 RAG 흐름 (프롬프트 체인):
  ┌─ Chain 1: 쟁점 분석 ─────────────────────────────────────┐
  │ 이슈 제목 + 스니펫                                        │
  │     → LLM: "이 이슈의 핵심 법적 쟁점 3개를 추출하세요.      │
  │            각 쟁점에 관련될 수 있는 법령명/조문번호를        │
  │            함께 제시하세요."                               │
  │     → 출력: [{"쟁점": "...", "예상_법령": "민법 제750조"}]  │
  └──────────────────────────────────────────────────────────┘
            │
            ▼
  ┌─ Chain 2: RAG 심화 검색 ─────────────────────────────────┐
  │ Chain 1 출력의 각 쟁점을 쿼리로 변환                       │
  │     → RAGPipeline.execute(query=쟁점, doc_type="law")     │
  │     → RAGPipeline.execute(query=쟁점, doc_type="precedent")│
  │     → PipelineConfig(n_results=10, enable_rerank=True,    │
  │                      rerank_top_k=5)                      │
  │ Chain 1이 제시한 법령명이 실제 RAG 결과에 포함되는지 교차검증 │
  └──────────────────────────────────────────────────────────┘
            │
            ▼
  ┌─ Chain 3: 컨텍스트 구성 ─────────────────────────────────┐
  │ RAG 결과를 쟁점별로 그룹화하여 대본 생성 컨텍스트 구성      │
  │     → {                                                  │
  │         "쟁점_1": {                                       │
  │           "description": "...",                           │
  │           "laws": [RAG 검색 법령],                         │
  │           "cases": [RAG 검색 판례],                        │
  │           "key_article": "민법 제750조 전문"               │
  │         },                                                │
  │         "쟁점_2": { ... }                                  │
  │       }                                                   │
  └──────────────────────────────────────────────────────────┘
```

### 4.2 할루시네이션 방지 대책

| 단계 | 대책 | 구현 위치 |
|------|------|----------|
| 프롬프트 | "제공된 법령/판례만 인용하세요. 제공되지 않은 것은 절대 만들어내지 마세요." 명시 | `templates.py` |
| RAG 검색 | `enable_rerank=True`로 관련성 높은 결과만 선별 | `generator.py` |
| 교차 검증 | Chain 1의 예상 법령이 Chain 2 RAG 결과에 포함되는지 확인. 미포함 시 해당 인용 제거 | `generator.py` (신규) |
| 후처리 | 대본에서 `[📋 인용: ...]` 형식의 인용문이 제공된 RAG 컨텍스트에 존재하는지 최종 검증 | `generator.py` (신규) |

### 4.3 페르소나 맥락 주입 대본 생성

```
기존 (v1.0):
  프롬프트 = 주제 + 페르소나(professional/casual) + RAG 컨텍스트

v2.0:
  프롬프트 = 주제 + 상세 페르소나 + RAG 컨텍스트 + 채널 스타일 가이드

상세 페르소나 주입 예시:
  "당신은 {specialty_areas} 전문 변호사입니다.
   영상 톤: {preferred_tone_description}
   타겟 시청자: {target_audience_description}
   채널 스타일: {channel_style_description}
   자주 다루는 주제: {focus_topics}

   위 프로필에 맞는 자연스러운 말투와 전문성 수준으로 대본을 작성하세요.
   타겟 시청자가 이해할 수 있는 수준으로 법률 용어를 조절하세요."
```

---

## 5. UI/UX 스토리보드 (Agent C: UX/UI & Prompt Designer 기획)

### 5.1 전체 사용자 플로우

```
[변호사가 /content-marketing 진입]
        │
        ▼
  ┌── 페르소나 존재? ──┐
  │                    │
  Yes                  No
  │                    │
  │               ┌────▼────────────────────┐
  │               │  Track 판별              │
  │               │  대화 이력 ≥ 10건?       │
  │               └────┬───────────┬────────┘
  │                   Yes          No
  │                    │            │
  │            ┌───────▼──┐  ┌─────▼──────────┐
  │            │Track 1    │  │Track 2          │
  │            │자동 분석   │  │온보딩 위저드     │
  │            │(로딩 2~3초)│  │(4단계 질문)      │
  │            └───────┬──┘  └─────┬──────────┘
  │                    │           │
  │                    ▼           ▼
  │               LawyerPersona 생성
  │                    │
  ▼                    ▼
  ┌──────────────────────────────────────────┐
  │         메인 대시보드                      │
  │                                           │
  │  ┌─ 상단 바 ──────────────────────────┐  │
  │  │ [페르소나: 형사법 전문, 전문가형]      │  │
  │  │ [편집 ✏️]                           │  │
  │  └────────────────────────────────────┘  │
  │                                           │
  │  [트렌드 분석] [대본 생성] [내 콘텐츠]     │
  │                                           │
  │  ┌─ 트렌드 카드 목록 ─────────────────┐  │
  │  │  (페르소나 기반 랭킹 적용)           │  │
  │  │  채널 적합도 배지 표시               │  │
  │  └────────────────────────────────────┘  │
  └──────────────────────────────────────────┘
```

### 5.2 페르소나 배너 (상단 고정)

```
┌───────────────────────────────────────────────────────────┐
│  👤 형사법·가사법 전문 | 전문가형 톤 | 일반 대중 타겟        │
│  관심 쟁점: 이혼 재산분할, 양육권 분쟁, 위자료               │
│                                    [페르소나 수정 ✏️]       │
└───────────────────────────────────────────────────────────┘
```

### 5.3 트렌드 카드 (v2.0 Enhanced)

```
┌──────────────────────────────────────┐
│ 🔥 1위  ⭐ 채널 적합도 92%            │
│                                       │
│ "XX 사건 손해배상 판결 논란"            │
│                                       │
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
│ 쟁점:                                 │
│ 1. 과실 상계 비율 적정성               │
│ 2. 기업 안전 관리 의무 범위             │
│ 3. 피해자 과실 입증 책임               │
│                                       │
│ 관련 법령: 민법 제750조, 산업안전보건법  │
│ 관련 판례: 대법원 2024다12345          │
│                                       │
│  [상세 보기]  [이 주제로 대본 생성 →]   │
└──────────────────────────────────────┘
```

### 5.4 대본 생성 화면 (v2.0 Enhanced)

```
┌─────────────────────────────────────────────────────────────┐
│  대본 생성기                                                  │
│                                                              │
│  주제: [XX 사건 손해배상 판결 논란                    ]        │
│                                                              │
│  ┌─ 페르소나 적용 (자동) ───────────────────────────────┐    │
│  │ 전문 분야: 형사법·가사법 | 톤: 전문가형                │    │
│  │ 타겟: 일반 대중 | 스타일: 전문가형                     │    │
│  │ [변경]                                               │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│  영상 길이: [5분] [10분 ✓] [15분]                            │
│                                                              │
│  [대본 생성하기 ▶]                                            │
│                                                              │
│  ┌─ RAG 컨텍스트 미리보기 (접기/펼치기) ──────────────────┐   │
│  │ 검색된 법령 3건: 민법 제750조, 산업안전보건법 제38조...  │   │
│  │ 검색된 판례 3건: 대법원 2024다12345, 2023나67890...     │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌─ 대본 미리보기 ──────────────────────────────────────┐   │
│  │                                                       │   │
│  │  ## 1. 도입 (Hooking)                                │   │
│  │  ▍ (스트리밍 커서) 최근 XX 사건...                     │   │
│  │                                                       │   │
│  │  ## 2. 본론 (Legal Analysis)                          │   │
│  │  (생성 대기 중...)                                     │   │
│  │                                                       │   │
│  │  ## 3. 결론 (Advice & CTA)                            │   │
│  │  (생성 대기 중...)                                     │   │
│  │                                                       │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌─ 메타데이터 ──────────────────────────────────────────┐  │
│  │ 영상 제목 제안: ...                                    │  │
│  │ 설명문: ...                                            │  │
│  │ SEO 태그: ...                                          │  │
│  │ 해시태그: ...                                          │  │
│  │ CTA 문구: ...                                          │  │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  [클립보드 복사 📋]  [TXT 다운로드 ⬇]  [MD 다운로드 ⬇]      │
└─────────────────────────────────────────────────────────────┘
```

### 4.4 RAG 체인 레이턴시 최적화 (Red Team [보완] 반영)

```
문제: Chain 1 → Chain 2 → Chain 3 직렬 실행 시 총 레이턴시 15~25초 예상

최적화 전략:

1. Chain 2 병렬화 (기존 설계에 포함):
   Chain 1에서 추출한 N개 쟁점을 asyncio.gather()로 동시 RAG 검색
   → 3개 쟁점 직렬 시 ~12초 → 병렬 시 ~4초

2. Chain 1 결과 캐시 (NEW):
   동일 이슈 ID에 대한 쟁점 분석 결과를 TTL=1h로 캐시
   → 동일 이슈 재조회 시 Chain 1 스킵 (2~3초 절감)

3. 스트리밍 파이프라인:
   Chain 3 컨텍스트 구성 완료 전에 대본 생성 시작
   → 첫 번째 쟁점 컨텍스트 완성 시 즉시 도입부(Hooking) 생성 시작
   → 나머지 쟁점은 본론(Analysis) 진행 중 도착

4. 레이턴시 목표:
   | 단계 | 목표 시간 |
   |------|----------|
   | Chain 1 (쟁점 분석) | ≤ 3초 |
   | Chain 2 (RAG 검색, 병렬) | ≤ 4초 |
   | Chain 3 (컨텍스트 구성) | ≤ 1초 |
   | 대본 생성 첫 토큰 | ≤ 5초 (기존 목표 유지) |
   | 총 체감 대기 시간 | ≤ 8초 (스트리밍 시작까지) |
```

### 4.5 페르소나 피드백 루프 (Red Team [보완] 반영)

```
문제: 생성된 대본이 변호사 기대와 다를 때 페르소나를 자동으로 보정하는 메커니즘 부재

피드백 루프 설계:

[대본 생성 완료 후]
        │
        ▼
  ┌── 사용자 평가 UI ──────────────────────────┐
  │  "이 대본이 선생님의 스타일에 맞나요?"         │
  │                                              │
  │  ⭐⭐⭐⭐⭐  (1~5점 별점)                    │
  │                                              │
  │  [선택] 개선 요청:                            │
  │  □ 톤이 너무 딱딱해요 / 너무 가벼워요          │
  │  □ 전문 분야가 안 맞아요                      │
  │  □ 타겟 시청자 수준이 안 맞아요                │
  │  □ 기타: [자유 입력]                          │
  └──────────────────────────────────────────────┘
        │
        ▼
  [백엔드 처리]
  ├── 별점 ≥ 4: 페르소나 유지, 피드백 로그 저장
  ├── 별점 ≤ 3 + "톤" 피드백: preferred_tone 자동 조정 제안
  ├── 별점 ≤ 3 + "분야" 피드백: specialty_areas 수정 온보딩 트리거
  └── 3회 연속 별점 ≤ 2: Track 2 재온보딩 자동 제안

  피드백 저장: lawyer_persona_feedback 테이블
  ├── persona_id, script_id, rating, feedback_type, feedback_text
  └── 추후 분석용 (페르소나 정확도 지표 계산)
```

---

## 6. 데이터 모델 (신규/변경)

### 6.1 Pydantic 스키마 (신규 추가)

```python
# ── 페르소나 관련 (NEW) ──

class PersonaTone(str, Enum):
    """영상 톤"""
    PROFESSIONAL = "professional"    # 전문가형
    CASUAL = "casual"                # 캐주얼형
    STORYTELLING = "storytelling"    # 스토리텔링형
    EDUCATIONAL = "educational"      # 교육형

class TargetAudience(str, Enum):
    """타겟 시청자"""
    GENERAL_PUBLIC = "general_public"     # 일반 대중
    BUSINESS = "business"                  # 사업자/기업
    LEGAL_STUDENT = "legal_student"        # 법학 전공자
    LEGAL_PROFESSIONAL = "legal_professional"  # 법조인

class ChannelStyle(str, Enum):
    """채널 스타일"""
    EXPERT = "expert"              # 전문가형
    CASUAL_FRIENDLY = "casual_friendly"  # 친근형
    STORYTELLING = "storytelling"   # 스토리텔링형
    LECTURE = "lecture"             # 강의형

class LawyerPersona(BaseModel):
    """변호사 페르소나"""
    id: str
    specialty_areas: list[TrendCategory]
    focus_topics: list[str] = Field(default_factory=list, max_length=5)
    preferred_tone: PersonaTone = PersonaTone.PROFESSIONAL
    target_audience: TargetAudience = TargetAudience.GENERAL_PUBLIC
    channel_style: ChannelStyle | None = None
    source: Literal["passive", "active"]
    confidence: float = Field(default=1.0, ge=0, le=1)
    created_at: datetime
    updated_at: datetime

class PersonaAnalysisRequest(BaseModel):
    """Track 1: 자동 분석 요청"""
    user_id: str
    max_history: int = Field(default=100, ge=10, le=500)
    days_back: int = Field(default=30, ge=7, le=90)

class PersonaOnboardingRequest(BaseModel):
    """Track 2: 온보딩 결과"""
    specialty_areas: list[TrendCategory] = Field(min_length=1, max_length=3)
    target_audience: TargetAudience
    preferred_tone: PersonaTone
    channel_style: ChannelStyle | None = None
    focus_topics: list[str] = Field(default_factory=list, max_length=5)


# ── 스코어링 관련 (ENHANCED) ──

class LegalStage(str, Enum):
    """법적 단계"""
    LITIGATION = "litigation"        # 소송/재판 진행
    LEGISLATION = "legislation"      # 법 개정/입법
    PROSECUTION = "prosecution"      # 검찰 수사/기소
    DISPUTE = "dispute"              # 민사 분쟁/중재
    MENTION = "mention"              # 단순 언급

class TrendScoreDetail(BaseModel):
    """트렌드 세부 점수 (v2.0)"""
    mention_score: float = Field(ge=0, le=1)
    legal_score: float = Field(ge=0, le=1)
    controversy_score: float = Field(ge=0, le=1)
    spread_score: float = Field(ge=0, le=1)
    fitness_score: float = Field(ge=0, le=1)
    legal_stage: LegalStage
    combined_score: float = Field(ge=0, le=100)
```

### 6.2 기존 스키마 변경 사항

| 스키마 | 변경 | 내용 |
|--------|------|------|
| `TrendIssue` | 필드 추가 | `score_detail: TrendScoreDetail` (세부 점수), `fitness_label: str` (채널 적합도 라벨) |
| `ScriptRequest` | 필드 변경 | `persona: PersonaType` → `persona_id: str \| None` (LawyerPersona 참조) |
| `PersonaType` | 확장 | `PersonaTone`으로 대체 (4가지 톤), 하위호환 유지 |

---

## 7. API 엔드포인트 (신규/변경)

### 7.1 신규 엔드포인트

```
# 페르소나 관련 (NEW)
POST /api/content-marketing/persona/analyze
  Body: PersonaAnalysisRequest
  Response: LawyerPersona
  설명: Track 1 — 대화 이력 기반 자동 페르소나 분석

POST /api/content-marketing/persona/onboarding
  Body: PersonaOnboardingRequest
  Response: LawyerPersona
  설명: Track 2 — 온보딩 결과로 페르소나 생성

GET /api/content-marketing/persona/current
  Response: LawyerPersona | null
  설명: 현재 세션의 페르소나 조회

PUT /api/content-marketing/persona/update
  Body: Partial<LawyerPersona> (수정할 필드만)
  Response: LawyerPersona
  설명: 페르소나 수정
```

### 7.2 기존 엔드포인트 변경

```
POST /api/content-marketing/trends (ENHANCED)
  Body: TrendRequest + persona_id (optional)
  Response: TrendResponse (score_detail 포함)
  변경: 페르소나 기반 fitness_score 추가, score_detail 필드 추가

POST /api/content-marketing/script/generate (ENHANCED)
  Body: ScriptRequest (persona_id 추가)
  Response: SSE stream (기존 유지)
  변경: 페르소나 맥락이 프롬프트에 주입됨
```

---

## 8. 백엔드 구조 변경

```
backend/app/
├── modules/content_marketing/
│   ├── router/
│   │   └── __init__.py            # persona/ 엔드포인트 추가
│   └── schema/
│       └── __init__.py            # LawyerPersona, TrendScoreDetail 등 추가
│
├── services/service_function/
│   └── content_marketing_service.py  # persona_analyzer, onboarding 함수 추가
│
├── tools/trend/
│   ├── collector.py               # (변경 없음)
│   ├── scorer.py                  # 4차원 스코어링으로 전면 재설계
│   ├── summarizer.py              # 통합 프롬프트로 LLM 호출 최적화
│   └── sources/                   # (변경 없음)
│
├── tools/script/
│   ├── generator.py               # 페르소나 맥락 주입, 프롬프트 체인
│   └── templates.py               # 페르소나별 프롬프트 템플릿 확장
│
├── tools/persona/                 # (NEW) 페르소나 분석 도구
│   ├── __init__.py
│   ├── analyzer.py                # PersonaAnalyzer (Track 1)
│   ├── onboarding.py              # OnboardingProcessor (Track 2)
│   ├── pii_masker.py              # (NEW — Red Team [심각]) PII 마스킹 유틸리티
│   └── models.py                  # 페르소나 내부 모델
│
├── models/
│   └── lawyer_persona.py          # (NEW — Red Team [심각]) SQLAlchemy ORM 모델
│
└── services/service_function/
    └── persona_db_service.py      # (NEW — Red Team [심각]) DB CRUD 서비스
```

---

## 9. 프론트엔드 구조 변경

```
frontend/src/features/content-marketing/
├── components/
│   ├── PersonaBanner.tsx          # (NEW) 상단 페르소나 배너
│   ├── PersonaOnboarding.tsx      # (NEW) 온보딩 위저드 (4단계)
│   ├── PersonaEditor.tsx          # (NEW) 페르소나 수정 모달
│   ├── TrendDashboard.tsx         # (변경) 채널 적합도 배지 추가
│   ├── TrendCard.tsx              # (변경) score_detail 세부 지표 표시
│   ├── TrendDetailView.tsx        # (변경) 프롬프트 체인 RAG 결과 표시
│   ├── ScriptGenerator.tsx        # (변경) 페르소나 맥락 표시
│   ├── ScriptPreview.tsx          # (변경 없음)
│   ├── ScriptEditor.tsx           # (변경 없음)
│   ├── MetadataPanel.tsx          # (변경 없음)
│   ├── PersonaSelector.tsx        # (변경) PersonaTone 4가지로 확장
│   ├── ExportButton.tsx           # (변경 없음)
│   ├── DisclaimerBanner.tsx       # (변경 없음)
│   └── ScoreBar.tsx               # (변경) 5차원 레이더 차트 옵션
│
├── hooks/
│   ├── useTrends.ts               # (변경) persona_id 파라미터 추가
│   ├── useScript.ts               # (변경) persona_id 파라미터 추가
│   └── usePersona.ts              # (NEW) 페르소나 상태 관리
│
├── services/
│   └── index.ts                   # (변경) persona API 함수 추가
│
└── types/
    └── index.ts                   # (변경) 신규 타입 추가
```

---

## 10. 구현 우선순위 및 단계

### Phase 1: 페르소나 시스템 (핵심)

| Step | 작업 | 난이도 | Red Team |
|------|------|--------|----------|
| 1-1 | 스키마 확장 (LawyerPersona, PersonaTone, TargetAudience 등) | Low | - |
| **1-1a** | **DB 마이그레이션: `lawyer_personas` + `lawyer_persona_feedback` 테이블** | **Medium** | **[심각]** |
| 1-2 | Track 2 온보딩 API + 프론트엔드 위저드 | Medium | - |
| **1-2a** | **PIIMasker 유틸리티 구현 (정규식 + 패턴 매칭)** | **Medium** | **[심각]** |
| 1-3 | Track 1 자동 분석 백엔드 로직 + 할루시네이션 검증 (PII 마스킹 포함) | High | [심각] |
| 1-4 | 페르소나 배너 + 수정 모달 (프론트엔드) | Medium | - |
| **1-5** | **페르소나 피드백 루프 (별점 UI + 백엔드 저장)** | **Medium** | **[보완]** |

### Phase 2: 트렌드 스코어링 v2.0

| Step | 작업 | 난이도 | Red Team |
|------|------|--------|----------|
| **2-1** | **TrendScorer Legal Gate + 승수 방식 스코어링 재설계** | **High** | **[심각]** |
| 2-2 | 통합 LLM 프롬프트 (호출 50% 감소) | Medium | - |
| 2-3 | 채널 적합도(F) 계산 로직 | Medium | - |
| 2-4 | 프론트엔드 세부 지표 UI (ScoreBar + Legal Gate 시각화) | Medium | [심각] |

### Phase 3: RAG 프롬프트 체인 + 대본 생성 강화

| Step | 작업 | 난이도 |
|------|------|--------|
| 3-1 | 프롬프트 체인 3단계 구현 (쟁점 분석 → RAG 심화 → 컨텍스트 구성) | High |
| 3-2 | 페르소나 맥락 주입 대본 생성 | Medium |
| 3-3 | 인용 교차 검증 후처리 | Medium |
| 3-4 | 프론트엔드 RAG 컨텍스트 미리보기 | Low |

### Phase 4: 통합 검증

| Step | 작업 | 난이도 |
|------|------|--------|
| 4-1 | E2E 흐름 테스트 (온보딩 → 트렌드 → 대본) | Medium |
| 4-2 | 정적 검증 (ruff, mypy, npm run build) | Low |
| 4-3 | Gap Analysis (기획서 ↔ 구현 매칭) | Low |

---

## 11. 리스크 및 대응

| 리스크 | 영향 | 대응 | Red Team 검증 |
|--------|------|------|--------------|
| **Track 1 PII 유출** | **Critical** | PIIMasker 전처리 필수, 마스킹 후에만 LLM 전달, 원본은 서버 메모리만 | ✅ [심각] 반영 |
| **페르소나 데이터 유실** | **High** | PostgreSQL primary + localStorage cache 2-Layer 저장 | ✅ [심각] 반영 |
| **비법률 가십 상위 노출** | **High** | Legal Gate (L ≥ 0.3 필터) + Legal Score 승수 방식 | ✅ [심각] 반영 |
| Track 1 페르소나 분석 할루시네이션 | High | 4중 검증 (Enum + 키워드 + RAG + 사용자 확인), 실패 시 Track 2 폴백 | ✅ 기존 반영 |
| Track 1 데이터 부족 | Medium | 최소 이력 30회로 상향, 10~29회는 Track 2 안내 | ✅ [보완] 반영 |
| 스코어링 LLM 비용 증가 | Medium | 통합 프롬프트로 호출 50% 감소, 캐시 TTL 유지, 계층형 모델 전략(v2.1) | ✅ [대안] 참고 |
| 페르소나 온보딩 이탈률 | Medium | 4단계 이하 간결한 질문, Skip 옵션 제공, 기본값 자동 설정 | - |
| RAG 프롬프트 체인 지연 | Medium | Chain 1 캐시, Chain 2 병렬, 스트리밍 파이프라인 (목표: 첫 토큰 ≤ 5초) | ✅ [보완] 반영 |
| 페르소나 정확도 저하 | Medium | 피드백 루프 (별점 + 개선 요청), 3회 연속 저평가 시 재온보딩 | ✅ [보완] 반영 |
| 기존 v1.0 하위호환 | Low | PersonaType → PersonaTone 매핑 유지, persona_id 없으면 기본 페르소나 적용 | - |

---

## 12. v2.1 로드맵 — Gemini Red Team [대안] 제안 반영

> 아래 항목은 v2.0 구현 완료 후 확장 가능한 개선안. v2.0 아키텍처에서 자연스럽게 확장 가능하도록 인터페이스를 미리 설계.

### 12.1 [대안 1] Shorts/Reels 지원

```
현재: 5분/10분/15분 롱폼 대본만 지원
확장: 30초/60초 숏폼 대본 생성 옵션 추가

변경 사항:
  - ScriptDuration enum에 SHORT_30, SHORT_60 추가
  - SECTION_RATIO 숏폼 전용: {"hooking": 0.3, "core_message": 0.5, "cta": 0.2}
  - 숏폼 프롬프트 템플릿: "핵심 한 줄 + 법률 포인트 1개 + CTA"
  - 메타데이터: YouTube Shorts 태그 자동 생성, 세로 영상 비율 안내

설계 원칙: 기존 ScriptGenerator의 generate_stream()에 duration 분기 추가로 구현 가능
```

### 12.2 [대안 2] LangGraph 중간 단계 시각화

```
현재: 대본 생성 중 로딩 스피너만 표시
확장: RAG 프롬프트 체인의 각 단계를 실시간 표시

UI 확장:
  ┌─ 생성 진행 상황 ──────────────────────────┐
  │  ✅ Step 1: 법적 쟁점 3개 추출 완료         │
  │  ✅ Step 2: RAG 검색 (법령 5건, 판례 3건)   │
  │  🔄 Step 3: 컨텍스트 조립 중...             │
  │  ⏳ Step 4: 대본 생성 대기                  │
  └──────────────────────────────────────────┘

구현: SSE 이벤트에 stage_update 타입 추가
  {"type": "stage_update", "stage": "chain_2", "status": "completed", "detail": "법령 5건 검색"}
```

### 12.3 [대안 3] 계층형 모델 전략 (비용 최적화)

```
현재: 모든 LLM 호출에 동일 모델 사용
확장: 작업 복잡도에 따라 모델 계층화

┌─────────────────┬────────────────┬───────────┐
│ 작업             │ 모델           │ 비용 절감  │
├─────────────────┼────────────────┼───────────┤
│ 스코어링 통합 분석│ GPT-4o-mini   │ -70%      │
│ 쟁점 추출 (Chain1)│ GPT-4o       │ 기준      │
│ 대본 생성        │ GPT-4o        │ 기준      │
│ 메타데이터 생성   │ GPT-4o-mini   │ -70%      │
│ 페르소나 분석     │ GPT-4o       │ 기준      │
│ 감성 분석        │ GPT-4o-mini   │ -70%      │
└─────────────────┴────────────────┴───────────┘

예상 효과: 전체 LLM 비용 ~40% 절감
설정: config.py에 MODEL_TIER 환경변수로 작업별 모델 지정
```

---

## 13. 성공 지표

| 지표 | 목표 | 측정 방법 |
|------|------|----------|
| 페르소나 설정 완료율 | Track 2 온보딩 완료율 ≥ 80% | 온보딩 시작/완료 이벤트 비율 |
| Track 1 분석 정확도 | 변호사 확인 시 전문 분야 일치율 ≥ 70% | 4차 검증 UI "맞습니다" 비율 |
| **PII 마스킹 정확도** | **민감 정보 유출 0건** | PII 패턴 로깅 + 수동 샘플링 |
| **Legal Gate 필터링** | **비법률 이슈 상위 10위 내 0건** | legal_score < 0.3 이슈가 Top 10에 없는지 |
| 트렌드 스코어링 정밀도 | 법률 이슈에 legal_score ≥ 0.7 비율 ≥ 80% | 법률 전문가 100건 샘플 평가 |
| 대본 생성 시간 | 10분 대본 기준 첫 토큰 ≤ 5초, 전체 ≤ 60초 | SSE 타임스탬프 측정 |
| **RAG 체인 레이턴시** | **첫 토큰까지 ≤ 8초** (체인 포함) | Chain 단계별 타이머 |
| 인용 정확성 | RAG 제공 법령/판례만 인용 (환각 0%) | 인용 교차 검증 통과율 |
| **피드백 평균 별점** | **≥ 3.5 / 5.0** | lawyer_persona_feedback 집계 |

---

## 14. Gemini CLI Red Team 교차 검증 요약

### 검증 결과 총괄

| 분류 | 건수 | 상태 |
|------|------|------|
| [심각] Critical Issues | 3건 | ✅ 전량 PRD 반영 완료 |
| [보완] Improvements | 3건 | ✅ 전량 PRD 반영 완료 |
| [확인] Confirmed Good | 3건 | 설계 유지 |
| [대안] Alternatives | 3건 | v2.1 로드맵으로 분리 |

### [심각] 반영 내역

| # | 이슈 | 원래 설계 | 변경 후 |
|---|------|----------|--------|
| 1 | PII 유출 위험 | 대화 이력 그대로 LLM 전달 | PIIMasker 전처리 필수 (§2.2.2 Step 0) |
| 2 | 페르소나 저장소 | localStorage 전용 | PostgreSQL primary + localStorage cache (§2.4) |
| 3 | 비법률 가십 상위 노출 | 단순 가중합 | Legal Gate (L ≥ 0.3) + Legal Score 승수 방식 (§3.2) |

### [보완] 반영 내역

| # | 이슈 | 변경 후 |
|---|------|--------|
| 1 | Track 1 최소 이력 | 10회 → 30회 상향 (§2.2.1, §2.3.1) |
| 2 | RAG 체인 레이턴시 | Chain 2 병렬 + Chain 1 캐시 + 스트리밍 파이프라인 (§4.4) |
| 3 | 페르소나 피드백 루프 | 별점 + 개선 요청 UI, 3회 연속 저평가 시 재온보딩 (§4.5) |

### [확인] (설계 유지)

| # | 항목 | 평가 |
|---|------|------|
| 1 | 확산 속도 + 논란 지수 설계 | 법률 콘텐츠에 적합한 차별화된 지표 |
| 2 | RAG 교차 검증 | Chain 1 예상 법령 ↔ Chain 2 실제 RAG 결과 비교 설계 우수 |
| 3 | 4중 할루시네이션 검증 | Enum → 키워드 → RAG → 사용자 확인 계층적 방어 |

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-02-20 | Initial draft (v1.0) | Claude |
| 2.0 | 2026-02-22 | 전면 재설계: 페르소나 초기화 시스템, 4차원 스코어링, RAG 프롬프트 체인, UI 스토리보드 | Lead Manager TF |
| 2.0-final | 2026-02-22 | Gemini CLI Red Team 교차 검증 반영: PII 마스킹, DB 저장, Legal Gate, 피드백 루프, 레이턴시 최적화, v2.1 로드맵 | Lead Manager TF + Gemini Senior Manager |
