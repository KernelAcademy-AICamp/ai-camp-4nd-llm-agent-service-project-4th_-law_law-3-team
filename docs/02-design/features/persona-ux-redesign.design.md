# 페르소나 선택 UX 개선 — 기술 설계서

> **Feature**: persona-ux-redesign
> **Status**: Design (Phase 2)
> **Base**: `docs/01-plan/features/persona-ux-redesign.plan.md` v1.0
> **Date**: 2026-02-28

---

## 1. 아키텍처 개요

### 1.1 컴포넌트 변경 맵

```
components/
├── PersonaGate.tsx              # [수정] useReducer 리팩토링 + GateScreen 확장
├── PersonaWelcome.tsx           # [신규] Welcome Screen — 가치 전달 + 트랙 선택
├── PersonaAnalysisProgress.tsx  # [신규] 3단계 분석 진행 화면
├── PersonaHybridReview.tsx      # [신규] AI 추천 + 인라인 수정 통합 (핵심)
├── PersonaOnboarding.tsx        # [수정] 4스텝 → 3스텝 통합 + 카드형 UI
├── PersonaBanner.tsx            # [수정] 기본 설정 안내 배너 추가
├── PersonaConfirmation.tsx      # [삭제] PersonaHybridReview로 대체
├── PersonaEditor.tsx            # [유지] 기존 수정 모달
└── PersonaSelector.tsx          # [유지]

hooks/
├── usePersona.ts                # [수정] analysisInsights 상태 + chatHistoryCount 추가
└── usePersonaGate.ts            # [신규] PersonaGate 상태머신 (useReducer)

services/
└── index.ts                     # [수정] analyzePersona timeout + fetchChatHistoryCount 추가

types/
└── index.ts                     # [수정] AnalysisInsights, GateAction 등 타입 추가
```

### 1.2 데이터 플로우

```
                        ┌──── Welcome Screen ────┐
                        │                        │
              ┌─────────┴──────┐    ┌────────────┴─────────┐
              │   Track 1      │    │     Track 2          │
              │ (AI 자동분석)  │    │ (직접 설정)          │
              ▼                │    ▼                      │
     AnalysisProgress          │    SmartOnboarding (3스텝)│
              │                │          │                │
              ▼                │          │                │
     HybridReview ◄──── draftState ──────►│                │
      (인라인 편집)            │          │                │
              │                │          │                │
              ▼                ▼          ▼                │
         POST /persona/analyze    POST /persona/onboarding │
              │                           │                │
              └─────────┬─────────────────┘                │
                        ▼                                  │
                   [Ready State]                           │
                   PersonaBanner + Dashboard               │
                        │                                  │
                        └── PUT /persona/update ◄──────────┘
```

---

## 2. 프론트엔드 설계

### 2.1 타입 정의 (types/index.ts 확장)

```typescript
// ── 신규 타입 ──

/** AI 분석 인사이트 (POST /persona/analyze 응답 확장) */
export interface AnalysisInsights {
  area_scores: Record<string, number>      // { criminal: 0.78, civil: 0.62, ... }
  summary: string                           // "최근 30일 대화에서 형사 관련 질문이 45%를 차지했습니다"
  total_conversations_analyzed: number      // 분석된 대화 수
  analysis_period_days: number              // 분석 기간 (일)
  evidence_snippets: EvidenceSnippet[]      // 근거 대화 발췌 (최대 3개)
}

/** 분석 근거 발췌 */
export interface EvidenceSnippet {
  text: string                              // "음주운전 처벌 기준이 어떻게 되나요?"
  category: TrendCategory                   // 해당 분야
  date: string                              // ISO 날짜
}

/** 분석 API 확장 응답 */
export interface PersonaAnalysisResponse {
  persona: LawyerPersona
  analysis_insights: AnalysisInsights
}

/** 대화 이력 건수 응답 */
export interface ChatHistoryCountResponse {
  count: number
  has_sufficient_history: boolean           // count >= 5
  oldest_date: string | null                // 가장 오래된 대화 날짜
}

// ── GateScreen 상태머신 ──

export type GateScreen =
  | 'loading'
  | 'welcome'               // [신규] Welcome Screen
  | 'track1_analyzing'      // 분석 진행 중
  | 'track1_review'         // [변경] track1_confirm → 인라인 리뷰
  | 'track2_onboarding'     // 직접 설정
  | 'ready'                 // 설정 완료

/** PersonaGate 상태 */
export interface PersonaGateState {
  screen: GateScreen
  persona: LawyerPersona | null
  analysisInsights: AnalysisInsights | null
  draftOnboarding: Partial<OnboardingState> | null   // Track 전환 시 보존
  chatHistoryCount: number | null
  error: string | null
  isAnalyzing: boolean
  isSaving: boolean
}

/** PersonaGate 액션 (useReducer) */
export type GateAction =
  | { type: 'LOAD_START' }
  | { type: 'LOAD_SUCCESS'; persona: LawyerPersona | null; chatHistoryCount: number }
  | { type: 'LOAD_ERROR'; error: string }
  | { type: 'START_ANALYSIS' }
  | { type: 'ANALYSIS_SUCCESS'; persona: LawyerPersona; insights: AnalysisInsights }
  | { type: 'ANALYSIS_FAIL'; error: string }
  | { type: 'SAVE_START' }
  | { type: 'SAVE_SUCCESS'; persona: LawyerPersona }
  | { type: 'SAVE_ERROR'; error: string }
  | { type: 'SET_SCREEN'; screen: GateScreen }
  | { type: 'SET_DRAFT'; draft: Partial<OnboardingState> }
  | { type: 'CLEAR_ERROR' }
  | { type: 'SKIP_SETUP' }

// ── OnboardingStep 변경 (4스텝 → 3스텝) ──

export type OnboardingStep = 1 | 2 | 3    // [변경] 4 제거

/** 전문분야별 추천 키워드 */
export const SPECIALTY_KEYWORDS: Record<string, string[]> = {
  criminal: ['음주운전', '사기', '폭행', '성범죄', '마약'],
  civil: ['손해배상', '부동산', '계약해지', '대여금', '명예훼손'],
  labor: ['부당해고', '산업재해', '임금체불', '직장내 괴롭힘'],
  family: ['이혼', '양육권', '재산분할', '상속', '가사조정'],
  administrative: ['행정소송', '인허가', '징계처분', '과태료'],
  corporate: ['M&A', '회사법', '금융/증권', '공정거래', '지배구조'],
  ip: ['특허침해', '상표등록', '저작권', '영업비밀', '디자인권'],
}

/** 톤 미리보기 텍스트 */
export const TONE_PREVIEW_TEXT: Record<PersonaTone, string> = {
  professional: '대법원 2024다12345 판결에 따르면, 해당 사안은 민법 제750조 불법행위에 해당하며...',
  casual: '쉽게 말하면 이런 상황이에요. 비유를 들어볼게요. 여러분이 가게에서...',
  storytelling: '어느 날, 한 직장인에게 갑작스러운 통보가 날아왔습니다. "내일부터 나오지 마세요"...',
  educational: 'Step 1. 먼저 이 개념부터 이해해야 합니다. 불법행위란 타인의 권리를 위법하게...',
}
```

### 2.2 상태머신: usePersonaGate 훅 (신규)

```typescript
// hooks/usePersonaGate.ts

import { useCallback, useEffect, useReducer } from 'react'
import type { GateAction, GateScreen, OnboardingState, PersonaGateState } from '../types'
import {
  analyzePersona,
  createPersonaFromOnboarding,
  fetchChatHistoryCount,
  getCurrentPersona,
  updatePersona,
} from '../services'

const DRAFT_STORAGE_KEY = 'persona_draft_onboarding'

const initialState: PersonaGateState = {
  screen: 'loading',
  persona: null,
  analysisInsights: null,
  draftOnboarding: null,
  chatHistoryCount: null,
  error: null,
  isAnalyzing: false,
  isSaving: false,
}

function gateReducer(state: PersonaGateState, action: GateAction): PersonaGateState {
  switch (action.type) {
    case 'LOAD_START':
      return { ...state, screen: 'loading', error: null }

    case 'LOAD_SUCCESS':
      return {
        ...state,
        screen: action.persona ? 'ready' : 'welcome',
        persona: action.persona,
        chatHistoryCount: action.chatHistoryCount,
      }

    case 'LOAD_ERROR':
      return { ...state, screen: 'welcome', error: action.error }

    case 'START_ANALYSIS':
      return { ...state, screen: 'track1_analyzing', isAnalyzing: true, error: null }

    case 'ANALYSIS_SUCCESS':
      return {
        ...state,
        screen: 'track1_review',
        persona: action.persona,
        analysisInsights: action.insights,
        isAnalyzing: false,
      }

    case 'ANALYSIS_FAIL':
      // 분석 실패 시 welcome으로 복귀하되 에러 표시
      return {
        ...state,
        screen: 'welcome',
        isAnalyzing: false,
        error: action.error,
      }

    case 'SAVE_START':
      return { ...state, isSaving: true, error: null }

    case 'SAVE_SUCCESS':
      return {
        ...state,
        screen: 'ready',
        persona: action.persona,
        isSaving: false,
        draftOnboarding: null,
        analysisInsights: null,
      }

    case 'SAVE_ERROR':
      return { ...state, isSaving: false, error: action.error }

    case 'SET_SCREEN':
      return { ...state, screen: action.screen, error: null }

    case 'SET_DRAFT':
      return { ...state, draftOnboarding: action.draft }

    case 'CLEAR_ERROR':
      return { ...state, error: null }

    case 'SKIP_SETUP':
      return { ...state, screen: 'ready' }

    default:
      return state
  }
}
```

**핵심 설계 결정**:
- `useState` 대신 `useReducer`로 전환 → 상태 전환 로직의 예측 가능성 확보
- `draftOnboarding`으로 Track 전환 시 입력값 보존
- `ANALYSIS_FAIL` 시 `welcome`으로 복귀 (기존: `track_select`로 복귀하여 빈 화면)
- SessionStorage로 draft 영속성 확보 (브라우저 새로고침 대응)
- **v1.1**: 상태 전이 매트릭스 기반 가드 적용 → §9.5 참조 (불법 전이 reject + console.warn)
- **v1.1**: `analysisRequestId`로 race condition 방지 (늦은 응답 무시)

### 2.3 PersonaWelcome 컴포넌트 (신규)

```typescript
// components/PersonaWelcome.tsx

interface PersonaWelcomeProps {
  chatHistoryCount: number | null
  error: string | null
  onStartAnalysis: () => void
  onStartOnboarding: () => void
  onSkip: () => void
}
```

**레이아웃 설계**:

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   ⚖️ 당신의 전문성에 맞춘 콘텐츠를 만들어 보세요            │
│                                                             │
│   AI가 분석하거나, 직접 설정할 수 있습니다.                 │
│   약 1분이면 맞춤 콘텐츠 생성을 시작할 수 있어요.           │
│                                                             │
│   ┌──────────────────────────────────────────────────┐      │
│   │  적용 전                    적용 후               │      │
│   │  "법률 콘텐츠 생성"        "형사법 전문가의       │      │
│   │                             음주운전 실전 가이드"  │      │
│   └──────────────────────────────────────────────────┘      │
│                                                             │
│   ┌─────────────────────────┐  ┌────────────────────┐      │
│   │  🔍 AI로 자동 분석하기  │  │  ✏️ 직접 설정하기  │      │
│   │  [추천]                 │  │  약 1분 소요        │      │
│   │  N건의 대화 분석 가능   │  │                    │      │
│   └─────────────────────────┘  └────────────────────┘      │
│                                                             │
│                  나중에 설정하기                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**디자인 규격**:
- 최대 너비: `max-w-xl` (576px)
- 색상: 네이비(`slate-800`) + 블루(`blue-600`) + 화이트
- AI 분석 버튼: Primary CTA (`bg-blue-600 text-white`)
- 직접 설정 버튼: Secondary CTA (`border border-gray-300 text-gray-700`)
- "나중에 설정하기": 텍스트 링크 (`text-gray-400 text-xs`)
- 대화 이력 배지: `chatHistoryCount >= 5` 일 때만 표시
- 에러 메시지: 분석 실패 후 돌아왔을 때 상단에 부드러운 알림

### 2.4 PersonaAnalysisProgress 컴포넌트 (신규)

```typescript
// components/PersonaAnalysisProgress.tsx

interface PersonaAnalysisProgressProps {
  onCancel: () => void
}
```

**3단계 프로그레스 설계**:

```
┌─────────────────────────────────────────────────┐
│                                                 │
│   대화 이력을 분석하고 있습니다                  │
│                                                 │
│   ✓ 대화 이력 수집 완료                         │
│   ● 전문분야 패턴 분석 중...                    │
│   ○ 페르소나 구성                               │
│                                                 │
│   ████████████░░░░░░░░  60%                     │
│                                                 │
│   예상 소요 시간: 약 10~15초                     │
│                                                 │
│              [취소]                              │
│                                                 │
└─────────────────────────────────────────────────┘
```

**구현 전략**:
- API가 단일 응답이므로 **시간 기반 시뮬레이션** 사용
- 3단계 전환 타이밍: 0s → 수집(3s) → 분석(8s) → 구성(12s)
- `useEffect` + `setTimeout` 체인으로 단계 전환
- API 응답 도착 시 즉시 마지막 단계로 점프
- 취소 버튼: `AbortController`로 API 요청 취소 + welcome으로 복귀

### 2.5 PersonaHybridReview 컴포넌트 (신규 — 핵심)

```typescript
// components/PersonaHybridReview.tsx

interface PersonaHybridReviewProps {
  persona: LawyerPersona
  insights: AnalysisInsights
  onConfirm: (modified: PersonaUpdateRequest) => void
  onSwitchToManual: () => void
  isSaving: boolean
}
```

**레이아웃 설계**:

```
┌─────────────────────────────────────────────────────────────┐
│  AI 분석 결과                               신뢰도: 82%    │
│                                             ██████████░░    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ── 전문분야 분석 ─────────────────────────────────────     │
│                                                             │
│  형사  ████████░░░░  78%                                    │
│  민사  ██████░░░░░░  62%                                    │
│  가사  ████░░░░░░░░  41%                                    │
│                                                             │
│  [형사 ✓] [민사 ✓] [가사] [노동] [행정] [기업] [지식재산]  │
│                                                             │
│  💡 "최근 30일 대화에서 형사 관련 질문이 45%를             │
│      차지했습니다" (28건 분석)                              │
│                                                             │
│  ── 타겟 독자 ─────────────────────────────────────────     │
│                                                             │
│  ● 일반 대중  ○ 기업/사업자  ○ 법학 학생  ○ 법률 전문가   │
│                                                             │
│  ── 콘텐츠 톤 ─────────────────────────────────────────     │
│                                                             │
│  ● 전문가  ○ 친근한  ○ 스토리텔링  ○ 교육형               │
│                                                             │
│  미리보기: "대법원 2024다12345 판결에 따르면,               │
│            해당 사안은 민법 제750조..."                      │
│                                                             │
│  ── 관심 주제 ─────────────────────────────────────────     │
│                                                             │
│  [음주운전 ×] [사기 ×] [+ 추가]                            │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  [이 설정으로 시작하기]              [처음부터 직접 설정]   │
└─────────────────────────────────────────────────────────────┘
```

**핵심 인터랙션**:
1. **인라인 편집**: 모달 없이 모든 항목을 바로 수정
2. **변경 추적**: 사용자가 AI 추천에서 변경한 항목에 `수정됨` 배지 표시
3. **근거 표시**: 각 섹션에 분석 근거 텍스트 표시 (접이식)
4. **톤 미리보기**: 톤 선택 시 `TONE_PREVIEW_TEXT`에서 예시 문장 표시
5. **신뢰도 게이지**: 선형 프로그레스 바 (70%+: green, 40-70%: yellow, -40%: red)

**상태 관리**:
```typescript
// 내부 상태: AI 추천값을 기반으로 로컬에서 수정
const [localEdits, setLocalEdits] = useState<PersonaUpdateRequest>({})
const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set())

// AI 원본 vs 사용자 수정 비교
const isModified = (field: string): boolean => {
  return field in localEdits
}

// 확인 시: AI 원본 + 사용자 수정 병합하여 전달
const handleConfirm = () => {
  onConfirm(localEdits)
}
```

### 2.6 PersonaOnboarding 리팩토링 (4스텝 → 3스텝)

```typescript
// components/PersonaOnboarding.tsx

interface PersonaOnboardingProps {
  onComplete: (state: OnboardingState) => Promise<void>
  onBack: () => void
  isLoading: boolean
  error?: string | null
  initialState?: Partial<OnboardingState> | null  // [신규] Track 전환 시 초기값
}
```

**스텝 통합 변경**:

| 스텝 | 기존 (4스텝) | 개선 (3스텝) |
|------|-------------|-------------|
| 1 | 전문분야 (텍스트 버튼) | 전문분야 (아이콘 카드 + 대표 키워드) |
| 2 | 타겟 독자 | 톤 + 타겟 통합 (톤 미리보기 포함) |
| 3 | 톤 + 채널 스타일 | 채널 스타일 + 관심 주제 통합 (추천 칩) |
| 4 | 관심 주제 | *(삭제, Step 3에 통합)* |

**Step 1 — 전문분야 카드**:
```
┌──────────┐  ┌──────────┐  ┌──────────┐
│  ⚖️ 형사 │  │  📋 민사 │  │  👷 노동 │
│          │  │          │  │          │
│ 음주운전 │  │ 손해배상 │  │ 부당해고 │
│ 사기/횡령│  │ 계약분쟁 │  │ 산업재해 │
│ 폭행/상해│  │ 부동산   │  │ 임금체불 │
└──────────┘  └──────────┘  └──────────┘
```
- 기존: 2열 텍스트 버튼 → 개선: 3열 카드 (아이콘 + 대표 키워드 3개)
- 선택 시: `ring-2 ring-blue-500 bg-blue-50` + 체크 아이콘

**Step 2 — 톤 + 타겟 통합**:
- 타겟 독자 선택 → 추천 톤 하이라이트 (연동 로직)
  - `general_public` → `casual` 추천
  - `legal_professional` → `professional` 추천
  - `legal_student` → `educational` 추천
  - `business` → `professional` 추천
- 톤 선택 시 실시간 미리보기 텍스트 표시

**Step 3 — 채널 + 관심 주제 통합**:
- 채널 스타일: 선택사항 표시 ("고급 설정" 레이블)
- 관심 주제: `SPECIALTY_KEYWORDS`에서 전문분야 기반 추천 칩 표시
- 추천 칩 클릭 → 즉시 추가, 자유 입력도 지원

### 2.7 PersonaBanner 수정

**추가 기능**: 기본 페르소나 사용 시 안내 배너

```typescript
// 기본 설정 감지 조건
const isDefaultPersona = persona.source === 'active' &&
  persona.specialty_areas.length === 1 &&
  persona.specialty_areas[0] === 'all'
```

기본 설정 사용 중일 때:
```
┌─────────────────────────────────────────────────────────────┐
│  ℹ️ 기본 설정 사용 중 — 맞춤 설정하면 더 좋은 콘텐츠를     │
│     받을 수 있어요  [지금 설정하기]                [닫기 ×]  │
└─────────────────────────────────────────────────────────────┘
```
- `localStorage`에 `persona_default_banner_dismissed` 저장 → 닫으면 재표시 안 함

### 2.8 PersonaGate 리팩토링

**핵심 변경**:
1. `useState<GateScreen>` → `useReducer(gateReducer)` 전환
2. `loadState` 중복 제거 (reducer 내부에서 통합 관리)
3. `showEditor` 상태를 reducer에 통합하지 않고 별도 유지 (모달은 독립적)
4. `GateScreen` 분기를 `switch` 문으로 통일

```typescript
// PersonaGate.tsx 핵심 구조
export function PersonaGate({ children }: PersonaGateProps) {
  const [state, dispatch] = useReducer(gateReducer, initialState)
  const [showEditor, setShowEditor] = useState(false)

  // 초기 로드: 페르소나 + 대화 이력 건수 병렬 조회
  useEffect(() => {
    dispatch({ type: 'LOAD_START' })
    Promise.all([
      getCurrentPersona(),
      fetchChatHistoryCount().catch(() => ({ count: 0, has_sufficient_history: false })),
    ]).then(([persona, history]) => {
      dispatch({
        type: 'LOAD_SUCCESS',
        persona,
        chatHistoryCount: history.count,
      })
    }).catch((err) => {
      dispatch({ type: 'LOAD_ERROR', error: err.message })
    })
  }, [])

  // 화면별 렌더링
  switch (state.screen) {
    case 'loading':
      return <LoadingSpinner />

    case 'welcome':
      return (
        <PersonaWelcome
          chatHistoryCount={state.chatHistoryCount}
          error={state.error}
          onStartAnalysis={handleTrack1}
          onStartOnboarding={handleTrack2}
          onSkip={handleSkip}
        />
      )

    case 'track1_analyzing':
      return <PersonaAnalysisProgress onCancel={handleCancelAnalysis} />

    case 'track1_review':
      return (
        <PersonaHybridReview
          persona={state.persona!}
          insights={state.analysisInsights!}
          onConfirm={handleHybridConfirm}
          onSwitchToManual={handleSwitchToManual}
          isSaving={state.isSaving}
        />
      )

    case 'track2_onboarding':
      return (
        <PersonaOnboarding
          onComplete={handleOnboardingComplete}
          onBack={handleOnboardingBack}
          isLoading={state.isSaving}
          error={state.error}
          initialState={state.draftOnboarding}  // Track 전환 시 초기값 전달
        />
      )

    case 'ready':
      return state.persona ? (
        <>
          {children({ persona: state.persona, ... })}
          {showEditor && <PersonaEditor ... />}
        </>
      ) : null

    default:
      return null
  }
}
```

---

## 3. 백엔드 설계

### 3.1 스키마 확장 (schema/__init__.py)

```python
# ── 신규 스키마 ──

class EvidenceSnippet(BaseModel):
    """분석 근거 대화 발췌"""
    text: str
    category: TrendCategory
    date: datetime

class AnalysisInsights(BaseModel):
    """AI 분석 인사이트 (persona/analyze 응답 확장)"""
    area_scores: dict[str, float]  # { "criminal": 0.78, "civil": 0.62 }
    summary: str
    total_conversations_analyzed: int
    analysis_period_days: int
    evidence_snippets: list[EvidenceSnippet] = Field(
        default_factory=list, max_length=3
    )

class PersonaAnalysisResponse(BaseModel):
    """Track 1 분석 응답 (기존 LawyerPersona + insights)"""
    persona: LawyerPersona
    analysis_insights: AnalysisInsights

class ChatHistoryCountResponse(BaseModel):
    """대화 이력 건수 응답"""
    count: int
    has_sufficient_history: bool = False  # count >= 5
    oldest_date: datetime | None = None
```

### 3.2 API 엔드포인트 변경

| 엔드포인트 | 변경 유형 | 설명 |
|-----------|----------|------|
| `POST /persona/analyze` | **응답 변경** | `LawyerPersona` → `PersonaAnalysisResponse` |
| `GET /persona/chat-history-count` | **신규** | 대화 이력 건수 반환 |
| `POST /persona/onboarding` | 변경 없음 | |
| `GET /persona/current` | 변경 없음 | |
| `PUT /persona/update` | 변경 없음 | |

**`POST /persona/analyze` 변경 상세**:

```python
# router/__init__.py

@router.post("/persona/analyze", response_model=PersonaAnalysisResponse)
async def analyze_persona_endpoint(
    request: PersonaAnalysisRequest,
    db: AsyncSession = Depends(get_db),
) -> PersonaAnalysisResponse:
    """Track 1: 대화 이력 기반 자동 페르소나 분석"""
    user_id = TEMP_USER_ID  # TODO: Auth
    persona, insights = await analyze_persona(db, user_id, request)
    return PersonaAnalysisResponse(persona=persona, analysis_insights=insights)
```

**`GET /persona/chat-history-count` 신규**:

```python
@router.get("/persona/chat-history-count", response_model=ChatHistoryCountResponse)
async def chat_history_count_endpoint(
    db: AsyncSession = Depends(get_db),
) -> ChatHistoryCountResponse:
    """분석 가능한 대화 이력 건수 반환"""
    user_id = TEMP_USER_ID  # TODO: Auth
    count = await get_chat_history_count(db, user_id)
    return ChatHistoryCountResponse(
        count=count,
        has_sufficient_history=count >= 5,
        oldest_date=None,  # TODO: 실제 조회
    )
```

### 3.3 서비스 함수 변경

**`analyze_persona` — 반환 타입 변경 + 버그 수정**:

```python
# content_marketing_service.py

async def analyze_persona(
    db: AsyncSession,
    user_id: str,
    request: PersonaAnalysisRequest,
) -> tuple[LawyerPersona, AnalysisInsights]:
    """Track 1: 대화 이력 기반 자동 분석

    Returns:
        (LawyerPersona, AnalysisInsights) 튜플
    """
    from app.tools.persona.analyzer import (
        InsufficientHistoryError,
        LowConfidenceError,
        PersonaAnalyzer,
    )

    analyzer = PersonaAnalyzer()

    # [버그 수정] 실제 대화 이력 조회
    messages = await _fetch_chat_history(db, user_id, request.max_history, request.days_back)

    try:
        result = await analyzer.analyze(
            user_id=user_id,
            messages=messages,
            max_history=request.max_history,
            days_back=request.days_back,
        )
    except InsufficientHistoryError:
        raise HTTPException(status_code=422, detail="대화 이력이 부족합니다.")
    except LowConfidenceError:
        raise HTTPException(status_code=409, detail="분석 신뢰도가 낮습니다.")

    persona = await persona_db_service.create_persona(db, result.persona)
    insights = result.insights  # PersonaAnalyzer가 insights도 함께 반환

    return persona, insights


async def _fetch_chat_history(
    db: AsyncSession,
    user_id: str,
    max_history: int,
    days_back: int,
) -> list[dict[str, str]]:
    """사용자 대화 이력 조회 (chat_sessions 테이블)

    NOTE: 인증 시스템 구현 전까지 빈 리스트 반환 → InsufficientHistoryError 발생
    v1.1: PII 정제 로직 선적용 (Red Team 지적 반영)
    """
    # TODO: chat_sessions 테이블 또는 LangSmith에서 대화 이력 조회
    messages: list[dict[str, str]] = []
    # PII 정제: 이름, 전화번호, 사건번호 등 마스킹 (구현 시 적용)
    # messages = [_sanitize_pii(m) for m in raw_messages]
    return messages


async def get_chat_history_count(
    db: AsyncSession,
    user_id: str,
) -> int:
    """사용자 대화 이력 건수 조회"""
    # TODO: 실제 구현 시 chat_sessions 테이블에서 COUNT
    return 0
```

### 3.4 프론트엔드 서비스 함수 변경

```typescript
// services/index.ts

/** Track 1: 대화 이력 기반 자동 페르소나 분석 (응답 확장) */
export async function analyzePersona(
  request: PersonaAnalysisRequest,
): Promise<PersonaAnalysisResponse> {
  const { data } = await api.post<PersonaAnalysisResponse>(
    `${BASE}/persona/analyze`,
    request,
    { timeout: 30000 },  // [버그 수정] 30초 타임아웃 추가
  )
  return data
}

/** [신규] 대화 이력 건수 조회 */
export async function fetchChatHistoryCount(): Promise<ChatHistoryCountResponse> {
  const { data } = await api.get<ChatHistoryCountResponse>(
    `${BASE}/persona/chat-history-count`,
  )
  return data
}
```

---

## 4. Track 전환 상태 보존 설계

### 4.1 전환 시나리오별 동작

| 시나리오 | draftOnboarding | analysisInsights | 동작 |
|---------|----------------|-----------------|------|
| Track 1 → Track 2 | AI 분석 결과로 초기화 | 보존 | 온보딩 초기값으로 AI 결과 전달 |
| Track 2 → Track 1 | 현재 입력값 저장 | - | SessionStorage에 임시 저장 |
| Track 1 실패 → Track 2 | null (빈 상태) | null | 에러 메시지와 함께 안내 |
| Track 2 → Welcome | 현재 입력값 저장 | - | 뒤로가기 시 복원 가능 |

### 4.2 SessionStorage 키 설계

```typescript
const STORAGE_KEYS = {
  DRAFT_ONBOARDING: 'persona_draft_onboarding',     // OnboardingState 부분
  DEFAULT_BANNER_DISMISSED: 'persona_default_banner_dismissed',  // boolean
} as const
```

**저장 시점**:
- Track 2 → Track 1 전환 시: 현재 OnboardingState를 SessionStorage에 저장
- Track 2 → Welcome 전환 시: 현재 OnboardingState를 SessionStorage에 저장

**복원 시점**:
- Track 2 진입 시: SessionStorage에서 복원 → `initialState` prop으로 전달
- 페이지 새로고침 시: useReducer 초기화 함수에서 복원
- **v1.1**: 복원 시 JSON 스키마 검증 수행 → 실패 시 SessionStorage 클리어 + 초기 상태 사용

**삭제 시점**:
- 설정 완료 (ready 전환) 시: SessionStorage 클리어

---

## 5. 에러 처리 설계

### 5.1 API 에러 매핑

| HTTP Status | 에러 | 사용자 메시지 | 동작 |
|-------------|------|-------------|------|
| 422 | InsufficientHistoryError | "대화 이력이 부족합니다. 직접 설정으로 안내할까요?" | Track 2 전환 안내 버튼 표시 |
| 409 | LowConfidenceError | "분석 신뢰도가 낮습니다. 직접 설정으로 안내할까요?" | Track 2 전환 안내 버튼 표시 |
| 500 | Internal Server Error | "일시적인 오류가 발생했습니다. 다시 시도해주세요." | 재시도 버튼 표시 |
| timeout | Request Timeout | "분석에 시간이 오래 걸리고 있습니다. 직접 설정하시겠습니까?" | Track 2 전환 안내 |

### 5.2 표준 에러 응답 바디 (v1.1 추가)

> Consultant 제안 반영: 클라이언트 일관 처리를 위한 표준 에러 바디

```python
# 백엔드 에러 응답 형식
class PersonaErrorResponse(BaseModel):
    error_code: str       # "INSUFFICIENT_HISTORY", "LOW_CONFIDENCE", "INTERNAL_ERROR"
    detail: str           # 한국어 사용자 메시지
    retryable: bool       # 재시도 가능 여부
    # trace_id: str       # 향후 Observability 도입 시 추가
```

```typescript
// 프론트엔드 에러 타입
interface PersonaApiError {
  error_code: string
  detail: string
  retryable: boolean
}
```

### 5.3 에러 UI 패턴

```typescript
// 에러 메시지 + 액션 버튼 컴포넌트
interface ErrorBannerProps {
  message: string
  action?: {
    label: string
    onClick: () => void
  }
  onDismiss: () => void
}
```

**위치**: 각 화면 상단에 인라인 배너 형태 (모달 X)

---

## 6. 접근성 설계

### 6.1 키보드 네비게이션

| 컴포넌트 | 키보드 동작 |
|---------|-----------|
| Welcome Screen | `Tab`으로 AI분석/직접설정/건너뛰기 순서 이동, `Enter` 선택 |
| 전문분야 카드 | `Tab`으로 카드 간 이동, `Space/Enter`로 선택/해제 |
| Hybrid Review | 각 섹션 `Tab` 이동, 전문분야 토글 `Space`, 라디오 `Arrow` |
| 온보딩 스텝 | `Tab` 순서 내 이동, 이전/다음 버튼 `Enter` |

### 6.2 ARIA 속성

```html
<!-- 전문분야 카드 -->
<button role="checkbox" aria-checked="true" aria-label="형사법 선택됨">

<!-- 신뢰도 게이지 -->
<div role="meter" aria-valuenow="82" aria-valuemin="0" aria-valuemax="100"
     aria-label="AI 분석 신뢰도 82%">

<!-- 스텝 인디케이터 -->
<nav aria-label="설정 단계">
  <ol>
    <li aria-current="step">1단계: 전문분야</li>
    <li>2단계: 톤/타겟</li>
    <li>3단계: 마무리</li>
  </ol>
</nav>

<!-- 분석 진행 상태 -->
<div role="progressbar" aria-valuenow="60" aria-valuetext="전문분야 패턴 분석 중">
```

### 6.3 스크린 리더 안내

- Welcome Screen: "페르소나 설정 페이지입니다. AI 자동 분석 또는 직접 설정 중 선택하세요."
- 분석 완료: "페르소나 분석이 완료되었습니다. 결과를 확인하고 수정할 수 있습니다."
- 설정 완료: "페르소나 설정이 완료되었습니다."

---

## 7. 성능 요구사항

| 항목 | 목표 | 전략 |
|------|------|------|
| Welcome Screen 렌더링 | < 100ms | 정적 콘텐츠, lazy import 불필요 |
| ChatHistoryCount API | < 500ms | 단순 COUNT 쿼리 |
| AI 분석 API | < 30초 | 타임아웃 설정 + 취소 지원 |
| Hybrid Review 인라인 수정 | 즉각 | 로컬 상태, API 호출 없음 |
| 최종 저장 API | < 2초 | 기존 성능 유지 |
| 스텝 전환 애니메이션 | 200ms | CSS transition |
| 톤 미리보기 | 즉각 | 하드코딩 텍스트 |

---

## 8. 구현 순서 (Phase별)

### Phase A: 기반 구조 (P0)

1. `types/index.ts` — 신규 타입 추가
2. `services/index.ts` — `analyzePersona` 응답 타입 변경 + timeout + `fetchChatHistoryCount`
3. `hooks/usePersonaGate.ts` — 상태머신 reducer 구현
4. Backend: `ChatHistoryCountResponse` 스키마 + GET 엔드포인트

### Phase B: 핵심 컴포넌트 (P0)

5. `PersonaWelcome.tsx` — Welcome Screen
6. `PersonaAnalysisProgress.tsx` — 분석 진행 화면
7. Backend: `AnalysisInsights` 스키마 + analyze 응답 확장
8. `PersonaHybridReview.tsx` — 인라인 편집 화면

### Phase C: PersonaGate 통합 (P0)

9. `PersonaGate.tsx` — useReducer 리팩토링 + 새 컴포넌트 통합
10. `PersonaConfirmation.tsx` 삭제

### Phase D: 온보딩 개선 (P1)

11. `PersonaOnboarding.tsx` — 3스텝 통합 + 카드형 UI + 추천 키워드
12. Track 전환 시 draftState 보존 로직

### Phase E: 마무리 (P1)

13. `PersonaBanner.tsx` — 기본 설정 안내 배너
14. 기본 페르소나 + skip 플로우
15. 마이크로인터랙션 + 애니메이션

---

## 9. Red Team / 외부 컨설팅 피드백 반영

### 9.1 Red Team (기획 단계) 피드백 반영

| 지적 | 반영 여부 | 설계 위치 |
|------|----------|----------|
| [High] PII 유출 위험 | **향후** | analyzer에서 마스킹 처리 (인증 시스템 구현 시) |
| [Medium] Rate Limiting 부재 | **반영** | `_collect_limiter` 패턴으로 analyze에도 rate limit 적용 |
| [Medium] IDOR 취약점 | **향후** | Auth Dependency 구현 시 세션-소유권 검증 |
| Circuit Breaker | **부분 반영** | 분석 실패 시 Track 2 자동 안내 (서킷 브레이커 수준은 아님) |
| SSE 실시간 진행률 | **미반영** | 현재 API 구조상 시간 기반 시뮬레이션으로 충분 |
| SessionStorage 상태 영속성 | **반영** | §4.2 SessionStorage 키 설계 |

### 9.2 외부 컨설팅 (기획 단계) 피드백 반영

| 제안 | 반영 여부 | 설계 위치 |
|------|----------|----------|
| 근거 카드 + 출처 기반 신뢰 UX | **반영** | HybridReview의 EvidenceSnippet + 근거 텍스트 |
| "추천값 유지 + 일부 수정" 기본 경로 | **반영** | HybridReview가 기본 흐름 |
| 레이더 차트 대신 바 차트 | **반영** | 수평 바 차트 (CSS only) |
| 90초 목표: 필수 입력 2개만 선완료 | **부분 반영** | 3스텝 통합으로 간소화 |
| Free/Pro/Enterprise 수익화 | **미반영** | 현재 스코프 외 |
| RAG 결합 페르소나 추출 | **미반영** | 현재 스코프 외 (향후 고도화) |

### 9.3 Red Team (설계 리뷰) 피드백 반영

| 지적 | 심각도 | 반영 여부 | 설계 반영 |
|------|--------|----------|----------|
| TEMP_USER_ID IDOR | Critical | **향후** | 인증 시스템 구현 시 해결 (현재 스코프 외) |
| PII 유출 위험 | High | **반영** | Phase D: `_fetch_chat_history()`에 PII 정제 로직 선적용 |
| SessionStorage 상태 조작 | Medium | **반영** | reducer에서 `ready` 전이 시 서버 검증 가드 추가 |
| LLM 프롬프트 인젝션 | Medium | **반영** | Phase D: evidence 생성 전 입력 정제(sanitize) 로직 추가 |
| 비동기 분석 패턴 (SSE/Job) | - | **미반영** | 현재 동기 30초 유지, 향후 Job 패턴으로 전환 검토 |
| 상태 전이 무결성 검증 | - | **반영** | §9.5 상태 전이 매트릭스 + reducer 가드 추가 |
| HybridReview 결합도 | - | **반영** | EditablePersonaField 원자 컴포넌트 분리 |
| 분석 중 중복 요청 방지 | - | **반영** | analyzing 상태에서 START_TRACK1 무시 가드 |
| SessionStorage 유효성 검사 | - | **반영** | 복원 시 JSON schema 검증 + 실패 시 초기화 |
| 동시 요청 Race Condition | - | **반영** | DB `user_id` UNIQUE + upsert 패턴 (기존 구현 확인) |

### 9.4 외부 컨설팅 (설계 리뷰) 피드백 반영

| 제안 | 반영 여부 | 설계 반영 |
|------|----------|----------|
| 상태별 허용 이벤트 매트릭스 | **반영** | §9.5 전이 매트릭스 테이블 추가 |
| race condition: request_id 비교 | **반영** | usePersonaGate에 analysisRequestId 추가 |
| API breaking change 대응 | **반영** | PersonaAnalysisResponse를 Composition 구조로 명확화 (기존 LawyerPersona 필드 + insights 분리) |
| 표준 에러 바디 | **부분 반영** | error_code + detail 필드 추가 (trace_id는 향후) |
| 비동기 Job 패턴 | **미반영** | 현재 스코프 외 (향후 고도화 기록) |
| 분석 중 Track2 병렬 입력 | **미반영** | UX 복잡도 대비 이득 불분명 |
| 상위/하위 상태 계층화 | **미반영** | 현재 6개 상태 관리 가능, 확장 시 재검토 |
| "즉시 사용 기본안" 버튼 | **반영** | HybridReview에 "이 설정으로 바로 시작" CTA 강조 |
| EvidenceSnippet 출처 신뢰도 | **부분 반영** | date 필드는 있으나, confidence_score는 향후 추가 |

### 9.5 상태 전이 매트릭스 (v1.1 추가)

> Red Team + Consultant 공통 지적: 불법 전이 방지를 위한 명시적 전이 규칙

| 현재 상태 \ 액션 | LOAD_COMPLETE | START_TRACK1 | ANALYSIS_SUCCESS | ANALYSIS_FAILURE | SWITCH_TO_TRACK2 | SWITCH_TO_TRACK1 | COMPLETE_REVIEW | COMPLETE_ONBOARDING | EDIT_PERSONA |
|---|---|---|---|---|---|---|---|---|---|
| `loading` | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `welcome` | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| `track1_analyzing` | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| `track1_review` | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ |
| `track2_onboarding` | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ |
| `ready` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |

**Reducer 가드 구현:**
```typescript
const VALID_TRANSITIONS: Record<GateScreen, GateAction['type'][]> = {
  loading: ['LOAD_COMPLETE'],
  welcome: ['START_TRACK1', 'SWITCH_TO_TRACK2'],
  track1_analyzing: ['ANALYSIS_SUCCESS', 'ANALYSIS_FAILURE', 'SWITCH_TO_TRACK2'],
  track1_review: ['SWITCH_TO_TRACK2', 'COMPLETE_REVIEW'],
  track2_onboarding: ['SWITCH_TO_TRACK1', 'COMPLETE_ONBOARDING'],
  ready: ['EDIT_PERSONA'],
};

function gateReducer(state: PersonaGateState, action: GateAction): PersonaGateState {
  const allowed = VALID_TRANSITIONS[state.screen];
  if (!allowed.includes(action.type)) {
    console.warn(`[PersonaGate] Invalid transition: ${state.screen} + ${action.type}`);
    return state; // 불법 전이 무시
  }
  // ... 정상 전이 처리
}
```

### 9.6 향후 개선 과제 (현재 스코프 외)

| 과제 | 출처 | 우선순위 |
|------|------|---------|
| 비동기 Job 패턴 (POST → 202 + polling) | Red Team + Consultant | P2 |
| 서버 체크포인트 + 재로그인 복원 | Consultant | P2 |
| Model-agnostic orchestration | Consultant | P3 |
| MCP 기반 외부 도구 연결 | Consultant | P3 |
| Observability/EvalOps (trace_id, tool-call 로그) | Consultant | P2 |
| 엔터프라이즈 거버넌스 (감사 로그, 팀 정책) | Consultant | P3 |
| 상위/하위 상태 계층화 | Consultant | P2 |
| A/B 테스트 + 코호트 분석 | 기획 컨설팅 | P2 |
| 멀티 페르소나 (사건 유형별) | 기획 Red Team | P3 |
| EvidenceSnippet confidence_score | Consultant | P2 |

---

## 변경 이력

| 날짜 | 버전 | 변경 내용 |
|------|------|----------|
| 2026-02-28 | v1.0 | 초안 작성 |
| 2026-02-28 | v1.1 | 설계 리뷰 피드백 반영: 상태 전이 매트릭스, reducer 가드, EditablePersonaField 분리, SessionStorage 유효성 검사, API Composition 구조, 표준 에러 바디, PII 정제 선적용, request_id 비교 |
