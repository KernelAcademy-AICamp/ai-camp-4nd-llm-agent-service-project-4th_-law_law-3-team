# Mock Trial (모의 법정) Design Document

> **Summary**: Phaser.js 픽셀아트 법정 + LangGraph 다중 에이전트 재판 시뮬레이터의 상세 설계
>
> **Project**: law-3-team (법률 서비스 플랫폼)
> **Version**: 0.1.0
> **Author**: Claude
> **Date**: 2026-02-12
> **Status**: Draft
> **Planning Doc**: [mock-trial.plan.md](../01-plan/features/mock-trial.plan.md)

### Pipeline References

| Phase | Document | Status |
|-------|----------|--------|
| Phase 1 | [Schema Definition](../01-plan/schema.md) | N/A |
| Phase 2 | [Coding Conventions](../01-plan/conventions.md) | N/A |
| Phase 3 | Mockup | N/A (Phaser.js 법정 씬으로 대체) |
| Phase 4 | API Spec (본 문서 Section 4) | ✅ |

---

## 1. Overview

### 1.1 Design Goals

1. **기존 아키텍처와의 일관성**: small_claims 서브그래프 패턴(interrupt + Command)을 그대로 재사용하여 mock_trial 서브그래프를 구현한다.
2. **한국 법정 절차 정확 반영**: 형사 공판절차(형사소송법 §275~§321) 6단계, 민사 변론절차(민사소송법 §134~) 6단계를 별도 서브그래프로 구현한다.
3. **Phaser.js ↔ React 분리 설계**: 게임 엔진(Phaser.js)과 UI 프레임워크(React)를 EventBus로 느슨하게 결합한다.
4. **에이전트 모듈화**: SimCourt의 Profile/Memory/Strategy 패턴을 CourtAgent 클래스로 추상화하여 5개 역할에 적용한다.

### 1.2 Design Principles

- **Single Responsibility**: 각 노드 함수는 하나의 재판 단계만 담당
- **Open/Closed**: CourtAgent 클래스를 상속하여 역할별 에이전트 확장 가능
- **Dependency Inversion**: RAG 검색은 인터페이스(프로토콜)로 추상화하여 기존 RAG와 독립 인터페이스 모두 지원
- **기존 패턴 준수**: BaseChatAgent, ModuleRegistry, AGENT_NODE_MAP 등 기존 컨벤션 100% 유지

---

## 2. Architecture

### 2.1 Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Next.js App (/mock-trial)                     │
│                                                                      │
│  ┌──────────────────────────────────┐  ┌──────────────────────────┐ │
│  │     Phaser.js Canvas (게임 씬)    │  │   React Overlay UI       │ │
│  │  ┌────────────────────────────┐  │  │  ┌────────────────────┐  │ │
│  │  │ CourtScene (타일맵+스프라이트)│  │  │  │ ChatPanel          │  │ │
│  │  │ LobbyScene (설정 화면)      │  │  │  │ EvidencePanel      │  │ │
│  │  │ SpeechBubble (말풍선 UI)    │  │  │  │ StageProgress      │  │ │
│  │  └────────────────────────────┘  │  │  │ JudgmentDisplay     │  │ │
│  │           ▲           │           │  │  └────────────────────┘  │ │
│  └───────────┼───────────┼───────────┘  └──────────┼───────────────┘ │
│              │   EventBus (CustomEvent)             │                 │
│              └──────────────┬────────────────────────┘                │
│                             │                                        │
│                     ┌───────┴───────┐                                │
│                     │  API Service  │ SSE/fetch                      │
│                     └───────┬───────┘                                │
└─────────────────────────────┼────────────────────────────────────────┘
                              │ HTTP
┌─────────────────────────────┼────────────────────────────────────────┐
│                     FastAPI Backend                                    │
│                             │                                        │
│  ┌──────────────────────────┴──────────────────────────┐             │
│  │               /api/chat (통합 채팅 API)               │             │
│  │  router_node → mock_trial_subgraph → END             │             │
│  └──────────────────────────┬──────────────────────────┘             │
│                             │                                        │
│  ┌──────────────────────────┴──────────────────────────┐             │
│  │          LangGraph mock_trial Subgraph               │             │
│  │                                                      │             │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │             │
│  │  │ setup    │→ │ stage_1  │→ │ stage_N  │→ verdict  │             │
│  │  │ (설정)   │  │ (형사/민사│  │ ...      │  (판결)   │             │
│  │  └──────────┘  │  분기)   │  └──────────┘          │             │
│  │                └──────────┘                         │             │
│  │  ┌──────────────────────────────────────────┐      │             │
│  │  │ CourtAgent (Profile/Memory/Strategy)     │      │             │
│  │  │ × 5 (판사/검사/변호사/피고인/서기)        │      │             │
│  │  └──────────────────────────────────────────┘      │             │
│  └─────────────────────────────────────────────────────┘             │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐          │
│  │  Solar LLM   │  │  LanceDB RAG │  │  PostgreSQL      │          │
│  │  (에이전트)   │  │  (판례/법령)  │  │  (체크포인터)     │          │
│  └──────────────┘  └──────────────┘  └──────────────────┘          │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

#### 전체 흐름
```
사용자 → React UI → EventBus → Phaser.js (애니메이션)
                  ↘ API call → /api/chat → router_node → mock_trial_subgraph
                                              ↓
                              (interrupt) ← stage_node → CourtAgent.generate()
                                              ↓           → RAG 검색 (Tool)
                              resume(사용자 입력) → 다음 노드 → ... → verdict_node
                                              ↓
                              response (SSE 스트리밍) → React UI → EventBus → Phaser.js (말풍선)
```

#### 단일 단계 내부 흐름
```
stage_node 진입
  │
  ├─ _sync_from_ui_state(): 프론트엔드 상태 동기화
  ├─ CourtAgent.generate(): 해당 단계 AI 발언 생성
  │   ├─ Profile 확인 (역할별 시스템 프롬프트)
  │   ├─ Memory 조회 (이전 단계 요약 + 현재 단계 기록)
  │   ├─ Strategy 적용 (단계별 전략)
  │   └─ [검사/변호사] RAG Tool 호출 (판례/법령 검색)
  │
  ├─ court_record 업데이트 (서기 기록)
  ├─ interrupt(): 사용자 입력 대기 (응답 + 액션 전송)
  │
  └─ resume → Command(update={...}, goto="next_node")
```

### 2.3 Dependencies

| Component | Depends On | Purpose |
|-----------|-----------|---------|
| Phaser.js CourtScene | 타일맵 에셋, 스프라이트시트 | 법정 렌더링 |
| React Overlay UI | EventBus, API Service | 사용자 입력/출력 |
| mock_trial Subgraph | CourtAgent, RAG Tool | 재판 절차 진행 |
| CourtAgent | Solar LLM, Profile/Memory/Strategy | AI 발언 생성 |
| RAG Tool | LanceDB (법령/판례 테이블) | 법률 검색 |
| Checkpointer | PostgreSQL | 세션 상태 저장/복원 |

---

## 3. Data Model

### 3.1 Entity Definition — MockTrialState (서브그래프 상태)

```python
from typing import Any, Optional
from typing_extensions import TypedDict


class MockTrialState(TypedDict, total=False):
    """모의재판 서브그래프 상태 (SmallClaimsState 패턴 준수)"""

    # ── Input (부모 ChatState에서 전달) ──
    message: str
    history: list[dict[str, str]]
    session_data: dict[str, Any]

    # ── Setup (설정 단계에서 결정) ──
    case_type: str         # "criminal" | "civil"
    case_category: str     # "형사_사기", "민사_손해배상" 등
    user_role: str         # "prosecutor" | "attorney" | "plaintiff" | "defendant"
    case_summary: str      # 사건 개요 (사용자 입력, 500자 이내)

    # ── Agent State ──
    agents: dict[str, "AgentState"]
    # 키: "judge" | "prosecutor" | "attorney" | "defendant" | "clerk"
    # 값: { "profile": AgentProfile, "memory": AgentMemory, "strategy": str }

    # ── Evidence (증거) ──
    evidence_cases: list[dict[str, Any]]     # RAG 판례 검색 결과
    evidence_articles: list[dict[str, Any]]  # RAG 법령 검색 결과
    selected_evidence: list[str]             # 사용자 선택 증거 ID

    # ── Trial Progress (재판 진행) ──
    stage: str
    # criminal: "setup"|"identity"|"opening"|"evidence"|"examination"|"closing"|"verdict"
    # civil:    "setup"|"pretrial"|"claims"|"evidence"|"argument"|"closing"|"verdict"
    current_round: int     # 현재 라운드 (1-based)
    max_rounds: int        # 최대 라운드 (기본 3)
    court_record: list["CourtRecord"]  # 서기 기록

    # ── Output (부모 그래프로 전달) ──
    response: str               # 현재 에이전트 발언
    speaking_agent: str         # 발언 중인 에이전트 역할명
    actions: list[dict[str, Any]]
    judgment: Optional[str]     # 최종 판결문
    feedback: Optional[str]     # 강점/약점 피드백
    is_complete: bool
    agent_used: str             # "mock_trial" (고정)
    output_session_data: dict[str, Any]
```

### 3.2 보조 타입 정의

```python
class AgentProfile(TypedDict):
    """에이전트 프로필 모듈"""
    name: str          # "재판장 김법관"
    role: str          # "judge" | "prosecutor" | "attorney" | "defendant" | "clerk"
    personality: str   # 성향 설명 (프롬프트에 반영)
    expertise: str     # 전문 분야


class AgentMemory(TypedDict):
    """에이전트 기억 모듈"""
    short_term: list[str]   # 현재 단계 발언 기록
    long_term: list[str]    # 이전 단계 요약 (reflection 결과)


class AgentState(TypedDict):
    """에이전트 전체 상태"""
    profile: AgentProfile
    memory: AgentMemory
    strategy: str          # 현재 단계 전략 (단계 전환 시 업데이트)


class CourtRecord(TypedDict):
    """서기 기록 엔트리"""
    stage: str         # 재판 단계명
    speaker: str       # 발언자 역할
    content: str       # 발언 내용 (요약)
    timestamp: str     # ISO 형식
```

### 3.3 Stage Enum 상수

```python
class CriminalStage:
    """형사 공판절차 단계 (형사소송법 기반)"""
    SETUP = "setup"              # 사건 설정
    IDENTITY = "identity"        # 인정신문 (§284)
    OPENING = "opening"          # 모두진술 (§285~§286)
    EVIDENCE = "evidence"        # 증거조사 (§290~§313)
    EXAMINATION = "examination"  # 피고인신문 (§296-2)
    CLOSING = "closing"          # 최종변론 (§302~§303)
    VERDICT = "verdict"          # 판결선고 (§318-4)

    ALL = [SETUP, IDENTITY, OPENING, EVIDENCE, EXAMINATION, CLOSING, VERDICT]


class CivilStage:
    """민사 변론절차 단계 (민사소송법 기반)"""
    SETUP = "setup"          # 사건 설정
    PRETRIAL = "pretrial"    # 변론준비 (§258~§268)
    CLAIMS = "claims"        # 주장/답변 (§256~§257)
    EVIDENCE = "evidence"    # 증거조사 (§288~§344)
    ARGUMENT = "argument"    # 변론 (§134~§148)
    CLOSING = "closing"      # 변론종결 (§200)
    VERDICT = "verdict"      # 판결선고 (§206~§208)

    ALL = [SETUP, PRETRIAL, CLAIMS, EVIDENCE, ARGUMENT, CLOSING, VERDICT]
```

### 3.4 Entity Relationships

```
[MockTrialState] 1 ──── N [AgentState]   (agents dict)
                 1 ──── N [CourtRecord]  (court_record list)
                 1 ──── N [Evidence]     (evidence_cases, evidence_articles)

[AgentState] 1 ──── 1 [AgentProfile]
             1 ──── 1 [AgentMemory]
```

---

## 4. API Specification

### 4.1 통합 채팅 API (기존 엔드포인트 활용)

모의재판은 **기존 `/api/chat` 통합 API**를 통해 진입한다. `router_node`가 `AgentType.MOCK_TRIAL`로 라우팅하면 `mock_trial_subgraph`가 실행된다.

```
POST /api/chat
{
  "message": "모의재판 시작",
  "history": [],
  "session_data": {}
}
→ router_node → mock_trial_subgraph → interrupt (설정 단계)
```

### 4.2 모듈 전용 엔드포인트 (modules/mock_trial/router)

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| GET | `/api/mock-trial/case-types` | 사건 유형 목록 반환 | - |
| GET | `/api/mock-trial/roles/{case_type}` | 사건 유형별 선택 가능 역할 반환 | - |
| POST | `/api/mock-trial/search-evidence` | 모의재판 전용 판례/법령 검색 | - |
| GET | `/api/mock-trial/stage-info/{case_type}` | 사건 유형별 단계 정보 반환 | - |

### 4.3 상세 API 명세

#### `GET /api/mock-trial/case-types`

**Response (200 OK):**
```json
{
  "case_types": [
    {
      "id": "criminal",
      "name": "형사 재판",
      "categories": [
        { "id": "criminal_assault", "name": "폭행/상해", "description": "폭행죄, 상해죄 등" },
        { "id": "criminal_fraud", "name": "사기", "description": "사기죄, 횡령죄 등" },
        { "id": "criminal_theft", "name": "절도", "description": "절도죄, 강도죄 등" }
      ]
    },
    {
      "id": "civil",
      "name": "민사 재판",
      "categories": [
        { "id": "civil_damages", "name": "손해배상", "description": "불법행위, 채무불이행 등" },
        { "id": "civil_contract", "name": "계약 분쟁", "description": "계약 해제, 이행 청구 등" },
        { "id": "civil_property", "name": "부동산", "description": "임대차, 소유권 분쟁 등" }
      ]
    }
  ]
}
```

#### `GET /api/mock-trial/roles/{case_type}`

**Request:** `GET /api/mock-trial/roles/criminal`

**Response (200 OK):**
```json
{
  "case_type": "criminal",
  "roles": [
    { "id": "prosecutor", "name": "검사", "description": "공소 유지 및 범죄 입증" },
    { "id": "attorney", "name": "변호사", "description": "피고인 방어 및 무죄/감형 논증" }
  ]
}
```

#### `POST /api/mock-trial/search-evidence`

**Request:**
```json
{
  "query": "사기죄 양형 기준",
  "search_type": "all",
  "limit": 5
}
```

**Response (200 OK):**
```json
{
  "cases": [
    {
      "id": "case_001",
      "title": "대법원 2020도12345",
      "summary": "사기죄의 양형 기준에 관한 판례...",
      "relevance_score": 0.92,
      "source": "lancedb"
    }
  ],
  "articles": [
    {
      "id": "art_001",
      "title": "형법 제347조 (사기)",
      "content": "사람을 기망하여 재물의 교부를 받거나...",
      "relevance_score": 0.95,
      "source": "lancedb"
    }
  ]
}
```

#### `GET /api/mock-trial/stage-info/{case_type}`

**Request:** `GET /api/mock-trial/stage-info/criminal`

**Response (200 OK):**
```json
{
  "case_type": "criminal",
  "stages": [
    {
      "id": "identity",
      "name": "인정신문",
      "order": 1,
      "legal_basis": "형사소송법 §284",
      "description": "재판장이 피고인 인적사항 확인 및 진술거부권 고지",
      "user_action": "자동 진행 (관전)",
      "duration_hint": "1-2분"
    },
    {
      "id": "opening",
      "name": "모두진술",
      "order": 2,
      "legal_basis": "형사소송법 §285~§286",
      "description": "검사 공소사실 요지 진술, 피고인/변호인 의견 진술",
      "user_action": "역할에 따라 진술 입력",
      "duration_hint": "3-5분"
    }
  ]
}
```

---

## 5. UI/UX Design

### 5.1 Screen Layout — 법정 씬

```
┌────────────────────────────────────────────────────────────────┐
│  ┌──────────────────────────────────────────────────────────┐ │
│  │                 Phaser.js Canvas (800×480)                │ │
│  │                                                           │ │
│  │    ┌───────┐       ┌──────────────────┐                  │ │
│  │    │ 서기  │       │  "피고인은..."    │  ← 말풍선        │ │
│  │    │ (clerk)│       │  (타이핑 효과)    │                  │ │
│  │    └───────┘       └──────┬───────────┘                  │ │
│  │                    ┌──────┴───────┐                       │ │
│  │                    │   👨‍⚖️ 판사    │                       │ │
│  │                    └──────────────┘                       │ │
│  │     ┌──────────┐              ┌──────────┐               │ │
│  │     │ 👨‍💼 검사  │              │ 👩‍💼 변호사 │ ← 사용자     │ │
│  │     └──────────┘              └──────────┘               │ │
│  │              ┌──────────┐                                 │ │
│  │              │ 🧑 피고인 │                                 │ │
│  │              └──────────┘                                 │ │
│  │  ─────────────────────────────────────────────            │ │
│  │  ⚖️ 단계: 3/6 증거조사     👤 역할: 검사                  │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  React Overlay Panel                                      │ │
│  │  ┌──────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐│ │
│  │  │ 💬   │ │ 📋 증거  │ │ 📖 법령  │ │ ⏩ 다음 단계 →   ││ │
│  │  │ 대화 │ │ 목록     │ │ 검색     │ │                  ││ │
│  │  └──────┘ └──────────┘ └──────────┘ └──────────────────┘│ │
│  │                                                           │ │
│  │  ┌────────────────────────────────────────┐  ┌────────┐ │ │
│  │  │ 검사 측 주장을 입력하세요...             │  │ 전송 → │ │ │
│  │  └────────────────────────────────────────┘  └────────┘ │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
```

### 5.2 Screen Layout — 로비 씬

```
┌────────────────────────────────────────────────────────────────┐
│  ┌──────────────────────────────────────────────────────────┐ │
│  │                 Phaser.js Canvas (800×480)                │ │
│  │                                                           │ │
│  │                    🏛️ 모의 법정                            │ │
│  │                                                           │ │
│  │         "실제 법률 자문이 아닙니다" (면책 고지)             │ │
│  │                                                           │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  Step 1: 사건 유형 선택                                   │ │
│  │  ┌──────────────┐  ┌──────────────┐                      │ │
│  │  │ ⚖️ 형사 재판  │  │ 📜 민사 재판  │                      │ │
│  │  └──────────────┘  └──────────────┘                      │ │
│  │                                                           │ │
│  │  Step 2: 세부 유형 선택                                   │ │
│  │  [폭행/상해] [사기] [절도] [횡령] [기타]                   │ │
│  │                                                           │ │
│  │  Step 3: 역할 선택                                        │ │
│  │  ┌──────────────┐  ┌──────────────┐                      │ │
│  │  │ 👨‍💼 검사 역할  │  │ 👩‍💼 변호사 역할│                      │ │
│  │  └──────────────┘  └──────────────┘                      │ │
│  │                                                           │ │
│  │  Step 4: 사건 개요                                        │ │
│  │  ┌──────────────────────────────────┐  ┌──────────────┐  │ │
│  │  │ 사건 개요를 입력하세요... (500자)  │  │ 재판 시작 →  │  │ │
│  │  └──────────────────────────────────┘  └──────────────┘  │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
```

### 5.3 User Flow

```
/mock-trial 접속
    │
    ▼
[로비 씬] 사건유형(형사/민사) → 세부유형 → 역할 → 사건개요 입력
    │
    ▼
[법정 씬 전환] 캐릭터 배치 애니메이션
    │
    ├─ (형사) identity → opening → evidence → examination → closing → verdict
    │
    └─ (민사) pretrial → claims → evidence → argument → closing → verdict
                │
                ▼ (각 단계)
        AI 에이전트 발언 (말풍선 스트리밍)
            │
            ▼
        사용자 입력 (interrupt 대기)
            │
            ▼
        다음 단계로 전환 (씬 전환 효과)
    │
    ▼
[판결 씬] AI 판사 판결문 스트리밍 → 피드백 표시
    │
    ▼
[결과 화면] 강점/약점 분석 + 인용 판례/법령 + "다시 시작" 버튼
```

### 5.4 Component List

| Component | Location | Responsibility |
|-----------|----------|----------------|
| **Phaser.js (Game)** | | |
| `CourtScene` | `features/mock-trial/game/CourtScene.ts` | 메인 법정 씬 (타일맵, 캐릭터 배치, 단계별 연출) |
| `LobbyScene` | `features/mock-trial/game/LobbyScene.ts` | 로비/설정 씬 (법정 외관, 면책 고지) |
| `CharacterBase` | `features/mock-trial/game/sprites/CharacterBase.ts` | 캐릭터 공통 (스프라이트 로드, 애니메이션, 위치) |
| `SpeechBubble` | `features/mock-trial/game/ui/SpeechBubble.ts` | 말풍선 (9-patch 배경, 타이핑 효과, 스트리밍) |
| `StageIndicator` | `features/mock-trial/game/ui/StageIndicator.ts` | 단계 표시 바 (현재 단계 하이라이트) |
| `EventBus` | `features/mock-trial/game/EventBus.ts` | Phaser ↔ React 이벤트 통신 |
| **React (Overlay)** | | |
| `MockTrialGame` | `features/mock-trial/components/MockTrialGame.tsx` | Phaser 게임 래퍼 (dynamic import, canvas 마운트) |
| `MockTrialSetup` | `features/mock-trial/components/MockTrialSetup.tsx` | 설정 UI (사건유형/역할/개요 입력) |
| `ChatPanel` | `features/mock-trial/components/ChatPanel.tsx` | 하단 대화 입력/출력 패널 |
| `EvidencePanel` | `features/mock-trial/components/EvidencePanel.tsx` | 증거 카드 목록 (판례/법령 검색 결과) |
| `StageProgress` | `features/mock-trial/components/StageProgress.tsx` | 재판 단계 진행률 표시 |
| `JudgmentDisplay` | `features/mock-trial/components/JudgmentDisplay.tsx` | 판결문 모달 + 피드백 |
| `DisclaimerBanner` | `features/mock-trial/components/DisclaimerBanner.tsx` | 면책 고지 배너 (상시 표시) |
| **Page** | | |
| `page.tsx` | `app/mock-trial/page.tsx` | 페이지 엔트리포인트 |

---

## 6. Backend 상세 설계

### 6.1 CourtAgent 클래스 (Profile/Memory/Strategy)

```python
# subgraphs/mock_trial_agents.py

from dataclasses import dataclass, field
from typing import Any

from app.tools.llm.solar_client import get_solar_response_stream


@dataclass
class CourtAgent:
    """법정 에이전트 (SimCourt Profile/Memory/Strategy 패턴)"""

    # Profile Module
    role: str                    # "judge"|"prosecutor"|"attorney"|"defendant"|"clerk"
    name: str                    # 표시 이름 ("재판장 김법관")
    system_prompt: str           # 역할별 시스템 프롬프트
    temperature: float = 0.5    # LLM 온도

    # Memory Module
    short_term: list[str] = field(default_factory=list)  # 현재 단계 발언
    long_term: list[str] = field(default_factory=list)   # 이전 단계 요약

    # Strategy Module
    strategy: str = ""           # 현재 단계 전략

    # Tools
    tools: list[str] = field(default_factory=list)  # ["case_retriever", "article_retriever"]

    async def generate(
        self,
        stage: str,
        context: str,
        court_record: list[dict],
    ) -> str:
        """에이전트 발언 생성

        Args:
            stage: 현재 재판 단계
            context: 사건 맥락 (case_summary + 현재 상황)
            court_record: 서기 기록 (이전 발언 내역)

        Returns:
            생성된 발언 텍스트
        """
        memory_context = self._build_memory_context()
        prompt = f"{self.system_prompt}\n\n"
        prompt += f"[전략] {self.strategy}\n\n"
        prompt += f"[기억] {memory_context}\n\n"
        prompt += f"[현재 단계] {stage}\n\n"
        prompt += f"[사건 맥락] {context}\n\n"
        prompt += f"[법정 기록]\n{self._format_record(court_record)}\n\n"
        prompt += "위 맥락을 바탕으로 발언하세요."

        response = await get_solar_response_stream(
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        # 단기 기억에 추가
        self.short_term.append(response)
        return response

    def reflect(self) -> str:
        """단계 종료 시 기억 요약 (reflection)"""
        summary = f"[{len(self.short_term)}개 발언 요약]: "
        # 단기 기억을 요약하여 장기 기억으로 이동
        summary += " / ".join(self.short_term[-3:])  # 최근 3개 요약
        self.long_term.append(summary)
        self.short_term.clear()
        return summary

    def update_strategy(self, new_strategy: str) -> None:
        """다음 단계 전략 업데이트"""
        self.strategy = new_strategy

    def _build_memory_context(self) -> str:
        """기억 컨텍스트 문자열 빌드"""
        parts = []
        if self.long_term:
            parts.append("[이전 단계 요약]\n" + "\n".join(self.long_term[-3:]))
        if self.short_term:
            parts.append("[현재 단계 발언]\n" + "\n".join(self.short_term[-5:]))
        return "\n\n".join(parts) if parts else "(없음)"

    def _format_record(self, court_record: list[dict]) -> str:
        """법정 기록 포맷팅"""
        lines = []
        for r in court_record[-10:]:  # 최근 10개
            lines.append(f"[{r.get('stage', '')}] {r.get('speaker', '')}: {r.get('content', '')}")
        return "\n".join(lines) if lines else "(기록 없음)"

    def to_state(self) -> dict[str, Any]:
        """AgentState TypedDict로 직렬화"""
        return {
            "profile": {
                "name": self.name,
                "role": self.role,
                "personality": "",
                "expertise": "",
            },
            "memory": {
                "short_term": list(self.short_term),
                "long_term": list(self.long_term),
            },
            "strategy": self.strategy,
        }

    @classmethod
    def from_state(
        cls, state: dict[str, Any], system_prompt: str, temperature: float = 0.5
    ) -> "CourtAgent":
        """AgentState TypedDict에서 복원"""
        profile = state.get("profile", {})
        memory = state.get("memory", {})
        return cls(
            role=profile.get("role", ""),
            name=profile.get("name", ""),
            system_prompt=system_prompt,
            temperature=temperature,
            short_term=list(memory.get("short_term", [])),
            long_term=list(memory.get("long_term", [])),
            strategy=state.get("strategy", ""),
        )
```

### 6.2 에이전트 프로필 설정

```python
# subgraphs/mock_trial_prompts.py

JUDGE_SYSTEM_PROMPT = """당신은 대한민국 법원의 재판장입니다.
- 공정하고 중립적인 태도로 재판을 진행합니다
- 형사소송법/민사소송법에 따른 절차를 엄격히 준수합니다
- 양측의 주장을 균형 있게 청취합니다
- 발언은 간결하고 권위 있게 합니다
- 면책 고지: 이 재판은 교육 목적의 모의재판입니다"""

PROSECUTOR_CRIMINAL_PROMPT = """당신은 대한민국 검찰의 검사입니다.
- 공소사실을 입증하는 것이 목표입니다
- 증거와 판례를 인용하여 논리적으로 주장합니다
- 피고인의 범죄 사실을 구체적으로 적시합니다
- 구형 시 양형 기준을 참고합니다"""

PLAINTIFF_CIVIL_PROMPT = """당신은 원고측 대리인(변호사)입니다.
- 청구원인을 구체적으로 입증하는 것이 목표입니다
- 손해 발생 사실과 인과관계를 논증합니다
- 관련 판례와 법령을 인용합니다"""

ATTORNEY_CRIMINAL_PROMPT = """당신은 피고인의 변호인입니다.
- 피고인의 무죄 또는 감형을 논증하는 것이 목표입니다
- 검사 측 증거의 약점을 지적합니다
- 유리한 판례와 정상참작 사유를 제시합니다"""

DEFENDANT_CIVIL_PROMPT = """당신은 피고측 대리인(변호사)입니다.
- 원고의 청구를 기각시키는 것이 목표입니다
- 항변 사유를 구체적으로 제시합니다
- 원고 측 주장의 약점을 지적합니다"""

DEFENDANT_PERSON_PROMPT = """당신은 재판의 피고인/당사자입니다.
- 자신의 입장을 감정적이면서도 사실에 기반하여 표현합니다
- 질문에 성실히 답변합니다
- 법률 용어보다 일상 언어를 사용합니다"""

CLERK_SYSTEM_PROMPT = """당신은 법원 서기입니다.
- 재판 진행을 간결하게 기록합니다
- 중립적이고 객관적인 서술을 합니다
- 각 단계의 핵심 내용만 요약합니다"""

# 에이전트별 기본 설정
AGENT_CONFIGS = {
    "judge": {
        "name": "재판장",
        "temperature": 0.3,
        "tools": [],
    },
    "prosecutor": {
        "name": "검사",
        "temperature": 0.7,
        "tools": ["case_retriever", "article_retriever"],
    },
    "attorney": {
        "name": "변호인",
        "temperature": 0.7,
        "tools": ["case_retriever", "article_retriever"],
    },
    "defendant": {
        "name": "피고인",
        "temperature": 0.8,
        "tools": [],
    },
    "clerk": {
        "name": "서기",
        "temperature": 0.2,
        "tools": [],
    },
}

# 시스템 프롬프트 매핑 (case_type × role)
SYSTEM_PROMPTS = {
    ("criminal", "judge"): JUDGE_SYSTEM_PROMPT,
    ("criminal", "prosecutor"): PROSECUTOR_CRIMINAL_PROMPT,
    ("criminal", "attorney"): ATTORNEY_CRIMINAL_PROMPT,
    ("criminal", "defendant"): DEFENDANT_PERSON_PROMPT,
    ("criminal", "clerk"): CLERK_SYSTEM_PROMPT,
    ("civil", "judge"): JUDGE_SYSTEM_PROMPT,
    ("civil", "prosecutor"): PLAINTIFF_CIVIL_PROMPT,    # 민사에서는 원고측
    ("civil", "attorney"): DEFENDANT_CIVIL_PROMPT,       # 민사에서는 피고측
    ("civil", "defendant"): DEFENDANT_PERSON_PROMPT,
    ("civil", "clerk"): CLERK_SYSTEM_PROMPT,
}
```

### 6.3 서브그래프 구현 설계

```python
# subgraphs/mock_trial.py

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt, Command


def setup_node(state: MockTrialState) -> Command[str]:
    """사건 설정 노드 (형사/민사 공통)"""
    interrupt_value = interrupt({
        "response": "모의 법정에 오신 것을 환영합니다. 사건 유형, 역할, 사건 개요를 입력해주세요.",
        "actions": _case_type_actions(),
        "step": "setup",
    })
    # resume 시 사용자 입력 파싱
    case_type, user_role, case_summary = _parse_setup(interrupt_value)
    agents = _init_agents(case_type)

    return Command(
        update={
            "case_type": case_type,
            "user_role": user_role,
            "case_summary": case_summary,
            "agents": agents,
            "stage": "setup",
            "current_round": 1,
            "max_rounds": 3,
            "court_record": [],
            "agent_used": "mock_trial",
            "output_session_data": {"active_agent": "mock_trial"},
        },
        goto=_route_first_stage(case_type),
    )


def _route_first_stage(case_type: str) -> str:
    """사건 유형에 따른 첫 번째 재판 노드 결정"""
    if case_type == "criminal":
        return "identity_node"     # 형사: 인정신문
    else:
        return "pretrial_node"     # 민사: 변론준비


# ── 형사 전용 노드 ──

def identity_node(state: MockTrialState) -> Command[str]:
    """[형사] 인정신문 (형사소송법 §284)"""
    # 재판장 자동 발언: 인적사항 확인 + 진술거부권 고지
    judge = _get_agent(state, "judge")
    response = await judge.generate("identity", state["case_summary"], state["court_record"])
    _record(state, "identity", "judge", response)
    return Command(
        update={"stage": "identity", "response": response, "speaking_agent": "judge"},
        goto="opening_node",
    )


def opening_node(state: MockTrialState) -> Command[str]:
    """[형사] 모두진술 (형사소송법 §285~§286)"""
    prosecutor = _get_agent(state, "prosecutor")
    pros_stmt = await prosecutor.generate("opening", state["case_summary"], state["court_record"])
    _record(state, "opening", "prosecutor", pros_stmt)

    interrupt_value = interrupt({
        "response": f"[검사 모두진술]\n{pros_stmt}\n\n피고인/변호인 측 의견을 진술해주세요.",
        "speaking_agent": "prosecutor",
        "stage": "opening",
        "actions": [{"type": "button", "label": "인정", "action": "admit"},
                    {"type": "button", "label": "부인", "action": "deny"}],
    })
    user_input = str(interrupt_value)
    _record(state, "opening", state["user_role"], user_input)

    return Command(
        update={"stage": "opening", "response": user_input, "speaking_agent": state["user_role"]},
        goto="evidence_node",
    )


# (examination_node, closing_node 동일 패턴)


# ── 민사 전용 노드 ──

def pretrial_node(state: MockTrialState) -> Command[str]:
    """[민사] 변론준비 (민사소송법 §258~§268)"""
    judge = _get_agent(state, "judge")
    response = await judge.generate("pretrial", state["case_summary"], state["court_record"])
    _record(state, "pretrial", "judge", response)
    return Command(
        update={"stage": "pretrial", "response": response, "speaking_agent": "judge"},
        goto="claims_node",
    )


def claims_node(state: MockTrialState) -> Command[str]:
    """[민사] 주장/답변 (민사소송법 §256~§257)"""
    # 사용자 역할에 따라 원고/피고 입력 대기
    interrupt_value = interrupt({
        "response": "원고 측 청구원인을 진술해주세요.",
        "stage": "claims",
        "actions": [],
    })
    user_input = str(interrupt_value)
    _record(state, "claims", state["user_role"], user_input)

    # AI 상대측 답변 생성
    opponent_role = "attorney" if state["user_role"] == "prosecutor" else "prosecutor"
    opponent = _get_agent(state, opponent_role)
    opponent_response = await opponent.generate("claims", state["case_summary"], state["court_record"])
    _record(state, "claims", opponent_role, opponent_response)

    return Command(
        update={"stage": "claims", "response": opponent_response, "speaking_agent": opponent_role},
        goto="evidence_node",
    )


def argument_node(state: MockTrialState) -> Command[str]:
    """[민사] 변론 (민사소송법 §134~§148) — 2-3 라운드 루프"""
    current_round = state.get("current_round", 1)
    max_rounds = state.get("max_rounds", 3)

    interrupt_value = interrupt({
        "response": f"[변론 라운드 {current_round}/{max_rounds}] 주장을 입력하세요.",
        "stage": "argument",
        "actions": [{"type": "button", "label": "변론 종결 요청", "action": "end_argument"}],
    })
    user_input = str(interrupt_value)

    if user_input == "end_argument" or current_round >= max_rounds:
        return Command(
            update={"stage": "argument", "current_round": current_round},
            goto="civil_closing_node",
        )

    # AI 반론 생성
    opponent = _get_agent(state, _get_opponent_role(state))
    rebuttal = await opponent.generate("argument", state["case_summary"], state["court_record"])
    _record(state, "argument", _get_opponent_role(state), rebuttal)

    return Command(
        update={"current_round": current_round + 1, "response": rebuttal},
        goto="argument_node",  # 자기 자신으로 루프
    )


# ── 공통 노드 ──

def evidence_node(state: MockTrialState) -> Command[str]:
    """[공통] 증거조사 (형사: §290~§313 / 민사: §288~§344)"""
    # RAG 검색
    cases = await _search_cases(state["case_summary"])
    articles = await _search_articles(state["case_summary"])

    interrupt_value = interrupt({
        "response": f"증거조사를 시작합니다. {len(cases)}건의 관련 판례, {len(articles)}건의 관련 법령이 검색되었습니다.",
        "stage": "evidence",
        "evidence_cases": cases,
        "evidence_articles": articles,
        "actions": [{"type": "button", "label": "증거 제출 완료", "action": "submit_evidence"}],
    })
    selected = _parse_evidence_selection(interrupt_value)

    next_node = "examination_node" if state["case_type"] == "criminal" else "argument_node"
    return Command(
        update={
            "stage": "evidence",
            "evidence_cases": cases,
            "evidence_articles": articles,
            "selected_evidence": selected,
        },
        goto=next_node,
    )


def verdict_node(state: MockTrialState) -> Command[str]:
    """[공통] 판결선고"""
    judge = _get_agent(state, "judge")
    judgment = await judge.generate("verdict", state["case_summary"], state["court_record"])
    feedback = _generate_feedback(state)
    _record(state, "verdict", "judge", judgment)

    return Command(
        update={
            "stage": "verdict",
            "judgment": judgment,
            "feedback": feedback,
            "is_complete": True,
            "response": f"[판결]\n{judgment}\n\n[피드백]\n{feedback}",
            "speaking_agent": "judge",
        },
        goto=END,
    )


def build_mock_trial_subgraph() -> CompiledStateGraph:
    """모의재판 서브그래프 빌드

    Returns:
        컴파일된 모의재판 서브그래프
    """
    builder = StateGraph(MockTrialState)

    # 공통 노드
    builder.add_node("setup_node", setup_node)
    builder.add_node("evidence_node", evidence_node)
    builder.add_node("verdict_node", verdict_node)

    # 형사 전용 노드
    builder.add_node("identity_node", identity_node)
    builder.add_node("opening_node", opening_node)
    builder.add_node("examination_node", examination_node)
    builder.add_node("criminal_closing_node", criminal_closing_node)

    # 민사 전용 노드
    builder.add_node("pretrial_node", pretrial_node)
    builder.add_node("claims_node", claims_node)
    builder.add_node("argument_node", argument_node)
    builder.add_node("civil_closing_node", civil_closing_node)

    # 엣지
    builder.add_edge(START, "setup_node")
    # setup_node → identity_node / pretrial_node (Command로 분기)
    # 중간 노드 → 다음 노드 (모두 Command(goto=...)로 라우팅)
    builder.add_edge("verdict_node", END)

    return builder.compile()
```

### 6.4 기존 시스템 통합 변경점

#### router.py 추가 항목

```python
# AgentType enum 추가
class AgentType(str, Enum):
    # ... (기존)
    MOCK_TRIAL = "mock_trial"

# ROLE_AGENTS 추가 (user, lawyer 모두 접근 가능)
ROLE_AGENTS[UserRole.USER].append(AgentType.MOCK_TRIAL)
ROLE_AGENTS[UserRole.LAWYER].append(AgentType.MOCK_TRIAL)

# INTENT_PATTERNS 추가
INTENT_PATTERNS[AgentType.MOCK_TRIAL] = [
    ("모의재판", 0.95),
    ("모의 재판", 0.95),
    ("모의법정", 0.95),
    ("모의 법정", 0.95),
    ("재판 시뮬", 0.9),
    ("법정 체험", 0.9),
    ("법정 시뮬", 0.9),
    ("재판 연습", 0.85),
    ("재판 게임", 0.85),
    ("법정 게임", 0.85),
]
```

#### nodes.py 추가 항목

```python
# AGENT_NODE_MAP 추가
AGENT_NODE_MAP["mock_trial"] = "mock_trial_subgraph"
```

#### graph.py 추가 항목

```python
from app.multi_agent.subgraphs.mock_trial import build_mock_trial_subgraph

# build_graph() 내
builder.add_node("mock_trial_subgraph", build_mock_trial_subgraph())

# END 엣지 목록에 추가
for node_name in (..., "mock_trial_subgraph"):
    builder.add_edge(node_name, END)
```

### 6.5 RAG 검색 인터페이스

```python
# services/service_function/mock_trial_service.py

from typing import Any, Protocol


class EvidenceSearcher(Protocol):
    """모의재판 증거 검색 인터페이스 (기존 RAG와 독립)"""

    async def search_cases(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """판례 검색"""
        ...

    async def search_articles(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """법령 검색"""
        ...


class LanceDBEvidenceSearcher:
    """기존 LanceDB RAG를 활용한 구현체"""

    async def search_cases(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        from app.services.rag.pipeline import search_pipeline
        results = await search_pipeline(query, search_type="precedent", top_k=limit)
        return [
            {
                "id": r.get("id", ""),
                "title": r.get("title", ""),
                "summary": r.get("content", "")[:300],
                "relevance_score": r.get("score", 0.0),
                "source": "lancedb",
            }
            for r in results
        ]

    async def search_articles(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        from app.services.rag.pipeline import search_pipeline
        results = await search_pipeline(query, search_type="law", top_k=limit)
        return [
            {
                "id": r.get("id", ""),
                "title": r.get("title", ""),
                "content": r.get("content", "")[:300],
                "relevance_score": r.get("score", 0.0),
                "source": "lancedb",
            }
            for r in results
        ]
```

---

## 7. Frontend 상세 설계

### 7.1 Phaser.js + Next.js 통합 패턴

```typescript
// features/mock-trial/components/MockTrialGame.tsx

'use client'

import { useEffect, useRef, useState } from 'react'
import type { Game as PhaserGame } from 'phaser'

export function MockTrialGame() {
  const gameRef = useRef<PhaserGame | null>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [isLoaded, setIsLoaded] = useState(false)

  useEffect(() => {
    // dynamic import로 SSR 회피
    const initGame = async () => {
      const Phaser = (await import('phaser')).default
      const { CourtScene } = await import('../game/CourtScene')
      const { LobbyScene } = await import('../game/LobbyScene')

      if (containerRef.current && !gameRef.current) {
        gameRef.current = new Phaser.Game({
          type: Phaser.AUTO,
          parent: containerRef.current,
          width: 800,
          height: 480,
          pixelArt: true,         // 픽셀아트 선명하게
          roundPixels: true,
          scene: [LobbyScene, CourtScene],
          physics: { default: 'arcade' },
          scale: {
            mode: Phaser.Scale.FIT,
            autoCenter: Phaser.Scale.CENTER_BOTH,
          },
        })
        setIsLoaded(true)
      }
    }

    initGame()

    return () => {
      gameRef.current?.destroy(true)
      gameRef.current = null
    }
  }, [])

  return <div ref={containerRef} style={{ width: '100%', maxWidth: 800 }} />
}
```

### 7.2 EventBus 설계

```typescript
// features/mock-trial/game/EventBus.ts

type EventMap = {
  // Phaser → React
  'agent:speak': { agent: string; text: string; streaming: boolean }
  'stage:change': { from: string; to: string; stageNumber: number; totalStages: number }
  'evidence:presented': { cases: EvidenceItem[]; articles: EvidenceItem[] }
  'trial:complete': { judgment: string; feedback: string }
  'game:ready': {}

  // React → Phaser
  'user:input': { text: string }
  'user:select_evidence': { evidenceIds: string[] }
  'game:advance_stage': {}
  'agent:animate': { agent: string; animation: 'idle' | 'speak' | 'react' }
  'setup:complete': { caseType: string; userRole: string; caseSummary: string }
}

class CourtEventBus {
  private target = new EventTarget()

  emit<K extends keyof EventMap>(event: K, data: EventMap[K]): void {
    this.target.dispatchEvent(new CustomEvent(event, { detail: data }))
  }

  on<K extends keyof EventMap>(event: K, handler: (data: EventMap[K]) => void): () => void {
    const listener = (e: Event) => handler((e as CustomEvent).detail)
    this.target.addEventListener(event, listener)
    return () => this.target.removeEventListener(event, listener)
  }
}

export const eventBus = new CourtEventBus()
```

### 7.3 TypeScript 타입 정의

```typescript
// features/mock-trial/types/index.ts

export type CaseType = 'criminal' | 'civil'

export type CriminalCategory =
  | 'criminal_assault'
  | 'criminal_fraud'
  | 'criminal_theft'
  | 'criminal_embezzlement'
  | 'criminal_other'

export type CivilCategory =
  | 'civil_damages'
  | 'civil_contract'
  | 'civil_property'
  | 'civil_other'

export type CaseCategory = CriminalCategory | CivilCategory

export type CriminalRole = 'prosecutor' | 'attorney'
export type CivilRole = 'plaintiff' | 'defendant'
export type UserRole = CriminalRole | CivilRole

export type CriminalStage =
  | 'setup' | 'identity' | 'opening' | 'evidence'
  | 'examination' | 'closing' | 'verdict'

export type CivilStage =
  | 'setup' | 'pretrial' | 'claims' | 'evidence'
  | 'argument' | 'closing' | 'verdict'

export type TrialStage = CriminalStage | CivilStage

export interface EvidenceItem {
  id: string
  title: string
  summary: string
  relevance_score: number
  source: string
}

export interface CourtEvent {
  stage: string
  speaker: string
  content: string
  timestamp: string
}

export interface TrialSetup {
  case_type: CaseType
  case_category: CaseCategory
  user_role: UserRole
  case_summary: string
}

export interface JudgmentResult {
  judgment: string
  feedback: string
  cited_cases: EvidenceItem[]
  cited_articles: EvidenceItem[]
}

export interface StageInfo {
  id: string
  name: string
  order: number
  legal_basis: string
  description: string
  user_action: string
  duration_hint: string
}

// 사건 유형 옵션
export const CASE_TYPE_OPTIONS = [
  { id: 'criminal' as const, name: '형사 재판', icon: '⚖️' },
  { id: 'civil' as const, name: '민사 재판', icon: '📜' },
]

export const CRIMINAL_CATEGORIES: { id: CriminalCategory; name: string; description: string }[] = [
  { id: 'criminal_assault', name: '폭행/상해', description: '폭행죄, 상해죄 등' },
  { id: 'criminal_fraud', name: '사기', description: '사기죄, 횡령죄 등' },
  { id: 'criminal_theft', name: '절도', description: '절도죄, 강도죄 등' },
  { id: 'criminal_embezzlement', name: '횡령/배임', description: '횡령죄, 배임죄 등' },
  { id: 'criminal_other', name: '기타', description: '기타 형사 사건' },
]

export const CIVIL_CATEGORIES: { id: CivilCategory; name: string; description: string }[] = [
  { id: 'civil_damages', name: '손해배상', description: '불법행위, 채무불이행 등' },
  { id: 'civil_contract', name: '계약 분쟁', description: '계약 해제, 이행 청구 등' },
  { id: 'civil_property', name: '부동산', description: '임대차, 소유권 분쟁 등' },
  { id: 'civil_other', name: '기타', description: '기타 민사 사건' },
]

export const CRIMINAL_STAGES: StageInfo[] = [
  { id: 'identity', name: '인정신문', order: 1, legal_basis: '형사소송법 §284', description: '피고인 인적사항 확인, 진술거부권 고지', user_action: '자동 진행', duration_hint: '1-2분' },
  { id: 'opening', name: '모두진술', order: 2, legal_basis: '형사소송법 §285~§286', description: '검사 공소사실, 피고인 의견 진술', user_action: '역할에 따라 진술 입력', duration_hint: '3-5분' },
  { id: 'evidence', name: '증거조사', order: 3, legal_basis: '형사소송법 §290~§313', description: '판례/법령 검색, 증거 제출', user_action: '증거 선택/제출', duration_hint: '5-10분' },
  { id: 'examination', name: '피고인신문', order: 4, legal_basis: '형사소송법 §296-2', description: '검사/변호인이 피고인에게 질문', user_action: '질문 입력', duration_hint: '3-5분' },
  { id: 'closing', name: '최종변론', order: 5, legal_basis: '형사소송법 §302~§303', description: '검사 구형, 변호인 최후변론, 피고인 최후진술', user_action: '최후변론 입력', duration_hint: '3-5분' },
  { id: 'verdict', name: '판결선고', order: 6, legal_basis: '형사소송법 §318-4', description: 'AI 판사 판결문 낭독', user_action: '관전', duration_hint: '2-3분' },
]

export const CIVIL_STAGES: StageInfo[] = [
  { id: 'pretrial', name: '변론준비', order: 1, legal_basis: '민사소송법 §258~§268', description: '쟁점 정리, 증거 목록 확인', user_action: '자동 진행', duration_hint: '1-2분' },
  { id: 'claims', name: '주장/답변', order: 2, legal_basis: '민사소송법 §256~§257', description: '원고 청구원인, 피고 답변', user_action: '역할에 따라 입력', duration_hint: '3-5분' },
  { id: 'evidence', name: '증거조사', order: 3, legal_basis: '민사소송법 §288~§344', description: '판례/법령 검색, 서증 제출', user_action: '증거 선택/제출', duration_hint: '5-10분' },
  { id: 'argument', name: '변론', order: 4, legal_basis: '민사소송법 §134~§148', description: '양측 주장/반박 교환', user_action: '주장 입력 (2-3 라운드)', duration_hint: '5-10분' },
  { id: 'closing', name: '변론종결', order: 5, legal_basis: '민사소송법 §200', description: '양측 최종 주장 정리', user_action: '최종 주장 입력', duration_hint: '2-3분' },
  { id: 'verdict', name: '판결선고', order: 6, legal_basis: '민사소송법 §206~§208', description: 'AI 판사 판결문 낭독', user_action: '관전', duration_hint: '2-3분' },
]
```

### 7.4 API 서비스

```typescript
// features/mock-trial/services/index.ts

import { api, endpoints } from '@/lib/api'
import type {
  CaseType,
  EvidenceItem,
  StageInfo,
} from '../types'

export const mockTrialService = {
  getCaseTypes: async () => {
    const response = await api.get(`${endpoints.mockTrial}/case-types`)
    return response.data
  },

  getRoles: async (caseType: CaseType) => {
    const response = await api.get(`${endpoints.mockTrial}/roles/${caseType}`)
    return response.data
  },

  searchEvidence: async (
    query: string,
    searchType: 'all' | 'cases' | 'articles' = 'all',
    limit: number = 5
  ): Promise<{ cases: EvidenceItem[]; articles: EvidenceItem[] }> => {
    const response = await api.post(`${endpoints.mockTrial}/search-evidence`, {
      query,
      search_type: searchType,
      limit,
    })
    return response.data
  },

  getStageInfo: async (caseType: CaseType): Promise<{ stages: StageInfo[] }> => {
    const response = await api.get(`${endpoints.mockTrial}/stage-info/${caseType}`)
    return response.data
  },
}
```

### 7.5 모듈/API 등록

```typescript
// lib/modules.ts 추가 항목
{
  id: 'mock-trial',
  name: '모의 법정',
  description: '픽셀아트 법정에서 AI 에이전트와 함께하는 모의재판 시뮬레이션',
  href: '/mock-trial',
  icon: '🏛️',
  enabled: true,
  roles: ['user', 'lawyer'],
}

// lib/api.ts 추가 항목
export const endpoints = {
  // ... (기존)
  mockTrial: '/mock-trial',
}
```

---

## 8. Error Handling

### 8.1 Error Code Definition

| Code | Message | Cause | Handling |
|------|---------|-------|----------|
| 400 | Invalid setup | 사건유형/역할 입력 오류 | 설정 단계 재시도 |
| 404 | Stage not found | 잘못된 단계 전환 | 현재 단계 유지 |
| 408 | LLM timeout | AI 에이전트 응답 지연 | 재시도 버튼 표시 |
| 422 | Evidence search failed | RAG 검색 실패 | 수동 검색 안내 |
| 500 | Internal error | 서버 오류 | 에러 로깅 + "재시작" 버튼 |

### 8.2 Phaser.js 에러 처리

| Scenario | Handling |
|----------|----------|
| 에셋 로드 실패 | 플레이스홀더 이미지 표시 + 콘솔 경고 |
| Canvas 렌더링 실패 | React fallback UI (텍스트 기반 재판) |
| EventBus 이벤트 유실 | 메시지 큐 + 재전송 메커니즘 |
| 브라우저 미지원 | WebGL 미지원 안내 + Canvas 2D 폴백 |

---

## 9. Security Considerations

- [x] 면책 고지 상시 표시 ("실제 법률 자문이 아닙니다") — FR-28
- [x] 사용자 입력 검증 (XSS 방지: 말풍선 텍스트 이스케이프)
- [x] 사건 개요 입력 길이 제한 (500자)
- [x] LLM 프롬프트 인젝션 방지 (시스템 프롬프트에 역할 고정 지시)
- [x] RAG 결과만 인용 (환각 방지: "검색 결과에 없는 판례를 인용하지 마세요")
- [ ] Rate Limiting: 에이전트별 LLM 호출 횟수 제한 (세션당 최대 50회)

---

## 10. Test Plan

### 10.1 Test Scope

| Type | Target | Tool |
|------|--------|------|
| Unit Test | CourtAgent.generate(), Stage 노드 함수 | pytest |
| Unit Test | EventBus 이벤트 발행/수신 | Vitest |
| Integration Test | 서브그래프 전체 흐름 (setup → verdict) | pytest + LangGraph |
| Integration Test | /api/mock-trial 엔드포인트 | pytest + httpx |
| E2E Test | 로비→설정→재판→판결 전체 흐름 | Playwright |

### 10.2 Test Cases (Key)

- [ ] Happy path: 형사 재판 전체 6단계 완주 (setup → verdict)
- [ ] Happy path: 민사 재판 전체 6단계 완주 (setup → verdict)
- [ ] 형사/민사 분기: case_type="criminal" → identity_node로 라우팅
- [ ] 형사/민사 분기: case_type="civil" → pretrial_node로 라우팅
- [ ] 민사 변론 라운드: max_rounds 도달 시 자동 종결
- [ ] 민사 변론 조기 종결: 사용자 "변론 종결 요청" 시 closing으로 이동
- [ ] CourtAgent reflection: 단계 전환 시 short_term → long_term 이동
- [ ] RAG 검색: evidence_node에서 판례/법령 검색 결과 반환
- [ ] 에러: LLM 타임아웃 시 재시도 안내
- [ ] EventBus: Phaser→React agent:speak 이벤트 전달
- [ ] 면책 고지: 모든 화면에서 표시 확인

---

## 11. Clean Architecture

### 11.1 Layer Structure

| Layer | Responsibility | Location |
|-------|---------------|----------|
| **Presentation** | Phaser.js 게임 씬, React 오버레이 UI, 페이지 | `features/mock-trial/game/`, `features/mock-trial/components/`, `app/mock-trial/` |
| **Application** | 재판 상태 관리, API 서비스, EventBus 통신 | `features/mock-trial/services/`, `features/mock-trial/hooks/`, `features/mock-trial/game/EventBus.ts` |
| **Domain** | 타입 정의, 단계 상수, 사건 유형 정의 | `features/mock-trial/types/` |
| **Infrastructure** | API 클라이언트, Phaser.js 엔진 | `lib/api.ts`, `phaser` (npm) |

### 11.2 This Feature's Layer Assignment

| Component | Layer | Location |
|-----------|-------|----------|
| CourtScene, LobbyScene | Presentation | `features/mock-trial/game/` |
| MockTrialGame, ChatPanel | Presentation | `features/mock-trial/components/` |
| mockTrialService | Application | `features/mock-trial/services/` |
| useTrialState (향후) | Application | `features/mock-trial/hooks/` |
| EventBus | Application | `features/mock-trial/game/EventBus.ts` |
| types, stages 상수 | Domain | `features/mock-trial/types/` |
| api.ts endpoints | Infrastructure | `lib/api.ts` |

---

## 12. Implementation Guide

### 12.1 File Structure (최종)

```
# Backend
backend/app/
├── multi_agent/
│   ├── agents/mock_trial_agent.py          # MockTrialAgent (BaseChatAgent 상속)
│   ├── subgraphs/
│   │   ├── mock_trial.py                   # MockTrialState + 서브그래프 빌드
│   │   ├── mock_trial_agents.py            # CourtAgent 클래스
│   │   └── mock_trial_prompts.py           # 에이전트별 시스템 프롬프트
│   ├── router.py                           # +AgentType.MOCK_TRIAL
│   ├── nodes.py                            # +AGENT_NODE_MAP["mock_trial"]
│   └── graph.py                            # +add_node("mock_trial_subgraph")
├── modules/mock_trial/
│   ├── __init__.py
│   ├── router/__init__.py                  # /api/mock-trial 엔드포인트
│   └── schema/__init__.py                  # Pydantic 스키마
└── services/service_function/
    └── mock_trial_service.py               # RAG 검색 인터페이스 + 피드백 생성

# Frontend
frontend/src/
├── app/mock-trial/page.tsx                 # 페이지 엔트리 (dynamic import)
├── features/mock-trial/
│   ├── game/                               # Phaser.js 게임
│   │   ├── CourtScene.ts                   # 메인 법정 씬
│   │   ├── LobbyScene.ts                   # 로비/설정 씬
│   │   ├── config.ts                       # Phaser 게임 설정
│   │   ├── sprites/
│   │   │   ├── CharacterBase.ts            # 캐릭터 베이스 클래스
│   │   │   └── characters.ts              # 5종 캐릭터 설정
│   │   ├── ui/
│   │   │   ├── SpeechBubble.ts             # 말풍선 (9-patch, 타이핑)
│   │   │   └── StageIndicator.ts           # 단계 표시 바
│   │   └── EventBus.ts                     # Phaser ↔ React 통신
│   ├── components/
│   │   ├── MockTrialGame.tsx               # Phaser 게임 래퍼
│   │   ├── MockTrialSetup.tsx              # 설정 UI
│   │   ├── ChatPanel.tsx                   # 하단 대화 패널
│   │   ├── EvidencePanel.tsx               # 증거 목록 패널
│   │   ├── StageProgress.tsx               # 단계 진행률
│   │   ├── JudgmentDisplay.tsx             # 판결문 모달
│   │   └── DisclaimerBanner.tsx            # 면책 고지
│   ├── services/index.ts                   # API 서비스
│   └── types/index.ts                      # TypeScript 타입
├── public/assets/mock-trial/               # 픽셀아트 에셋
│   ├── tilemap/courtroom.json              # Tiled 타일맵
│   ├── tilemap/courtroom_tiles.png         # 타일셋 이미지
│   ├── sprites/judge.png                   # 판사 스프라이트시트
│   ├── sprites/prosecutor.png              # 검사
│   ├── sprites/attorney.png                # 변호사
│   ├── sprites/defendant.png               # 피고인
│   ├── sprites/clerk.png                   # 서기
│   └── ui/speech_bubble.png                # 말풍선 9-patch
├── lib/modules.ts                          # +mock-trial 모듈 등록
└── lib/api.ts                              # +mockTrial 엔드포인트
```

### 12.2 Implementation Order

| # | 작업 | 파일 | 의존성 | 난이도 |
|---|------|------|--------|--------|
| 1 | 픽셀아트 에셋 확보 (타일맵 + 스프라이트) | `public/assets/mock-trial/` | - | Medium |
| 2 | TypeScript 타입 + 상수 정의 | `types/index.ts` | - | Low |
| 3 | Backend 스키마 정의 (Pydantic) | `modules/mock_trial/schema/` | - | Low |
| 4 | CourtAgent 클래스 구현 | `subgraphs/mock_trial_agents.py` | #3 | Medium |
| 5 | 에이전트 프롬프트 정의 | `subgraphs/mock_trial_prompts.py` | #4 | Low |
| 6 | 서브그래프 구현 (형사 6단계 + 민사 6단계) | `subgraphs/mock_trial.py` | #4, #5 | High |
| 7 | RAG 검색 서비스 구현 | `services/mock_trial_service.py` | #6 | Medium |
| 8 | 기존 시스템 통합 (router, nodes, graph) | `router.py`, `nodes.py`, `graph.py` | #6 | Low |
| 9 | 모듈 라우터 구현 | `modules/mock_trial/router/` | #7 | Low |
| 10 | Phaser.js 설치 + Next.js 통합 | `package.json`, `MockTrialGame.tsx` | #1 | Medium |
| 11 | EventBus 구현 | `game/EventBus.ts` | #10 | Low |
| 12 | 법정 씬 구현 (타일맵 + 캐릭터) | `game/CourtScene.ts` | #1, #10 | High |
| 13 | 말풍선 + 캐릭터 애니메이션 | `game/ui/SpeechBubble.ts`, sprites | #12 | Medium |
| 14 | React 오버레이 UI | `components/` | #2, #11 | Medium |
| 15 | API 서비스 + Frontend 통합 | `services/index.ts`, `page.tsx` | #9, #14 | Medium |
| 16 | 모듈 등록 + 정적 검증 | `modules.ts`, `api.ts` | #15 | Low |
| 17 | 테스트 작성 | `tests/` | #16 | Medium |

### 12.3 추가 의존성

| Package | Version | Location | Purpose |
|---------|---------|----------|---------|
| `phaser` | `^3.80` | Frontend (npm) | 2D 게임 엔진 |

Backend는 추가 의존성 없음 (기존 LangGraph, Solar LLM, LanceDB 활용).

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-02-12 | Initial design document — Plan v0.3 기반 상세 설계 | Claude |
