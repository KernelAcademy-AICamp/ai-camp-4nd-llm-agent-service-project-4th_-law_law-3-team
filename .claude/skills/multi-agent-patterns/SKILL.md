---
name: multi-agent-patterns
description: LangGraph 기반 멀티 에이전트 시스템 구현 패턴. BaseChatAgent 상속, StateGraph 라우팅, Command 기반 노드 이동, 세션 관리 등. 에이전트 추가, 수정, 라우팅 로직 작업 시 사용.
---

# Multi-Agent Patterns (LangGraph)

LangGraph StateGraph 기반 멀티 에이전트 시스템 구현 패턴과 가이드라인.

## 1. 아키텍처 개요

```
사용자 메시지 (POST /api/chat)
     │
     ▼
┌─────────────────────┐
│  request_to_state() │  ← ChatRequest → ChatState 변환
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  LangGraph          │  ← graph.ainvoke(state) / graph.astream(state)
│  StateGraph         │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  router_node        │  ← RulesRouter.route() → Command(goto=target_node)
│                     │     키워드 매칭 + confidence 점수 + ROLE_AGENTS 검증
└────────┬────────────┘
         │
    ┌────┴────┬──────────┬───────────┬───────────┬──────────┬──────────┐
    ▼         ▼          ▼           ▼           ▼          ▼          ▼
┌────────┐┌────────┐┌─────────┐┌──────────┐┌────────┐┌────────┐┌────────┐
│Legal   ││Lawyer  ││Small    ││Storyboard││Lawyer  ││Law     ││Simple  │
│Search  ││Finder  ││Claims   ││          ││Stats   ││Study   ││Chat    │
│Node    ││Node    ││Subgraph ││Node      ││Node    ││Node    ││Node    │
└────────┘└────────┘└─────────┘└──────────┘└────────┘└────────┘└────────┘
    │         │          │           │           │          │          │
    └─────────┴──────────┴───────────┴───────────┴──────────┴──────────┘
                                     │
                                     ▼
                                    END
```

## 2. 핵심 클래스

### 2.1 위치
```
backend/app/multi_agent/
├── __init__.py              # 패키지 export (get_graph, ChatState 등)
├── graph.py                 # build_graph(), get_graph() 싱글톤
├── nodes.py                 # router_node + 에이전트 노드 함수 + AGENT_NODE_MAP
├── router.py                # RulesRouter, AgentType, UserRole, ROLE_AGENTS, detect_search_type
├── state.py                 # ChatState(TypedDict), request_to_state, state_to_response
├── agents/
│   ├── __init__.py          # Export all agents
│   ├── base_chat.py         # BaseChatAgent, SimpleChatAgent, ActionType, ChatAction
│   ├── legal_search_agent.py    # LegalSearchAgent (RAG 기반, search_focus)
│   ├── lawyer_finder_agent.py   # LawyerFinderAgent
│   ├── small_claims_agent.py    # SmallClaimsAgent
│   ├── storyboard_agent.py      # StoryboardAgent (LLM 타임라인)
│   ├── lawyer_stats_agent.py    # LawyerStatsAgent (통계 안내)
│   └── law_study_agent.py       # LawStudyAgent (학습 가이드)
├── subgraphs/
│   └── small_claims.py      # 소액소송 interrupt 서브그래프
└── schemas/
    ├── __init__.py          # Export all schemas
    ├── plan.py              # AgentPlan, AgentResult (dataclass)
    └── messages.py          # ChatMessage, ChatRequest, ChatResponse (Pydantic)
```

### 2.2 Import 패턴
```python
# 에이전트 구현 시
from app.multi_agent.agents.base_chat import BaseChatAgent, ActionType, ChatAction
from app.multi_agent.schemas.plan import AgentResult

# 라우터 사용 시
from app.multi_agent.router import AgentType, UserRole, RulesRouter, detect_search_type

# 그래프 사용 시
from app.multi_agent.graph import get_graph, build_graph

# State 사용 시
from app.multi_agent.state import ChatState, request_to_state, state_to_response

# 노드 함수 사용 시
from app.multi_agent.nodes import router_node, AGENT_NODE_MAP

# 스키마 사용 시
from app.multi_agent.schemas import AgentPlan, AgentResult, ChatRequest, ChatResponse, ChatMessage
```

## 3. BaseChatAgent 상속 패턴

### 3.1 기본 구조
```python
from typing import Any
from app.multi_agent.agents.base_chat import BaseChatAgent
from app.multi_agent.schemas.plan import AgentResult


class MyAgent(BaseChatAgent):
    """에이전트 설명 - 한 줄로"""

    @property
    def name(self) -> str:
        """에이전트 고유 식별자 (snake_case)"""
        return "my_agent"

    @property
    def description(self) -> str:
        """에이전트 역할 설명"""
        return "특정 기능을 수행하는 에이전트"

    async def process(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        session_data: dict[str, Any] | None = None,
        user_location: dict[str, float] | None = None,
    ) -> AgentResult:
        """
        메시지 처리 메인 로직 (비스트리밍)

        Args:
            message: 사용자 메시지
            history: 대화 기록 [{"role": "user/assistant", "content": "..."}]
            session_data: 세션 상태 (에이전트별 상태 저장)
            user_location: 위치 정보 {"latitude": float, "longitude": float}

        Returns:
            AgentResult (dataclass)
        """
        # 구현
        pass

    async def process_stream(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        session_data: dict[str, Any] | None = None,
        user_location: dict[str, float] | None = None,
    ):
        """
        스트리밍 메시지 처리 (AsyncGenerator)

        Yields:
            str 또는 dict (토큰 또는 이벤트)
        """
        # 구현 (스트리밍 에이전트만)
        pass
```

### 3.2 AgentResult 구조 (dataclass)
```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class AgentResult:
    """각 에이전트 노드가 반환하여 state_to_response()로 변환"""
    message: str                                            # AI 응답 메시지
    sources: list[dict[str, Any]] = field(default_factory=list)  # 참조 문서
    actions: list[dict[str, Any]] = field(default_factory=list)  # 액션 버튼
    session_data: dict[str, Any] = field(default_factory=dict)   # 다음 턴 세션
    agent_used: str | None = None                           # 사용된 에이전트명
    confidence: float = 1.0                                 # 라우팅 신뢰도
    processing_time_ms: float | None = None                 # 처리 시간
```

### 3.3 ChatAction 패턴
```python
from app.multi_agent.agents.base_chat import ActionType, ChatAction

# 버튼 액션
ChatAction(
    type=ActionType.BUTTON,
    label="다음 단계",
    action="next_step",  # 프론트엔드에서 처리할 액션 ID
)

# 외부 링크
ChatAction(
    type=ActionType.LINK,
    label="전자소송 바로가기",
    url="https://ecfs.scourt.go.kr",
)

# 위치 요청
ChatAction(
    type=ActionType.REQUEST_LOCATION,
    label="내 위치로 검색",
)

# 페이지 이동 (쿼리 파라미터 포함)
ChatAction(
    type=ActionType.NAVIGATE,
    label="검색 결과 보기",
    params={"query": "손해배상", "filter": "precedent"},
)
```

## 4. Intent 라우팅

### 4.1 AgentType 정의
```python
# backend/app/multi_agent/router.py

class AgentType(str, Enum):
    """에이전트 타입"""
    # 일반인 전용
    LAWYER_FINDER = "lawyer_finder"
    SMALL_CLAIMS = "small_claims"
    # 공통
    LEGAL_SEARCH = "legal_search"       # 판례+법령 통합
    CASE_SEARCH = "case_search"         # 판례 명시 (agent_override용)
    LAW_SEARCH = "law_search"           # 법령 명시 (agent_override용)
    STORYBOARD = "storyboard"
    # 변호사 전용
    LAWYER_STATS = "lawyer_stats"
    LAW_STUDY = "law_study"
    # 폴백
    GENERAL = "general"
```

### 4.2 Intent 패턴 등록
```python
# backend/app/multi_agent/router.py

# 키워드 + confidence 점수 쌍
INTENT_PATTERNS: dict[AgentType, list[tuple[str, float]]] = {
    AgentType.LAWYER_FINDER: [
        ("변호사 찾", 0.9),
        ("변호사 추천", 0.9),
        ("근처 변호사", 0.85),
    ],
    AgentType.SMALL_CLAIMS: [
        ("소액소송", 0.95),
        ("소액심판", 0.9),
    ],
    AgentType.STORYBOARD: [
        ("타임라인", 0.9),
        ("스토리보드", 0.95),
        ("사건 경위", 0.85),
    ],
    AgentType.LAWYER_STATS: [
        ("변호사 통계", 0.95),
        ("변호사 현황", 0.9),
    ],
    AgentType.LAW_STUDY: [
        ("로스쿨", 0.9),
        ("법학", 0.8),
    ],
    # 새 에이전트 추가 시 여기에 패턴 추가
}
```

### 4.3 역할별 접근 제어
```python
# backend/app/multi_agent/router.py

class UserRole(str, Enum):
    USER = "user"      # 일반 사용자
    LAWYER = "lawyer"  # 변호사

ROLE_AGENTS: dict[UserRole, list[AgentType]] = {
    UserRole.USER: [
        AgentType.LAWYER_FINDER,
        AgentType.SMALL_CLAIMS,
        AgentType.LEGAL_SEARCH,
        AgentType.CASE_SEARCH,
        AgentType.LAW_SEARCH,
        AgentType.STORYBOARD,
        AgentType.GENERAL,
    ],
    UserRole.LAWYER: [
        AgentType.LAWYER_STATS,
        AgentType.LAW_STUDY,
        AgentType.LEGAL_SEARCH,
        AgentType.CASE_SEARCH,
        AgentType.LAW_SEARCH,
        AgentType.STORYBOARD,
        AgentType.GENERAL,
    ],
}
```

### 4.4 RulesRouter 동작
```python
class RulesRouter:
    """규칙 기반 라우터"""

    def route(
        self,
        message: str,
        user_role: str = "user",
        session_data: dict[str, Any] | None = None,
    ) -> AgentPlan:
        """
        우선순위:
        1. 진행 중인 세션의 active_agent 유지
        2. 키워드 기반 Intent 매칭 (confidence 점수)
        3. 역할별 기본 에이전트
        """
```

### 4.5 search_focus 결정 (detect_search_type)
```python
# backend/app/multi_agent/router.py

def detect_search_type(message: str) -> str:
    """메시지에서 법령/판례 검색 타입을 분류한다."""
    law_keywords = ["법령", "법률", "조문", "시행령", "시행규칙", "법 제"]
    if any(kw in message for kw in law_keywords):
        return "law"
    return "precedent"  # 기본값: 판례
```

`router_node`에서 search_focus 결정 로직:
- `agent_override=case_search` → `search_focus="precedent"`
- `agent_override=law_search` → `search_focus="law"`
- `legal_search`로 라우팅 → `detect_search_type(message)`로 2차 분류
- 결정된 값은 `ChatState.search_focus`에 저장

## 5. LangGraph StateGraph

### 5.1 그래프 구조
```python
# backend/app/multi_agent/graph.py

def build_graph() -> StateGraph:
    builder = StateGraph(ChatState)

    # 노드 등록
    builder.add_node("router_node", router_node)
    builder.add_node("legal_search_node", legal_search_node)
    builder.add_node("lawyer_finder_node", lawyer_finder_node)
    builder.add_node("small_claims_subgraph", build_small_claims_subgraph())
    builder.add_node("storyboard_node", storyboard_node)
    builder.add_node("lawyer_stats_node", lawyer_stats_node)
    builder.add_node("law_study_node", law_study_node)
    builder.add_node("simple_chat_node", simple_chat_node)

    # 엣지
    builder.add_edge(START, "router_node")
    # router_node는 Command(goto=...)로 라우팅하므로 conditional edge 불필요
    for node in ("legal_search_node", "lawyer_finder_node", "small_claims_subgraph",
                 "storyboard_node", "lawyer_stats_node", "law_study_node", "simple_chat_node"):
        builder.add_edge(node, END)

    return builder
```

### 5.2 싱글톤 패턴
```python
from app.multi_agent.graph import get_graph

graph = get_graph()  # 컴파일된 StateGraph 싱글톤 (InMemorySaver 체크포인터)
```

### 5.3 사용 예시 (API Router)
```python
from app.multi_agent.graph import get_graph
from app.multi_agent.state import request_to_state, state_to_response
from app.multi_agent.schemas import ChatRequest, ChatResponse

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    graph = get_graph()
    initial_state = request_to_state(request)
    final_state = await graph.ainvoke(initial_state, config={"configurable": {"thread_id": thread_id}})
    return state_to_response(final_state)
```

### 5.4 AGENT_NODE_MAP (에이전트→노드 매핑)
```python
# backend/app/multi_agent/nodes.py

AGENT_NODE_MAP: dict[str, str] = {
    # 공통
    "legal_search": "legal_search_node",
    "case_search": "legal_search_node",   # agent_override 호환
    "law_search": "legal_search_node",    # agent_override 호환
    "storyboard": "storyboard_node",
    # 일반인
    "lawyer_finder": "lawyer_finder_node",
    "small_claims": "small_claims_subgraph",
    # 변호사
    "lawyer_stats": "lawyer_stats_node",
    "law_study": "law_study_node",
    # 폴백
    "general": "simple_chat_node",
    # 하위호환
    "legal_answer": "legal_search_node",
}
```

### 5.5 router_node (Command 기반 라우팅)
```python
# backend/app/multi_agent/nodes.py

def router_node(state: ChatState) -> Command:
    """라우터 노드 — RulesRouter로 에이전트 결정, Command(goto=...)로 이동"""
    message = state["message"]
    user_role = state.get("user_role", "user")
    agent_override = state.get("agent_override")

    # 1. agent_override가 있으면 직접 지정 (ROLE_AGENTS 검증)
    # 2. 없으면 RulesRouter.route()로 결정
    # 3. search_focus 결정 (case_search→precedent, law_search→law, legal_search→detect_search_type)

    return Command(
        update={
            "selected_agent": plan.agent_type,
            "search_focus": search_focus,
            "routing_confidence": plan.confidence,
            "routing_reason": plan.reason or "",
        },
        goto=target_node,
    )
```

### 5.6 노드 함수 헬퍼
```python
# backend/app/multi_agent/nodes.py

async def _run_streaming_node(
    state: ChatState,
    writer: StreamWriter,
    agent: BaseChatAgent,
) -> dict[str, Any]:
    """스트리밍 에이전트 노드 공통 로직"""
    # process_stream() → writer로 토큰 전송 → state 업데이트 반환

async def _run_nonstreaming_node(
    state: ChatState,
    writer: StreamWriter,
    agent: BaseChatAgent,
) -> dict[str, Any]:
    """비스트리밍 에이전트 노드 공통 로직"""
    # process() → 결과를 한번에 writer로 전송 → state 업데이트 반환
```

## 6. ChatState (TypedDict)

### 6.1 구조
```python
# backend/app/multi_agent/state.py

class ChatState(TypedDict):
    # 입력
    message: str
    user_role: str
    history: list[dict[str, str]]
    session_data: dict[str, Any]
    user_location: dict[str, float] | None
    agent_override: str | None
    # 라우팅 결과 (router_node가 설정)
    selected_agent: str
    search_focus: str           # "precedent" | "law" | ""
    routing_confidence: float
    routing_reason: str
    # 출력 (에이전트 노드가 설정)
    response: str
    sources: list[dict[str, Any]]
    actions: list[dict[str, Any]]
    output_session_data: dict[str, Any]
    agent_used: str
```

### 6.2 변환 함수
```python
def request_to_state(request: ChatRequest) -> ChatState:
    """ChatRequest → ChatState 변환"""

def state_to_response(state: ChatState) -> ChatResponse:
    """ChatState → ChatResponse 변환"""
```

## 7. 세션 상태 관리

### 7.1 세션 데이터 패턴
```python
async def process(
    self, message: str, ..., session_data: dict[str, Any] | None = None,
) -> AgentResult:
    session_data = session_data or {}

    # 현재 단계 읽기
    current_step = session_data.get("step", "init")

    # 새 세션 데이터 구성
    new_session = {
        **session_data,              # 기존 데이터 유지
        "active_agent": self.name,   # 필수: 다음 턴에서 같은 에이전트 유지
        "step": next_step,           # 상태 업데이트
        "collected_data": {...},     # 수집한 정보
    }

    return AgentResult(
        message="...",
        session_data=new_session,
    )
```

### 7.2 단계별 상태 머신
```python
class MyAgentStep:
    """에이전트 진행 단계"""
    INIT = "init"
    GATHER_INFO = "gather_info"
    PROCESS = "process"
    COMPLETE = "complete"

async def process(self, ...):
    current_step = session_data.get("step", MyAgentStep.INIT)

    if current_step == MyAgentStep.INIT:
        # 초기 단계 처리
        new_session["step"] = MyAgentStep.GATHER_INFO

    elif current_step == MyAgentStep.GATHER_INFO:
        # 정보 수집 단계
        if has_enough_info:
            new_session["step"] = MyAgentStep.PROCESS

    # ...
```

## 8. 새 에이전트 추가 체크리스트

### Step 1: 에이전트 클래스 생성
```bash
# 파일 생성
backend/app/multi_agent/agents/my_agent.py
```

### Step 2: AgentType 추가
```python
# backend/app/multi_agent/router.py
class AgentType(str, Enum):
    MY_AGENT = "my_agent"
```

### Step 3: Intent 패턴 추가
```python
# backend/app/multi_agent/router.py
INTENT_PATTERNS[AgentType.MY_AGENT] = [
    ("키워드1", 0.9),
    ("키워드2", 0.85),
]
```

### Step 4: ROLE_AGENTS에 추가
```python
# backend/app/multi_agent/router.py
# 해당 역할의 ROLE_AGENTS 리스트에 추가
ROLE_AGENTS[UserRole.USER].append(AgentType.MY_AGENT)
```

### Step 5: AGENT_NODE_MAP에 매핑 추가
```python
# backend/app/multi_agent/nodes.py
AGENT_NODE_MAP["my_agent"] = "my_agent_node"
```

### Step 6: 노드 함수 작성
```python
# backend/app/multi_agent/nodes.py

async def my_agent_node(state: ChatState, writer: StreamWriter) -> dict[str, Any]:
    """내 에이전트 노드"""
    from app.multi_agent.agents.my_agent import MyAgent
    agent = MyAgent()
    # 스트리밍이면 _run_streaming_node, 비스트리밍이면 _run_nonstreaming_node 사용
    return await _run_streaming_node(state, writer, agent)
```

### Step 7: graph.py에 노드 등록
```python
# backend/app/multi_agent/graph.py
builder.add_node("my_agent_node", my_agent_node)
builder.add_edge("my_agent_node", END)
```

### Step 8: Export 추가
```python
# backend/app/multi_agent/agents/__init__.py
from app.multi_agent.agents.my_agent import MyAgent

__all__ = [..., "MyAgent"]
```

## 9. 에이전트 구현 패턴

### 9.1 RAG 연동 에이전트 (스트리밍)
```python
from app.services.rag.retrieval import search_relevant_documents
from app.tools.llm import get_chat_model
from app.multi_agent.agents.base_chat import BaseChatAgent
from app.multi_agent.schemas.plan import AgentResult


class RagAgent(BaseChatAgent):
    async def process_stream(self, message: str, history: list | None = None, ...):
        # 1. 문서 검색
        docs = search_relevant_documents(message, n_results=5)

        # 2. 컨텍스트 구성 + LLM 스트리밍 호출
        llm = get_chat_model(temperature=0.7)
        async for chunk in llm.astream(messages):
            yield chunk.content

    async def process(self, message: str, history: list | None = None, ...) -> AgentResult:
        # 비스트리밍 폴백
        chunks = []
        async for chunk in self.process_stream(message, history, ...):
            chunks.append(chunk)

        return AgentResult(
            message="".join(chunks),
            sources=[...],
            session_data={"active_agent": self.name},
        )
```

### 9.2 단계별 가이드 에이전트 (비스트리밍)
```python
STEP_MESSAGES = {
    "init": "첫 번째 안내 메시지...",
    "step_1": "두 번째 단계 안내...",
    "step_2": "세 번째 단계 안내...",
}


class GuideAgent(BaseChatAgent):
    async def process(
        self, message: str, ..., session_data: dict | None = None,
    ) -> AgentResult:
        session_data = session_data or {}
        step = session_data.get("step", "init")

        # 단계별 처리
        response_msg = STEP_MESSAGES.get(step, STEP_MESSAGES["init"])
        next_step = self._determine_next_step(step, message)

        return AgentResult(
            message=response_msg,
            actions=self._get_actions_for_step(next_step),
            session_data={
                "active_agent": self.name,
                "step": next_step,
            },
        )
```

### 9.3 외부 API 연동 에이전트
```python
import httpx
from app.multi_agent.agents.base_chat import BaseChatAgent
from app.multi_agent.schemas.plan import AgentResult


class ExternalApiAgent(BaseChatAgent):
    async def process(self, message: str, ...) -> AgentResult:
        # 비동기 HTTP 호출
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.example.com/search",
                params={"query": message},
            )
            data = response.json()

        return AgentResult(
            message=self._format_response(data),
            sources=data.get("sources", []),
            session_data={"active_agent": self.name},
        )
```

## 10. 테스트 패턴

### 10.1 에이전트 단위 테스트
```python
import pytest
from app.multi_agent.agents import MyAgent

@pytest.mark.asyncio
async def test_my_agent_process():
    agent = MyAgent()
    result = await agent.process(
        message="테스트 메시지",
        history=None,
        session_data=None,
    )

    assert result.message
    assert result.session_data.get("active_agent") == "my_agent"

def test_my_agent_can_handle():
    agent = MyAgent()
    assert agent.can_handle("키워드1 포함 메시지")
    assert not agent.can_handle("관련 없는 메시지")
```

### 10.2 라우팅 테스트
```python
from app.multi_agent.nodes import router_node, AGENT_NODE_MAP
from app.multi_agent.state import ChatState

def test_router_my_agent_keyword():
    """키워드 매칭 → 올바른 노드로 라우팅"""
    state: ChatState = {
        "message": "키워드1 포함 메시지",
        "user_role": "user",
        "history": [],
        "session_data": {},
        "user_location": None,
        "agent_override": None,
        "selected_agent": "",
        "search_focus": "",
        "routing_confidence": 0.0,
        "routing_reason": "",
        "response": "",
        "sources": [],
        "actions": [],
        "output_session_data": {},
        "agent_used": "",
    }
    result = router_node(state)
    assert result.goto == "my_agent_node"

def test_agent_override_respects_role():
    """agent_override 시 ROLE_AGENTS 검증"""
    state = {
        ...,
        "user_role": "user",
        "agent_override": "lawyer_stats",
    }
    result = router_node(state)
    assert result.goto != "lawyer_stats_node"  # user에게 차단
```

### 10.3 그래프 구조 테스트
```python
from app.multi_agent.graph import build_graph

def test_graph_has_all_nodes():
    """모든 노드가 등록되어 있는지 확인"""
    builder = build_graph()
    compiled = builder.compile()
    node_names = set(compiled.nodes.keys())
    expected = {
        "router_node", "legal_search_node", "lawyer_finder_node",
        "small_claims_subgraph", "storyboard_node", "lawyer_stats_node",
        "law_study_node", "simple_chat_node",
    }
    assert expected.issubset(node_names)
```

## 11. 주의사항

### 11.1 세션 데이터
- `active_agent`는 반드시 설정 (다음 턴 라우팅에 사용)
- 민감 정보는 세션에 저장하지 않음
- 세션 크기가 커지지 않도록 주의

### 11.2 비동기 처리
- `process` 메서드는 `async def`로 정의
- 외부 API 호출 시 `httpx.AsyncClient` 사용
- 동기 함수 호출 시 `asyncio.to_thread()` 활용

### 11.3 에러 처리
```python
async def process(self, message: str, ...) -> AgentResult:
    try:
        # 메인 로직
        result = await self._do_something(message)
    except ExternalServiceError as e:
        logger.error(f"외부 서비스 오류: {e}")
        return AgentResult(
            message="죄송합니다. 일시적인 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            session_data={"active_agent": self.name},
        )
```

### 11.4 Command 기반 라우팅
- `router_node`는 `Command(goto=target_node, update={...})` 반환
- `conditional_edge` 대신 `Command`를 사용하여 동적 라우팅
- 에이전트 노드는 일반 `dict` 반환 (state 업데이트)

### 11.5 스트리밍 vs 비스트리밍
- **스트리밍**: `process_stream()` + `_run_streaming_node()` (LegalSearch, Storyboard, LawStudy, SimpleChat)
- **비스트리밍**: `process()` + `_run_nonstreaming_node()` (LawyerFinder, LawyerStats)
- StreamWriter로 SSE 이벤트 전송: `writer({"type": "token", "content": chunk})`
