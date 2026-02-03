---
name: multi-agent-patterns
description: 멀티 에이전트 시스템 구현 패턴. BaseChatAgent 상속, 오케스트레이터, Intent 라우팅, 세션 관리 등. 에이전트 추가, 수정, 라우팅 로직 작업 시 사용.
---

# Multi-Agent Patterns

멀티 에이전트 시스템 구현을 위한 패턴과 가이드라인.

## 1. 아키텍처 개요

```
사용자 메시지 (POST /api/chat)
     │
     ▼
┌─────────────────┐
│  Orchestrator   │  ← process(ChatRequest) → ChatResponse
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  RouterAgent    │  ← route(ChatContext) → AgentPlan
│  (RulesRouter)  │     키워드 매칭 + confidence 점수
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ AgentExecutor   │  ← execute(AgentPlan, ChatContext) → AgentResult
└────────┬────────┘
         │
    ┌────┴────┬────────────┬──────────────┐
    ▼         ▼            ▼              ▼
┌────────┐ ┌────────┐ ┌───────────┐ ┌──────────┐
│Legal   │ │Lawyer  │ │SmallClaims│ │Simple    │
│Answer  │ │Finder  │ │Agent      │ │ChatAgent │
└────────┘ └────────┘ └───────────┘ └──────────┘
```

## 2. 핵심 클래스

### 2.1 위치
```
backend/app/multi_agent/
├── __init__.py              # 패키지 export (Orchestrator, AgentExecutor 등)
├── orchestrator.py          # Orchestrator (process → route → execute)
├── executor.py              # AgentExecutor (에이전트 선택/실행)
├── router.py                # RouterAgent, RulesRouter, AgentType, UserRole
├── agents/
│   ├── __init__.py          # Export all agents
│   ├── base_chat.py         # BaseChatAgent, SimpleChatAgent, ActionType, ChatAction
│   ├── legal_answer_agent.py    # LegalAnswerAgent (RAG 기반)
│   ├── lawyer_finder_agent.py   # LawyerFinderAgent
│   └── small_claims_agent.py    # SmallClaimsAgent
└── schemas/
    ├── __init__.py          # Export all schemas
    ├── plan.py              # AgentPlan, AgentResult (dataclass)
    └── messages.py          # ChatMessage, ChatRequest, ChatResponse (Pydantic)
```

### 2.2 Import 패턴
```python
# 에이전트 구현 시
from app.multi_agent.agents.base_chat import BaseChatAgent, ActionType, ChatAction
from app.multi_agent.schemas.plan import AgentPlan, AgentResult

# 라우터 사용 시
from app.multi_agent.router import AgentType, UserRole, RulesRouter, RouterAgent

# 오케스트레이터 사용 시
from app.multi_agent import get_orchestrator
from app.multi_agent.orchestrator import Orchestrator

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
        메시지 처리 메인 로직

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

    def can_handle(self, message: str) -> bool:
        """키워드 기반 처리 가능 여부 (선택적 오버라이드, 기본: False)"""
        keywords = ["키워드1", "키워드2"]
        return any(kw in message for kw in keywords)
```

### 3.2 AgentResult 구조 (dataclass)
```python
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class AgentResult:
    """에이전트가 반환하여 Orchestrator가 응답으로 변환"""
    message: str                                            # AI 응답 메시지
    sources: List[Dict[str, Any]] = field(default_factory=list)  # 참조 문서
    actions: List[Dict[str, Any]] = field(default_factory=list)  # 액션 버튼
    session_data: Dict[str, Any] = field(default_factory=dict)   # 다음 턴 세션
    agent_used: Optional[str] = None                        # 사용된 에이전트명
    confidence: float = 1.0                                 # 라우팅 신뢰도
    processing_time_ms: Optional[float] = None              # 처리 시간
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
    LAWYER_FINDER = "lawyer_finder"
    CASE_SEARCH = "case_search"
    SMALL_CLAIMS = "small_claims"
    # 변호사 전용
    CASE_ANALYSIS = "case_analysis"
    CLIENT_MANAGEMENT = "client_management"
    DOCUMENT_DRAFTING = "document_drafting"
    # 기본
    GENERAL = "general"
```

### 4.2 Intent 패턴 등록
```python
# backend/app/multi_agent/router.py

# 키워드 + confidence 점수 쌍
INTENT_PATTERNS: Dict[AgentType, List[tuple[str, float]]] = {
    AgentType.LAWYER_FINDER: [
        ("변호사 찾", 0.9),
        ("변호사 추천", 0.9),
        ("근처 변호사", 0.85),
        # ...
    ],
    AgentType.SMALL_CLAIMS: [
        ("소액소송", 0.95),
        ("소액심판", 0.9),
        # ...
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

ROLE_AGENTS: Dict[UserRole, List[AgentType]] = {
    UserRole.USER: [
        AgentType.LAWYER_FINDER,
        AgentType.CASE_SEARCH,
        AgentType.SMALL_CLAIMS,
        AgentType.GENERAL,
    ],
    UserRole.LAWYER: [
        AgentType.CASE_ANALYSIS,
        AgentType.CLIENT_MANAGEMENT,
        AgentType.DOCUMENT_DRAFTING,
        AgentType.CASE_SEARCH,
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
        session_data: Optional[Dict[str, Any]] = None,
    ) -> AgentPlan:
        """
        우선순위:
        1. 진행 중인 세션의 active_agent 유지
        2. 키워드 기반 Intent 매칭 (confidence 점수)
        3. 역할별 기본 에이전트
        """
```

### 4.5 RouterAgent (하이브리드 라우터)
```python
class RouterAgent:
    """
    규칙 기반 라우팅 + LLM 기반 보완 (임계값 0.6)
    """
    LLM_ROUTING_THRESHOLD = 0.6

    def route(self, context: ChatContext) -> AgentPlan:
        """
        1. RulesRouter로 라우팅
        2. confidence >= 0.6이면 반환
        3. 미만이면 LLM 라우팅 (향후 확장)
        """

    def route_simple(
        self,
        message: str,
        user_role: str = "user",
        session_data: Optional[Dict[str, Any]] = None,
    ) -> AgentPlan:
        """ChatContext 없이 간단 라우팅"""
```

## 5. 오케스트레이터

### 5.1 구조
```python
# backend/app/multi_agent/orchestrator.py

class Orchestrator:
    """멀티 에이전트 오케스트레이터 - 라우팅 → 실행 → 응답 변환 관리"""

    def __init__(
        self,
        router: Optional[RouterAgent] = None,
        executor: Optional[AgentExecutor] = None,
        session_store: Optional[SessionStore] = None,
    ) -> None:
        # Lazy 초기화: 각 컴포넌트는 처음 접근 시 생성

    async def process(self, request: ChatRequest) -> ChatResponse:
        """
        전체 플로우:
        1. ChatRequest → ChatContext 변환
        2. request.agent 있으면 직접 지정, 아니면 RouterAgent로 라우팅
        3. AgentExecutor로 에이전트 실행
        4. 세션 데이터 저장
        5. AgentResult → ChatResponse 변환
        """
```

### 5.2 싱글톤 패턴
```python
from app.multi_agent import get_orchestrator

orchestrator = get_orchestrator()  # 싱글톤 인스턴스

# 하위 호환 별칭
AgentOrchestrator = Orchestrator
```

### 5.3 사용 예시 (API Router)
```python
from app.multi_agent import get_orchestrator
from app.multi_agent.schemas import ChatRequest, ChatResponse

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    orchestrator = get_orchestrator()
    response = await orchestrator.process(request)
    return response
```

### 5.4 AgentExecutor (에이전트 레지스트리)
```python
# backend/app/multi_agent/executor.py

class AgentExecutor:
    """에이전트 실행기 - AgentPlan을 받아 해당 에이전트 실행"""

    def __init__(self, agents: Dict[str, BaseChatAgent]):
        """agents: agent_name → agent_instance 매핑"""

    async def execute(self, plan: AgentPlan, context: ChatContext) -> AgentResult:
        """
        1. plan.agent_type으로 에이전트 선택
        2. agent.process() 실행
        3. 처리 시간 기록
        4. AgentResult 반환
        """

# 기본 에이전트 레지스트리 (get_agent_executor에서 생성):
# {
#     "legal_answer": LegalAnswerAgent(focus="precedent"),
#     "case_search":  LegalAnswerAgent(focus="precedent"),  # 하위 호환
#     "law_search":   LegalAnswerAgent(focus="law"),
#     "lawyer_finder": LawyerFinderAgent(),
#     "small_claims":  SmallClaimsAgent(),
#     "general":       SimpleChatAgent(),
# }
```

## 6. 스키마

### 6.1 AgentPlan (dataclass)
```python
@dataclass
class AgentPlan:
    """라우터가 생성하여 Executor에 전달"""
    agent_type: str              # "legal_answer", "lawyer_finder" 등
    use_rag: bool = True         # RAG 사용 여부
    confidence: float = 1.0      # 라우팅 신뢰도
    reason: Optional[str] = None # 라우팅 사유
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### 6.2 ChatRequest / ChatResponse (Pydantic)
```python
class ChatRequest(BaseModel):
    """POST /api/chat 요청"""
    message: str
    user_role: str = "user"
    history: List[ChatMessage] = []
    session_data: Dict[str, Any] = {}
    user_location: Optional[Dict[str, float]] = None  # {"latitude", "longitude"}
    agent: Optional[str] = None  # 에이전트 직접 지정 (라우팅 건너뜀)

class ChatResponse(BaseModel):
    """API 응답"""
    response: str
    agent_used: str
    sources: List[Dict[str, Any]] = []
    actions: List[Dict[str, Any]] = []
    session_data: Dict[str, Any] = {}
    confidence: float = 1.0
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
INTENT_PATTERNS: Dict[AgentType, List[tuple[str, float]]] = {
    AgentType.MY_AGENT: [
        ("키워드1", 0.9),
        ("키워드2", 0.85),
    ],
}
```

### Step 4: 에이전트 레지스트리에 등록
```python
# backend/app/multi_agent/executor.py
# get_agent_executor() 함수 내 agents dict에 추가
from app.multi_agent.agents.my_agent import MyAgent

agents = {
    # ... 기존 에이전트 ...
    "my_agent": MyAgent(),
}
```

### Step 5: Export 추가
```python
# backend/app/multi_agent/agents/__init__.py
from app.multi_agent.agents.my_agent import MyAgent

__all__ = [..., "MyAgent"]
```

## 9. 에이전트 구현 패턴

### 9.1 RAG 연동 에이전트
```python
from app.services.rag.retrieval import search_relevant_documents
from app.tools.llm import get_chat_model
from app.multi_agent.agents.base_chat import BaseChatAgent
from app.multi_agent.schemas.plan import AgentResult


class RagAgent(BaseChatAgent):
    async def process(self, message: str, history: list | None = None, ...) -> AgentResult:
        # 1. 문서 검색
        docs = search_relevant_documents(message, n_results=5)

        # 2. 컨텍스트 구성 + LLM 호출
        llm = get_chat_model(temperature=0.7)
        response = llm.invoke(messages)

        return AgentResult(
            message=response.content,
            sources=[...],
            session_data={"active_agent": self.name},
        )
```

### 9.2 단계별 가이드 에이전트
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

### 10.2 오케스트레이터 통합 테스트
```python
from app.multi_agent import get_orchestrator
from app.multi_agent.schemas import ChatRequest

@pytest.mark.asyncio
async def test_orchestrator_routing():
    orchestrator = get_orchestrator()
    response = await orchestrator.process(
        ChatRequest(message="변호사 찾아줘", user_role="user")
    )

    assert response.agent_used == "lawyer_finder"
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
