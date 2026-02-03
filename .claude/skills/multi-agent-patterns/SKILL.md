---
name: multi-agent-patterns
description: 멀티 에이전트 시스템 구현 패턴. BaseAgent 상속, 오케스트레이터, Intent 라우팅, 세션 관리 등. 에이전트 추가, 수정, 라우팅 로직 작업 시 사용.
---

# Multi-Agent Patterns

멀티 에이전트 시스템 구현을 위한 패턴과 가이드라인.

## 1. 아키텍처 개요

```
사용자 메시지
     │
     ▼
┌─────────────────┐
│  AgentRouter    │  ← Intent 감지
│  (detect_intent)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ AgentOrchestrator│  ← 에이전트 선택 & 실행
└────────┬────────┘
         │
    ┌────┴────┬────────────┐
    ▼         ▼            ▼
┌───────┐ ┌───────┐ ┌───────────┐
│Agent A│ │Agent B│ │Agent C    │
└───────┘ └───────┘ └───────────┘
```

## 2. 핵심 클래스

### 2.1 위치
```
backend/app/common/
├── agent_base.py      # BaseAgent, AgentResponse, ChatAction
├── agent_router.py    # AgentType, detect_intent, INTENT_PATTERNS
└── llm/               # get_chat_model

backend/app/modules/multi_agent/
├── service/__init__.py           # AgentOrchestrator
└── agents/
    ├── __init__.py               # Export all agents
    ├── case_search_agent.py
    ├── small_claims_agent.py
    └── lawyer_finder_agent.py
```

### 2.2 Import 패턴
```python
# 에이전트 구현 시
from app.common.agent_base import AgentResponse, BaseAgent, ChatAction, ActionType
from app.common.agent_router import AgentType, UserRole

# 오케스트레이터 사용 시
from app.modules.multi_agent.service import get_orchestrator
```

## 3. BaseAgent 상속 패턴

### 3.1 기본 구조
```python
from typing import Any

from app.common.agent_base import AgentResponse, BaseAgent


class MyAgent(BaseAgent):
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
    ) -> AgentResponse:
        """
        메시지 처리 메인 로직

        Args:
            message: 사용자 메시지
            history: 대화 기록 [{"role": "user/assistant", "content": "..."}]
            session_data: 세션 상태 (에이전트별 상태 저장)
            user_location: 위치 정보 {"latitude": float, "longitude": float}

        Returns:
            AgentResponse
        """
        # 구현
        pass

    def can_handle(self, message: str) -> bool:
        """키워드 기반 처리 가능 여부 (선택적 오버라이드)"""
        keywords = ["키워드1", "키워드2"]
        return any(kw in message for kw in keywords)
```

### 3.2 AgentResponse 구조
```python
class AgentResponse(BaseModel):
    message: str                        # AI 응답 메시지
    sources: list[dict[str, Any]] = []  # 참조 문서 (판례, 법령 등)
    actions: list[ChatAction] = []      # 액션 버튼
    session_data: dict[str, Any] = {}   # 다음 턴에 전달할 세션 데이터
```

### 3.3 ChatAction 패턴
```python
from app.common.agent_base import ActionType, ChatAction

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
# backend/app/common/agent_router.py

class AgentType(str, Enum):
    """에이전트 타입"""
    LAWYER_FINDER = "lawyer_finder"
    CASE_SEARCH = "case_search"
    SMALL_CLAIMS = "small_claims"
    # 새 에이전트 추가 시 여기에 추가
    MY_AGENT = "my_agent"
```

### 4.2 Intent 패턴 등록
```python
# backend/app/common/agent_router.py

INTENT_PATTERNS: dict[AgentType, list[str]] = {
    AgentType.MY_AGENT: [
        "키워드1",
        "키워드2",
        "특정 표현",
    ],
    # ...
}
```

### 4.3 역할별 접근 제어
```python
# backend/app/common/agent_router.py

class UserRole(str, Enum):
    USER = "user"      # 일반 사용자
    LAWYER = "lawyer"  # 변호사

ROLE_AGENTS: dict[UserRole, list[AgentType]] = {
    UserRole.USER: [
        AgentType.LAWYER_FINDER,
        AgentType.CASE_SEARCH,
        AgentType.SMALL_CLAIMS,
    ],
    UserRole.LAWYER: [
        AgentType.CASE_ANALYSIS,
        AgentType.DOCUMENT_DRAFTING,
        AgentType.CASE_SEARCH,
    ],
}
```

### 4.4 detect_intent 동작
```python
def detect_intent(
    message: str,
    user_role: UserRole = UserRole.USER,
    session_data: Optional[dict[str, Any]] = None,
) -> AgentType:
    """
    우선순위:
    1. 진행 중인 세션의 active_agent 유지
    2. 키워드 기반 Intent 매칭
    3. 기본 에이전트 (USER: CASE_SEARCH, LAWYER: CASE_ANALYSIS)
    """
```

## 5. 오케스트레이터

### 5.1 구조
```python
# backend/app/modules/multi_agent/service/__init__.py

class AgentOrchestrator:
    """에이전트 오케스트레이터"""

    def __init__(self) -> None:
        self._agents: dict[AgentType, BaseAgent] = {
            AgentType.CASE_SEARCH: CaseSearchAgent(),
            AgentType.LAWYER_FINDER: LawyerFinderAgent(),
            AgentType.SMALL_CLAIMS: SmallClaimsAgent(),
            # 새 에이전트 등록
        }

    async def process(
        self,
        message: str,
        user_role: str = "user",
        history: Optional[list[dict[str, str]]] = None,
        session_data: Optional[dict[str, Any]] = None,
        user_location: Optional[dict[str, float]] = None,
    ) -> tuple[AgentResponse, str]:
        """
        Returns:
            (AgentResponse, agent_name)
        """
```

### 5.2 싱글톤 패턴
```python
_orchestrator: Optional[AgentOrchestrator] = None

def get_orchestrator() -> AgentOrchestrator:
    """오케스트레이터 싱글톤 인스턴스 반환"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
    return _orchestrator
```

### 5.3 사용 예시 (Router)
```python
from app.modules.multi_agent.service import get_orchestrator

@router.post("/chat")
async def chat(request: ChatRequest):
    orchestrator = get_orchestrator()
    response, agent_name = await orchestrator.process(
        message=request.message,
        user_role=request.user_role,
        history=request.history,
        session_data=request.session_data,
        user_location=request.user_location,
    )
    return {
        "message": response.message,
        "sources": response.sources,
        "actions": [a.model_dump() for a in response.actions],
        "session_data": response.session_data,
        "agent": agent_name,
    }
```

## 6. 세션 상태 관리

### 6.1 세션 데이터 패턴
```python
async def process(self, message: str, ..., session_data: dict[str, Any] | None = None) -> AgentResponse:
    session_data = session_data or {}

    # 현재 단계 읽기
    current_step = session_data.get("step", "init")

    # 새 세션 데이터 구성
    new_session = {
        **session_data,  # 기존 데이터 유지
        "active_agent": self.name,  # 필수: 다음 턴에서 같은 에이전트 유지
        "step": next_step,  # 상태 업데이트
        "collected_data": {...},  # 수집한 정보
    }

    return AgentResponse(
        message="...",
        session_data=new_session,
    )
```

### 6.2 단계별 상태 머신
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

### 6.3 세션 초기화 (리셋)
```python
# 프론트엔드에서 reset 액션 처리 시
if request.action == "reset_session":
    session_data = {}  # 세션 초기화
```

## 7. 새 에이전트 추가 체크리스트

### Step 1: 에이전트 클래스 생성
```bash
# 파일 생성
backend/app/modules/multi_agent/agents/my_agent.py
```

### Step 2: AgentType 추가
```python
# backend/app/common/agent_router.py
class AgentType(str, Enum):
    MY_AGENT = "my_agent"
```

### Step 3: Intent 패턴 추가
```python
# backend/app/common/agent_router.py
INTENT_PATTERNS: dict[AgentType, list[str]] = {
    AgentType.MY_AGENT: ["키워드1", "키워드2"],
}
```

### Step 4: 역할 접근 권한 추가
```python
# backend/app/common/agent_router.py
ROLE_AGENTS: dict[UserRole, list[AgentType]] = {
    UserRole.USER: [..., AgentType.MY_AGENT],
}
```

### Step 5: 오케스트레이터에 등록
```python
# backend/app/modules/multi_agent/service/__init__.py
from app.modules.multi_agent.agents import MyAgent

class AgentOrchestrator:
    def __init__(self):
        self._agents = {
            AgentType.MY_AGENT: MyAgent(),
        }
```

### Step 6: Export 추가
```python
# backend/app/modules/multi_agent/agents/__init__.py
from .my_agent import MyAgent

__all__ = [..., "MyAgent"]
```

## 8. 에이전트 구현 패턴

### 8.1 RAG 연동 에이전트
```python
from app.common.chat_service import generate_chat_response, search_relevant_documents

class RagAgent(BaseAgent):
    async def process(self, message: str, history: list | None = None, ...) -> AgentResponse:
        # RAG 서비스 호출
        result = generate_chat_response(
            user_message=message,
            chat_history=history,
            n_context_docs=5,
        )

        return AgentResponse(
            message=result["response"],
            sources=result["sources"],
            session_data={"active_agent": self.name},
        )
```

### 8.2 단계별 가이드 에이전트
```python
STEP_MESSAGES = {
    "init": "첫 번째 안내 메시지...",
    "step_1": "두 번째 단계 안내...",
    "step_2": "세 번째 단계 안내...",
}

class GuideAgent(BaseAgent):
    async def process(self, message: str, ..., session_data: dict | None = None) -> AgentResponse:
        session_data = session_data or {}
        step = session_data.get("step", "init")

        # 단계별 처리
        response_msg = STEP_MESSAGES.get(step, STEP_MESSAGES["init"])
        next_step = self._determine_next_step(step, message)

        return AgentResponse(
            message=response_msg,
            actions=self._get_actions_for_step(next_step),
            session_data={
                "active_agent": self.name,
                "step": next_step,
            },
        )
```

### 8.3 외부 API 연동 에이전트
```python
import httpx

class ExternalApiAgent(BaseAgent):
    async def process(self, message: str, ...) -> AgentResponse:
        # 비동기 HTTP 호출
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.example.com/search",
                params={"query": message},
            )
            data = response.json()

        return AgentResponse(
            message=self._format_response(data),
            sources=data.get("sources", []),
            session_data={"active_agent": self.name},
        )
```

## 9. 테스트 패턴

### 9.1 에이전트 단위 테스트
```python
import pytest
from app.modules.multi_agent.agents import MyAgent

@pytest.mark.asyncio
async def test_my_agent_process():
    agent = MyAgent()
    response = await agent.process(
        message="테스트 메시지",
        history=None,
        session_data=None,
    )

    assert response.message
    assert response.session_data.get("active_agent") == "my_agent"

def test_my_agent_can_handle():
    agent = MyAgent()
    assert agent.can_handle("키워드1 포함 메시지")
    assert not agent.can_handle("관련 없는 메시지")
```

### 9.2 오케스트레이터 통합 테스트
```python
@pytest.mark.asyncio
async def test_orchestrator_routing():
    orchestrator = get_orchestrator()
    response, agent_name = await orchestrator.process(
        message="변호사 찾아줘",
        user_role="user",
    )

    assert agent_name == "lawyer_finder"
```

## 10. 주의사항

### 10.1 세션 데이터
- `active_agent`는 반드시 설정 (다음 턴 라우팅에 사용)
- 민감 정보는 세션에 저장하지 않음
- 세션 크기가 커지지 않도록 주의

### 10.2 비동기 처리
- `process` 메서드는 `async def`로 정의
- 외부 API 호출 시 `httpx.AsyncClient` 사용
- 동기 함수 호출 시 `asyncio.to_thread()` 활용

### 10.3 에러 처리
```python
async def process(self, message: str, ...) -> AgentResponse:
    try:
        # 메인 로직
        result = await self._do_something(message)
    except ExternalServiceError as e:
        logger.error(f"외부 서비스 오류: {e}")
        return AgentResponse(
            message="죄송합니다. 일시적인 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            session_data={"active_agent": self.name},
        )
```
