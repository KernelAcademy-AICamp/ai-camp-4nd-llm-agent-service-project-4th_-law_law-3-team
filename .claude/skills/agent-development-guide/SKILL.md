---
name: agent-development-guide
description: 멀티 에이전트 개발 시 반드시 따라야 할 규칙과 패턴. RAG 파이프라인(basic/focus), format_utils, LLM 호출, 스트리밍 프로토콜, 서비스 의존성 규칙. 새 에이전트 구현, 기존 에이전트 수정, RAG 연동 작업 시 사용. 구조/등록/라우팅은 `multi-agent-patterns` 스킬 참조.
---

# Agent Development Guide

에이전트 구현 시 반드시 따라야 할 규칙과 패턴.
구조/등록/라우팅은 `multi-agent-patterns` 스킬, 이 스킬은 **구현 규칙**에 집중.

---

## 1. 에이전트 유형 결정

에이전트 구현 전 아래 표로 유형을 먼저 결정한다.

| 유형 | RAG | LLM | 스트리밍 | 예시 |
|------|:---:|:---:|:-------:|------|
| **RAG + LLM** | O | O | O | LegalSearchAgent, LawStudyAgent |
| **Pure LLM** | X | O | O | StoryboardAgent, SimpleChatAgent |
| **상태 머신** | 보조 | X | X | SmallClaimsAgent |
| **서비스 직접** | X | X | X | LawyerFinderAgent, LawyerStatsAgent |

**규칙**: LLM을 사용하는 에이전트는 반드시 `supports_streaming = True`로 스트리밍 지원.

---

## 2. RAG 파이프라인 규칙 (필수)

### 2.1 호출 방식: search_with_pipeline_async 통일

RAG를 사용하는 에이전트는 **반드시** `search_with_pipeline_async()`를 사용한다.

```python
# 올바른 호출
from app.services.rag.pipeline import search_with_pipeline_async, PRESETS

result = await search_with_pipeline_async(query, PRESETS["my_preset"])
```

**금지 패턴:**
```python
# 직접 검색 함수 호출 금지
from app.services.rag.retrieval import search_relevant_documents  # 금지
from app.services.rag.retrieval import search_relevant_documents_async  # 금지

# asyncio.to_thread 래핑 금지
docs = await asyncio.to_thread(search_relevant_documents, ...)  # 금지
```

### 2.2 검색 모드: basic vs focus

| 모드 | 용도 | PipelineResult |
|------|------|---------------|
| **basic** | 전체 문서에서 단일 검색 | `result.documents` |
| **focus** | 주 타입 + 보충 타입 병렬 검색 | `result.documents` + `result.supplementary_documents` |

**basic 모드** (기본값): 대부분의 에이전트에 적합
```python
config = PipelineConfig(
    n_results=10,
    enable_rerank=True,
    rerank_top_k=5,
)
result = await search_with_pipeline_async(query, config)
documents = result.documents  # 검색 결과
```

**focus 모드**: 특정 타입 위주 검색 + 보충 자료가 필요할 때
```python
config = PipelineConfig(
    search_type="focus",
    n_results=15,
    doc_type="precedent",           # 주 타입
    enable_rewrite=True,            # 리라이팅 1회 (상위에서만)
    enable_rerank=True,
    rerank_top_k=5,
    supplementary_config=PipelineConfig(
        n_results=7,
        exclude_doc_types=["판례"],  # 주 타입 제외
        enable_rerank=True,
        rerank_top_k=3,
        # enable_rewrite는 무시됨 — 상위 config의 리라이팅 결과 공유
    ),
)
result = await search_with_pipeline_async(query, config)
focus_docs = result.documents                    # 주 타입 결과
supplementary_docs = result.supplementary_documents or []  # 보충 결과
```

**focus 모드 내부 동작:**
```
쿼리 → [리라이팅 1회] → 리라이팅된 쿼리
                          ├── Focus 검색 (병렬)
                          └── Supplementary 검색 (병렬)
                                   ↓ asyncio.gather
              PipelineResult (documents + supplementary_documents)
```

### 2.3 PRESETS 사용 (권장)

에이전트별 파이프라인 설정은 `pipeline.py`의 PRESETS에 중앙화한다.

```python
# backend/app/services/rag/pipeline.py

PRESETS: dict[str, PipelineConfig] = {
    # Focus 모드
    "legal_search_precedent": PipelineConfig(search_type="focus", ...),
    "legal_search_law": PipelineConfig(search_type="focus", ...),
    # Basic 모드
    "legal_search_all": PipelineConfig(n_results=20, ...),
    "law_study": PipelineConfig(n_results=10, ...),
    "small_claims": PipelineConfig(n_results=10, doc_type="precedent", ...),
    "quick_search": PipelineConfig(n_results=5, enable_rerank=False),
}
```

**새 프리셋 추가 시:**
1. `pipeline.py`의 `PRESETS` 딕셔너리에 추가
2. `__init__.py`에서 `PRESETS`가 이미 export됨 — 추가 작업 불필요
3. 에이전트에서 `PRESETS["my_preset"]`으로 참조

**에이전트에서 PRESETS 사용:**
```python
from app.services.rag.pipeline import PRESETS, search_with_pipeline_async

class MyAgent(BaseChatAgent):
    def __init__(self):
        self.config = PRESETS["my_preset"]

    async def _search(self, query: str):
        return await search_with_pipeline_async(query, self.config)
```

### 2.4 PipelineConfig 필드 레퍼런스

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `n_results` | int | 10 | 검색 후보 수 |
| `doc_type` | str \| None | None | 문서 타입 필터 (`"precedent"`, `"law"`) |
| `exclude_doc_types` | list[str] \| None | None | 제외할 data_type (한국어: `["판례"]`, `["법령"]`) |
| `enable_rewrite` | bool | True | 쿼리 리라이팅 활성화 |
| `enable_rerank` | bool | False | Cross-encoder 리랭킹 활성화 |
| `rerank_top_k` | int | 5 | 리랭킹 후 반환 수 |
| `use_llm_rewrite` | bool | True | LLM 기반 리라이팅 (False면 규칙 기반) |
| `search_type` | `"basic"` \| `"focus"` | `"basic"` | 검색 모드 |
| `supplementary_config` | PipelineConfig \| None | None | focus 모드 보충 검색 설정 |

**doc_type vs exclude_doc_types**:
- `doc_type`: 특정 타입만 검색 (`"precedent"` → 판례만)
- `exclude_doc_types`: 특정 타입 제외 (`["판례"]` → 판례 제외 나머지)
- 동시 지정 시 `doc_type` 우선, `exclude_doc_types` 무시

### 2.5 PipelineResult 필드 레퍼런스

| 필드 | 타입 | 설명 |
|------|------|------|
| `documents` | list[dict] | 검색 결과 (basic: 전체, focus: 주 타입) |
| `supplementary_documents` | list[dict] \| None | 보충 결과 (focus 모드 전용) |
| `original_query` | str | 원본 쿼리 |
| `rewritten_queries` | list[str] | 리라이팅된 쿼리 |
| `reranked` | bool | 리랭킹 적용 여부 |
| `total_retrieved` | int | 리랭킹 전 총 검색 수 |
| `metrics` | PipelineMetrics | 실행 메트릭 (시간 등) |

**문서(document) 구조:**
```python
{
    "content": str,         # 문서 본문
    "similarity": float,    # 유사도 점수
    "metadata": {
        "doc_id": str,      # 문서 고유 ID
        "data_type": str,   # "판례", "법령", "헌재결정례" 등
        "case_name": str,   # 사건명 / 법령명
        "case_number": str, # 사건번호 (판례)
        "court_name": str,  # 법원명
        "title": str,       # 제목 (법령)
        "date": str,        # 날짜
    }
}
```

---

## 3. format_utils 규칙 (필수)

### 3.1 컨텍스트/소스 포맷팅은 반드시 format_utils 사용

에이전트에서 자체 포맷팅 함수를 만들지 않는다.

```python
from app.services.rag.format_utils import (
    format_precedent_context,    # 판례 → LLM 컨텍스트
    format_law_context,          # 법령 → LLM 컨텍스트
    format_supplementary_context,# 보충 문서 → LLM 컨텍스트
    format_generic_context,      # 범용 → LLM 컨텍스트 (data_type 자동)
    format_precedent_sources,    # 판례 → 프론트엔드 소스
    format_law_sources,          # 법령 → 프론트엔드 소스
    format_supplementary_sources,# 보충 → 프론트엔드 소스
)
```

**금지 패턴:**
```python
# 에이전트 내부에 포맷팅 로직 금지
def _build_context(self, documents):  # 금지
    parts = ["## 관련 판례"]
    for doc in documents:
        ...
```

### 3.2 함수 선택 가이드

| 검색 대상 | 컨텍스트 함수 | 소스 함수 |
|----------|-------------|---------|
| 판례 | `format_precedent_context(docs, details)` | `format_precedent_sources(docs, details)` |
| 법령 | `format_law_context(docs)` | `format_law_sources(docs)` |
| 보충 자료 (다양한 타입) | `format_supplementary_context(docs)` | `format_supplementary_sources(docs)` |
| 타입 구분 없이 전체 | `format_generic_context(docs)` | `format_supplementary_sources(docs)` |

### 3.3 컨텍스트 함수 출력 형식

```python
# format_precedent_context (details 포함)
"## 관련 판례\n\n[판례 1] 사건명 (번호)\n본문\n[주문] ...\n[판결요지] ..."

# format_law_context
"## 관련 법령\n\n[법령 1] 법령명\n본문"

# format_supplementary_context
"## 관련 법률 자료 (보충)\n\n[헌재결정례 1] 제목\n본문"

# format_generic_context
"## 관련 자료\n\n[판례 1] 제목\n본문\n\n[법령 2] 제목\n본문"
```

모든 함수는 `documents`가 빈 리스트이면 빈 문자열 `""`을 반환한다.

### 3.4 판례 상세(details) 연동

`PrecedentService.get_details()`로 조회한 상세 정보를 context/sources에 전달:

```python
from app.services.service_function import PrecedentService, get_precedent_service

# 판례 상세 조회
precedent_service = get_precedent_service()
source_ids = [doc["metadata"]["doc_id"] for doc in docs if doc["metadata"].get("doc_id")]
details = await precedent_service.get_details(source_ids)

# context + sources에 details 전달
context = format_precedent_context(docs, details)
sources = format_precedent_sources(docs, details)
```

`details`는 `{doc_id: {ruling, reasoning, case_name, ...}}` 딕셔너리.
판례가 아닌 경우 `details` 파라미터 생략 가능 (기본값 None).

### 3.5 컨텍스트 조합 패턴

Focus 모드에서 focus + supplementary 컨텍스트를 합칠 때:

```python
def _build_context(self, focus_docs, details, supplementary_docs) -> str:
    parts: list[str] = []

    # Focus 컨텍스트
    text = format_precedent_context(focus_docs, details)  # 또는 format_law_context
    if text:
        parts.append(text)

    # Supplementary 컨텍스트
    sup_text = format_supplementary_context(supplementary_docs)
    if sup_text:
        parts.append(sup_text)

    return "\n\n".join(parts)
```

Basic 모드에서는 단일 함수 호출:

```python
context = format_generic_context(result.documents)
sources = format_supplementary_sources(result.documents)
```

---

## 4. LLM 호출 규칙

### 4.1 모델 획득

```python
from app.tools.llm import get_chat_model

model = get_chat_model()  # 기본 Solar 모델
```

`get_chat_model()`은 설정된 LLM을 반환. 직접 모델을 생성하지 않는다.

### 4.2 메시지 구성 패턴

```python
def _build_messages(
    self,
    message: str,
    context: str,
    history: list[dict[str, str]] | None = None,
) -> list[tuple[str, str]]:
    messages: list[tuple[str, str]] = [("system", _SYSTEM_PROMPT)]

    if history:
        for h in history:
            messages.append((h.get("role", "user"), h.get("content", "")))

    user_message = f"""참고 자료:
{context}

사용자 질문: {message}"""

    messages.append(("user", user_message))
    return messages
```

**규칙:**
- 시스템 프롬프트는 에이전트별 상수로 모듈 최상단에 정의
- 히스토리는 있으면 삽입, None이면 스킵
- RAG 컨텍스트는 사용자 메시지 앞에 `참고 자료:`로 주입
- LLM 없는 에이전트는 이 패턴 불필요

### 4.3 비동기 호출

```python
# 비스트리밍
response = await model.ainvoke(messages)
content = response.content
text = content if isinstance(content, str) else str(content)

# 스트리밍
async for chunk in model.astream(messages):
    if chunk.content and isinstance(chunk.content, str):
        yield ("token", {"content": chunk.content})
```

---

## 5. 스트리밍 프로토콜

### 5.1 이벤트 타입

`process_stream()`은 `AsyncGenerator[tuple[str, Any], None]`을 반환.

| 이벤트 | 데이터 | 시점 |
|--------|--------|------|
| `"token"` | `{"content": str}` | LLM 토큰 생성 중 |
| `"sources"` | `{"sources": list[dict]}` | LLM 완료 후 |
| `"metadata"` | `{"agent_used", "actions", "session_data"}` | 최종 |
| `"done"` | `{}` | 스트림 종료 |

### 5.2 스트리밍 에이전트 표준 구현

```python
async def process_stream(
    self,
    message: str,
    history: list[dict[str, str]] | None = None,
    session_data: dict[str, Any] | None = None,
    user_location: dict[str, float] | None = None,
) -> AsyncGenerator[tuple[str, Any], None]:
    # 1. RAG 검색 (스트리밍 전에 완료)
    context, sources = await self._prepare_rag_data(message)

    # 2. LLM 스트리밍
    model = get_chat_model()
    messages = self._build_messages(message, context, history)

    async for chunk in model.astream(messages):
        if chunk.content and isinstance(chunk.content, str):
            yield ("token", {"content": chunk.content})

    # 3. 후속 이벤트 (순서 반드시 유지)
    yield ("sources", {"sources": sources})
    yield ("metadata", {
        "agent_used": self.name,
        "actions": [],
        "session_data": {"active_agent": self.name},
    })
    yield ("done", {})
```

**규칙:**
- `"sources"` → `"metadata"` → `"done"` 순서 반드시 유지
- `"done"` 이벤트는 항상 마지막
- `session_data`에 `"active_agent": self.name` 필수

### 5.3 supports_streaming 플래그

```python
@property
def supports_streaming(self) -> bool:
    return True  # process_stream() 구현 시 True
```

`True`이면 노드에서 `_run_streaming_node()` 사용, `False`이면 `_run_nonstreaming_node()` 사용.

---

## 6. AgentResult 반환 규칙

### 6.1 필수 필드

```python
return AgentResult(
    message=response,                                    # 필수: 응답 메시지
    sources=sources,                                     # RAG 사용 시 필수
    actions=[],                                          # 액션 버튼 (없으면 빈 리스트)
    session_data={"active_agent": self.name},            # 필수: active_agent 포함
    agent_used=self.name,                                # 필수: 에이전트 식별
)
```

**`session_data`에 `"active_agent": self.name` 필수** — 다음 턴에서 같은 에이전트로 라우팅.

### 6.2 sources 필드

RAG를 사용하는 에이전트는 반드시 format_utils의 소스 함수로 생성한 `sources`를 반환.

```python
sources = format_precedent_sources(docs, details)
# 또는
sources = format_law_sources(docs)
# 또는
sources = format_supplementary_sources(docs)
```

RAG 미사용 에이전트는 `sources=[]`.

### 6.3 actions 필드

```python
from app.multi_agent.agents.base_chat import ActionType, ChatAction

actions = [
    ChatAction(type=ActionType.BUTTON, label="다음 단계", action="next_step").to_dict(),
    ChatAction(type=ActionType.NAVIGATE, label="결과 보기", params={"page": "/result"}).to_dict(),
    ChatAction(type=ActionType.LINK, label="외부 링크", url="https://...").to_dict(),
    ChatAction(type=ActionType.REQUEST_LOCATION, label="위치 검색").to_dict(),
]
```

---

## 7. 에이전트 유형별 구현 템플릿

### 7.1 RAG + LLM 에이전트 (Focus 모드)

```python
"""
에이전트 설명 (한 줄)

RAG 기반 검색 + LLM 응답 생성.
"""

import logging
from collections.abc import AsyncGenerator
from typing import Any, Literal

from app.multi_agent.agents.base_chat import BaseChatAgent
from app.multi_agent.schemas.plan import AgentResult
from app.services.rag.format_utils import (
    format_precedent_context,
    format_precedent_sources,
    format_supplementary_context,
    format_supplementary_sources,
)
from app.services.rag.pipeline import PRESETS, search_with_pipeline_async
from app.services.service_function import PrecedentService, get_precedent_service
from app.tools.llm import get_chat_model

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """시스템 프롬프트 내용"""


class MyFocusAgent(BaseChatAgent):
    def __init__(self, precedent_service: PrecedentService | None = None):
        self._precedent_service = precedent_service
        self.config = PRESETS["legal_search_precedent"]

    @property
    def precedent_service(self) -> PrecedentService:
        if self._precedent_service is None:
            self._precedent_service = get_precedent_service()
        return self._precedent_service

    @property
    def name(self) -> str:
        return "my_focus_agent"

    @property
    def description(self) -> str:
        return "Focus 모드 RAG 검색 에이전트"

    @property
    def supports_streaming(self) -> bool:
        return True

    async def _prepare_rag_data(
        self, message: str
    ) -> tuple[str, list[dict[str, Any]]]:
        result = await search_with_pipeline_async(message, self.config)

        focus_docs = result.documents
        sup_docs = result.supplementary_documents or []

        # 판례 상세 조회
        details: dict[str, dict[str, Any]] = {}
        if focus_docs:
            ids = [d["metadata"]["doc_id"] for d in focus_docs if d.get("metadata", {}).get("doc_id")]
            if ids:
                details = await self.precedent_service.get_details(ids)

        # 컨텍스트
        parts: list[str] = []
        text = format_precedent_context(focus_docs, details)
        if text:
            parts.append(text)
        sup_text = format_supplementary_context(sup_docs)
        if sup_text:
            parts.append(sup_text)
        context = "\n\n".join(parts)

        # 소스
        sources = format_precedent_sources(focus_docs, details) + format_supplementary_sources(sup_docs)

        return context, sources

    async def process(self, message, history=None, session_data=None, user_location=None) -> AgentResult:
        context, sources = await self._prepare_rag_data(message)
        response = await self._generate_response(message, context, history)
        return AgentResult(
            message=response, sources=sources, actions=[],
            session_data={"active_agent": self.name}, agent_used=self.name,
        )

    async def process_stream(self, message, history=None, session_data=None, user_location=None):
        context, sources = await self._prepare_rag_data(message)
        model = get_chat_model()
        messages = self._build_messages(message, context, history)

        async for chunk in model.astream(messages):
            if chunk.content and isinstance(chunk.content, str):
                yield ("token", {"content": chunk.content})
        yield ("sources", {"sources": sources})
        yield ("metadata", {"agent_used": self.name, "actions": [], "session_data": {"active_agent": self.name}})
        yield ("done", {})

    def _build_messages(self, message, context, history=None):
        msgs = [("system", _SYSTEM_PROMPT)]
        if history:
            for h in history:
                msgs.append((h.get("role", "user"), h.get("content", "")))
        msgs.append(("user", f"참고 자료:\n{context}\n\n사용자 질문: {message}"))
        return msgs

    async def _generate_response(self, message, context, history=None) -> str:
        model = get_chat_model()
        response = await model.ainvoke(self._build_messages(message, context, history))
        content = response.content
        return content if isinstance(content, str) else str(content)
```

### 7.2 RAG + LLM 에이전트 (Basic 모드)

```python
from app.services.rag.format_utils import format_generic_context, format_supplementary_sources
from app.services.rag.pipeline import PRESETS, search_with_pipeline_async

class MyBasicAgent(BaseChatAgent):
    def __init__(self):
        self.config = PRESETS["law_study"]  # basic 프리셋

    async def _prepare_rag_data(self, message: str) -> tuple[str, list[dict[str, Any]]]:
        result = await search_with_pipeline_async(message, self.config)
        context = format_generic_context(result.documents)
        sources = format_supplementary_sources(result.documents)
        return context, sources
    # ... process, process_stream 동일 패턴
```

### 7.3 Pure LLM 에이전트 (RAG 없음)

```python
class MyLlmAgent(BaseChatAgent):
    @property
    def supports_streaming(self) -> bool:
        return True

    async def process(self, message, history=None, **kwargs) -> AgentResult:
        model = get_chat_model()
        msgs = [("system", _SYSTEM_PROMPT)]
        if history:
            for h in history:
                msgs.append((h.get("role", "user"), h.get("content", "")))
        msgs.append(("user", message))

        response = await model.ainvoke(msgs)
        content = response.content
        return AgentResult(
            message=content if isinstance(content, str) else str(content),
            sources=[], actions=[],
            session_data={"active_agent": self.name}, agent_used=self.name,
        )

    async def process_stream(self, message, history=None, **kwargs):
        model = get_chat_model()
        msgs = [("system", _SYSTEM_PROMPT)]
        if history:
            for h in history:
                msgs.append((h.get("role", "user"), h.get("content", "")))
        msgs.append(("user", message))

        async for chunk in model.astream(msgs):
            if chunk.content and isinstance(chunk.content, str):
                yield ("token", {"content": chunk.content})
        yield ("sources", {"sources": []})
        yield ("metadata", {"agent_used": self.name, "actions": [], "session_data": {"active_agent": self.name}})
        yield ("done", {})
```

### 7.4 비스트리밍 서비스 에이전트 (RAG/LLM 없음)

```python
class MyServiceAgent(BaseChatAgent):
    @property
    def supports_streaming(self) -> bool:
        return False  # 비스트리밍

    async def process(self, message, history=None, session_data=None, **kwargs) -> AgentResult:
        # 서비스 로직
        data = await self._fetch_data(message)

        return AgentResult(
            message=self._format_response(data),
            sources=[], actions=[...],
            session_data={"active_agent": self.name}, agent_used=self.name,
        )
```

---

## 8. 서비스 의존성 규칙

### 8.1 Lazy Initialization

외부 서비스는 `__init__`에서 직접 생성하지 않고, property로 lazy init:

```python
class MyAgent(BaseChatAgent):
    def __init__(self, service: MyService | None = None):
        self._service = service  # 테스트에서 mock 주입 가능

    @property
    def service(self) -> MyService:
        if self._service is None:
            self._service = get_my_service()
        return self._service
```

### 8.2 Import 위치

- RAG 관련: `app.services.rag.pipeline`, `app.services.rag.format_utils`
- LLM: `app.tools.llm.get_chat_model`
- 판례 서비스: `app.services.service_function.PrecedentService`
- 변호사 서비스: `app.services.service_function.lawyer_service`
- Base 클래스: `app.multi_agent.agents.base_chat.BaseChatAgent`
- 결과 타입: `app.multi_agent.schemas.plan.AgentResult`
- 액션: `app.multi_agent.agents.base_chat.ActionType, ChatAction`

---

## 9. 체크리스트

### 새 에이전트 구현 시

- [ ] 유형 결정 (RAG+LLM / Pure LLM / 상태머신 / 서비스)
- [ ] `BaseChatAgent` 상속, `name`, `description`, `process()` 구현
- [ ] RAG 사용 시 `search_with_pipeline_async` + PRESETS 사용
- [ ] RAG 사용 시 `format_utils` 함수로 컨텍스트/소스 포맷팅
- [ ] 에이전트 내부에 자체 포맷팅 로직 없음
- [ ] LLM 사용 시 `supports_streaming = True` + `process_stream()` 구현
- [ ] 스트리밍 이벤트 순서: token → sources → metadata → done
- [ ] `session_data`에 `"active_agent": self.name` 포함
- [ ] 시스템 프롬프트는 모듈 상단 상수로 정의
- [ ] 외부 서비스는 lazy initialization (property 패턴)
- [ ] `ruff check` + `mypy` 통과

### 기존 에이전트 수정 시

- [ ] 직접 검색 함수 호출 → `search_with_pipeline_async` 전환
- [ ] 자체 포맷팅 함수 → `format_utils` 함수 전환
- [ ] `asyncio.to_thread(search_relevant_documents, ...)` → 파이프라인 async 직접 호출
- [ ] PRESETS에 해당 프리셋 존재 확인

---

## 10. 현재 에이전트 현황

| 에이전트 | RAG | LLM | 스트리밍 | 파이프라인 통일 | format_utils |
|---------|:---:|:---:|:-------:|:-------------:|:------------:|
| LegalSearchAgent | O | O | O | O | O |
| LawStudyAgent | O | O | O | **X** (전환 필요) | **X** |
| SmallClaimsAgent | O | X | X | **X** (전환 필요) | **X** |
| StoryboardAgent | X | O | O | - | - |
| LawyerFinderAgent | X | X | X | - | - |
| LawyerStatsAgent | X | X | X | - | - |
| MockTrialAgent | X | O | O | - | - |
| SimpleChatAgent | X | O | O | - | - |

> LawStudyAgent, SmallClaimsAgent는 파이프라인 전환 예정.
> PRESETS에 `law_study`, `small_claims` 프리셋이 이미 준비됨.
