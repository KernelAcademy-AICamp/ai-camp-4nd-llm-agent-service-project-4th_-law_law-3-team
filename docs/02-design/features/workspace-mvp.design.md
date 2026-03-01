# 워크스페이스 MVP 상세 설계서

> **Feature**: workspace-mvp
> **Plan**: `docs/01-plan/features/workspace-mvp.plan.md` (v0.2)
> **Date**: 2026-03-01
> **Status**: Draft
> **Level**: Enterprise

---

## 1. Executive Summary

챗봇 대화에서 크로스 에이전트 태그를 자동 수집하고, 사건 중심 워크스페이스로 묶어 스토리보드/타임라인을 생성하는 기능.

**핵심 변경 3가지:**
1. 모든 에이전트 노드에서 LLM 기반 태그 추출 → 서버사이드 DB 누적
2. 대화 영속화 (chat_conversations + chat_messages) + 구조화 요약 기반 이어가기
3. 사건 워크스페이스 CRUD + 태그 기반 타임라인 엔진

**영향 범위:**
- Backend: `multi_agent/` (nodes, graph, state, schemas, services), `api/router/chat.py`, 신규 모듈 `modules/workspace/`, 신규 미들웨어
- Frontend: ChatWidget 확장, 신규 페이지 3개 (/chat-history, /workspace, /workspace/[caseId])
- DB: Alembic 마이그레이션 6개 테이블 + 10개 인덱스

---

## 2. Non-Functional Requirements

| Category | Criteria | Target | Measurement |
|----------|----------|--------|-------------|
| Performance | 태그 추출 추가 지연 | < 500ms | 로그 P95 측정 |
| Performance | 대화 저장 지연 | < 100ms | DB 쿼리 시간 |
| Performance | 워크스페이스 목록 조회 | < 200ms | API 응답 시간 |
| Performance | 타임라인 재생성 | < 3s (태그 50개 기준) | LLM 호출 시간 |
| Storage | chat_messages 보존 | 30일 (MVP) | 자동 정리 배치 |
| Storage | 태그 상한 | conversation당 50개 | merge_tags MAX_TAGGED_ITEMS |
| Security | 세션 토큰 | HttpOnly/Secure/SameSite=Lax | 쿠키 속성 검증 |
| Security | 소유권 검증 | 모든 데이터 접근 시 session_token 검증 | 통합 테스트 |
| Availability | 태그 추출 실패 | 에이전트 응답에 영향 없음 | 실패 시 빈 태그 반환 |

---

## 3. System Architecture

### 3.1 컴포넌트 다이어그램

```
┌─────────────────────────────────────────────────────────────┐
│  Frontend (Next.js)                                         │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐ │
│  │ ChatWidget   │ │ /chat-history│ │ /workspace/[caseId] │ │
│  │ (확장)       │ │ (신규)       │ │ (신규)               │ │
│  └──────┬───────┘ └──────┬───────┘ └──────────┬───────────┘ │
│         │                │                     │             │
│         ▼                ▼                     ▼             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ API Client (api.ts) — session cookie 자동 전송       │   │
│  └──────────────────────────┬───────────────────────────┘   │
└─────────────────────────────┼───────────────────────────────┘
                              │ HTTPS
┌─────────────────────────────┼───────────────────────────────┐
│  Backend (FastAPI)          │                                │
│  ┌──────────────────────────▼───────────────────────────┐   │
│  │ SessionMiddleware (신규)                              │   │
│  │ — Set-Cookie: session_token (HttpOnly/Secure/SameSite)│   │
│  │ — request.state.session_token 주입                    │   │
│  └──────────────────────────┬───────────────────────────┘   │
│                             │                                │
│  ┌──────────────┐ ┌────────▼─────────┐ ┌─────────────────┐ │
│  │ /api/chat/*  │ │ /api/workspace/* │ │ /api/chat/      │ │
│  │ conversations│ │ cases, timeline  │ │ stream (확장)   │ │
│  └──────┬───────┘ └────────┬─────────┘ └────────┬────────┘ │
│         │                  │                     │          │
│  ┌──────▼──────────────────▼─────────────────────▼────────┐ │
│  │ Services Layer                                          │ │
│  │ ┌─────────────────┐ ┌──────────────┐ ┌───────────────┐ │ │
│  │ │ ChatPersistence │ │ WorkspaceCase│ │ TimelineEngine│ │ │
│  │ │ Service         │ │ Service      │ │               │ │ │
│  │ └─────────────────┘ └──────────────┘ └───────────────┘ │ │
│  └─────────────────────────┬──────────────────────────────┘ │
│                             │                                │
│  ┌──────────────────────────▼───────────────────────────┐   │
│  │ LangGraph StateGraph                                 │   │
│  │ router_node → agent_node → _append_tags() → END     │   │
│  │                ↕ (서브그래프: 래퍼에서 태그 병합)     │   │
│  └──────────────────────────┬───────────────────────────┘   │
│                             │                                │
│  ┌──────────────┐ ┌────────▼──────┐ ┌────────────────────┐ │
│  │ PostgreSQL   │ │ LangGraph     │ │ Tagger Service     │ │
│  │ (6 tables)   │ │ Checkpointer  │ │ (LLM 태그 추출)   │ │
│  └──────────────┘ └───────────────┘ └────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 크로스 에이전트 태그 수집 상세 흐름

**현재 구조 분석 (변경 전):**
- `nodes.py:_append_tags()` (L223-251): 일반 노드 후처리로 `tagger.extract_tags()` 호출
- `_run_streaming_node_inner()` (L101)와 `_run_nonstreaming_node()` (L212)에서 호출됨
- **Gap**: 서브그래프 (`small_claims_subgraph`, `mock_trial_subgraph`, `storyboard_subgraph`)는 `CompiledStateGraph`로 등록되어 `_append_tags()`를 우회

**변경 설계:**

```
일반 노드 (legal_search, lawyer_finder, lawyer_stats, law_study, simple_chat, content_marketing):
  에이전트 응답 → _append_tags() [기존, 이미 동작]
                      │
                      ▼
               tagger.extract_tags(message, agent_name, turn_index)
                      │
                      ▼
               tagger.merge_tags(existing, new_tags)
                      │
                      ▼
               output_session_data["tagged_items"] = merged
                      │
                      ▼ [NEW] 서버사이드 누적
               ChatPersistenceService.append_tags(conversation_id, new_tags)

서브그래프 (small_claims, mock_trial, storyboard):
  서브그래프 내부:
    각 노드에서 자체 태그 추출
    마지막 노드에서 output_session_data["tagged_items"]에 포함하여 반환
                      │
                      ▼
  부모 그래프로 복귀:
    nodes.py의 서브그래프 래퍼 함수 [NEW]가
    output_session_data["tagged_items"]를 읽어서
    ChatPersistenceService.append_tags() 호출
```

### 3.3 서브그래프 태그 전파 메커니즘

**문제**: 서브그래프는 `graph.py`에서 `build_*_subgraph()`로 등록된 `CompiledStateGraph`이며, 부모 그래프의 노드 후처리(`_append_tags`)가 서브그래프 내부에서는 호출되지 않음.

**해결 방안 — 래퍼 함수 삽입:**

```python
# nodes.py에 추가
async def _subgraph_tag_sync(
    state: ChatState,
    subgraph_output: dict[str, Any],
    conversation_id: str | None,
) -> dict[str, Any]:
    """서브그래프 출력에서 태그를 추출하여 DB에 누적"""
    output_session_data = subgraph_output.get("output_session_data", {})
    new_tags_raw = output_session_data.get("tagged_items", [])

    if new_tags_raw and conversation_id:
        from app.services.workspace.chat_persistence import ChatPersistenceService
        await ChatPersistenceService.append_tags(conversation_id, new_tags_raw)

    return output_session_data
```

**그러나** 현재 서브그래프는 `CompiledStateGraph`를 직접 `add_node()`에 전달하여 LangGraph가 내부적으로 실행하므로, 래퍼 함수를 중간에 삽입할 수 없음.

**대안 — 서브그래프를 래퍼 함수로 감싸기:**

```python
# graph.py 변경
# Before:
#   builder.add_node("small_claims_subgraph", build_small_claims_subgraph())
# After:
async def small_claims_wrapper(state: ChatState) -> dict[str, Any]:
    subgraph = build_small_claims_subgraph()
    result = await subgraph.ainvoke(state)
    # 서브그래프 출력에서 태그 추출 → DB 누적
    result = await _subgraph_tag_sync(state, result, state.get("conversation_id"))
    return result

builder.add_node("small_claims_subgraph", small_claims_wrapper)
```

**최종 결정: 래퍼 함수 방식 채택.**
- 서브그래프 3개 (small_claims, mock_trial, storyboard)에 래퍼 함수 적용
- 래퍼 함수에서 `_append_tags()` + DB 누적 수행
- 서브그래프 내부 코드 변경 최소화

### 3.4 태그 서버사이드 누적 전략

**현재 문제:**
`session_data.tagged_items`는 SSE 응답 → 클라이언트 → 다음 요청으로 라운드트립 누적됨.
- 클라이언트가 태그를 누락/변조할 수 있음
- 새로고침 시 유실됨 (ChatContext는 메모리 기반)

**변경 설계:**

```
[기존] 클라이언트 라운드트립 누적:
  서버 응답(tagged_items) → 클라이언트 메모리 → 다음 요청 session_data → 서버

[변경] 서버사이드 DB 누적:
  에이전트 응답 후:
    1. _append_tags()로 태그 추출
    2. ChatPersistenceService.append_tags(conversation_id, new_tags)
       → chat_conversations.tagged_items JSONB 직접 업데이트
    3. SSE done 이벤트에는 현재 턴 태그만 포함 (전체 태그 X)

  스토리보드 요청 시:
    1. DB에서 chat_conversations.tagged_items 조회
    2. case_id가 있으면 workspace_cases.tagged_items도 병합
    3. 통합 태그를 타임라인 엔진에 전달
```

**DB 업데이트 쿼리:**

```sql
-- 태그 추가 (JSONB concat, 중복 제거는 서비스 레이어에서)
UPDATE chat_conversations
SET tagged_items = tagged_items || $1::jsonb,
    updated_at = NOW()
WHERE id = $2;
```

### 3.5 세션 보안 아키텍처

**현재 취약점 (chat.py L133-143, L259-261):**
```python
# session_secret이 HTTP 응답 본문에 노출됨
done_data = {"thread_id": thread_id, "session_secret": session_secret}
```

**변경 설계:**

```
[NEW] SessionMiddleware:
  1. 요청에 session_token 쿠키 없으면:
     → secrets.token_hex(32) 생성
     → Set-Cookie: session_token=xxx; HttpOnly; Secure; SameSite=Lax; Path=/; Max-Age=2592000
     → request.state.session_token = xxx

  2. 요청에 session_token 쿠키 있으면:
     → request.state.session_token = cookie_value

  3. chat.py 변경:
     → session_secret 생성/검증 로직을 session_token 쿠키 기반으로 교체
     → SSE 응답에서 session_secret 필드 제거
     → thread_id 검증 시 session_token으로 소유권 확인

  4. 모든 데이터 접근 API:
     → request.state.session_token으로 소유권 필터링
     → WHERE session_token = :token 조건 필수
```

**SessionMiddleware 구현 위치:** `backend/app/core/session.py`

```python
# app/core/session.py
import secrets
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

SESSION_COOKIE_NAME = "session_token"
SESSION_MAX_AGE = 30 * 24 * 60 * 60  # 30일

class SessionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        token = request.cookies.get(SESSION_COOKIE_NAME)
        is_new = False
        if not token:
            token = secrets.token_hex(32)
            is_new = True
        request.state.session_token = token

        response: Response = await call_next(request)

        if is_new:
            response.set_cookie(
                key=SESSION_COOKIE_NAME,
                value=token,
                httponly=True,
                secure=True,     # HTTPS only (개발 시 False)
                samesite="lax",
                max_age=SESSION_MAX_AGE,
                path="/",
            )
        return response
```

---

## 4. Data Model

### 4.1 ERD

```
┌─────────────────────┐     ┌─────────────────────────────┐
│ chat_conversations  │     │ workspace_cases             │
│─────────────────────│     │─────────────────────────────│
│ id (PK, UUID)       │  ┌─▶│ id (PK, UUID)               │
│ thread_id           │  │  │ session_token               │
│ session_token       │  │  │ case_name                   │
│ case_id (FK) ───────┼──┘  │ case_type                   │
│ title               │     │ status                      │
│ case_type           │     │ summary (JSONB)             │
│ customer_name       │     │ tagged_items (JSONB)        │
│ is_title_manual     │     │ created_at, updated_at      │
│ summary (JSONB)     │     └──────────────┬──────────────┘
│ tagged_items (JSONB)│                    │
│ last_agent          │                    │ 1:N
│ created_at          │     ┌──────────────▼──────────────┐
│ updated_at          │     │ workspace_case_timeline_items│
└──────────┬──────────┘     │─────────────────────────────│
           │ 1:N            │ id (PK, UUID)               │
┌──────────▼──────────┐     │ case_id (FK)                │
│ chat_messages       │     │ date_text, date_normalized  │
│─────────────────────│     │ title, description          │
│ id (PK, UUID)       │     │ category, source_type       │
│ conversation_id(FK) │     │ source_ref (JSONB)          │
│ role                │     │ sort_order                  │
│ content             │     │ created_at, updated_at      │
│ agent_type          │     └─────────────────────────────┘
│ metadata (JSONB)    │
│ created_at          │     ┌─────────────────────────────┐
└─────────────────────┘     │ workspace_activity_logs     │
                            │─────────────────────────────│
┌─────────────────────┐     │ id (PK, UUID)               │
│ identity_links      │     │ case_id (FK, nullable)      │
│─────────────────────│     │ conversation_id (FK, null)  │
│ id (PK, UUID)       │     │ session_token               │
│ session_token       │     │ action                      │
│ user_id (nullable)  │     │ detail (JSONB)              │
│ provider            │     │ created_at                  │
│ linked_at           │     └─────────────────────────────┘
│ UNIQUE(session_token│
│        , provider)  │
└─────────────────────┘
```

### 4.2 ORM 모델

**`backend/app/models/chat_conversation.py`** (신규):

```python
import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class ChatConversation(Base):
    __tablename__ = "chat_conversations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    thread_id: Mapped[str] = mapped_column(String(100), nullable=False)
    session_token: Mapped[str] = mapped_column(String(100), nullable=False)
    case_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("workspace_cases.id"), nullable=True
    )
    title: Mapped[str | None] = mapped_column(String(200), nullable=True)
    case_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    customer_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    is_title_manual: Mapped[bool] = mapped_column(Boolean, default=False)
    summary: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    tagged_items: Mapped[list] = mapped_column(JSONB, default=list)
    last_agent: Mapped[str | None] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_chat_conversations_session", "session_token", updated_at.desc()),
        Index("idx_chat_conversations_tagged_items", "tagged_items", postgresql_using="gin"),
        Index("idx_chat_conversations_summary", "summary", postgresql_using="gin"),
        Index("idx_chat_conversations_thread", "thread_id", unique=True),
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chat_conversations.id"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(10), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    agent_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    conversation: Mapped["ChatConversation"] = relationship(back_populates="messages")

    __table_args__ = (
        Index("idx_chat_messages_conversation", "conversation_id", "created_at"),
    )
```

**`backend/app/models/workspace_case.py`** (신규):

```python
import uuid
from datetime import date, datetime

from sqlalchemy import Date, DateTime, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class WorkspaceCase(Base):
    __tablename__ = "workspace_cases"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_token: Mapped[str] = mapped_column(String(100), nullable=False)
    case_name: Mapped[str] = mapped_column(String(200), nullable=False)
    case_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="active")
    summary: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    tagged_items: Mapped[list] = mapped_column(JSONB, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    timeline_items: Mapped[list["WorkspaceCaseTimelineItem"]] = relationship(
        back_populates="case", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_workspace_cases_session", "session_token", updated_at.desc()),
        Index("idx_workspace_cases_tagged_items", "tagged_items", postgresql_using="gin"),
    )


class WorkspaceCaseTimelineItem(Base):
    __tablename__ = "workspace_case_timeline_items"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    case_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("workspace_cases.id"), nullable=False
    )
    date_text: Mapped[str | None] = mapped_column(String(50), nullable=True)
    date_normalized: Mapped[date | None] = mapped_column(Date, nullable=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    category: Mapped[str | None] = mapped_column(String(50), nullable=True)
    source_type: Mapped[str | None] = mapped_column(String(20), nullable=True)
    source_ref: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    sort_order: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    case: Mapped["WorkspaceCase"] = relationship(back_populates="timeline_items")

    __table_args__ = (
        Index("idx_timeline_items_case", "case_id", "sort_order"),
    )


class WorkspaceActivityLog(Base):
    __tablename__ = "workspace_activity_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    case_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("workspace_cases.id"), nullable=True
    )
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chat_conversations.id"), nullable=True
    )
    session_token: Mapped[str] = mapped_column(String(100), nullable=False)
    action: Mapped[str] = mapped_column(String(50), nullable=False)
    detail: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("idx_activity_logs_case", "case_id", created_at.desc()),
    )


class IdentityLink(Base):
    __tablename__ = "identity_links"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_token: Mapped[str] = mapped_column(String(100), nullable=False)
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    provider: Mapped[str | None] = mapped_column(String(50), nullable=True)
    linked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        # UniqueConstraint handled by Index
        Index("uq_identity_links_session_provider", "session_token", "provider", unique=True),
    )
```

### 4.3 Alembic 마이그레이션

**파일**: `backend/alembic/versions/020_add_workspace_mvp_tables.py`

마이그레이션 번호: `020` (현재 최신: `019_add_lawyer_personas_and_feedback.py`)

```python
# revision: 020
# depends_on: 019
# 테이블 생성 순서 (FK 의존성):
#   1. workspace_cases (독립)
#   2. chat_conversations (FK → workspace_cases)
#   3. chat_messages (FK → chat_conversations)
#   4. workspace_case_timeline_items (FK → workspace_cases)
#   5. workspace_activity_logs (FK → workspace_cases, chat_conversations)
#   6. identity_links (독립)
# downgrade: 역순 DROP
```

---

## 5. API Specification

### 5.1 세션 미들웨어 (공통)

모든 API에서 `request.state.session_token`으로 세션 식별. 쿠키 없으면 자동 발급.

### 5.2 Chat Streaming API 확장

**`POST /api/chat/stream`** (기존 확장)

Request Body 변경:
```json
{
  "message": "전세 보증금 5000만원 못 받고 있어요",
  "user_role": "user",
  "history": [],
  "session_data": {},
  "conversation_id": null,  // NEW: null이면 신규, 있으면 이어가기
  "case_id": null            // NEW: 사건에 연결할 경우
}
```

SSE `done` 이벤트 변경:
```json
{
  "type": "done",
  "data": {
    "thread_id": "...",
    "conversation_id": "uuid",     // NEW
    "case_id": "uuid|null",        // NEW
    "active_agent": "small_claims",
    "tags_extracted": 3            // NEW: 현재 턴에서 추출된 태그 수
    // session_secret 필드 제거
  }
}
```

### 5.3 대화 관리 API

**`GET /api/chat/conversations`**

```
Query params:
  - case_id: UUID (optional) — 특정 사건의 대화만
  - search: string (optional) — 제목/내용 검색
  - page: int (default: 1)
  - page_size: int (default: 20, max: 50)

Response 200:
{
  "items": [
    {
      "id": "uuid",
      "title": "전세 보증금 반환 문제",
      "case_type": "임대차",
      "customer_name": null,
      "last_agent": "small_claims",
      "tag_count": 5,
      "message_count": 12,
      "created_at": "2026-03-01T10:00:00Z",
      "updated_at": "2026-03-01T11:30:00Z"
    }
  ],
  "total": 15,
  "page": 1,
  "page_size": 20
}

소유권: WHERE session_token = request.state.session_token
```

**`GET /api/chat/conversations/{conversation_id}`**

```
Response 200:
{
  "id": "uuid",
  "thread_id": "...",
  "title": "전세 보증금 반환 문제",
  "case_id": "uuid|null",
  "case_type": "임대차",
  "customer_name": "홍길동",
  "is_title_manual": false,
  "summary": {
    "facts": ["..."],
    "issues": ["..."],
    "evidence": ["..."],
    "open_questions": ["..."],
    "next_steps": ["..."]
  },
  "tagged_items": [...],
  "last_agent": "small_claims",
  "messages": [
    {"role": "user", "content": "...", "created_at": "..."},
    {"role": "assistant", "content": "...", "agent_type": "small_claims", "created_at": "..."}
  ],
  "created_at": "...",
  "updated_at": "..."
}

소유권: session_token 검증, 불일치 시 404
```

**`PATCH /api/chat/conversations/{conversation_id}`**

```
Request Body:
{
  "title": "보증금 사건 (수정)",     // optional
  "case_id": "uuid",                // optional — 사건 연결
  "customer_name": "홍길동"          // optional
}

수동 보정 보호:
  title을 수동 변경하면 is_title_manual = true
  이후 자동 분류에서 title 덮어쓰기 방지

Response 200: 업데이트된 conversation 객체
소유권: session_token 검증
```

### 5.4 워크스페이스 API

**`POST /api/workspace/cases`**

```
Request Body:
{
  "case_name": "전세 보증금 반환 사건",
  "case_type": "임대차",              // optional
  "conversation_ids": ["uuid"]        // optional — 기존 대화 연결
}

Response 201:
{
  "id": "uuid",
  "case_name": "전세 보증금 반환 사건",
  "case_type": "임대차",
  "status": "active",
  "tagged_items": [...],  // 연결된 대화의 태그 병합
  "created_at": "..."
}

내부 동작:
  1. workspace_cases INSERT
  2. conversation_ids가 있으면 해당 대화의 case_id UPDATE
  3. 해당 대화들의 tagged_items를 workspace_cases.tagged_items에 병합
  4. workspace_activity_logs 기록 (action: "create_case")
```

**`GET /api/workspace/cases`**

```
Query params:
  - status: string (optional, default: "active")
  - search: string (optional)
  - page, page_size

Response 200:
{
  "items": [
    {
      "id": "uuid",
      "case_name": "...",
      "case_type": "...",
      "status": "active",
      "conversation_count": 3,
      "timeline_item_count": 8,
      "tag_count": 15,
      "created_at": "...",
      "updated_at": "..."
    }
  ],
  "total": 5
}
```

**`GET /api/workspace/cases/{case_id}`**

```
Response 200:
{
  "id": "uuid",
  "case_name": "...",
  "case_type": "...",
  "status": "active",
  "summary": {...},
  "tagged_items": [...],
  "conversations": [...],     // 연결된 대화 목록 (요약)
  "timeline": [...],          // 타임라인 항목
  "created_at": "...",
  "updated_at": "..."
}
```

**`PATCH /api/workspace/cases/{case_id}`**

```
Request Body:
{
  "case_name": "...",    // optional
  "case_type": "...",    // optional
  "status": "resolved"   // optional
}
```

**`DELETE /api/workspace/cases/{case_id}`**

```
Response 204
내부: 연결된 대화의 case_id를 null로 변경, 타임라인 항목 삭제, 사건 삭제
활동 로그: action: "delete_case"
```

### 5.5 타임라인 API

**`GET /api/workspace/cases/{case_id}/timeline`**

```
Response 200:
{
  "items": [
    {
      "id": "uuid",
      "date_text": "2025년 3월",
      "date_normalized": "2025-03-01",
      "title": "전세 계약 체결",
      "description": "보증금 5000만원, 계약기간 2년",
      "category": "계약",
      "source_type": "chat",
      "sort_order": 1
    }
  ],
  "total": 8
}
```

**`POST /api/workspace/cases/{case_id}/timeline/rebuild`**

```
Request Body:
{
  "include_conversations": true,   // 연결된 대화의 태그 포함
  "include_manual": true           // 수동 입력 항목 유지
}

내부 동작:
  1. workspace_cases.tagged_items + 연결 대화들의 tagged_items 수집
  2. timeline 태그 (tag_type: "timeline") 추출
  3. LLM으로 시간순 정리 + 카테고리 분류
  4. 기존 source_type="manual" 항목은 유지 (include_manual=true 시)
  5. 새 항목 INSERT, 기존 chat 항목 교체
  6. 활동 로그: action: "rebuild_timeline"

Response 200: 재생성된 타임라인
```

**`PATCH /api/workspace/cases/{case_id}/timeline/items/{item_id}`**

```
Request Body:
{
  "title": "...",
  "description": "...",
  "date_text": "...",
  "category": "..."
}

수정 시 source_type → "manual" (이후 rebuild에서 보존)
```

### 5.6 내보내기 API

**`GET /api/chat/conversations/{conversation_id}/export`**

```
Query params:
  - format: "json" | "txt" (default: "json")

Response 200: 파일 다운로드 (Content-Disposition: attachment)
```

**`GET /api/workspace/cases/{case_id}/export`**

```
Query params:
  - format: "json" | "txt"
  - include: "all" | "timeline" | "conversations" (default: "all")

Response 200: 파일 다운로드
```

---

## 6. Backend Service Layer

### 6.1 서비스 구조

```
backend/app/services/workspace/       # 신규 디렉토리
├── __init__.py
├── chat_persistence.py               # 대화 영속화
├── conversation_classifier.py        # 사건 자동 분류
├── structured_summarizer.py          # 구조화 요약
├── workspace_case_service.py         # 사건 CRUD
├── timeline_engine.py                # 타임라인 재생성
└── activity_logger.py                # 활동 로그
```

### 6.2 ChatPersistenceService

```python
class ChatPersistenceService:
    """대화 영속화 서비스"""

    @staticmethod
    async def get_or_create_conversation(
        db: AsyncSession,
        session_token: str,
        thread_id: str,
        conversation_id: str | None = None,
    ) -> ChatConversation:
        """conversation_id가 있으면 조회, 없으면 생성"""

    @staticmethod
    async def save_message(
        db: AsyncSession,
        conversation_id: uuid.UUID,
        role: str,
        content: str,
        agent_type: str | None = None,
        metadata: dict | None = None,
    ) -> ChatMessage:
        """메시지 저장"""

    @staticmethod
    async def append_tags(
        db: AsyncSession,
        conversation_id: uuid.UUID,
        new_tags: list[dict],
    ) -> None:
        """태그를 DB에 직접 누적 (서버사이드)"""
        # merge_tags() 사용하여 중복 제거 후 JSONB 업데이트

    @staticmethod
    async def update_summary(
        db: AsyncSession,
        conversation_id: uuid.UUID,
        summary: dict,
    ) -> None:
        """구조화 요약 업데이트"""

    @staticmethod
    async def list_conversations(
        db: AsyncSession,
        session_token: str,
        case_id: uuid.UUID | None = None,
        search: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[ChatConversation], int]:
        """대화 목록 조회 (페이지네이션)"""
```

### 6.3 StructuredSummarizer

```python
class StructuredSummarizer:
    """구조화 요약 생성기"""

    SUMMARY_FIELDS = ["facts", "issues", "evidence", "open_questions", "next_steps"]

    @staticmethod
    async def generate_summary(
        messages: list[ChatMessage],
        existing_summary: dict | None = None,
    ) -> dict:
        """대화 메시지에서 구조화 요약 생성

        Args:
            messages: 대화 메시지 리스트
            existing_summary: 기존 요약 (증분 업데이트용)

        Returns:
            {"facts": [...], "issues": [...], "evidence": [...],
             "open_questions": [...], "next_steps": [...]}
        """

    @staticmethod
    def build_resume_context(
        summary: dict,
        recent_messages: list[ChatMessage],
        token_budget: int = 2000,
    ) -> str:
        """이어가기용 컨텍스트 구성

        구조화 요약 + 토큰 예산 기반 최근 대화.
        고정 '최근 20턴' 규칙은 사용하지 않음.
        """
```

### 6.4 ConversationClassifier

```python
class ConversationClassifier:
    """대화 자동 분류기"""

    @staticmethod
    async def classify(
        messages: list[ChatMessage],
        tagged_items: list[dict],
    ) -> dict:
        """사건명/유형/고객명 자동 분류

        Returns:
            {"title": "...", "case_type": "...", "customer_name": "..."}
        """

    @staticmethod
    async def auto_classify_if_needed(
        db: AsyncSession,
        conversation: ChatConversation,
        messages: list[ChatMessage],
    ) -> None:
        """수동 보정 여부 확인 후 자동 분류 적용

        is_title_manual=True이면 title 덮어쓰기 방지
        """
```

### 6.5 TimelineEngine

```python
class TimelineEngine:
    """태그 기반 타임라인 재생성 엔진

    스토리보드 서브그래프와 워크스페이스가 공유하는 정본 타임라인 생성기.
    """

    @staticmethod
    async def rebuild(
        db: AsyncSession,
        case_id: uuid.UUID,
        tagged_items: list[dict],
        preserve_manual: bool = True,
    ) -> list[WorkspaceCaseTimelineItem]:
        """태그에서 타임라인 재생성

        1. timeline 태그 추출 (tag_type: "timeline")
        2. date_hint가 있는 태그 → 날짜 정규화
        3. LLM으로 시간순 정리 + 카테고리 분류 + 누락 날짜 추정
        4. 기존 manual 항목 보존
        5. 결과 INSERT/UPDATE
        """
```

### 6.6 WorkspaceAgent 라우팅 (RulesRouter 확장)

`router.py` 변경:

```python
# AgentType에 추가
class AgentType(str, Enum):
    ...
    WORKSPACE = "workspace"

# INTENT_PATTERNS에 추가
AgentType.WORKSPACE: [
    ("사건 목록", 0.9),
    ("워크스페이스", 0.95),
    ("내 사건", 0.85),
    ("사건 조회", 0.9),
    ("사건 정보", 0.85),
    ("타임라인 재생성", 0.9),
    ("타임라인 만들어", 0.9),
    ("타임라인 갱신", 0.85),
    ("사건 정리", 0.85),
    ("진행 상황", 0.7),
],

# ROLE_AGENTS에 추가 (양쪽 역할 모두)
UserRole.USER: [..., AgentType.WORKSPACE],
UserRole.LAWYER: [..., AgentType.WORKSPACE],
```

**WorkspaceAgent 구현** (`multi_agent/agents/workspace_agent.py`, 신규):
- 의도 판별: "조회" vs "생성/재생성"
- 조회: DB에서 사건/타임라인 조회 → 포맷팅 응답
- 재생성: TimelineEngine.rebuild() 호출 → 결과 포맷팅

---

## 7. Frontend Architecture

### 7.1 ChatWidget 확장

**현재 구조 (변경 전):**
- `ChatContext`: sessionData (메모리), userRole, userLocation
- `useStreamingChat`: SSE Fetch + ReadableStream
- 세션 데이터: React state (메모리, 새로고침 시 유실)

**변경 설계:**

```typescript
// ChatContext 확장
interface ChatContextState {
  sessionData: SessionData;
  userRole: string;
  userLocation: Location | null;
  conversationId: string | null;    // NEW
  caseId: string | null;            // NEW
}

// useStreamingChat 변경
// 1. request body에 conversation_id, case_id 추가
// 2. SSE done 이벤트에서 conversation_id, case_id 수신
// 3. session_secret 처리 로직 제거 (쿠키 기반으로 전환)
```

**API 클라이언트 변경 (`api.ts`):**

```typescript
// endpoints 추가
const endpoints = {
  ...existing,
  chatConversations: '/api/chat/conversations',
  workspaceCases: '/api/workspace/cases',
};

// fetch 옵션에 credentials: 'include' 추가 (쿠키 전송)
```

### 7.2 신규 페이지

**`/chat-history`:**
- 대화 목록 (사건별 그룹 가능)
- 검색 필터
- 이어가기 버튼 → ChatWidget에 conversation_id 전달

**`/workspace`:**
- 사건 목록 대시보드
- 사건 생성 버튼
- 사건별 요약 카드 (대화 수, 태그 수, 타임라인 수)

**`/workspace/[caseId]`:**
- 사건 상세 뷰
- 탭: 요약 | 태그 | 타임라인 | 대화
- 타임라인 뷰: 기존 스토리보드 컴포넌트 재활용
- 태그 뷰: 태그 카드 리스트 (유형별 색상)

### 7.3 프론트엔드 모듈 등록

`frontend/src/lib/modules.ts` 추가:

```typescript
{
  id: 'workspace',
  name: '사건 워크스페이스',
  description: '사건별 대화/태그/타임라인 관리',
  path: '/workspace',
  icon: 'Briefcase',
  enabled: true,
}
```

`frontend/src/lib/api.ts` endpoints 추가:
```typescript
workspace: '/api/workspace',
chatConversations: '/api/chat/conversations',
```

`frontend/next.config.js` rewrites 추가:
```javascript
{ source: '/api/workspace/:path*', destination: 'http://localhost:8000/api/workspace/:path*' },
{ source: '/api/chat/conversations/:path*', destination: 'http://localhost:8000/api/chat/conversations/:path*' },
```

---

## 8. chat.py 변경 상세

현재 `backend/app/api/router/chat.py` (299행)의 주요 변경:

### 8.1 session_secret 제거

```python
# 제거 대상 (L133-143, L259-261):
# - session_secret 생성 로직
# - SSE metadata/done 이벤트에서 session_secret 필드
# - _validate_session_secret() 함수

# 대체:
# - request.state.session_token (SessionMiddleware에서 주입)
# - thread_id 소유권은 chat_conversations.session_token으로 검증
```

### 8.2 대화 영속화 통합

```python
# _invoke_graph() 또는 chat_stream() 변경:
async def chat_stream(request: ChatRequest, ...):
    session_token = request.state.session_token

    # 1. conversation 조회/생성
    conversation = await ChatPersistenceService.get_or_create_conversation(
        db, session_token, thread_id, request.conversation_id
    )

    # 2. 사용자 메시지 저장
    await ChatPersistenceService.save_message(
        db, conversation.id, "user", request.message
    )

    # 3. 그래프 실행 (기존 로직)
    # ...

    # 4. 어시스턴트 응답 저장
    await ChatPersistenceService.save_message(
        db, conversation.id, "assistant", response_text, agent_used
    )

    # 5. 태그 서버사이드 누적 (이미 _append_tags에서 처리)

    # 6. 자동 분류 (N턴마다)
    if message_count % 5 == 0:
        await ConversationClassifier.auto_classify_if_needed(db, conversation, messages)

    # 7. 구조화 요약 (대화 종료 또는 10턴마다)
    if should_summarize:
        summary = await StructuredSummarizer.generate_summary(messages)
        await ChatPersistenceService.update_summary(db, conversation.id, summary)
```

### 8.3 ChatState 확장

`state.py` 변경:

```python
class ChatState(TypedDict, total=False):
    ...
    # NEW
    conversation_id: str  # chat_conversations.id
    case_id: str          # workspace_cases.id (optional)
    # session_secret는 유지하되 쿠키 기반으로 전환
```

`messages.py` 변경:

```python
class ChatRequest(BaseModel):
    ...
    conversation_id: str | None = None  # NEW
    case_id: str | None = None          # NEW
```

---

## 9. Storyboard 연동

### 9.1 현재 스토리보드 태그 흐름

스토리보드 서브그래프 (`subgraphs/storyboard.py`):
- `collect_node`: `session_data.get("tagged_items", [])` 읽기
- `_extract_tags_sync()`: 규칙 기반 태그 추출 (dates, amounts, parties, evidence)
- `generate_node`: `output_session_data`에 `tagged_items`, `generated_timeline_text` 포함

### 9.2 변경 사항

1. **입력**: 클라이언트 라운드트립 태그 대신 DB 조회
   ```python
   # collect_node 변경:
   # Before: tags = state.get("session_data", {}).get("tagged_items", [])
   # After:
   conversation_id = state.get("conversation_id")
   if conversation_id:
       tags = await ChatPersistenceService.get_tags(db, conversation_id)
   else:
       tags = state.get("session_data", {}).get("tagged_items", [])  # fallback
   ```

2. **출력**: 생성된 타임라인을 정본 테이블에 저장
   ```python
   # generate_node 변경:
   # case_id가 있으면 workspace_case_timeline_items에 저장
   case_id = state.get("case_id")
   if case_id:
       await TimelineEngine.save_timeline(db, case_id, timeline_items)
   ```

3. **래퍼 함수**: `graph.py`에서 storyboard_subgraph를 래퍼로 감싸서 태그 DB 동기화

---

## 10. Implementation Plan

### Phase 1: 보안 기반 + 대화 영속화 (Backend)

| # | Task | Files | Depends |
|---|------|-------|---------|
| 1.1 | SessionMiddleware 구현 | `app/core/session.py`, `app/main.py` | - |
| 1.2 | Alembic 마이그레이션 (6 tables) | `alembic/versions/020_*.py` | - |
| 1.3 | ORM 모델 생성 | `app/models/chat_conversation.py`, `app/models/workspace_case.py` | 1.2 |
| 1.4 | models/__init__.py 업데이트 | `app/models/__init__.py` | 1.3 |
| 1.5 | alembic/env.py 모델 import | `alembic/env.py` | 1.3 |
| 1.6 | ChatPersistenceService | `app/services/workspace/chat_persistence.py` | 1.3 |
| 1.7 | chat.py 리팩토링 (session_secret → cookie, 대화 영속화) | `app/api/router/chat.py` | 1.1, 1.6 |
| 1.8 | ChatState/ChatRequest 확장 | `app/multi_agent/state.py`, `schemas/messages.py` | - |
| 1.9 | 크로스 에이전트 태그 서버사이드 누적 | `app/multi_agent/nodes.py` | 1.6, 1.8 |
| 1.10 | 서브그래프 래퍼 함수 (태그 전파) | `app/multi_agent/graph.py` | 1.9 |
| 1.11 | StructuredSummarizer | `app/services/workspace/structured_summarizer.py` | 1.6 |
| 1.12 | ConversationClassifier | `app/services/workspace/conversation_classifier.py` | 1.6 |

### Phase 2: 워크스페이스 API (Backend)

| # | Task | Files | Depends |
|---|------|-------|---------|
| 2.1 | 워크스페이스 모듈 생성 | `app/modules/workspace/` | Phase 1 |
| 2.2 | WorkspaceCaseService | `app/services/workspace/workspace_case_service.py` | 1.3 |
| 2.3 | TimelineEngine | `app/services/workspace/timeline_engine.py` | 1.3 |
| 2.4 | Workspace CRUD API | `app/modules/workspace/router/` | 2.2 |
| 2.5 | Timeline API | `app/modules/workspace/router/` | 2.3 |
| 2.6 | Chat conversations API | `app/api/router/chat_conversations.py` | 1.6 |
| 2.7 | WorkspaceAgent + RulesRouter 확장 | `multi_agent/agents/workspace_agent.py`, `router.py` | 2.2, 2.3 |
| 2.8 | 내보내기 API | `app/modules/workspace/router/` | 2.4, 2.6 |
| 2.9 | ActivityLogger | `app/services/workspace/activity_logger.py` | 1.3 |
| 2.10 | chat_messages 30일 정리 배치 | `app/services/workspace/cleanup.py` | 1.6 |

### Phase 3: 프론트엔드

| # | Task | Files | Depends |
|---|------|-------|---------|
| 3.1 | ChatWidget 확장 (conversation_id, cookie) | `frontend/src/features/chat/` | Phase 1 |
| 3.2 | 모듈/API/rewrites 등록 | `modules.ts`, `api.ts`, `next.config.js` | - |
| 3.3 | /chat-history 페이지 | `frontend/src/app/chat-history/` | 2.6, 3.1 |
| 3.4 | /workspace 대시보드 | `frontend/src/app/workspace/` | 2.4 |
| 3.5 | /workspace/[caseId] 상세 | `frontend/src/app/workspace/[caseId]/` | 2.4, 2.5 |
| 3.6 | 스토리보드↔워크스페이스 타임라인 연결 | `frontend/src/features/storyboard/` | 2.5, 3.5 |

---

## 11. Test Plan

### 11.1 단위 테스트

| Target | Test Cases |
|--------|-----------|
| SessionMiddleware | 쿠키 없을 때 생성, 기존 쿠키 유지, HttpOnly 속성 검증 |
| ChatPersistenceService | 대화 생성/조회, 메시지 저장, 태그 누적/중복 제거, 소유권 필터 |
| StructuredSummarizer | 요약 생성, 증분 업데이트, 이어가기 컨텍스트 구성 |
| ConversationClassifier | 자동 분류, 수동 보정 보호 (is_title_manual) |
| TimelineEngine | 태그→타임라인 변환, manual 항목 보존, 날짜 정규화 |
| merge_tags | 중복 제거, MAX_TAGGED_ITEMS 초과 시 오래된 것 제거 |
| 서브그래프 래퍼 | 래퍼 함수에서 태그 병합 정상 동작 |

### 11.2 통합 테스트

| Scenario | Expected |
|----------|----------|
| 첫 대화 → conversation_id 생성 → 이어가기 | conversation 재사용, 메시지 추가됨 |
| 다른 세션으로 대화 접근 시도 | 404 반환 |
| 판례 검색 → 소액소송 → "타임라인 만들어줘" | 크로스 에이전트 태그 통합됨 |
| 수동 제목 변경 후 자동 분류 | 제목 덮어쓰기 방지됨 |
| 사건 생성 → 대화 연결 → 타임라인 재생성 | 태그 병합, 타임라인 생성됨 |
| interrupt 상태 재개 | LangGraph Checkpointer 정상 동작 |
| chat_messages 30일 초과 | 자동 정리 실행, 요약은 보존 |

### 11.3 보안 테스트

| Scenario | Expected |
|----------|----------|
| session_token 쿠키 없이 API 호출 | 쿠키 자동 발급 |
| 다른 사용자의 conversation_id로 조회 | 404 (소유권 불일치) |
| SSE 응답에 session_secret 포함 여부 | 포함되지 않음 |
| 쿠키 HttpOnly/Secure/SameSite 속성 | 모두 설정됨 |

---

## 12. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| 서브그래프 래퍼가 LangGraph 내부 동작과 충돌 | High | 래퍼 함수에서 `ainvoke` 호출 후 결과 변환만 수행, 상태 변경 최소화 |
| 태그 추출 LLM 호출로 응답 지연 증가 | Medium | `_append_tags()`는 이미 try/except로 감싸져 있어 실패 시 무시. 필요 시 비동기 백그라운드 전환 |
| chat.py 리팩토링 범위가 넓어 회귀 위험 | High | session_secret 제거와 대화 영속화를 분리 커밋, 단계별 검증 |
| 프론트엔드 cookie credentials 설정 누락 | Medium | api.ts에서 `credentials: 'include'` 전역 설정, CORS 미들웨어 `allow_credentials=True` |

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-03-01 | 초안 작성 — Plan v0.2 기반 상세 설계 | Claude |
