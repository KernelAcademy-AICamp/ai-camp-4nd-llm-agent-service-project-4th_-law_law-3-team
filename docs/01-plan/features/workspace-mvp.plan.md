# 워크스페이스 MVP 기획서

> **Summary**: 챗봇 대화에서 크로스 에이전트 태그를 자동 수집하고, 사건 중심 워크스페이스로 묶어 스토리보드/타임라인을 생성하는 기능
>
> **Project**: 법률 서비스 플랫폼
> **Author**: Claude + User
> **Date**: 2026-03-01
> **Status**: Draft
> **Base Document**: `workspace-mvp-no-ocr-handoff.plan.md` (핸드오프 계획)

---

## 1. Overview

### 1.1 Purpose

사용자가 **스토리보드 페이지에 직접 가지 않아도**, 챗봇과의 자연스러운 대화만으로 사건 정보가 자동 수집되고 타임라인이 생성되는 워크플로우를 구현한다.

핵심 변화:
1. **크로스 에이전트 태그 수집** — 변호사 질문, 판례 검색 등 모든 에이전트 대화에서 LLM이 자동으로 사건 관련 태그를 부착
2. **챗봇 주도 워크플로우** — 챗봇이 능동적으로 질문하며 정보를 수집
3. **워크스페이스** — 사건 단위로 대화/태그/타임라인을 묶는 상위 구조

### 1.2 Background

현재 한계:
- 스토리보드 서브그래프는 **자체 태그만 수집** (다른 에이전트에서 나온 정보 미활용)
- 대화 기록이 **영속적으로 저장되지 않음** (브라우저 새로고침 시 유실)
- 사건별로 대화를 묶어 관리하는 구조가 없음
- 스토리보드 에이전트로 **명시적으로 라우팅**되어야만 정보 수집 시작

### 1.3 Related Documents

- 핸드오프 계획: `docs/01-plan/features/workspace-mvp-no-ocr-handoff.plan.md`
- 현재 스토리보드 서브그래프: `backend/app/multi_agent/subgraphs/storyboard.py`
- 태그 스키마: `backend/app/multi_agent/schemas/tag.py`
- 라우터: `backend/app/multi_agent/router.py`

---

## 2. Scope

### 2.1 In Scope

- [ ] **크로스 에이전트 태그 수집**: 모든 에이전트 노드에서 LLM 기반 태그 추출 → session_data 누적
- [ ] **대화 영속화**: chat_conversations + chat_messages 테이블 (PostgreSQL)
- [ ] **사건 워크스페이스 CRUD**: workspace_cases 테이블 + 기본 API
- [ ] **챗봇 주도 정보 수집**: 태그 부족 시 챗봇이 능동적으로 질문
- [ ] **태그 기반 타임라인 생성**: 수집된 태그를 엮어 스토리보드 타임라인으로 변환
- [ ] **대화 이어가기**: 구조화 요약 기반 컨텍스트 재개
- [ ] **사건 자동 분류**: 대화에서 사건명/유형 자동 감지 + 수동 보정
- [ ] **프론트엔드 워크스페이스 UI**: /workspace, /workspace/[caseId] 페이지

### 2.2 Out of Scope

- OCR / 문서 업로드 자동 텍스트화 (Phase 2)
- 계정/인증 시스템 (별도 피처)
- 실시간 협업 / 공유 기능
- 자동 병합 (Opt-in 가져오기만)

---

## 3. Requirements

### 3.1 Functional Requirements

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| FR-01 | 모든 에이전트(판례/변호사/소액소송/일반채팅)에서 대화 내 사건 관련 정보를 LLM이 태그로 추출하여 session_data에 누적 | High | Pending |
| FR-02 | 스토리보드 요청 시, 누적된 크로스 에이전트 태그만 엮어서 타임라인 생성 | High | Pending |
| FR-03 | 챗봇이 부족한 태그(timeline/party/evidence/amount) 감지 시 능동적으로 질문 | High | Pending |
| FR-04 | 대화 메시지를 chat_conversations + chat_messages 테이블에 영속 저장 | High | Pending |
| FR-05 | 사건 워크스페이스 CRUD (생성/조회/수정/삭제) API | High | Pending |
| FR-06 | 대화→사건 자동 분류 (사건명/유형/고객명) + 수동 보정 보호 | Medium | Pending |
| FR-07 | 구조화 요약 기반 대화 이어가기 (facts/issues/evidence/open_questions/next_steps) | Medium | Pending |
| FR-08 | /chat-history 페이지: 사건명/유형별 그룹 조회 + 이어가기 | Medium | Pending |
| FR-09 | /workspace, /workspace/[caseId] 페이지 기본 UI | Medium | Pending |
| FR-10 | 워크스페이스↔스토리보드 타임라인 정본 연결 | Medium | Pending |
| FR-11 | WorkspaceAgent 라우팅 — 챗봇에서 사건 조회/타임라인 재생성 의도 판별 (RulesRouter 키워드 추가) | High | Pending |
| FR-12 | 대화/타임라인 내보내기 (JSON/TXT) — 최소 백업 대안 | Low | Pending |
| FR-13 | 활동 로그 기록 — workspace_activity_logs 테이블로 사건별 주요 이벤트 추적 | Low | Pending |

### 3.2 Non-Functional Requirements

| Category | Criteria | Measurement Method |
|----------|----------|-------------------|
| Performance | 태그 추출 지연 < 500ms (에이전트 응답에 추가되는 시간) | 로그 측정 |
| Performance | 대화 저장 지연 < 100ms | DB 쿼리 시간 |
| Storage | 대화 기록 30일 보존 (MVP) | 자동 정리 배치 |
| UX | 스토리보드 페이지 방문 없이 챗봇만으로 타임라인 생성 가능 | E2E 테스트 |

---

## 4. Architecture Design

### 4.1 크로스 에이전트 태그 수집 흐름

```
사용자: "전세 보증금 5000만원 못 받고 있어요"
    │
    ▼
router_node → small_claims_subgraph
    │
    ├─ 에이전트 응답 생성 (기존 로직)
    │
    └─ [NEW] 태그 추출 미들웨어 (post-processing)
        ├─ LLM이 응답에서 태그 추출:
        │   - [amount] 보증금 5000만원
        │   - [party] 임차인(사용자) ↔ 임대인
        │   - [timeline] 전세 계약 관련 분쟁
        │
        └─ session_data.tagged_items에 누적
            (source_agent="small_claims", turn_index=N)

사용자: "계약서는 있는데 카톡 내역도 있어요"
    │
    ▼
small_claims_subgraph (세션 유지)
    │
    └─ [NEW] 태그 추출:
        - [evidence] 계약서
        - [evidence] 카카오톡 대화 내역

사용자: "이런 경우 판례 있어?"
    │
    ▼
router_node → legal_search_node (세션 전환, 신뢰도 0.9)
    │
    └─ [NEW] 태그 추출:
        - [precedent] 전세보증금 반환 관련 판례
        - [timeline] 판례에서 추출된 시점 정보

사용자: "타임라인 만들어줘"
    │
    ▼
router_node → storyboard_subgraph
    │
    └─ session_data.tagged_items에서 모든 태그 로드
        → 4개 에이전트에서 수집된 태그를 통합
        → 부족한 정보 있으면 추가 질문
        → 타임라인 생성
```

### 4.2 태그 추출 위치 (구현 전략)

**현재 상태 분석:**
- `_append_tags()` 함수가 `nodes.py`에 이미 존재 (부분 구현)
- `multi_agent/services/tagger` 모듈에 `extract_tags`, `merge_tags` 인터페이스 존재
- **Critical Gap**: 서브그래프(small_claims, mock_trial, storyboard)는 별도 `CompiledStateGraph`로 실행되어 `_append_tags()` 후처리를 우회함

**Option A: 각 에이전트 노드 후처리** (선택)
- `nodes.py`의 각 에이전트 노드 함수에서, 에이전트 응답 생성 후 태그 추출 함수 호출
- 장점: 에이전트별 컨텍스트를 활용한 정밀 태그 추출
- 단점: 모든 에이전트 노드에 코드 추가 필요

**Option B: graph.py에 공통 후처리 노드 삽입**
- 모든 에이전트 노드 → `tag_extraction_node` → END
- 장점: 한 곳에서 관리
- 단점: 에이전트별 맥락 손실, 그래프 구조 복잡화

→ **Option A 선택**: 각 에이전트가 자신의 도메인 지식으로 태그를 추출하는 것이 더 정확함

**서브그래프 태그 후처리 전략:**
서브그래프는 자체 StateGraph 내부에서 태그를 추출하고, 서브그래프 종료 시 부모 그래프의 `session_data.tagged_items`로 결과를 전파한다.
1. 서브그래프 내부 마지막 노드(또는 별도 `sync_tags_node`)에서 태그 추출
2. 서브그래프 출력 state에 `tagged_items` 포함
3. 부모 `nodes.py`의 서브그래프 래퍼 함수가 출력 태그를 `session_data`에 병합

### 4.2.1 태그 서버사이드 누적 전략

**현재 문제**: `session_data.tagged_items`는 SSE 응답 → 클라이언트 → 다음 요청으로 라운드트립하며 누적됨. 클라이언트가 태그를 누락/변조할 수 있고, 새로고침 시 유실됨.

**해결**: 서버사이드 태그 누적으로 전환
1. `chat_conversations.tagged_items` (JSONB) 컬럼이 태그 정본
2. 에이전트 응답 생성 후 태그 추출 → DB에 직접 `jsonb_concat` 업데이트
3. SSE `done` 이벤트에는 현재 턴 태그만 포함 (전체 태그 노출 X)
4. 스토리보드 요청 시 DB에서 `conversation.tagged_items` 조회하여 사용

### 4.3 데이터 모델

```sql
-- 대화 메타데이터 (state 정본은 LangGraph Checkpointer)
CREATE TABLE chat_conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    thread_id VARCHAR(100) NOT NULL,         -- LangGraph thread_id 포인터
    session_token VARCHAR(100) NOT NULL,      -- 소유권 검증
    case_id UUID REFERENCES workspace_cases(id),  -- 사건 연결 (nullable)
    title VARCHAR(200),                       -- 대화 제목 (자동 생성)
    case_type VARCHAR(50),                    -- 사건 유형 (auto-classified)
    customer_name VARCHAR(100),               -- 고객명 (auto-classified)
    is_title_manual BOOLEAN DEFAULT FALSE,    -- 수동 보정 여부
    summary JSONB,                            -- 구조화 요약
    tagged_items JSONB DEFAULT '[]',          -- 누적 태그
    last_agent VARCHAR(50),                   -- 마지막 활성 에이전트
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 대화 메시지 (검색/표시용)
CREATE TABLE chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES chat_conversations(id),
    role VARCHAR(10) NOT NULL,                -- user | assistant
    content TEXT NOT NULL,
    agent_type VARCHAR(50),                   -- 응답 에이전트
    metadata JSONB,                           -- sources, actions 등
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 사건 워크스페이스
CREATE TABLE workspace_cases (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_token VARCHAR(100) NOT NULL,
    case_name VARCHAR(200) NOT NULL,
    case_type VARCHAR(50),
    status VARCHAR(20) DEFAULT 'active',      -- active | resolved | archived
    summary JSONB,                            -- 사건 요약
    tagged_items JSONB DEFAULT '[]',          -- 통합 태그 (대화에서 병합)
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 타임라인 정본 (워크스페이스 + 스토리보드 공유)
CREATE TABLE workspace_case_timeline_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id UUID NOT NULL REFERENCES workspace_cases(id),
    date_text VARCHAR(50),                    -- 표시용 날짜
    date_normalized DATE,                     -- 정규화 날짜 (정렬용)
    title VARCHAR(200) NOT NULL,
    description TEXT,
    category VARCHAR(50),                     -- 계약, 이행, 분쟁, 소송 등
    source_type VARCHAR(20),                  -- chat | manual | ocr(Phase2)
    source_ref JSONB,                         -- 출처 참조
    sort_order INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 활동 로그 (사건별 이벤트 추적)
CREATE TABLE workspace_activity_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id UUID REFERENCES workspace_cases(id),
    conversation_id UUID REFERENCES chat_conversations(id),
    session_token VARCHAR(100) NOT NULL,
    action VARCHAR(50) NOT NULL,              -- create_case | add_conversation | rebuild_timeline | manual_edit 등
    detail JSONB,                             -- 액션별 상세 데이터
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 신원 연결 (로그인 연동 대비, Phase 2)
CREATE TABLE identity_links (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_token VARCHAR(100) NOT NULL,
    user_id UUID,                             -- 추후 계정 시스템 연동
    provider VARCHAR(50),                     -- local | google | kakao 등
    linked_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(session_token, provider)
);

-- JSONB GIN 인덱스 (태그/요약 검색 성능)
CREATE INDEX idx_chat_conversations_tagged_items ON chat_conversations USING GIN (tagged_items);
CREATE INDEX idx_chat_conversations_summary ON chat_conversations USING GIN (summary);
CREATE INDEX idx_workspace_cases_tagged_items ON workspace_cases USING GIN (tagged_items);

-- 조회 성능 인덱스
CREATE INDEX idx_chat_conversations_session ON chat_conversations(session_token, updated_at DESC);
CREATE INDEX idx_chat_messages_conversation ON chat_messages(conversation_id, created_at);
CREATE INDEX idx_workspace_cases_session ON workspace_cases(session_token, updated_at DESC);
CREATE INDEX idx_timeline_items_case ON workspace_case_timeline_items(case_id, sort_order);
CREATE INDEX idx_activity_logs_case ON workspace_activity_logs(case_id, created_at DESC);
```

### 4.4 API 설계

```
# 대화 관리
POST   /api/chat/stream              -- 기존 (확장: conversation_id, case_id 파라미터)
GET    /api/chat/conversations        -- 대화 목록 (그룹/검색)
GET    /api/chat/conversations/:id    -- 대화 상세
PATCH  /api/chat/conversations/:id    -- 대화 메타 수정 (제목, 사건 연결)

# 워크스페이스
POST   /api/workspace/cases           -- 사건 생성
GET    /api/workspace/cases           -- 사건 목록
GET    /api/workspace/cases/:id       -- 사건 상세 (+ 대화 목록 + 타임라인)
PATCH  /api/workspace/cases/:id       -- 사건 수정
DELETE /api/workspace/cases/:id       -- 사건 삭제

# 타임라인
GET    /api/workspace/cases/:id/timeline          -- 타임라인 조회
POST   /api/workspace/cases/:id/timeline/rebuild  -- 타임라인 재생성 (태그 기반)
PATCH  /api/workspace/cases/:id/timeline/items/:itemId  -- 항목 수정
```

### 4.5 프론트엔드 페이지 구조

```
/chat-history                    -- 대화 기록 (사건별 그룹)
/workspace                       -- 사건 목록 대시보드
/workspace/[caseId]              -- 사건 상세
  ├─ 사건 요약 카드
  ├─ 수집된 태그 목록
  ├─ 타임라인 뷰 (정본)
  └─ 관련 대화 목록
/storyboard                      -- 기존 스토리보드 (워크스페이스 타임라인과 연결)
```

---

## 5. 구현 순서

### Phase 1: 보안 기반 + 대화 영속화 + 태그 수집 (Backend)

1. **세션 보안 전환 (Critical, 최우선)**
   - HttpOnly/Secure/SameSite 세션 쿠키 발급 미들웨어 구현
   - `session_secret` HTTP 응답 본문 노출 제거 (현재 보안 취약점)
   - `X-Device-Id` 단독 신뢰 폐기, 세션 쿠키 기반 principal로 전환
2. Alembic 마이그레이션 (chat_conversations, chat_messages, workspace_cases, timeline_items, activity_logs, identity_links)
3. Chat persistence 서비스 (대화 저장/조회/이어가기)
4. `/api/chat/stream` 확장 — conversation_id 파라미터, 메시지 자동 저장
5. **크로스 에이전트 태그 추출** — 각 에이전트 노드 후처리 + 서브그래프 태그 전파
6. **태그 서버사이드 누적** — DB 직접 업데이트, 클라이언트 라운드트립 의존 제거
7. 구조화 요약기 구현 (대화 종료/일정 턴 후 자동 요약)
8. chat_messages 30일 자동 정리 배치 (비동기 스케줄러)

### Phase 2: 워크스페이스 API (Backend)

9. 워크스페이스 CRUD API
10. 대화↔사건 연결 API
11. 사건 자동 분류 (LLM 기반 사건명/유형 추출)
12. 타임라인 엔진 (태그 기반 재생성)
13. WorkspaceAgent 라우팅 추가 (RulesRouter에 워크스페이스 키워드 등록, 조회 vs 재생성 의도 판별)
14. 대화/타임라인 내보내기 API (JSON/TXT)

### Phase 3: 프론트엔드

15. **ChatWidget 확장 (conversation_id 관리, 이어가기)** — Phase 3 초반으로 이동, 백엔드 연동 즉시 확인
16. /chat-history 페이지 (대화 목록, 그룹, 이어가기)
17. /workspace 대시보드
18. /workspace/[caseId] 상세 페이지
19. 스토리보드↔워크스페이스 타임라인 정본 연결

---

## 6. Success Criteria

### 6.1 Definition of Done

- [ ] 판례/변호사/소액소송 대화에서 태그가 자동 수집됨
- [ ] "타임라인 만들어줘" 시 이전 대화의 태그가 모두 반영됨
- [ ] 스토리보드 페이지 방문 없이 챗봇만으로 타임라인 생성 가능
- [ ] 대화 기록이 DB에 저장되어 새로고침 후에도 유지됨
- [ ] 사건별로 대화/태그/타임라인이 묶여 관리됨

### 6.2 Quality Criteria

- [ ] 린트/타입 에러 없음 (ruff + mypy + tsc)
- [ ] 기존 에이전트 동작에 회귀 없음
- [ ] 태그 추출 지연 < 500ms

---

## 7. Risks and Mitigation

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R-01 | 태그 추출 LLM 호출로 응답 지연 | Medium | High | 비동기 후처리, 응답 후 백그라운드 태그 추출 |
| R-02 | 기존 에이전트 동작 회귀 | High | Medium | 태그 추출을 독립 후처리로 분리, 실패해도 에이전트 응답에 영향 없음 |
| R-03 | DB 마이그레이션 복잡성 | Medium | Medium | 단계별 마이그레이션, Feature Flag 사용 |
| R-04 | 세션 토큰 보안 | High | Low | HttpOnly 쿠키 기반, X-Device-Id 단독 신뢰 금지 |
| R-05 | 서브그래프 태그 전파 실패 — 서브그래프가 별도 CompiledStateGraph로 실행되어 부모 state 직접 수정 불가 | High | High | 서브그래프 출력 state에 tagged_items 포함, 래퍼 함수에서 병합 |
| R-06 | session_secret 노출 — 현재 HTTP 응답 본문에 session_secret 원문이 포함됨 | Critical | Confirmed | Phase 1 최우선으로 HttpOnly 쿠키 전환, 응답 본문에서 secret 제거 |
| R-07 | 대화 데이터 무한 증가 — chat_messages 테이블이 정리 없이 계속 증가 | Medium | High | 30일 자동 정리 배치 + 구조화 요약으로 핵심 정보 보존 |
| R-08 | 태그 품질 저하 — LLM 추출 태그의 중복/오분류 누적 | Medium | Medium | merge_tags에서 중복 제거 + confidence 기반 필터링, 수동 보정 UI 제공 |

---

## 8. Architecture Considerations

### 8.1 Project Level

Enterprise (기존 아키텍처 유지)

### 8.2 Key Decisions

| Decision | Selected | Rationale |
|----------|----------|-----------|
| 태그 추출 위치 | 에이전트 노드 후처리 + 서브그래프 래퍼 병합 | 도메인 맥락 활용 + 서브그래프 태그 누락 방지 |
| 대화 state 정본 | LangGraph Checkpointer (AsyncPostgresSaver, Feature Flag 이미 구현) | 핸드오프 계획 결정사항 유지 |
| 태그 추출 방식 | 규칙 기반 + LLM 하이브리드 | 규칙 기반으로 빠르게 추출, LLM은 보충 |
| 세션 식별 | 서버 발급 HttpOnly 세션 쿠키 | 보안 계획 결정사항 유지, session_secret 응답 노출 제거 |
| 태그 누적 위치 | 서버사이드 (DB 직접 업데이트) | 클라이언트 라운드트립 의존 제거, 보안/안정성 확보 |
| 로그인 시 데이터 귀속 | Opt-in 가져오기 (자동 병합 금지) | 공용 기기 오귀속 방지 |

### 8.3 보안 결정 (핸드오프 계획 반영)

| # | 결정 | 상세 |
|---|------|------|
| S-01 | X-Device-Id 단독 식별 금지 | 서버 세션 토큰(HttpOnly/Secure/SameSite)으로 소유권 검증 |
| S-02 | session_secret 원문 반환 금지 | HTTP 응답 본문에서 제거, 필요 시 만료형 capability token |
| S-03 | 로그인 시 자동 병합 금지 | Opt-in 가져오기만 허용, 명시적 확인 UX |
| S-04 | 공용 기기 오귀속 방지 | 세션 만료 + 명시적 "내 기기가 아닙니다" UX |

---

## 9. 핸드오프 계획 반영 현황

`workspace-mvp-no-ocr-handoff.plan.md`의 13개 핵심 항목 반영 상태:

| # | 핸드오프 항목 | 반영 | 비고 |
|---|-------------|------|------|
| 1 | 대화 분류/요약/이어가기 | ✅ | FR-06, FR-07 |
| 2 | 사건 중심 워크스페이스 | ✅ | FR-05, FR-09 |
| 3 | HttpOnly 세션 토큰 기반 전환 | ✅ | S-01, Phase 1 최우선 |
| 4 | X-Device-Id 단독 신뢰 폐기 | ✅ | S-01 |
| 5 | LangGraph Checkpointer 단일 정본 | ✅ | 8.2 Key Decisions |
| 6 | 타임라인 텍스트/대화 기반 추출만 | ✅ | FR-02, FR-10 |
| 7 | OCR/문서 파이프라인 Phase 2 | ✅ | Out of Scope |
| 8 | workspace_activity_logs | ✅ | FR-13, DDL 추가 |
| 9 | identity_links | ✅ | DDL 추가 (Phase 2 대비) |
| 10 | WorkspaceAgent 라우팅 | ✅ | FR-11 |
| 11 | 내보내기 (JSON/TXT) | ✅ | FR-12 |
| 12 | session_secret 원문 반환 금지 | ✅ | S-02, R-06 |
| 13 | Opt-in 가져오기 | ✅ | S-03 |

## 10. Next Steps

1. [ ] `/pdca design workspace-mvp` — 상세 설계 문서 작성
2. [ ] Alembic 마이그레이션 설계 (6개 테이블 + 인덱스)
3. [ ] 크로스 에이전트 태그 추출 프로토타입 (서브그래프 태그 전파 포함)
4. [ ] HttpOnly 세션 쿠키 미들웨어 프로토타입

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-03-01 | 초안 작성 (핸드오프 계획 기반) | Claude + User |
| 0.2 | 2026-03-01 | 팀 리뷰 반영: FR-11~13 추가, 서브그래프 태그 전파 전략, 서버사이드 태그 누적, 보안 결정 S-01~04, DDL 보완 (activity_logs, identity_links, GIN 인덱스), R-05~08 추가, 구현 순서 재배치, 핸드오프 반영 현황표 추가 | Claude |
