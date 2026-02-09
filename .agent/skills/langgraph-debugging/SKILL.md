---
name: langgraph-debugging
description: LangGraph 멀티 에이전트 라우팅/상태 전파 디버깅 패턴. router_node 분기 추적, ChatState 변환 검증, Command 흐름 분석, 에이전트 실패 진단. 멀티에이전트 버그 수정, 라우팅 오류, 상태 누락 문제 해결 시 사용.
---

# LangGraph Debugging Skill

LangGraph StateGraph 기반 멀티 에이전트 시스템의 디버깅 패턴 가이드.

> **참조**: `multi-agent-patterns/SKILL.md` (아키텍처 개요), `Antigravity.md` (에이전트 목록)

## 1. 디버깅 대상 아키텍처

```
POST /api/chat
    │
    ▼
request_to_state()  ← 여기서 상태 변환 오류 발생 가능
    │
    ▼
router_node         ← 라우팅 분기 오류 발생 가능
    │
    ├→ legal_search_node    (RAG)
    ├→ lawyer_finder_node
    ├→ small_claims_subgraph (서브그래프)
    ├→ storyboard_node
    ├→ lawyer_stats_node
    ├→ law_study_node       (RAG)
    └→ simple_chat_node     (폴백)
```

## 2. 문제 유형별 디버깅 가이드

### 2.1 라우팅 오류 (잘못된 에이전트로 분기)

**증상**: "판례 검색해줘" 입력 → `simple_chat_node`로 라우팅

**디버깅 순서**:

1. **router.py 키워드 매칭 확인**
```bash
# 키워드 사전 확인
grep -n "keyword\|keywords\|KEYWORDS" backend/app/multi_agent/router.py
```

2. **RulesRouter.route() 로직 추적**
```python
# backend/app/multi_agent/router.py
# route() 메서드에서 confidence 점수가 threshold 이상인지 확인
# ROLE_AGENTS에서 현재 사용자 role에 해당 에이전트가 허용되는지 확인
```

3. **Command 반환값 확인**
```python
# backend/app/multi_agent/nodes.py - router_node 함수
# Command(goto=target_node) 반환값이 올바른 노드명인지 확인
# graph.py의 add_node() 등록명과 일치해야 함
```

**핵심 파일**:
- `backend/app/multi_agent/router.py` - RulesRouter, AgentType, ROLE_AGENTS
- `backend/app/multi_agent/nodes.py` - router_node 함수
- `backend/app/multi_agent/graph.py` - StateGraph 노드 등록

### 2.2 상태 전파 오류 (ChatState 필드 누락)

**증상**: 에이전트에서 이전 대화 히스토리가 비어있음

**디버깅 순서**:

1. **ChatState 정의 확인**
```python
# backend/app/multi_agent/state.py
# ChatState TypedDict의 필드 목록 확인
# 필수 필드가 누락되지 않았는지 확인
```

2. **request_to_state() 변환 확인**
```python
# backend/app/multi_agent/state.py
# ChatRequest → ChatState 변환 시 모든 필드가 매핑되는지 확인
# history, session_data 등이 올바르게 전달되는지 확인
```

3. **에이전트 run() 반환값 확인**
```python
# 에이전트의 run() 메서드 반환값이 ChatState 필드를 올바르게 업데이트하는지 확인
# 특히 messages 필드에 응답이 추가되는지 확인
```

**핵심 파일**:
- `backend/app/multi_agent/state.py` - ChatState, request_to_state()
- `backend/app/multi_agent/agents/base.py` - BaseChatAgent.run()

### 2.3 RAG 에이전트 실패 (검색 결과 없음)

**증상**: "판례 검색" → "관련 정보를 찾을 수 없습니다"

**디버깅 순서**:

1. **query_rewrite 확인** - 쿼리 변환이 올바른지
```python
# backend/app/services/rag/query_rewrite.py
# 입력 쿼리 → 검색용 쿼리 변환 결과 확인
```

2. **retrieval 확인** - LanceDB 검색 결과가 있는지
```python
# backend/app/services/rag/retrieval.py
# 벡터 검색 결과 건수, 유사도 점수 확인
```

3. **rerank 확인** - 재순위화 후 결과가 필터링되지 않았는지
```python
# backend/app/services/rag/rerank.py
# rerank 후 threshold 이상 결과가 있는지 확인
```

4. **search_focus 분기 확인**
```python
# backend/app/multi_agent/agents/legal_search_agent.py
# search_focus (precedent/statute/both) 값에 따른 분기 확인
```

**핵심 파일**:
- `backend/app/services/rag/` - query_rewrite, retrieval, rerank, pipeline
- `backend/app/multi_agent/agents/legal_search_agent.py`
- `backend/app/multi_agent/agents/law_study_agent.py`

### 2.4 서브그래프 오류 (small_claims)

**증상**: 소액소송 가이드 중 단계가 건너뛰어지거나 반복됨

**디버깅 순서**:

1. **서브그래프 상태 확인**
```python
# backend/app/multi_agent/subgraphs/small_claims.py
# 서브그래프 내부 상태(step, collected_info 등) 확인
```

2. **단계 전이 로직 확인**
```python
# 현재 단계 → 다음 단계 결정 로직
# conditional_edge 또는 Command(goto=...) 확인
```

3. **부모 그래프 → 서브그래프 상태 전달 확인**

### 2.5 스트리밍 오류 (프론트엔드)

**증상**: 응답이 한번에 출력되거나, 중간에 끊김

**디버깅 순서**:

1. **Backend SSE 확인**
```python
# backend/app/api/router/chat.py
# StreamingResponse, astream() 호출 확인
# yield 형식이 SSE 규격(data: ...\n\n)인지 확인
```

2. **Frontend useStreamingChat 확인**
```typescript
// frontend/src/hooks/useStreamingChat.ts
// EventSource 또는 fetch + ReadableStream 처리 확인
// onmessage 핸들러에서 파싱 오류 없는지 확인
```

3. **ChatWidget 렌더링 확인**
```typescript
// frontend/src/components/ChatWidget.tsx
// 스트리밍 중 상태 업데이트가 올바른지 확인
```

## 3. 빠른 진단 명령어

```bash
# 1. 멀티에이전트 관련 파일 전체 구조 확인
find backend/app/multi_agent -type f -name "*.py" | sort

# 2. 에이전트 등록 현황 (graph.py에서 add_node 확인)
grep -n "add_node\|add_edge\|add_conditional" backend/app/multi_agent/graph.py

# 3. 라우터 키워드 사전 확인
grep -n "AgentType\|ROLE_AGENTS" backend/app/multi_agent/router.py

# 4. ChatState 필드 확인
grep -n "TypedDict\|class ChatState" backend/app/multi_agent/state.py

# 5. 에이전트별 run() 시그니처 확인
grep -rn "async def run\|def run" backend/app/multi_agent/agents/

# 6. RAG 파이프라인 import 확인
grep -rn "from.*rag.*import\|import.*rag" backend/app/multi_agent/agents/
```

## 4. 로깅 추가 패턴

디버깅 시 임시 로깅을 추가할 위치:

```python
import logging
logger = logging.getLogger(__name__)

# router_node에 추가
logger.info(f"[ROUTER] message='{message[:50]}', route={agent_type}, confidence={confidence}")

# 에이전트 run()에 추가
logger.info(f"[{self.agent_type}] state_keys={list(state.keys())}, history_len={len(state.get('history', []))}")

# RAG pipeline에 추가
logger.info(f"[RAG] query='{query[:50]}', results_count={len(results)}, top_score={results[0].score if results else 'N/A'}")
```

## 5. 일반적인 실수 패턴

| 실수 | 증상 | 해결 |
|------|------|------|
| graph.py에 노드 미등록 | KeyError: 'node_name' | `add_node()` 추가 |
| Command(goto=...) 오타 | 라우팅 실패 | 노드명 상수화 |
| state 필드 오타 | 데이터 누락 | TypedDict 타입 체크 |
| import 순환 | ImportError | 지연 import 사용 |
| astream vs ainvoke 혼용 | 스트리밍 불가 | 일관된 호출 방식 |
| search_focus 미전달 | 항상 both 검색 | state에서 명시적 전달 |

