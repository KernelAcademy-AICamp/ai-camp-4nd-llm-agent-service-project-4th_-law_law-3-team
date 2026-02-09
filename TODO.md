# LangGraph 대화 기억 문제 TODO

> 분석일: 2026-02-09
> 브랜치: `fix/langgraph-spec-panel-improvements` (기존 개선 작업 위)

---

## 문제 1: 라우팅 리셋 (Follow-up 메시지가 다른 에이전트로 빠짐)

### 증상

사용자가 "손해배상 판례 알려줘" → legal_search 응답 후,
"더 자세히 알려줘"라고 후속 질문하면 **GENERAL 에이전트**로 라우팅됨.

### 원인

`router.py`의 키워드 매칭이 매 메시지마다 재실행됨.
"더 자세히 알려줘"에는 법률 관련 키워드가 없어서 기본 폴백(GENERAL)으로 분류.

### 현재 상태 (90% 구현됨)

`active_agent` 세션 고정 메커니즘이 이미 존재하지만, 데이터 흐름이 불안정함:

1. **에이전트 → state 설정**: `nodes.py:94` - 모든 에이전트가 `output_session_data["active_agent"]` 설정 ✅
2. **router에서 읽기**: `router.py:220-225` - `session_data.get("active_agent")` 읽어서 세션 유지 ✅
3. **스트리밍 done 이벤트**: `chat.py:238-247` - `active_agent` 누락 ❌

### 해결 방법

**`chat.py` 스트리밍 done 이벤트에 `active_agent` 포함:**

```python
# chat.py:238-247 수정
# 스트리밍 종료 후 최종 state에서 active_agent 추출하여 done 이벤트에 포함
graph_state_final = await graph.aget_state(config)
final_session_data = graph_state_final.values.get("output_session_data", {})

yield {
    "event": "done",
    "data": json.dumps({
        "thread_id": thread_id,
        "session_secret": session_secret,
        "active_agent": final_session_data.get("active_agent", ""),
    }, ensure_ascii=False),
}
```

**프론트엔드 `ChatWidget.tsx:514-518` 세션 병합 확인:**
- done 이벤트의 `active_agent`가 `sessionData`에 정상 병합되는지 확인 필요

### 관련 파일

| 파일 | 위치 | 역할 |
|------|------|------|
| `backend/app/api/router/chat.py` | 238-247행 | done 이벤트 (수정 필요) |
| `backend/app/multi_agent/router.py` | 220-225행 | active_agent 읽기 (이미 구현) |
| `backend/app/multi_agent/nodes.py` | 94행 | active_agent 설정 (이미 구현) |
| `frontend/src/components/ChatWidget.tsx` | 514-518행 | 세션 병합 (확인 필요) |

### 예상 작업량

- 백엔드: chat.py done 이벤트 수정 (~10줄)
- 프론트엔드: ChatWidget 세션 병합 검증/수정 (~5줄)
- 테스트: 스트리밍 응답에 active_agent 포함 확인

---

## 문제 2: RAG 검색이 현재 메시지만 사용 (대화 맥락 무시)

### 증상

사용자가 "손해배상 판례 알려줘" → 관련 판례 응답 후,
"더 자세히 알려줘"라고 하면 RAG가 "더 자세히 알려줘"로 벡터 검색 → 엉뚱한 결과 반환.

### 원인

`legal_search_agent.py`의 `_prepare_rag_data(message)`가 `message` 파라미터만 사용.
`history`는 `_build_messages()`에서 LLM 프롬프트 구성에만 사용되고, RAG 검색에는 반영되지 않음.

```python
# legal_search_agent.py:180행
_, _, _, _, context, sources = await self._prepare_rag_data(message)
# ↑ message만 전달, history 무시
```

### 해결 방법: Conversational Query Rewriting

후속 질문을 이전 대화 맥락을 포함한 독립 질문으로 변환 후 RAG 검색에 사용.

**변환 예시:**
```
history: [
  {"role": "user", "content": "손해배상 판례 알려줘"},
  {"role": "assistant", "content": "손해배상 관련 판례를 안내드립니다..."}
]
message: "더 자세히 알려줘"

→ rewritten_query: "손해배상 관련 판례의 상세 내용과 판결 요지"
```

**구현 방안:**

```python
# 1. query_rewrite.py에 대화 기반 리라이트 함수 추가
async def rewrite_conversational_query(
    message: str,
    history: list[dict[str, str]],
) -> str:
    """후속 질문을 독립적인 검색 쿼리로 변환"""
    # 키워드 기반 빠른 판단: 대명사/지시어 감지
    needs_rewrite = _is_follow_up(message)  # "더", "그거", "자세히" 등
    if not needs_rewrite:
        return message

    # LLM으로 히스토리 기반 쿼리 리라이트
    prompt = f"""이전 대화를 참고하여, 사용자의 후속 질문을 독립적인 검색 쿼리로 변환하세요.

이전 대화:
{format_history(history[-4:])}  # 최근 2턴만

후속 질문: {message}

독립적 검색 쿼리:"""
    return await llm_call(prompt)

# 2. legal_search_agent.py의 process()에서 사용
async def process(self, message, history, ...):
    rewritten = await rewrite_conversational_query(message, history)
    _, _, _, _, context, sources = await self._prepare_rag_data(rewritten)
    # ... 이후 동일
```

**주의사항:**
- LLM 호출 추가로 인한 지연 (약 0.5~1초)
- 단순 질문(키워드 충분한 경우)은 리라이트 스킵하여 비용 절감
- `query_rewrite.py`에 이미 `rewrite_query()` 함수가 존재하므로, 이를 확장하는 방향으로 구현
- `law_study_agent.py`에도 동일 패턴 적용 필요 (RAG 사용 에이전트)

### 관련 파일

| 파일 | 위치 | 역할 |
|------|------|------|
| `backend/app/multi_agent/agents/legal_search_agent.py` | 180행 | RAG 검색 호출 (수정 필요) |
| `backend/app/multi_agent/agents/law_study_agent.py` | - | 동일 패턴 적용 필요 |
| `backend/app/services/rag/query_rewrite.py` | - | 기존 쿼리 리라이트 (확장 필요) |

### 예상 작업량

- `query_rewrite.py` 확장: 대화 기반 리라이트 함수 추가 (~50줄)
- `legal_search_agent.py` 수정: process/process_stream에서 리라이트 호출 (~10줄)
- `law_study_agent.py` 수정: 동일 패턴 (~10줄)
- 테스트: 리라이트 함수 단위 테스트 + 통합 테스트

---

## 구현 순서 권장

1. **문제 1 먼저** (라우팅 리셋) — 변경량 적고, 기존 메커니즘 안정화만 필요
2. **문제 2** (RAG 컨텍스트) — LLM 호출 추가, 프롬프트 튜닝 필요

## 참고: thread_id 필요성

- `thread_id`는 현재 **소액소송(small_claims) interrupt/resume 패턴에서만** 실질적으로 필요
- 나머지 6개 에이전트는 프론트엔드가 history를 직접 전송하므로 thread_id 불필요
- 향후 multi-step 워크플로우 에이전트 추가 시 thread_id 필요성 증가
