# query-rewrite-fix Planning Document

> **Summary**: 파이프라인 쿼리 리라이팅(`rewrite_query`)이 검색에 실제 적용되지 않는 치명적 버그 수정
>
> **Project**: law-3-team (법률 서비스 플랫폼)
> **Author**: Claude
> **Date**: 2026-02-24
> **Status**: Draft

---

## 1. Overview

### 1.1 Purpose

LangSmith 트레이싱에서 발견된 **치명적 버그 수정**: 법률 검색 최적화를 위한 쿼리 리라이팅(`rewrite_query`)이 실행되지 않아, 사용자 원본 쿼리가 그대로 벡터/키워드 검색에 전달되고 있음.

### 1.2 Background

#### 현재 쿼리 리라이팅 체계 (2가지)

| 함수 | 목적 | 위치 | 실태 |
|------|------|------|------|
| `rewrite_query()` | **법률 검색 최적화** — 일상 표현을 법률 용어로 변환 | `pipeline.py` | `enable_rewrite=False`로 **실행 안 됨** |
| `rewrite_conversational_query()` | follow-up 해소 — "그거 더 알려줘" → 독립 쿼리 | 에이전트 레벨 | 대부분 원본 반환 (사실상 무의미) |

#### 핵심 문제

1. **`rewrite_query()` 미실행**: `enable_rewrite=False`로 설정되어 법률 용어 변환이 전혀 안 됨
2. **`rewrite_conversational_query()` 무의미**: 법률 상담 쿼리는 거의 독립적 질문이라 follow-up 판정이 안 됨. 대부분 원본 그대로 반환하는 pass-through 함수에 불과

```
현재 흐름 (버그):
"사기당했어" → conversational_rewrite() → "사기당했어" (그대로 반환)
             → pipeline(enable_rewrite=False) → 검색("사기당했어")
             ← 법률 용어 변환 없이 원본 검색!

수정 후 흐름:
"사기당했어" → pipeline(enable_rewrite=True)
             → rewrite_query("사기당했어")
             → "사기죄 형사고소 손해배상청구 절차"
             → 검색("사기죄 형사고소 손해배상청구 절차")
```

#### 영향 범위

- `LegalSearchAgent` (판례/법령 검색) — 현재 유일한 파이프라인 사용 에이전트
- 향후 4개+ 에이전트가 파이프라인을 추가 사용 예정

### 1.3 Related Documents

- 코드: `backend/app/services/rag/query_rewrite.py`
- 코드: `backend/app/services/rag/pipeline.py`
- 코드: `backend/app/multi_agent/agents/legal_search_agent.py`
- 아카이브: `docs/archive/2026-02/rag-performance-optimization/`

---

## 2. Scope

### 2.1 In Scope

- [x] `LegalSearchAgent`의 `enable_rewrite=False` → `True` 변경
- [x] `rewrite_conversational_query()` 호출 제거 (LegalSearchAgent)
- [x] `rewrite_query()`의 동기 `model.invoke()` → 비동기 `model.ainvoke()` 전환
- [x] `pipeline.py`의 `execute_async()`에서 `asyncio.to_thread` 래핑 제거 → `await` 직접 호출
- [x] LangSmith 트레이싱에서 리라이팅 결과가 정확히 기록되는지 확인

### 2.2 Out of Scope

- `rewrite_conversational_query()` 함수 자체 삭제 (호출만 제거, 함수는 유지)
- 리라이팅 프롬프트 품질 개선 (별도 실험)
- 리라이팅 결과 캐싱 레이어 추가

---

## 3. Requirements

### 3.1 Functional Requirements

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| FR-01 | `rewrite_query()` 결과가 실제 벡터/키워드 검색 쿼리로 사용되어야 함 | **Critical** | Pending |
| FR-02 | `rewrite_query()`를 async로 전환하여 이벤트 루프 블로킹 방지 | High | Pending |
| FR-03 | `rewrite_conversational_query()` 호출 제거 (에이전트 레벨 불필요 리라이팅 삭제) | High | Pending |
| FR-04 | LangSmith 트레이싱에 original_query / rewritten_queries 정확히 기록 | Medium | Pending |

### 3.2 Non-Functional Requirements

| Category | Criteria | Measurement Method |
|----------|----------|-------------------|
| 레이턴시 | 리라이팅 추가로 인한 전체 응답시간 증가 ≤ 500ms | LangSmith 메트릭 비교 |
| 정확도 | 리라이팅 후 검색 Recall@10 기존 대비 개선 | RAG 평가 시스템 |
| 안정성 | 리라이팅 실패 시 원본 쿼리로 안전 폴백 | 기존 try-except 유지 |

---

## 4. Success Criteria

### 4.1 Definition of Done

- [ ] LangSmith에서 `rewrite_query` 트레이스가 정상 실행되고, 리라이팅 결과로 검색 수행 확인
- [ ] `ruff check` + `mypy` 통과
- [ ] 기존 테스트 통과

### 4.2 Quality Criteria

- [ ] `asyncio.to_thread` 래핑 제거 (async 네이티브로 전환)
- [ ] Zero lint errors

---

## 5. Risks and Mitigation

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| LLM 리라이팅 지연으로 전체 응답 느려짐 | Medium | Medium | Focus+Supplementary 병렬 실행이므로 레이턴시 영향 최소 |
| 리라이팅이 원본 의도를 왜곡 | Medium | Low | 기존 프롬프트 유지 + 리랭킹에서 보정 |
| `model.ainvoke()` 전환 시 호환성 | Low | Low | `get_chat_model()` 이미 LangChain 모델, ainvoke 지원 확인됨 |
| Focus+Supplementary에서 리라이팅 2회 호출 | Low | High | 병렬 실행이라 레이턴시 동일, LLM 비용 미미 |

---

## 6. Architecture Considerations

### 6.1 설계 원칙

파이프라인은 여러 에이전트가 공통으로 사용하는 **RAG 인프라**. 리라이팅은 파이프라인 내부에서 처리.

```
LegalSearchAgent ──┐
LawStudyAgent ─────┤
(향후 에이전트 A) ──┤→ search_with_pipeline_async(query, config)
(향후 에이전트 B) ──┘     └─ rewrite_query()     ← 파이프라인이 알아서 처리
                          └─ 검색 → 리랭킹 → 원문조회
```

에이전트는 사용자 쿼리를 넘기기만 하면 됨. 리라이팅 로직을 알 필요 없음.

### 6.2 수정 대상 파일

| 파일 | 변경 내용 |
|------|----------|
| `backend/app/services/rag/query_rewrite.py` | `rewrite_query_async()` 추가 (`model.ainvoke`) |
| `backend/app/services/rag/pipeline.py` | `execute_async()`에서 `asyncio.to_thread` → `await rewrite_query_async()` |
| `backend/app/multi_agent/agents/legal_search_agent.py` | `rewrite_conversational_query` 호출 제거, `enable_rewrite=True`로 변경 |

### 6.3 수정 후 데이터 흐름 (LegalSearchAgent)

```
사용자 메시지 ("사기당했어")
    │
    ▼
[Agent] process_stream() / process()
    │  (rewrite_conversational_query 호출 없음 — 제거됨)
    │  message를 그대로 _prepare_rag_data()에 전달
    ▼
[Agent] _prepare_rag_data(message)
    │
    ├─── [Pipeline: Focus]  enable_rewrite=True
    │       └─ await rewrite_query_async("사기당했어")
    │       → "사기죄 형사고소 손해배상청구 절차"
    │       └─ 벡터+FTS 검색 → 리랭킹 → 원문 조회
    │
    └─── [Pipeline: Supplementary]  enable_rewrite=True  (병렬)
            └─ await rewrite_query_async("사기당했어")
            → "사기죄 형사고소 손해배상청구 절차"
            └─ 벡터+FTS 검색 → 리랭킹 → 원문 조회
```

Focus + Supplementary 각각에서 리라이팅이 실행되지만, `asyncio.gather`로 **병렬 실행**이므로 레이턴시 영향 없음.

---

## 7. Convention Prerequisites

### 7.1 준수 사항

| Category | Rule |
|----------|------|
| Async | `model.invoke()` → `model.ainvoke()` (기존 패턴 준수) |
| 타입 | 모든 함수에 타입 힌트 필수 |
| 트레이싱 | `@traceable` 데코레이터 유지 |
| 폴백 | LLM 실패 시 원본 쿼리 반환 (기존 패턴) |
| 하위호환 | 동기 `rewrite_query()` 유지 (동기 `execute()`에서 사용) |

---

## 8. Implementation Steps

### Step 1: `rewrite_query_async()` 추가

`query_rewrite.py`에 비동기 버전 추가.
- `model.invoke()` → `model.ainvoke()`
- `@traceable` 유지
- 기존 동기 `rewrite_query()`는 하위 호환을 위해 유지

### Step 2: `pipeline.py` 수정

`execute_async()`의 리라이팅 호출을 async 네이티브로 변경.
- `asyncio.to_thread(rewrite_query, ...)` → `await rewrite_query_async(...)`

### Step 3: `LegalSearchAgent` 수정

1. `rewrite_conversational_query` import/호출 제거
2. `FOCUS_CONFIG` / `SUPPLEMENTARY_CONFIG`에서 `enable_rewrite=False` → 삭제 (기본값 `True` 사용)
3. `process()` / `process_stream()`에서 message를 그대로 `_prepare_rag_data()`에 전달

### Step 4: 검증

- `ruff check` + `mypy` 통과
- LangSmith 트레이싱에서 리라이팅 결과 확인

---

## 9. Next Steps

1. [ ] Design 문서 작성 (`query-rewrite-fix.design.md`)
2. [ ] 구현
3. [ ] LangSmith 트레이싱 검증

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-02-24 | Initial draft | Claude |
| 0.2 | 2026-02-24 | conversational_rewrite 제거, pre_rewritten_queries 제거, 파이프라인 공통 인프라 설계로 단순화 | Claude |
