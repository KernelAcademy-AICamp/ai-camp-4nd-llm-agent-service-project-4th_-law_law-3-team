# Plan: LLM Context 구조화 필드 기반 포맷팅

> **Feature**: llm-context-structured-fields
> **Phase**: Plan
> **Created**: 2026-02-24
> **Status**: Draft

---

## 1. 배경 및 목적

### 현재 상태

`format_*_context()` 함수들이 `doc.get("content", "")`로 원문을 통째로 LLM context에 삽입.
`content`는 `fetch_document_contents()`가 `DOCUMENT_TABLE_REGISTRY`의 `content_columns`를 `\n\n`으로 조인한 단일 문자열.

**문제점:**
- 컬럼 구분 없이 하나의 blob으로 LLM에 전달 → 구조 정보 소실
- `format_precedent_context()`에서 `content`(ruling+reasoning 조인)와 `details`(ruling, reasoning 개별)가 **중복**
- 문서 타입별로 어떤 필드인지 LLM이 인식 불가

### 목적

`DOCUMENT_TABLE_REGISTRY`에 정의된 **컬럼별 구조화 데이터**를 활용하여, 각 문서의 LLM context를 `[라벨] 값` 형태로 명확하게 포맷팅한다.

### 범위

- **포함**: `fetch_document_contents` 반환 구조 변경, `_populate_content` 변경, `format_*_context` 함수 리팩토링, `COLUMN_LABELS` 매핑 추가
- **제외**: rerank 로직, ai_summary 조회, 프론트엔드 sources 포맷팅, PrecedentService.get_details()

---

## 2. 목표

| 항목 | Before | After |
|------|--------|-------|
| LLM context 내 필드 구분 | 없음 (blob) | `[주문]`, `[판결요지]` 등 라벨 구분 |
| 판례 content/details 중복 | 중복 발생 | 제거 |
| 문서 식별 | content만 | data_type + title + id + 구조화 필드 |

---

## 3. 현재 구조 → 변경 후 구조

### 3.1 현재 LLM context 출력 (판례 예시)

```
[판례 1] 손해배상(기) (2020다12345)
주문 텍스트\n\n판결요지 텍스트     ← content (blob)
[주문] 주문 텍스트                  ← details에서 또 추가 (중복)
[판결요지] 판결요지 텍스트          ← details에서 또 추가 (중복)
```

### 3.2 변경 후 LLM context 출력 (판례 예시)

```
[판례 1] 손해배상(기) (serial: 76396)
[주문] 피고는 원고에게 금 1,000만원을...
[판결요지] 불법행위로 인한 손해배상...
```

### 3.3 변경 후 LLM context 출력 (부처유권해석 예시)

```
[부처유권해석 1] 근로자 퇴직금 지급 의무 (serial: 12345)
[회답] ...
[이유] ...
```

---

## 4. 구현 계획

### Step 1: `COLUMN_LABELS` 매핑 추가 (`format_utils.py`)

컬럼명 → 한국어 라벨 매핑. ORM 모델의 `comment`와 일치.

```python
COLUMN_LABELS: dict[str, str] = {
    "ruling": "주문",
    "reasoning": "판결요지",
    "content": "내용",
    "answer": "회답",
    "reason": "이유",
    "judgment_summary": "판정요지",
    "judgment_result": "판정결과",
    "action_reason": "조치이유",
    "action_content": "조치내용",
    "evaluation_opinion": "평가의견",
}
```

### Step 2: `DOCUMENT_TITLE_COLUMN` 매핑 추가 (`format_utils.py`)

각 data_type별 제목/ID로 쓸 컬럼 정의. 현재 `metadata`에서 가져올 수 있는 필드 활용.

```python
# metadata에서 title로 쓸 필드 (기본: case_name → title 순 fallback)
# metadata에서 id로 쓸 필드 (기본: doc_id)
# → 이미 metadata에 case_name, doc_id가 들어있으므로 추가 매핑 불필요
```

### Step 3: `fetch_document_contents` 반환 타입 변경 (`retrieval.py`)

**Before**: `dict[str, str]` — `{source_id: "col1\n\ncol2"}`
**After**: `dict[str, dict[str, str]]` — `{source_id: {"ruling": "...", "reasoning": "..."}}`

동기 `fetch_document_contents`, 비동기 `fetch_document_contents_async`, 헬퍼 `_fetch_contents_for_type` 모두 변경.

### Step 4: `_populate_content` 변경 (`retrieval.py`)

**Before**: `doc["content"] = contents[sid]` (단일 문자열)
**After**: `doc["content_fields"] = contents[sid]` (구조화 dict) + `doc["content"]` = 조인 문자열 (하위 호환)

```python
def _populate_content(
    docs: list[dict[str, Any]],
    contents: dict[str, dict[str, str]],
) -> None:
    for doc in docs:
        sid = doc.get("metadata", {}).get("doc_id", "")
        if sid and sid in contents:
            fields = contents[sid]
            doc["content_fields"] = fields
            doc["content"] = "\n\n".join(fields.values())  # 하위 호환
```

### Step 5: `format_utils.py` — 통합 포맷 함수 구현

기존 3개 함수(`format_precedent_context`, `format_law_context`, `format_supplementary_context`)를 유지하되, 내부에서 `content_fields`를 사용하도록 변경.

**공통 헬퍼:**

```python
def _format_fields(doc: dict[str, Any]) -> str:
    """content_fields를 [라벨] 값 형태로 포맷."""
    fields = doc.get("content_fields", {})
    if not fields:
        return doc.get("content", "")  # fallback
    parts = []
    for col_name, value in fields.items():
        label = COLUMN_LABELS.get(col_name, col_name)
        parts.append(f"[{label}] {value}")
    return "\n".join(parts)
```

**format_precedent_context 변경:**

```python
def format_precedent_context(documents, details=None) -> str:
    # details 파라미터는 유지 (프론트엔드 sources에서 사용)
    # 그러나 LLM context에서는 content_fields만 사용, details 중복 삽입 제거
    for doc in documents:
        metadata = doc.get("metadata", {})
        title = metadata.get("case_name", "")
        doc_id = metadata.get("doc_id", "")
        part = f"[판례 {i}] {title} (serial: {doc_id})\n{_format_fields(doc)}"
```

### Step 6: `legal_search_agent.py` — details 전달 정리

`_build_context()`에서 `format_precedent_context(docs, details)` 호출은 유지하되, 함수 내부에서 details를 LLM context에 중복 삽입하지 않으므로 자연스럽게 해결.

`_format_sources()`는 변경 없음 (프론트엔드용 — details 그대로 사용).

---

## 5. 영향 범위

| 파일 | 변경 내용 |
|------|----------|
| `backend/app/services/rag/format_utils.py` | `COLUMN_LABELS` 추가, `_format_fields` 헬퍼, `format_*_context` 4개 함수 리팩토링 |
| `backend/app/services/rag/retrieval.py` | `fetch_document_contents` 반환 타입 변경, `_fetch_contents_for_type` 변경, `_populate_content` 변경 |
| `backend/app/services/rag/__init__.py` | `COLUMN_LABELS` export 추가 (선택) |

### 변경 없는 파일

| 파일 | 이유 |
|------|------|
| `pipeline.py` | `_populate_content` 시그니처 유지, 호출 코드 변경 불필요 |
| `legal_search_agent.py` | `_build_context` 호출 방식 동일 |
| `rerank.py` | rerank는 `content` 문자열 사용 → 하위 호환 유지 |
| `format_utils.py` (sources 함수들) | 프론트엔드용 — 변경 불필요 |

---

## 6. 하위 호환성

| 항목 | 보장 방식 |
|------|----------|
| `doc["content"]` | 조인 문자열 유지 (rerank, 기타 코드에서 사용) |
| `format_*_sources()` | 변경 없음 |
| `PrecedentService.get_details()` | 변경 없음 |
| `ai_summary` 조회 | `fetch_ai_summaries`는 변경 없음 (`dict[str, str]` 유지) |

---

## 7. 체크리스트

- [ ] `COLUMN_LABELS` 매핑이 모든 content_columns를 커버하는지 확인
- [ ] `_populate_content` 하위 호환 (`content` 필드 유지)
- [ ] rerank 동작 확인 (`content` 문자열 기반 — 영향 없음)
- [ ] `format_precedent_context`에서 details 중복 제거 확인
- [ ] ruff check + mypy 통과
