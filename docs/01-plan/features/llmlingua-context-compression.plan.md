# Plan: LLMLingua 기반 RAG Context 압축

> **Feature**: llmlingua-context-compression
> **Phase**: Plan
> **Created**: 2026-02-26
> **Status**: Draft
> **선행 작업**: `llm-context-structured-fields` (구조화 필드 기반 포맷팅, 이미 구현 완료)

---

## 1. 배경 및 목적

### 현재 상태

RAG 파이프라인이 리랭킹 후 top-k 문서의 **원문 전체**를 PostgreSQL에서 가져와 LLM context로 전달한다.

**문서별 원문 구조** (`DOCUMENT_TABLE_REGISTRY` 기반):

| data_type | content_columns | 평균 길이 (추정) |
|-----------|----------------|-----------------|
| 판례 | `ruling`, `reasoning` | ruling 200자 + reasoning 2,000~10,000자 |
| 법령 | `content` | 500~5,000자 |
| 헌재결정례 | `ruling`, `reasoning` | 유사 |
| 행정심판례 | `ruling`, `reason` | 유사 |
| 부처유권해석 | `answer`, `reason` | 500~3,000자 |
| 법령해석례 | `answer`, `reason` | 유사 |
| 위원회결정례 (11종) | 각 테이블별 2~3개 컬럼 | 다양 |
| 자치법규 | `content`, `overall_summary` | 1,000~10,000자 |

**문제:**
- focus 5건 + supplementary 2건 = 7건의 원문이 LLM context에 삽입
- 판례 `reasoning`이 10,000자를 넘는 경우 빈번
- **총 context가 30,000~50,000자(15K~25K 토큰)에 달하여 LLM 비용 증가 + 응답 품질 저하**
- Solar Pro3 등 context window 제한이 있는 모델에서 truncation 위험

### 목적

**LLMLingua-2**를 활용하여 `DOCUMENT_TABLE_REGISTRY`의 **컬럼별 압축 전략**을 적용, LLM context 길이를 50~70% 수준으로 줄이면서 핵심 법률 정보를 보존한다.

### 범위

- **포함**: LLMLingua-2 통합, 컬럼별 압축률 설정, 파이프라인 삽입, Feature Flag
- **제외**: 리랭킹 로직 변경, 프론트엔드 sources 포맷팅 (원문 그대로 전달), ai_summary 변경

---

## 2. 전략 분석 (LLMLingua-2 적합성 판단)

### 2.1 장점 (채택 근거)

| 항목 | 설명 |
|------|------|
| **컬럼별 차등 압축** | `ruling`(주문)은 압축 안 함, `reasoning`(판결요지)은 0.3~0.5 비율로 압축 가능 |
| **structured_compress_prompt** | `<llmlingua>` 태그로 섹션별 압축률 지정 → `DOCUMENT_TABLE_REGISTRY`와 1:1 매핑 가능 |
| **다국어 지원** | XLM-RoBERTa 기반 → 한국어 토큰 분류 가능 |
| **task-agnostic** | 법률 도메인 fine-tuning 없이도 토큰 중요도 기반 압축 |
| **기존 아키텍처 호환** | `_populate_content` → `format_*_context` 사이에 압축 단계 삽입 가능 |

### 2.2 리스크 및 대응

| 리스크 | 심각도 | 대응 방안 |
|--------|--------|----------|
| **한국어 법률 용어 손실** | 높음 | `force_tokens`에 법률 핵심 토큰 등록 + 컬럼별 최소 압축률 설정 |
| **모델 로딩 지연 (XLM-RoBERTa ~1.2GB)** | 중간 | 서버 시작 시 lazy loading, 임베딩/리랭커와 동일한 패턴 |
| **추론 지연 (+200~500ms/문서)** | 중간 | 문서 수 5~7건 × 200ms = 1~1.5s 추가. 비동기 병렬 압축으로 완화 |
| **압축 후 법률 의미 변질** | 높음 | 핵심 컬럼(ruling, answer) 압축 제외 + 압축률 보수적 설정 |
| **GPU 의존성** | 낮음 | CPU에서도 동작 (XLM-RoBERTa는 BERT급, ~100ms/문서) |

### 2.3 대안 비교

| 방안 | 장점 | 단점 | 선택 |
|------|------|------|------|
| **LLMLingua-2** | 컬럼별 차등 압축, 의미 보존, 한국어 지원 | 모델 추가 로딩 | **채택** |
| 단순 truncation (앞 N자 자르기) | 구현 간단 | 뒤쪽 핵심 내용 유실, 법적 결론 누락 | X |
| LLM 기반 요약 (GPT/Solar) | 요약 품질 높음 | 추가 API 호출 비용 + 1~3초 지연 | X |
| ai_summary만 전달 (원문 제외) | 즉시 구현 | reasoning 등 세부 근거 누락, 상담 품질 저하 | X |

### 2.4 결론

**LLMLingua-2 컬럼별 차등 압축 전략은 적합하다.**

핵심 이유:
1. `DOCUMENT_TABLE_REGISTRY`에 이미 컬럼별 구조가 정의되어 있어, 압축 설정을 자연스럽게 매핑 가능
2. `content_fields` dict 구조(기 구현)가 컬럼별 독립 압축의 기반
3. 법적 결론(`ruling`) 보존 + 장문 근거(`reasoning`) 압축이라는 차등 전략이 법률 도메인에 적합

---

## 3. 컬럼별 압축 전략

### 3.1 압축 정책 매핑

| 컬럼명 | 한국어 라벨 | 압축 정책 | 압축률 (rate) | 근거 |
|--------|-----------|----------|-------------|------|
| `ruling` | 주문 | **압축 안 함** | 1.0 | 판결 결론, 단어 하나가 결과를 바꿈 |
| `reasoning` | 판결요지 | **압축** | 0.4 | 장문 근거, 핵심 논리만 보존 |
| `content` (법령) | 내용 | **압축** | 0.5 | 조문 전체보다 핵심 조항 보존 |
| `answer` | 회답 | **압축 안 함** | 1.0 | 유권해석 결론 |
| `reason` | 이유 | **압축** | 0.4 | 장문 근거 |
| `judgment_summary` | 판정요지 | **압축 안 함** | 1.0 | 판정 결론 |
| `judgment_result` | 판정결과 | **압축 안 함** | 1.0 | 판정 결과 |
| `action_reason` | 조치이유 | **압축** | 0.5 | 중간 길이 근거 |
| `action_content` | 조치내용 | **압축 안 함** | 1.0 | 조치 결론 |
| `evaluation_opinion` | 평가의견 | **압축** | 0.5 | 장문 의견 |
| `overall_summary` | 전체요약 | **압축 안 함** | 1.0 | 이미 요약된 텍스트 |

### 3.2 압축 규칙

1. **결론/결과 컬럼** (ruling, answer, judgment_summary, judgment_result, action_content, overall_summary) → **압축 안 함** (rate=1.0)
2. **근거/이유 컬럼** (reasoning, reason, evaluation_opinion) → **적극 압축** (rate=0.3~0.4)
3. **혼합 컬럼** (content, action_reason) → **중간 압축** (rate=0.5)
4. **짧은 텍스트** (500자 미만) → **압축 스킵** (오버헤드 대비 효과 미미)

### 3.3 `force_tokens` (압축 시 보존할 토큰)

```python
FORCE_TOKENS: list[str] = [
    "\n", ".", ",", "?",           # 구조 보존
    "제", "조", "항", "호",         # 법조문 번호 (제1조, 제2항)
    "원고", "피고", "법원",          # 소송 당사자
    "기각", "인용", "각하", "취소",  # 판결 결과
    "손해배상", "위법", "위반",      # 핵심 법률 용어
]
```

---

## 4. 아키텍처

### 4.1 파이프라인 삽입 위치

```
현재:
  검색 → 리랭킹 → top-k 원문 조회 → _populate_content → format_*_context → LLM

변경:
  검색 → 리랭킹 → top-k 원문 조회 → _populate_content → [★ 압축] → format_*_context → LLM
```

압축은 `_populate_content` 이후, `format_*_context` 이전에 수행.
`content_fields` dict의 각 value를 컬럼별 정책에 따라 독립 압축.

### 4.2 모듈 구조

```
backend/app/services/rag/
├── compression.py          # [신규] LLMLingua 래퍼 + 컬럼별 압축 로직
├── pipeline.py             # 압축 단계 호출 추가
├── retrieval.py            # 변경 없음
└── format_utils.py         # 변경 없음 (이미 content_fields 기반)
```

### 4.3 `compression.py` 설계

```python
"""RAG Context 압축 모듈 (LLMLingua-2 기반)"""

from dataclasses import dataclass
from typing import Any

@dataclass
class ColumnCompressionPolicy:
    """컬럼별 압축 정책."""
    rate: float = 0.4              # 압축률 (0.0~1.0, 1.0=압축 안 함)
    min_length: int = 500          # 이 길이 미만이면 압축 스킵
    force_tokens: list[str] | None = None  # 추가 보존 토큰

# DOCUMENT_TABLE_REGISTRY 컬럼 → 압축 정책 매핑
COLUMN_COMPRESSION_POLICIES: dict[str, ColumnCompressionPolicy] = {
    # 결론/결과 → 압축 안 함
    "ruling": ColumnCompressionPolicy(rate=1.0),
    "answer": ColumnCompressionPolicy(rate=1.0),
    "judgment_summary": ColumnCompressionPolicy(rate=1.0),
    "judgment_result": ColumnCompressionPolicy(rate=1.0),
    "action_content": ColumnCompressionPolicy(rate=1.0),
    "overall_summary": ColumnCompressionPolicy(rate=1.0),
    # 근거/이유 → 적극 압축
    "reasoning": ColumnCompressionPolicy(rate=0.4),
    "reason": ColumnCompressionPolicy(rate=0.4),
    "evaluation_opinion": ColumnCompressionPolicy(rate=0.5),
    # 혼합 → 중간 압축
    "content": ColumnCompressionPolicy(rate=0.5),
    "action_reason": ColumnCompressionPolicy(rate=0.5),
}

DEFAULT_POLICY = ColumnCompressionPolicy(rate=0.5)


class ContextCompressor:
    """LLMLingua-2 기반 컨텍스트 압축기."""

    def __init__(self) -> None:
        self._compressor = None  # lazy loading

    def _get_compressor(self):
        if self._compressor is None:
            from llmlingua import PromptCompressor
            self._compressor = PromptCompressor(
                model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                use_llmlingua2=True,
                device_map="cpu",  # GPU 사용 시 "cuda"
            )
        return self._compressor

    def compress_document_fields(
        self,
        content_fields: dict[str, str],
    ) -> dict[str, str]:
        """content_fields를 컬럼별 정책에 따라 압축."""
        compressed: dict[str, str] = {}
        compressor = self._get_compressor()

        for col_name, text in content_fields.items():
            policy = COLUMN_COMPRESSION_POLICIES.get(col_name, DEFAULT_POLICY)

            # 압축 안 함 or 짧은 텍스트 → 원문 유지
            if policy.rate >= 1.0 or len(text) < policy.min_length:
                compressed[col_name] = text
                continue

            # LLMLingua-2 압축
            result = compressor.compress_prompt(
                [text],
                rate=policy.rate,
                force_tokens=FORCE_TOKENS + (policy.force_tokens or []),
            )
            compressed[col_name] = result["compressed_prompt"]

        return compressed

    def compress_documents(
        self,
        documents: list[dict[str, Any]],
    ) -> None:
        """문서 리스트의 content_fields를 in-place 압축."""
        for doc in documents:
            fields = doc.get("content_fields")
            if fields:
                doc["content_fields"] = self.compress_document_fields(fields)
                doc["content"] = "\n\n".join(doc["content_fields"].values())
```

### 4.4 파이프라인 통합 (`pipeline.py`)

```python
# _rerank_and_fetch 내부, _populate_content 이후에 삽입

if settings.ENABLE_CONTEXT_COMPRESSION:
    from app.services.rag.compression import get_context_compressor
    compressor = get_context_compressor()
    await asyncio.to_thread(compressor.compress_documents, reranked)
```

---

## 5. 환경 변수 (Feature Flag)

| 변수 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `ENABLE_CONTEXT_COMPRESSION` | bool | `false` | 컨텍스트 압축 활성화 |
| `COMPRESSION_MODEL` | str | `microsoft/llmlingua-2-xlm-roberta-large-meetingbank` | 압축 모델명 |
| `COMPRESSION_DEFAULT_RATE` | float | `0.5` | 미등록 컬럼 기본 압축률 |
| `COMPRESSION_MIN_LENGTH` | int | `500` | 압축 스킵 최소 글자 수 |
| `COMPRESSION_DEVICE` | str | `cpu` | 디바이스 (`cpu` / `cuda`) |

---

## 6. 구현 순서

### Phase 1: 기반 모듈 (compression.py)

1. `compression.py` 신규 생성 — `ContextCompressor` 클래스 + 컬럼별 정책
2. `pyproject.toml`에 `llmlingua` 의존성 추가
3. `config.py`에 Feature Flag 추가
4. 단위 테스트: 압축률 검증, force_tokens 보존 확인

### Phase 2: 파이프라인 통합

5. `pipeline.py` — `_rerank_and_fetch` 내에 압축 단계 삽입 (동기 + async)
6. 통합 테스트: 압축 전/후 context 길이 비교

### Phase 3: 한국어 품질 검증

7. 법률 도메인 압축 품질 평가
   - 판례 `reasoning` 10건 수동 검증 (핵심 법리 보존 여부)
   - 법조문 번호 (`제1조`, `제2항`) 보존 확인
   - 압축률별 (0.3, 0.4, 0.5) 품질 비교
8. RAG 평가 시스템과 연동하여 압축 전/후 응답 품질 비교

### Phase 4: 최적화

9. 문서별 병렬 압축 (`asyncio.gather` + `to_thread`)
10. 모델 캐시 경로 통합 (`data/models/`)
11. 압축 메트릭 LangSmith 기록 (압축 전/후 토큰 수, 소요 시간)

---

## 7. 영향 범위

| 파일 | 변경 내용 |
|------|----------|
| `backend/app/services/rag/compression.py` | **신규** — 압축 모듈 |
| `backend/app/services/rag/pipeline.py` | 압축 단계 호출 추가 |
| `backend/app/core/config.py` | Feature Flag 추가 |
| `backend/pyproject.toml` | `llmlingua` 의존성 추가 |
| `backend/app/services/rag/__init__.py` | export 추가 |

### 변경 없는 파일

| 파일 | 이유 |
|------|------|
| `retrieval.py` | `_populate_content`는 변경 없음 (content_fields 이미 구현) |
| `format_utils.py` | `_format_fields`가 content_fields dict를 그대로 사용 |
| `legal_search_agent.py` | 파이프라인 내부 변경이므로 호출 코드 불변 |
| `rerank.py` | 리랭킹은 ai_summary 기반, 압축 대상 아님 |

---

## 8. 성능 예상

| 지표 | Before | After (예상) |
|------|--------|-------------|
| LLM context 길이 (판례 5건) | ~25,000자 (~12K 토큰) | ~12,000자 (~6K 토큰) |
| LLM context 길이 (법령 5건) | ~15,000자 (~7K 토큰) | ~8,000자 (~4K 토큰) |
| 압축 소요 시간 (5건 병렬) | 0ms | ~300~500ms (CPU) |
| LLM API 비용 | baseline | ~50% 절감 |
| 전체 파이프라인 지연 | baseline | +300~500ms |

---

## 9. 하위 호환성

| 항목 | 보장 방식 |
|------|----------|
| `ENABLE_CONTEXT_COMPRESSION=false` | 기존 동작 100% 유지 (기본값) |
| `doc["content"]` | 압축 후에도 `"\n\n".join(fields.values())`로 재조인 |
| `doc["content_fields"]` | 압축된 값으로 교체 (format_utils 호환) |
| 프론트엔드 sources | `format_*_sources()`는 원문 사용 (압축 대상 아님) |

---

## 10. 체크리스트

- [ ] `llmlingua` 패키지 설치 및 XLM-RoBERTa 모델 다운로드 확인
- [ ] 컬럼별 압축 정책이 `DOCUMENT_TABLE_REGISTRY` 모든 컬럼 커버
- [ ] `ruling`, `answer` 등 결론 컬럼이 압축되지 않는지 확인
- [ ] `force_tokens`에 법률 핵심 토큰 등록
- [ ] 압축 전/후 RAG 응답 품질 비교 (최소 10건)
- [ ] Feature Flag off 시 기존 동작 유지 확인
- [ ] ruff check + mypy 통과
- [ ] LangSmith에 압축 메트릭 기록

---

## 11. 참고 자료

- [LLMLingua-2 논문](https://arxiv.org/abs/2403.12968) (ACL 2024)
- [LLMLingua GitHub](https://github.com/microsoft/LLMLingua)
- [structured_compress_prompt API](https://llmlingua.com/llmlingua2.html)
- 선행 작업: `llm-context-structured-fields.plan.md` (content_fields 구조화)
