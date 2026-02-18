# Plan: RAG 서비스 고도화

> **Feature**: rag-service
> **Phase**: Plan
> **Created**: 2026-02-19
> **Updated**: 2026-02-19 (데이터 탐색 결과 반영)
> **Status**: Draft

---

## 1. 배경 및 목적

### 현재 상태

`backend/app/services/rag/` 에 RAG 파이프라인 초안이 존재하나, 파일 구조 정의 시점에 작성된 코드로 **프로덕션 수준의 고도화가 필요**함.

현재 파일 구조:
```
app/services/rag/
├── __init__.py          # 모듈 export
├── embedding.py         # 임베딩 모델 (KURE-v1) ✅ 완성
├── retrieval.py         # 벡터 검색 (LanceDB) — 고도화 필요
├── rerank.py            # 리랭킹 (Cross-encoder) — 고도화 필요
├── pipeline.py          # 통합 파이프라인 — 고도화 필요
├── keyword_search.py    # PostgreSQL FTS ✅ 완성
├── fusion.py            # RRF 병합 ✅ 완성
├── query_rewrite.py     # 쿼리 리라이팅 ✅ 완성
└── tsvector_builder.py  # tsvector 생성 ✅ 완성
```

### 고도화 대상 (3개 파일)

| 파일 | 현재 상태 | 고도화 방향 |
|------|----------|------------|
| `retrieval.py` | LanceDB 벡터검색 + FTS 하이브리드 검색 구현되어 있으나, content를 LanceDB/PG에서 개별 조회하는 비효율 | **ID 기반 배치 원문 조회** 도입, 벡터/FTS에서는 ID만 추출 |
| `rerank.py` | 단순 Cross-encoder predict + sigmoid만 구현 | **배치 처리, 스코어 정규화, 최소 점수 필터링, content 길이 적응형 truncation** |
| `pipeline.py` | PipelineConfig 데이터클래스 기반, 단순 순차 실행 | **명확한 4단계 파이프라인 (검색→원문조회→리랭킹→결과)**, 에이전트별 프리셋, 단계별 로깅/메트릭 |

### 고도화 목적

RAG 파이프라인은 다음 기능들의 **공통 인프라**로 사용됨:

| 사용 에이전트 | 용도 | RAG 설정 |
|-------------|------|---------|
| `LegalSearchAgent` | 판례/법령 RAG 검색 | 판례 4 + 법령 1 (focus="precedent") 또는 역방향 |
| `LawStudyAgent` | 로스쿨 학습 자료 | 법령 3 (doc_type="law") |
| `SmallClaimsAgent` | 소액소송 가이드 | 판례/법령 혼합 |

→ 에이전트마다 다른 설정을 파이프라인 프리셋으로 제공해야 함.

---

## 2. 목표 아키텍처

### 2.1 핵심 설계 원칙

사용자가 지정한 RAG 파이프라인 흐름:

```
사용자 쿼리
    │
    ▼
┌─────────────────────────────────────────┐
│  Step 1: 병렬 검색 (ID 추출)              │
│  ├─ LanceDB 벡터검색 → 문서 ID 목록      │
│  └─ PostgreSQL FTS → 문서 ID 목록         │
└──────────────┬──────────────────────────┘
               │ RRF 병합된 source_id 목록
               ▼
┌─────────────────────────────────────────┐
│  Step 2: 원문 조회 (PostgreSQL)           │
│  - source_id + data_type → 19개 document │
│    테이블에서 원문 배치 조회               │
│  - data_type → 테이블 매핑 레지스트리 사용 │
└──────────────┬──────────────────────────┘
               │ 원문이 포함된 문서 목록
               ▼
┌─────────────────────────────────────────┐
│  Step 3: Cross-encoder 리랭킹            │
│  - (쿼리, 원문) 쌍으로 관련성 재평가       │
│  - top_k 선택                            │
└──────────────┬──────────────────────────┘
               │ 최종 정렬된 문서 목록
               ▼
           PipelineResult
```

### 2.2 현재 코드와의 핵심 차이점

| 항목 | 현재 (초안) | 고도화 후 |
|------|-----------|---------|
| 벡터검색 반환 | content 포함 (LanceDB에서 가져옴) | **source_id만 반환** (content 빈 문자열) |
| 원문 조회 | `_get_chunk_content()`로 개별 조회 | **Step 2에서 배치 조회** (`WHERE source_id IN (...)`) |
| FTS 검색 반환 | content 빈 문자열 (이미 올바름) | 동일 |
| 리랭킹 입력 | LanceDB 요약문 (최대 2048자) | **PostgreSQL 원문** (항상 원문, ai_summary 사용 안 함) |
| 파이프라인 | 단순 순차 호출 | **4단계 구조화 + 프리셋 + 메트릭** |

---

## 3. 파일별 고도화 상세

### 3.1 `retrieval.py` 고도화

#### 현재 문제점
1. `_search_vector()` 가 content를 LanceDB에서 가져오고, 없으면 PG에서 개별 조회 → **N+1 쿼리 문제**
2. `search_relevant_documents()` 에서 벡터 결과와 FTS 결과를 source_id 기준으로 병합할 때 content가 일관되지 않음
3. deprecated `RetrievalService` 클래스가 하위 호환으로 남아있음

#### 고도화 방향

**A. `_search_vector()` → ID만 반환하도록 변경**
- LanceDB 검색 후 `(source_id, similarity)` 튜플 목록만 반환
- content 조회 로직 제거 (`_get_chunk_content()` 삭제 대상)
- 메타데이터는 유지 (case_name, case_number, doc_type 등)

**B. 원문 배치 조회 함수 추가**
- `fetch_documents_by_ids(source_ids, data_type)` → PostgreSQL에서 배치 조회
- `DOCUMENT_TABLE_REGISTRY`로 data_type → 테이블 매핑 (19개 테이블, 10개 data_type)
- 판례/법령은 `precedent_documents`/`law_documents` 테이블 사용
- 위원회결정례는 10개 dec_* 테이블 순차 조회
- 원문 content + 메타데이터를 한 번에 조회 (리랭킹용 content는 항상 원문)

**C. `search_relevant_documents()` 흐름 변경**
1. `_search_vector()` → source_id + similarity 목록
2. `search_by_keyword()` → source_id + rank 목록 (기존 유지)
3. RRF 병합 → 정렬된 source_id 목록
4. **content는 포함하지 않음** (pipeline.py에서 배치 조회)

**D. deprecated 코드 정리**
- `_RetrievalServiceCompat`, `get_retrieval_service()` 삭제
- 에이전트들이 직접 `search_relevant_documents()` 사용하도록 확인 (이미 완료)

### 3.2 `rerank.py` 고도화

#### 현재 문제점
1. content를 단순 `[:2048]`로 truncation → 토큰 수와 불일치
2. 모델 로드 실패 시 원본 순서 그대로 반환만 하고 로깅 부족
3. 리랭킹 점수의 최소 임계값 필터링 없음
4. 배치 사이즈 미고려 (문서가 많으면 OOM 위험)

#### 고도화 방향

**A. 배치 처리**
- 문서 수가 많을 때 배치 단위로 predict 수행
- 기본 배치 사이즈: 32 (메모리 안전)

**B. 스코어 정규화 및 필터링**
- sigmoid 출력 (0~1) 이미 적용됨 → 유지
- 최소 점수 임계값 (기본 0.01) 아래 문서 제거 (노이즈 필터)
- 스코어 기반 상위 k개 + 임계값 동시 적용

**C. 원문 적응형 truncation**
- 리랭킹 입력은 항상 원문 (ai_summary 사용 안 함)
- 리랭커 모델(bge-reranker-v2-m3-ko)의 최대 입력: 8192 토큰, 한국어 ~4000자
- 짧은 문서(≤4000자): 전문 사용
- 긴 문서(>4000자): 앞 3000자 + 뒤 1000자 결합 (head+tail)
- 향후 RAG eval 결과에 따라 MaxP(청크별 최고 점수) 방식으로 업그레이드 가능

**D. async 래퍼 추가**
- `rerank_documents_async()` — `asyncio.to_thread()` 래핑
- 에이전트에서 비동기 호출 가능

**E. deprecated 코드 정리**
- `_RerankerServiceCompat`, `get_reranker_service()` 삭제

### 3.3 `pipeline.py` 고도화

#### 현재 문제점
1. 파이프라인이 단순히 retrieval → rerank 순차 호출
2. **원문 배치 조회 단계 없음** → retrieval에서 content를 가져와야 함
3. 에이전트별 프리셋 없음 → 에이전트마다 PipelineConfig 직접 생성 필요
4. 실행 메트릭 (소요 시간, 단계별 결과 수) 미수집

#### 고도화 방향

**A. 4단계 파이프라인 구조**
```python
class RAGPipeline:
    """통합 RAG 파이프라인"""

    async def execute(self, query, config) -> PipelineResult:
        # Step 1: 병렬 검색 (벡터 + FTS → ID 추출)
        search_results = await self._step_retrieve(query, config)

        # Step 2: 원문 배치 조회 (PostgreSQL)
        documents = await self._step_fetch_content(search_results, config)

        # Step 3: 리랭킹 (Cross-encoder)
        ranked_docs = await self._step_rerank(query, documents, config)

        # Step 4: 결과 포맷팅
        return self._step_format(ranked_docs, config)
```

**B. PipelineConfig 고도화**
```python
@dataclass
class PipelineConfig:
    # 검색 설정
    n_vector_results: int = 30        # 벡터검색 후보 수
    n_fts_results: int = 30           # FTS 후보 수
    doc_type: Optional[str] = None    # 문서 유형 필터

    # 리랭킹 설정
    enable_rerank: bool = True        # 리랭킹 활성화
    rerank_top_k: int = 5             # 리랭킹 후 반환 수
    rerank_min_score: float = 0.01    # 최소 리랭킹 점수

    # 쿼리 리라이팅 설정
    enable_rewrite: bool = False
    num_rewrite_queries: int = 3

    # 결과 설정
    final_top_k: int = 5             # 최종 반환 수
```

**C. 에이전트별 프리셋**
```python
PRESETS = {
    "legal_search_precedent": PipelineConfig(
        n_vector_results=30, doc_type="precedent",
        enable_rerank=True, rerank_top_k=4,
    ),
    "legal_search_law": PipelineConfig(
        n_vector_results=30, doc_type="law",
        enable_rerank=True, rerank_top_k=4,
    ),
    "law_study": PipelineConfig(
        n_vector_results=20, doc_type="law",
        enable_rerank=True, rerank_top_k=3,
    ),
    "small_claims": PipelineConfig(
        n_vector_results=20,
        enable_rerank=True, rerank_top_k=5,
    ),
}
```

**D. PipelineResult 고도화**
```python
@dataclass
class PipelineResult:
    documents: List[Dict[str, Any]]       # 최종 문서 목록
    original_query: str                   # 원본 쿼리
    rewritten_queries: List[str]          # 리라이팅된 쿼리들
    metrics: PipelineMetrics              # 실행 메트릭
```

**E. 단계별 메트릭**
```python
@dataclass
class PipelineMetrics:
    total_duration_ms: float
    retrieve_duration_ms: float
    fetch_duration_ms: float
    rerank_duration_ms: float
    n_vector_candidates: int
    n_fts_candidates: int
    n_fused: int                   # RRF 병합 후 고유 ID 수
    n_after_rerank: int            # 리랭킹 후 결과 수
    hybrid_search_used: bool
    rerank_used: bool
```

---

## 4. 에이전트 연동 변경 사항

### 현재 에이전트 호출 방식 (변경 전)

```python
# LegalSearchAgent
results = await search_relevant_documents_async(
    query=message, n_results=4, doc_type="precedent"
)

# LawStudyAgent
results = await asyncio.to_thread(
    search_relevant_documents, query=search_query, n_results=3, doc_type="law"
)
```

### 고도화 후 에이전트 호출 방식

```python
from app.services.rag.pipeline import RAGPipeline, PRESETS

pipeline = RAGPipeline()

# LegalSearchAgent
result = await pipeline.execute(
    query=message,
    config=PRESETS["legal_search_precedent"],
)

# LawStudyAgent
result = await pipeline.execute(
    query=search_query,
    config=PRESETS["law_study"],
)
```

→ **에이전트 코드 수정도 포함** (pipeline 호출로 전환)

---

## 5. 변경 범위

### 수정 파일

| 파일 | 변경 유형 | 설명 |
|------|----------|------|
| `app/services/rag/retrieval.py` | 대규모 수정 | ID 기반 검색 + 원문 배치 조회 |
| `app/services/rag/rerank.py` | 중규모 수정 | 배치 처리, 스코어 필터링, async |
| `app/services/rag/pipeline.py` | 대규모 수정 | 4단계 파이프라인, 프리셋, 메트릭 |
| `app/services/rag/__init__.py` | 소규모 수정 | export 업데이트 |
| `app/multi_agent/agents/legal_search_agent.py` | 중규모 수정 | pipeline 호출로 전환 |
| `app/multi_agent/agents/law_study_agent.py` | 소규모 수정 | pipeline 호출로 전환 |
| `app/multi_agent/agents/small_claims_agent.py` | 소규모 수정 | pipeline 호출로 전환 |

### 변경하지 않는 파일 (이미 완성)

| 파일 | 이유 |
|------|------|
| `embedding.py` | 임베딩 로직은 변경 불필요 |
| `keyword_search.py` | FTS 검색은 이미 source_id 기반 반환 |
| `fusion.py` | RRF 병합 로직은 범용적 |
| `query_rewrite.py` | 쿼리 리라이팅은 독립적 |
| `tsvector_builder.py` | tsvector 유틸리티는 독립적 |

---

## 6. 구현 순서

```
Phase 1: retrieval.py 고도화
  ├─ 1.1 _search_vector() ID 기반 반환으로 변경
  ├─ 1.2 fetch_documents_by_ids() 배치 조회 함수 추가
  ├─ 1.3 search_relevant_documents() 흐름 재구성
  └─ 1.4 deprecated 코드 정리

Phase 2: rerank.py 고도화
  ├─ 2.1 배치 처리 로직 추가
  ├─ 2.2 스코어 필터링 + 적응형 truncation
  ├─ 2.3 async 래퍼 추가
  └─ 2.4 deprecated 코드 정리

Phase 3: pipeline.py 고도화
  ├─ 3.1 RAGPipeline 클래스 구현 (4단계)
  ├─ 3.2 PipelineConfig 고도화 + 프리셋
  ├─ 3.3 PipelineMetrics 구현
  └─ 3.4 기존 편의 함수 유지 (하위 호환)

Phase 4: 에이전트 연동
  ├─ 4.1 LegalSearchAgent pipeline 호출 전환
  ├─ 4.2 LawStudyAgent pipeline 호출 전환
  ├─ 4.3 SmallClaimsAgent pipeline 호출 전환
  └─ 4.4 __init__.py export 업데이트

Phase 5: 검증
  ├─ 5.1 정적 검증 (ruff check, mypy)
  └─ 5.2 기존 테스트 통과 확인
```

---

## 7. 기술적 고려사항

### 7.1 원문 배치 조회 설계

#### PostgreSQL 데이터 현황 (2026-02-19 확정)

| fts_index.data_type | PostgreSQL 테이블 | 건수 | ID 컬럼 | 리랭킹용 원문 컬럼 |
|---------------------|------------------|------|---------|------------------|
| 판례 | `precedent_documents` | 92,055 | `serial_number` | ruling, reasoning |
| 법령 | `law_documents` | 5,548 | `law_id` | content |
| 행정규칙 | `admin_rule_documents` | 5,257 | `serial_number` | content |
| 부처유권해석 | `interpretation_ministry_documents` | 37,325 | `serial_number` | answer, reason |
| 헌재결정례 | `constitutional_documents` | 8,000 | `serial_number` | ruling, reasoning |
| 행정심판례 | `administration_documents` | 34,254 | `serial_number` | ruling, reason |
| 법령해석례 | `legislation_documents` | 8,597 | `serial_number` | answer, reason |
| 조약 | `treaty_documents` | 3,589 | `serial_number` | content |
| 특별행정심판 | `special_admin_appeal_documents` | 148,778 | `serial_number` | ruling, reason |
| 위원회결정례 | `dec_labor_documents` | 40,714 | `serial_number` | judgment_summary, judgment_result |
| 위원회결정례 | `dec_human_rights_documents` | 3,721 | `serial_number` | ruling, reason |
| 위원회결정례 | `dec_privacy_documents` | 1,448 | `serial_number` | reason |
| 위원회결정례 | `dec_employment_documents` | 118 | `serial_number` | ruling, reason |
| 위원회결정례 | `dec_financial_documents` | 662 | `serial_number` | action_reason, action_content |
| 위원회결정례 | `dec_industrial_documents` | 782 | `serial_number` | ruling, reason |
| 위원회결정례 | `dec_environment_documents` | 358 | `serial_number` | ruling, case_overview |
| 위원회결정례 | `dec_securities_documents` | 636 | `serial_number` | action_reason, action_content |
| 위원회결정례 | `dec_civil_rights_documents` | 635 | `serial_number` | ruling, reason |
| 위원회결정례 | `dec_fair_trade_documents` | 7,728 | `serial_number` | ruling, reason |

> **참고:** 10개 data_type (판례, 법령, 행정규칙, 부처유권해석, 헌재결정례, 행정심판례, 법령해석례, 조약, 특별행정심판, 위원회결정례) → 19개 테이블 매핑

#### data_type → 테이블 매핑 전략

```python
# data_type은 fts_index.data_type 또는 LanceDB의 data_type 필드
DOCUMENT_TABLE_REGISTRY: dict[str, list[TableConfig]] = {
    "판례": [TableConfig("precedent_documents", "serial_number", ["ruling", "reasoning"])],
    "법령": [TableConfig("law_documents", "law_id", ["content"])],
    "행정규칙": [TableConfig("admin_rule_documents", "serial_number", ["content"])],
    "부처유권해석": [TableConfig("interpretation_ministry_documents", "serial_number", ["answer", "reason"])],
    "헌재결정례": [TableConfig("constitutional_documents", "serial_number", ["ruling", "reasoning"])],
    "행정심판례": [TableConfig("administration_documents", "serial_number", ["ruling", "reason"])],
    "법령해석례": [TableConfig("legislation_documents", "serial_number", ["answer", "reason"])],
    "조약": [TableConfig("treaty_documents", "serial_number", ["content"])],
    "특별행정심판": [TableConfig("special_admin_appeal_documents", "serial_number", ["ruling", "reason"])],
    "위원회결정례": [
        TableConfig("dec_labor_documents", "serial_number", ["judgment_summary", "judgment_result"]),
        TableConfig("dec_human_rights_documents", "serial_number", ["ruling", "reason"]),
        TableConfig("dec_privacy_documents", "serial_number", ["reason"]),
        TableConfig("dec_employment_documents", "serial_number", ["ruling", "reason"]),
        TableConfig("dec_financial_documents", "serial_number", ["action_reason", "action_content"]),
        TableConfig("dec_industrial_documents", "serial_number", ["ruling", "reason"]),
        TableConfig("dec_environment_documents", "serial_number", ["ruling", "case_overview"]),
        TableConfig("dec_securities_documents", "serial_number", ["action_reason", "action_content"]),
        TableConfig("dec_civil_rights_documents", "serial_number", ["ruling", "reason"]),
        TableConfig("dec_fair_trade_documents", "serial_number", ["ruling", "reason"]),
    ],
}
```

#### 위원회결정례 (1:N 매핑) 처리 전략

`fts_index.data_type="위원회결정례"`는 10개 dec_* 테이블에 분산. source_id 충돌 가능성은 낮지만 보장 불가.

**방안: 순차 조회 (채택)**
- 10개 dec_* 테이블을 순차로 `WHERE serial_number IN (...)` 조회
- 먼저 찾은 결과를 반환 (early return)
- 장점: 구현 간단, 실용적 (1개 요청에 위원회결정례는 보통 소수)
- 단점: 최악의 경우 10회 쿼리 (하지만 각 쿼리는 인덱스 활용으로 <5ms)

#### 리랭킹용 content 선택 원칙

**항상 원문 사용 (ai_summary 사용 안 함).**

리랭킹 시 Cross-encoder에 입력할 텍스트는 테이블별 원문 컬럼을 사용.
원문이 길 경우 적응형 truncation으로 리랭커 최대 입력 길이에 맞춤.

```
리랭킹 입력: 테이블별 원문 컬럼 (위 표 참조)
  - 판례: ruling + reasoning
  - 법령/행정규칙/조약: content
  - 부처유권해석/법령해석례: answer + reason
  - 헌재결정례: ruling + reasoning
  - 행정심판례/특별행정심판: ruling + reason
  - 위원회결정례: 테이블별 상이 (judgment_summary+judgment_result, ruling+reason,
                  action_reason+action_content, ruling+case_overview 등)
```

```python
async def fetch_documents_by_ids(
    source_ids: list[str],
    data_type: Optional[str] = None,
) -> dict[str, dict[str, Any]]:
    """
    source_id 목록으로 PostgreSQL에서 원문 배치 조회.

    data_type이 지정되면 해당 테이블 그룹만 조회,
    미지정이면 fts_index에서 data_type을 먼저 확인 후 조회.

    Returns:
        {source_id: {"content": str, "metadata": dict}} 매핑
        content는 항상 원문 (ai_summary 아님)
    """
```

### 7.2 하위 호환성

- `search_relevant_documents()` / `search_relevant_documents_async()` 함수 시그니처 유지
- 기존 에이전트가 바로 깨지지 않도록 **점진적 전환**
- deprecated 함수는 내부적으로 pipeline을 호출하도록 래핑 가능

### 7.3 성능 고려

| 단계 | 예상 소요 | 최적화 |
|------|----------|--------|
| 벡터검색 (LanceDB) | ~50ms | 이미 빠름 |
| FTS 검색 (PostgreSQL) | ~30ms | 이미 인덱스 적용 |
| 원문 배치 조회 | ~20ms | `IN (...)` 쿼리, 인덱스 활용 |
| 리랭킹 (Cross-encoder) | ~200-500ms | 배치 처리, GPU 활용 |
| **총합** | ~300-600ms | 벡터/FTS 병렬 실행으로 단축 가능 |

### 7.4 리스크

| 리스크 | 영향 | 대응 |
|--------|------|------|
| 에이전트 호출 변경으로 기존 동작 깨짐 | 높음 | 하위 호환 래퍼 유지, 점진적 전환 |
| 리랭커 모델 OOM (GPU 메모리) | 중간 | 배치 사이즈 제한, CPU fallback |
| 원문 조회 시 테이블 불일치 (data_type) | 낮음 | doc_type → 테이블 매핑 명확화 |

---

## 8. 성공 기준

- [ ] retrieval.py: 벡터/FTS에서 ID만 추출, 원문은 별도 배치 조회
- [ ] rerank.py: 배치 처리, 스코어 필터링, async 지원
- [ ] pipeline.py: 4단계 구조, 프리셋 지원, 메트릭 수집
- [ ] 에이전트 3개 모두 pipeline 호출로 전환
- [ ] 정적 검증 통과 (ruff check, mypy)
- [ ] 기존 테스트 통과
- [ ] 하위 호환성 유지 (search_relevant_documents 함수 시그니처)
