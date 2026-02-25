# PDCA Plan: RAG 다중 타입 검색 (Focus + Supplementary)

## 배경

### 현재 상태
- 판례/법령 2개 타입만 검색, 18개 타입(헌재결정례, 행정심판례 등 584K+ 문서) 미활용
- 쿼리 리라이팅 2회 중복 실행 (파이프라인별 `rewrite_query` 1회 × 2)
- Focus/비Focus 파이프라인 순차 실행

### 요구사항
1. Focus 타입 ~70% + 나머지 타입 ~30% 비율로 검색
2. 쿼리 리라이팅 1회만 실행
3. 병렬 검색으로 시간 단축
4. 전략 적절성 검토 + 제안

## 전략: Focus + Supplementary 추가 검색

기존 30건 풀 분할 대신 보충 검색 추가 방식:

| | Focus (기존 유지) | Supplementary (신규) |
|---|---|---|
| 대상 | focus 타입만 | focus 제외 전체 |
| n_results | 15 | 7 |
| rerank | top_k=5 | top_k=3 |
| 최종 | 5건 | 2-3건 |

- 총 7-8건 (실효 비율 ~65:35)
- 병렬 실행으로 현재보다 빨라짐 (순차 3단계 → 2단계)
- 기존 Focus 품질 유지하면서 다양한 타입 보강
- 법령 보장 없음: Supplementary는 relevance 기반 자연 선택 (기존 n_results=1 법령 서브파이프라인 제거)

## 수정 대상 파일 (5개)

| 파일 | 변경 |
|------|------|
| `backend/app/tools/vectorstore/lancedb.py` | `_build_filter_conditions`에 NOT IN (`$not_in`) 필터 지원 |
| `backend/app/services/rag/pipeline.py` | `PipelineConfig`에 `exclude_doc_types` 필드 추가 |
| `backend/app/services/rag/retrieval.py` | 벡터 검색 + 하이브리드 검색에 exclude 필터 전달 |
| `backend/app/services/rag/keyword_search.py` | FTS에 `exclude_doc_types` WHERE 조건 추가 |
| `backend/app/multi_agent/agents/legal_search_agent.py` | Focus + Supplementary 구조로 전면 재구성 |

## 상세 변경 사항

### 1. LanceDB NOT IN 필터

```python
# _build_filter_conditions에서 dict 값 처리
where = {"data_type": {"$not_in": ["판례", "법령"]}}
# → "data_type NOT IN ('판례', '법령')"
```

### 2. PipelineConfig 확장

```python
@dataclass
class PipelineConfig:
    exclude_doc_types: list[str] | None = None  # 신규: 제외할 data_type 목록 (한국어)
```

파이프라인 `execute()`에서 `exclude_doc_types`를 검색 함수로 전달:
- `doc_type`과 `exclude_doc_types`는 상호 배타적 (동시 지정 시 `doc_type` 우선)

### 3. retrieval.py / keyword_search.py exclude 필터

벡터 검색 (`_search_vector_ids`):
```python
# exclude_doc_types가 있으면 NOT IN 필터
where = {"data_type": {"$not_in": exclude_doc_types}}
```

FTS 검색 (`_execute_fts_query`):
```python
# exclude_doc_types가 있으면 WHERE NOT IN 조건
stmt = stmt.where(FtsIndex.data_type.not_in(exclude_doc_types))
```

### 4. SEARCH_CONFIG 데이터 구조 변경

기존:
```python
SEARCH_CONFIG = {
    "precedent": {
        "precedent": PipelineConfig(n_results=15, doc_type="precedent", ...),
        "law": PipelineConfig(n_results=1, doc_type="law"),
    },
}
```

변경 후:
```python
FOCUS_CONFIG: dict[str, PipelineConfig] = {
    "precedent": PipelineConfig(
        n_results=15, doc_type="precedent",
        enable_rewrite=False, enable_rerank=True, rerank_top_k=5,
    ),
    "law": PipelineConfig(
        n_results=15, doc_type="law",
        enable_rewrite=False, enable_rerank=True, rerank_top_k=5,
    ),
}

SUPPLEMENTARY_CONFIG: dict[str, PipelineConfig] = {
    "precedent": PipelineConfig(
        n_results=7, exclude_doc_types=["판례"],
        enable_rewrite=False, enable_rerank=True, rerank_top_k=3,
    ),
    "law": PipelineConfig(
        n_results=7, exclude_doc_types=["법령"],
        enable_rewrite=False, enable_rerank=True, rerank_top_k=3,
    ),
}
```

- `enable_rewrite=False`: 에이전트 레벨에서 1회만 리라이팅
- `exclude_doc_types`: 한국어 data_type 기준 (LanceDB/FTS 모두 한국어 사용)

### 5. legal_search_agent.py 재구성

#### 쿼리 리라이팅 1회
```python
# process() / process_stream()에서 1회만 호출
search_query = await rewrite_conversational_query(message, history)
# 이후 Focus/Supplementary 모두 search_query 사용
```

#### Focus + Supplementary 병렬 실행
```python
focus_result, supplementary_result = await asyncio.gather(
    search_with_pipeline_async(search_query, self.focus_config),
    search_with_pipeline_async(search_query, self.supplementary_config),
)
```

#### Supplementary 컨텍스트/소스 포맷팅

Supplementary 결과는 판례/법령 이외의 다양한 타입을 포함하므로 **범용 포맷터** 추가:

```python
def _build_supplementary_context(self, documents: list[dict]) -> list[str]:
    """Supplementary 문서 컨텍스트 구성 (data_type별 섹션)"""
    # "## 관련 법률 자료 (보충)" 제목
    # 각 문서: [data_type] 제목\n내용
```

```python
def _format_supplementary_sources(self, documents: list[dict]) -> list[dict]:
    """Supplementary 소스 포맷팅"""
    # doc_type은 metadata의 data_type 한국어 원본 사용
    # PrecedentService 상세조회 스킵 (판례 타입이 아님)
```

#### PrecedentService 상세조회 범위

- Focus 결과: 기존과 동일하게 `PrecedentService.get_details()` 호출 (판례인 경우만)
- Supplementary 결과: 상세조회 스킵 (다양한 타입이므로 원문 content만 사용)

#### doc_type 매핑 전략

- `_DOC_TYPE_TO_DATA_TYPE` / `_DATA_TYPE_TO_DOC_TYPE`은 기존 2개 매핑 유지
- Supplementary 결과의 `doc_type` 필드: metadata의 `data_type` 한국어 원본 그대로 사용
  - 예: `"doc_type": "헌재결정례"`, `"doc_type": "행정심판례"`
- 프론트엔드 소스 표시에 data_type 한국어가 더 직관적 (변환 불필요)

### 6. 기존 Law 서브파이프라인 제거

| 항목 | 기존 | 변경 후 |
|------|------|---------|
| focus=precedent 시 법령 | n_results=1로 1건 보장 | Supplementary에서 relevance 기반 자연 선택 |
| focus=law 시 판례 | n_results=1로 1건 보장 | Supplementary에서 relevance 기반 자연 선택 |

- 법령/판례가 Supplementary top-3에 포함되지 않을 수 있음
- 이는 의도된 동작: 사용자 쿼리에 가장 관련 높은 타입이 자연 선택됨

## 실행 흐름 (변경 후)

```
사용자 메시지
    │
    ▼
rewrite_conversational_query() ─── 1회만 실행
    │
    ├──────────────────────┐
    ▼                      ▼
Focus Pipeline         Supplementary Pipeline
(doc_type 필터)        (exclude_doc_types 필터)
    │                      │
    ▼                      ▼
15건 검색 → 리랭킹     7건 검색 → 리랭킹
    → top 5건              → top 3건
    │                      │
    └──────────┬───────────┘
               ▼
         결과 병합 (5 + 2~3건)
               │
               ▼
         컨텍스트 구성 + LLM 응답
```

## 검증
- `uv run ruff check backend/app/` 린트 통과
- LangSmith에서 리라이팅 1회 + 병렬 검색 확인
- Supplementary 결과에 다양한 타입 포함 확인
- Focus 결과 품질이 기존과 동일한지 확인
