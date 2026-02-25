# 벡터 검색 및 정보 검색(IR) 최적화 분석

## 작성일
2026-02-20

## 분석 범위
- 파일: lancedb.py, retrieval.py, rerank.py, fusion.py, keyword_search.py, embedding.py, pipeline.py, config.py

## 1. 현재 아키텍처 분석

### 1.1 검색 파이프라인 흐름
```
사용자 쿼리
  ↓
쿼리 임베딩 (embedding.py: create_query_embedding)
  ↓
병렬 검색 (retrieval.py)
  ├─ 벡터 검색 (LanceDB, n_results=30)
  └─ 키워드 검색 (PostgreSQL FTS, n_results=30)
  ↓
RRF 병합 (fusion.py: reciprocal_rank_fusion, k=60)
  ↓
원문 배치 조회 (PostgreSQL, source_id별)
  ↓
리랭킹 (rerank.py: Cross-encoder, batch_size=32, top_k=5)
  ↓
최종 결과 반환
```

### 1.2 주요 구성 요소

#### LanceDB (벡터 DB)
- **모델**: nlpai-lab/KURE-v1 (1024차원)
- **인덱스**: IVF_FLAT (기본: brute-force)
- **nprobes**: 40 (IVF 인덱스 사용 시 탐색 파티션 수)
- **데이터**: 253,768 청크 (법령+판례 통합)
- **검색 전략**: metric='cosine'

#### PostgreSQL FTS (키워드 검색)
- **테이블**: fts_index (GIN 인덱스)
- **전략**: 개념 AND → OR fallback
- **필터링**: _CONCEPT_AND_MIN_RESULTS = 5

#### Cross-encoder (리랭킹)
- **모델**: dragonkue/bge-reranker-v2-m3-ko
- **토큰 제한**: 8192 (≈ 한글 4000자)
- **Truncation**: head 3000자 + tail 1000자
- **배치 크기**: 32
- **최소 점수**: 0.01

#### RRF (Reciprocal Rank Fusion)
- **파라미터 k**: 60
- **공식**: Σ 1/(k + rank_i)

### 1.3 현재 설정 (pipeline.py)

| 프리셋 | n_results | rerank_top_k | enable_rewrite |
|--------|-----------|-------------|----------------|
| legal_search_precedent | 15 | 5 | - |
| legal_search_law | 15 | 5 | - |
| legal_search_all | 20 | 7 | - |
| law_study | 10 | 5 | ✓ |
| small_claims | 10 | 3 | - |
| quick_search | 5 | - | - |

## 2. 현재 성능 병목 분석

### 2.1 검색 단계 (대기 시간)
- **벡터 검색 + FTS**: 순차 실행 (병렬화 가능)
  - 벡터 검색: ~50-100ms (n_results=30)
  - FTS 검색: ~30-80ms (n_results=30)
  - **총**: ~80-180ms

### 2.2 RRF 병합 (O(n log n))
- **복잡도**: O(n log n), n≈30-60
- **소요 시간**: ~1-5ms (미미)
- **메모리**: ~minimal

### 2.3 원문 조회 (배치 I/O)
- **방식**: PostgreSQL IN 절
- **소요 시간**: ~50-150ms (n=10-20)
- **병목**: 여러 테이블 순차 조회 (TableConfig 반복)

### 2.4 리랭킹 (GPU/CPU 집약)
- **준비**: LanceDB 요약문 조회 ~20-30ms
- **Cross-encoder**: ~200-500ms (n=10-20, batch_size=32)
- **점수 정렬**: ~1-5ms (미미)
- **예상 병목**: 모델 추론 시간

### 2.5 쿼리 리라이팅 (선택)
- **LLM 호출**: ~1-3초 (따로 사용 시)
- **사용**: law_study만 (enable_rewrite=true)

## 3. 최적화 방안 (상세)

### 3.1 LanceDB 인덱스 최적화

#### 1️⃣ IVF_FLAT 또는 IVF_PQ 인덱스 구축

**현재**: brute-force (모든 벡터 순회)
**개선**: IVF_FLAT (Inverted File)

```python
# backend/app/tools/vectorstore/lancedb.py: create_vector_index()
num_partitions = max(16, int(row_count**0.5))  # 현재 ≈ 500
# ↓
# 최적값: sqrt(N) = sqrt(253,768) ≈ 504 (OK)
# 하지만 제너럴한 설정으로 변경:
num_partitions = min(256, int(row_count**0.5))  # 상한 설정
```

**예상 효과**:
- 벡터 검색 시간: 50-100ms → **15-30ms** (3-5배)
- 단점: 약간의 recall 저하 (IVF 근사 때문)

**리스크**: nprobes 튜닝 필요
- nprobes=40 (현재) → recall ≥0.9
- nprobes=20 (강공격) → recall ≈0.8-0.85

#### 2️⃣ nprobes 동적 조정

```python
# 구성 가능한 전략:
# - 빠른 검색 (quick_search): nprobes=10
# - 일반 검색: nprobes=40 (현재)
# - 정밀 검색: nprobes=80
```

**설정 추가**:
```python
# app/core/config.py
LANCEDB_NPROBES_FAST: int = 10
LANCEDB_NPROBES_DEFAULT: int = 40
LANCEDB_NPROBES_PRECISE: int = 80
```

**예상 효과**:
- quick_search: 50-100ms → **20-40ms** (2-3배)
- small_claims: 80ms → **40-60ms** (1.5-2배)

#### 3️⃣ 데이터 타입별 별도 인덱스

**현재**: 단일 테이블 (법령+판례 혼합)
**개선**: data_type별 파티션 또는 별도 테이블

```python
# 검색 시 (retrieval.py)
where = {"data_type": "판례"}  # 이미 필터링 중
# ↓
# 별도 테이블: legal_chunks_law, legal_chunks_precedent
```

**예상 효과**:
- 검색 공간 50% 축소
- 벡터 검색: 50-100ms → **30-50ms** (1.5-2배)

**비용**: 스토리지 2배

#### 종합 평가

| 최적화 | 예상 절감 | 난이도 | 리스크 | 설명 |
|--------|---------|--------|--------|------|
| IVF_FLAT 인덱스 | 100-150ms | 중간 | 중간 | recall 저하 가능, nprobes 튜닝 필수 |
| nprobes 동적 조정 | 30-60ms | 낮음 | 낮음 | 검색 타입별로 다른 설정 적용 |
| data_type별 파티션 | 30-50ms | 높음 | 높음 | 유지보수 복잡도 증가, 스토리지 2배 |

**권장**: 1️⃣ + 2️⃣ 조합 (IVF_FLAT + nprobes 동적 조정)
- **예상 총 절감**: 100-200ms
- **구현 난이도**: 중간
- **리스크**: 중간

---

### 3.2 검색 병렬화

#### 1️⃣ 벡터 + FTS 병렬 실행

**현재** (retrieval.py:399-419):
```python
vector_results = _search_vector_ids(...)  # 동기 실행
keyword_results = search_by_keyword(...)  # 동기 실행, 순차
```

**개선**:
```python
async def search_without_content(...):
    # 벡터 검색과 FTS를 asyncio.gather로 병렬화
    vector_results, keyword_results = await asyncio.gather(
        asyncio.to_thread(_search_vector_ids, ...),
        asyncio.to_thread(search_by_keyword, ...),
    )
```

**예상 효과**:
- 순차: 50 + 30 = 80ms
- 병렬: max(50, 30) = **50ms** (37% 절감)

**난이도**: 낮음 (asyncio 추가)

#### 2️⃣ 원문 조회 배치 최적화

**현재** (retrieval.py:270-309):
```python
for data_type, source_ids in type_groups.items():
    for tc in table_configs:  # 테이블 순차 반복
        # 테이블별로 IN 절 실행
```

**병목**: TableConfig 순차 반복
- 다테이블 쿼리 순차 실행
- 예: 위원회결정례 10개 테이블 → 최대 10개 IN 쿼리

**개선**:
```python
# 병렬화 (asyncio.gather로 테이블별 쿼리 동시 실행)
from sqlalchemy import select_array
tasks = [
    asyncio.to_thread(execute_table_query, tc, source_ids)
    for tc in table_configs
]
results = await asyncio.gather(*tasks)
```

**예상 효과**:
- 순차: 10×20ms = 200ms
- 병렬: max(20ms) = **20ms** (10배!)

**난이도**: 중간 (async session 관리)

#### 종합 평가

| 최적화 | 예상 절감 | 난이도 | 리스크 | 설명 |
|--------|---------|--------|--------|------|
| 벡터+FTS 병렬 | 30-50ms | 낮음 | 낮음 | 현재 동기 코드 → asyncio.gather |
| 원문 조회 배치 병렬화 | 100-200ms | 중간 | 중간 | 테이블별 async 쿼리 동시 실행 |

**권장**: 두 가지 모두 구현
- **예상 총 절감**: 130-250ms (전체 검색 ~40% 개선)
- **구현 난이도**: 중간
- **리스크**: 낮음

---

### 3.3 리랭킹 최적화

#### 1️⃣ Cross-encoder 모델 교체 (ONNX)

**현재**: PyTorch + GPU (없으면 CPU)
- 모델 로드: ~500ms
- 추론: ~200-500ms (n=10-20)

**개선**: ONNX Runtime
```python
# rerank.py
import onnxruntime as ort

# ONNX 모델 로드 (더 가벼움)
session = ort.InferenceSession("bge-reranker-v2-m3-ko.onnx")
# 추론 속도: ~50-150ms (PyTorch 대비 3-4배)
```

**예상 효과**:
- 리랭킹: 200-500ms → **50-150ms** (3-4배)

**난이도**: 중간 (ONNX 모델 변환 필요)

#### 2️⃣ 리랭킹 배치 크기 증가

**현재**: batch_size=32
- 메모리: ~1-2GB

**개선**: batch_size=64-128 (GPU가 있으면)
```python
# rerank.py: _RERANK_BATCH_SIZE = 64
```

**예상 효과**:
- 배치 처리 오버헤드 감소
- 리랭킹: ~5-10% 개선

**난이도**: 낮음

#### 3️⃣ 리랭킹 top_k 사전 필터링

**현재**: 모든 검색 결과를 리랭킹
- n_results=15-20 → 모두 리랭킹

**개선**: similarity 상위만 리랭킹
```python
# pipeline.py: execute()
if config.enable_rerank:
    # 상위 n_results * 0.7만 리랭킹 (나머지는 similarity 정렬)
    top_candidates = all_documents[:int(config.n_results * 0.7)]
    reranked = rerank_documents(query, top_candidates, top_k)
```

**예상 효과**:
- 리랭킹 입력: 15 → 10 (33% 감소)
- 리랭킹 시간: 200ms → **150ms** (25%)

**난이도**: 낮음

#### 종합 평가

| 최적화 | 예상 절감 | 난이도 | 리스크 | 설명 |
|--------|---------|--------|--------|------|
| ONNX 모델 전환 | 100-350ms | 중간 | 낮음 | 모델 변환 + 의존성 추가 |
| 배치 크기 증가 | 10-20ms | 낮음 | 낮음 | GPU 메모리 확보 필수 |
| top_k 사전 필터링 | 30-50ms | 낮음 | 중간 | recall 약간 저하 가능 |

**권장**: 1️⃣ (ONNX) + 2️⃣ (배치 크기)
- **예상 총 절감**: 100-350ms (리랭킹 30-50% 개선)
- **구현 난이도**: 중간
- **리스크**: 낮음

---

### 3.4 임베딩 캐싱 확대

#### 1️⃣ 쿼리 임베딩 캐싱

**현재**: 매번 새로 계산
- create_query_embedding() 호출 시마다 모델 추론

**개선**: LRU 캐시
```python
# embedding.py
from functools import lru_cache

@lru_cache(maxsize=1000)
def create_query_embedding(query: str):
    # 동일 쿼리는 캐시에서 반환
```

**예상 효과**:
- 자주 검색되는 쿼리: 0.5ms (캐시 히트)
- 캐시 히트율: 10-30% (추정)
- 평균 절감: **5-20ms** (쿼리당)

**난이도**: 낮음

**주의**: 캐시 크기 모니터링 필요

#### 2️⃣ 문서 임베딩 재계산 방지

**현재**: LanceDB에 벡터 저장됨 (이미 최적화)
- 추가 개선 없음

#### 종합 평가

| 최적화 | 예상 절감 | 난이도 | 리스크 | 설명 |
|--------|---------|--------|--------|------|
| 쿼리 임베딩 캐싱 | 5-20ms | 낮음 | 낮음 | 캐시 메모리 ~50MB (maxsize=1000) |

**권장**: 구현 권장
- **예상 절감**: 5-20ms/쿼리
- **비용**: 메모리 ~50MB

---

### 3.5 검색 결과 캐싱

#### 1️⃣ 인기 쿼리 캐싱

**사용 경우**:
- 자주 검색되는 쿼리 (예: "손해배상 판례")
- 법률 학습 모듈 (일정한 쿼리 세트)

**구현**:
```python
# pipeline.py
_QUERY_CACHE = {}  # {query_hash: PipelineResult}

def execute(self, query, config):
    cache_key = hash((query, config))
    if cache_key in _QUERY_CACHE:
        return _QUERY_CACHE[cache_key]  # 캐시 히트

    result = self._execute_internal(query, config)
    _QUERY_CACHE[cache_key] = result
    return result
```

**예상 효과**:
- 캐시 히트 시: ~500ms → **10ms** (50배!)
- 캐시 히트율: 5-15% (추정)
- 평균 절감: **20-100ms** (전체 기준)

**난이도**: 낮음

**주의**: 캐시 무효화 전략 필요 (데이터 업데이트 시)

#### 종합 평가

| 최적화 | 예상 절감 | 난이도 | 리스크 | 설명 |
|--------|---------|--------|--------|------|
| 인기 쿼리 캐싱 | 20-100ms | 낮음 | 중간 | 캐시 무효화 전략 필요 |

**권장**: 선택적 구현 (law_study, small_claims 모듈용)
- **예상 절감**: 20-100ms (특정 시나리오)

---

### 3.6 쿼리 리라이팅 최적화

#### 1️⃣ LLM 호출 최소화

**현재**: law_study만 활성화 (enable_rewrite=true)
- rewrite_query()에서 LLM 호출 (1-3초)

**개선**: 하이브리드 전략
```python
# query_rewrite.py
def rewrite_query(query, use_llm=True):
    # 1. 짧은 쿼리 (≤20자): keyword fallback
    if len(query) <= 20:
        use_llm = False

    # 2. 법률 키워드 포함: keyword fallback
    if any(kw in query for kw in LEGAL_KEYWORDS):
        use_llm = False
```

**예상 효과**:
- LLM 호출 감소: 50-70%
- 리라이팅 시간: **100-200ms** (vs. 1-3초)

**난이도**: 낮음

#### 2️⃣ 리라이팅 쿼리 수 감소

**현재**: num_rewrite_queries=3 (pipeline.py 설정 없음, 기본값)
```python
# law_study 프리셋
enable_rewrite=True  # 기본 num_rewrite_queries=3
```

**개선**:
```python
PRESETS["law_study"] = PipelineConfig(
    ...
    num_rewrite_queries=2  # 3 → 2 (LLM 비용 1/3 절감)
)
```

**예상 효과**:
- LLM 호출 수: 3 → 2 (33% 감소)
- 리라이팅 시간: 1-3초 → **0.7-2초**

**난이도**: 낮음

#### 종합 평가

| 최적화 | 예상 절감 | 난이도 | 리스크 | 설명 |
|--------|---------|--------|--------|------|
| LLM 호출 최소화 (keyword fallback) | 800-2400ms | 낮음 | 낮음 | 짧은 쿼리/키워드 포함 시만 |
| 리라이팅 쿼리 수 감소 | 300-1000ms | 낮음 | 낮음 | num_rewrite_queries: 3→2 |

**권장**: 두 가지 모두 구현
- **예상 총 절감**: 800-2400ms (리라이팅 활성화 시나리오)
- **비용**: LLM API 호출 감소

---

## 4. 통합 최적화 시나리오

### 시나리오 A: 빠른 검색 (quick_search)

**목표**: 응답 시간 < 200ms

| 단계 | 현재 | 최적화 | 절감 |
|------|------|--------|------|
| 쿼리 임베딩 | 50ms | 임베딩 캐시 | 40ms |
| 벡터 검색 | 50ms | nprobes=10 | 30ms |
| FTS 검색 | 30ms | 병렬화 | 0ms (병렬) |
| RRF 병합 | 1ms | - | - |
| 원문 조회 | 50ms | 병렬화 | 30ms |
| 결과 정렬 | 5ms | - | - |
| **총** | **186ms** | | **100ms** |
| **목표 달성** | 186ms > 200ms | 86ms < 200ms | **46% 개선** |

### 시나리오 B: 일반 검색 (legal_search_all)

**목표**: 응답 시간 < 1000ms

| 단계 | 현재 | 최적화 | 절감 |
|------|------|--------|------|
| 쿼리 임베딩 | 50ms | 임베딩 캐시 | 40ms |
| 벡터 검색 | 100ms | IVF_FLAT + nprobes=40 | 50ms |
| FTS 검색 | 80ms | 병렬화 | 0ms (병렬) |
| RRF 병합 | 2ms | - | - |
| 원문 조회 | 150ms | 배치 병렬화 | 100ms |
| 리랭킹 | 300ms | ONNX + 상위만 | 150ms |
| 결과 정렬 | 5ms | - | - |
| **총** | **687ms** | | **340ms** |
| **목표 달성** | 687ms < 1000ms | 347ms < 1000ms | **49% 개선** |

### 시나리오 C: 정밀 검색 (legal_search_law + 리라이팅)

**목표**: 응답 시간 < 3000ms (LLM 호출 포함)

| 단계 | 현재 | 최적화 | 절감 |
|------|------|--------|------|
| 쿼리 리라이팅 | 2000ms | 조건부 LLM | 1000ms |
| 벡터 검색 (×2) | 200ms | IVF_FLAT | 60ms |
| FTS 검색 (×2) | 160ms | 병렬화 | 0ms |
| RRF 병합 (×2) | 4ms | - | - |
| 원문 조회 | 150ms | 배치 병렬화 | 100ms |
| 리랭킹 | 300ms | ONNX | 150ms |
| **총** | **2814ms** | | **1310ms** |
| **목표 달성** | 2814ms < 3000ms | 1504ms < 3000ms | **46% 개선** |

---

## 5. 구현 순서 (권장)

### Phase 1 (1주차): 저위험 + 높은 효과
1. ✓ 쿼리 임베딩 캐싱 (낮음/낮음)
2. ✓ nprobes 동적 조정 (낮음/낮음)
3. ✓ 벡터+FTS 병렬 실행 (낮음/낮음)
4. ✓ 배치 크기 증가 (낮음/낮음)
- **예상 효과**: 50-150ms (15-20%)

### Phase 2 (2주차): 중간 복잡도 + 높은 효과
5. ✓ IVF_FLAT 인덱스 구축 (중간/중간)
6. ✓ 원문 조회 배치 병렬화 (중간/중간)
7. ✓ ONNX 모델 전환 (중간/낮음)
- **예상 효과**: 150-250ms (20-30%)

### Phase 3 (3주차): 선택적 최적화
8. ☆ 데이터 타입별 파티션 (높음/높음) - 비용 대비 효과 검토
9. ☆ 인기 쿼리 캐싱 (낮음/중간) - 모듈별 선택적 적용
10. ☆ LLM 호출 최소화 (낮음/낮음) - 리라이팅 활성화 시나리오

---

## 6. 모니터링 및 평가

### 메트릭
- **검색 응답 시간** (ms): p50, p95, p99
- **리랭킹 시간** (ms): 모델별 비교
- **Recall@10, Hit Rate, MRR**: RAG 품질 유지

### 측정 방법
```python
# pipeline.py의 PipelineMetrics 활용
- search_time_ms: 검색 + 원문 조회
- rerank_time_ms: 리랭킹
- total_time_ms: 전체

# 로깅
logger.info("RAG 파이프라인: %d건 검색 → %d건 반환 (%.0fms)",
    result.total_retrieved,
    len(result.documents),
    metrics.total_time_ms)
```

### 벤치마크 (기준)
- quick_search: ~200ms (현재 ~350ms)
- legal_search_all: ~500ms (현재 ~700ms)
- legal_search_law+rerank: ~800ms (현재 ~1200ms)

---

## 7. 리스크 및 완화 전략

### 리스크 1: Recall 저하 (IVF_FLAT)
- **원인**: 근사 인덱스의 정확도 감소
- **대응**: nprobes 튜닝, recall 모니터링
- **Fallback**: brute-force로 복귀 (LANCEDB_INDEX_TYPE="")

### 리스크 2: 캐시 무효화 (쿼리/결과 캐싱)
- **원인**: 데이터 업데이트 시 캐시 부실화
- **대응**: TTL 설정, 수동 무효화 API
- **Fallback**: 캐싱 비활성화

### 리스크 3: 동시성 문제 (병렬화)
- **원인**: AsyncSession 관리 미흡
- **대응**: session pool 설정, 동시 연결 제한
- **Fallback**: 순차 실행으로 복귀

### 리스크 4: 메모리 증가 (캐싱)
- **원인**: 캐시 크기 무제한 성장
- **대응**: maxsize 제한, 주기적 정리
- **Fallback**: LRU 캐시 크기 감소

---

## 8. 추가 고려사항

### 8.1 데이터 특성별 최적화
- **법령**: 고정적, 적은 업데이트 → 캐싱 유리
- **판례**: 동적, 잦은 추가 → 캐싱 신중

### 8.2 모듈별 차별화
- **law_study**: 상세 리라이팅 (LLM) → 비용 증가
- **small_claims**: 빠른 응답 (캐싱/병렬화) → 저비용
- **legal_search**: 정확성 (리랭킹) → 높은 비용

### 8.3 인프라 제약
- **GPU 부족**: ONNX 모델 권장
- **메모리 부족**: 캐싱 크기 제한, 배치 크기 감소
- **DB 연결 제한**: 병렬화 수준 조정 (asyncio semaphore)

---

## 9. 결론

### 핵심 최적화 (Phase 1-2)
1. **벡터 검색**: IVF_FLAT + nprobes 동적 조정 (100-150ms 절감)
2. **병렬화**: 벡터+FTS, 원문 조회 배치 (100-200ms 절감)
3. **리랭킹**: ONNX 모델 (100-350ms 절감)
4. **캐싱**: 쿼리 임베딩 + 선택적 결과 (20-100ms 절감)

### 기대 효과
- **전체 파이프라인**: 25-50% 개선 (700ms → 350-525ms)
- **구현 난이도**: 중간 (3-4주)
- **리스크**: 낮음-중간 (모니터링으로 완화)

### 추천 우선순위
1. 병렬화 (저비용, 높은 효과)
2. IVF_FLAT + nprobes 조정 (중간 비용, 높은 효과)
3. ONNX 모델 (중간 비용, 높은 효과)
4. 캐싱 (선택적, 시나리오 의존)
