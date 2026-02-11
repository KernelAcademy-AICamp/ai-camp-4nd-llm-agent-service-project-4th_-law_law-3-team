# LanceDB 벡터 인덱스 최적화 개발 로그 (2026-02-11)

## 개요

LanceDB 벡터 검색 속도를 개선하기 위해 벡터 인덱스 타입별 벤치마크를 수행하고, 최적 인덱스(IVF_FLAT)를 서비스에 적용한 작업 기록.

---

## 1. 문제

### 1.1 검색 속도 병목

- 법률 데이터 ~253K 청크 (법령 118,922 + 판례 134,846)에 대한 벡터 검색이 **brute-force**(전수 비교) 방식으로 동작
- 단일 쿼리 평균 **~91ms** (median 89ms, p95 114ms)
- RAG 파이프라인에서 벡터 검색 → FTS → RRF 순으로 호출되므로, 벡터 검색 지연이 전체 응답 시간에 직접 영향

### 1.2 인덱스 미적용 원인

- `LanceDBStore` 클래스에 벡터 인덱스 생성 메서드가 없었음
- 앱 시작(lifespan)에서 FTS 인덱스만 생성하고 벡터 인덱스는 처리하지 않았음
- 어떤 인덱스 타입이 적합한지 데이터 기반 판단 자료가 없었음

---

## 2. 벤치마크

### 2.1 벤치마크 설계

**스크립트**: `backend/scripts/benchmark_lancedb_search.py`

| 항목 | 값 |
|------|-----|
| 데이터셋 | `legal_chunks` 테이블 (~253K 청크) |
| 쿼리 수 | 5개 (법률 도메인 대표 쿼리) |
| n_results | 10 |
| Warm-up | 2회 |
| 측정 반복 | 10회 |
| 측정 지표 | Mean, Median, P95, Min, Max (ms) |
| 정확도 지표 | Recall@10 (brute-force 대비 상위 결과 일치율) |

**테스트 쿼리**:
```
1. 손해배상 청구 요건과 입증 책임
2. 임대차 계약 해지 사유
3. 교통사고 과실 비율 산정 기준
4. 명예훼손 성립 요건
5. 상속 포기 절차와 기한
```

**인덱스 파라미터**:
- `num_partitions`: `max(16, sqrt(row_count))` 자동 계산 → 약 503
- `metric`: cosine
- `vector_column_name`: "vector" (1024차원, KURE-v1)

### 2.2 비교 대상 인덱스 타입

| 인덱스 | 알고리즘 | 특징 |
|--------|---------|------|
| **Brute-force** | 전수 비교 | 정확도 100%, 속도 느림 |
| **IVF_PQ** | IVF + Product Quantization | 벡터 압축, 속도 빠름, recall 손실 |
| **IVF_FLAT** | IVF + 원본 벡터 | 파티션 내 전수 비교, recall 유지 |
| **IVF_HNSW_SQ** | IVF + HNSW + Scalar Quantization | 그래프 탐색, 가장 빠름 |

### 2.3 벤치마크 결과

| Index Type | Mean (ms) | Median (ms) | P95 (ms) | Min (ms) | Max (ms) | Build Time | Recall |
|:-----------|----------:|------------:|---------:|---------:|---------:|-----------:|-------:|
| Brute-force | 90.86 | 89.06 | 114.20 | 78.45 | 125.03 | - | 100% |
| IVF_PQ | 3.41 | 3.18 | 4.76 | 2.57 | 10.26 | 26.3s | 60% |
| **IVF_FLAT** | **6.37** | **5.95** | **8.34** | **4.67** | **18.08** | **35.8s** | **100%** |
| IVF_HNSW_SQ | 3.68 | 3.16 | 5.35 | 2.41 | 22.52 | 46.3s | 90% |

### 2.4 결과 분석

```
속도 순위:  IVF_PQ (3.4ms) > IVF_HNSW_SQ (3.7ms) > IVF_FLAT (6.4ms) >> Brute-force (90.9ms)
정확도 순위: Brute-force = IVF_FLAT (100%) > IVF_HNSW_SQ (90%) >> IVF_PQ (60%)
```

**IVF_FLAT 선택 근거**:
1. **Recall 100%**: brute-force와 동일한 검색 결과 — 법률 도메인에서 정확도 손실은 허용 불가
2. **~14x 속도 향상**: 91ms → 6.4ms (median 기준 89ms → 6.0ms)
3. **빌드 시간 허용 범위**: 36초 (서버 시작 시 1회, 이미 존재하면 스킵)

**IVF_PQ 탈락 사유**:
- Recall 60% — 10개 결과 중 4개가 brute-force와 다름
- 법률 검색에서 관련 판례/법령을 놓치는 것은 치명적

**IVF_HNSW_SQ 탈락 사유**:
- Recall 90% — 10% 결과 차이 존재
- IVF_FLAT 대비 2ms 빠르지만 정확도 손실 정당화 불가
- 빌드 시간도 가장 김 (46초)

---

## 3. 서비스 적용

### 3.1 변경 파일

| 파일 | 변경 내용 |
|------|----------|
| `backend/app/core/config.py` | `LANCEDB_INDEX_TYPE` 설정 추가 (기본값: 빈 문자열 = brute-force) |
| `backend/app/tools/vectorstore/lancedb.py` | `create_vector_index()` 메서드 추가 |
| `backend/app/main.py` | lifespan에 벡터 인덱스 초기화 추가 |

### 3.2 설정 (`config.py`)

```python
# 빈 문자열이면 인덱스 미사용 (brute-force)
LANCEDB_INDEX_TYPE: str = ""

# .env에서 활성화
# LANCEDB_INDEX_TYPE=IVF_FLAT
```

- 기본값 빈 문자열 → 기존 동작(brute-force) 유지
- Feature flag 패턴으로 안전한 롤백 보장

### 3.3 인덱스 생성 메서드 (`lancedb.py`)

`create_vector_index(index_type)` 메서드 핵심 로직:

```
1. 테이블 없거나 비어있으면 → 스킵 (False 반환)
2. list_indices()로 기존 벡터 인덱스 확인 → 이미 있으면 스킵 (매 시작마다 36초 대기 방지)
3. num_partitions = max(16, sqrt(row_count)) 자동 계산
4. table.create_index(metric="cosine", index_type=..., replace=True) 실행
```

**스킵 로직이 중요한 이유**:
- IVF_FLAT 인덱스 생성에 ~36초 소요
- 서버 재시작마다 36초 대기는 개발 경험 저하
- `list_indices()`로 "vector" 컬럼에 인덱스가 이미 존재하는지 확인 후 스킵

### 3.4 앱 초기화 (`main.py`)

```
lifespan 실행 순서:
1. 임베딩 모델 로드 (기존)
2. 벡터 인덱스 생성 ← 추가됨 (LANCEDB_INDEX_TYPE 설정 시에만)
3. LangGraph 체크포인터 초기화 (기존)
4. 법률 용어 사전 초기화 (기존)
```

- `try/except`로 감싸 인덱스 생성 실패해도 앱 시작은 차단하지 않음
- 실패 시 brute-force로 자동 fallback (기능 정상 동작, 속도만 느림)

---

## 4. 사용법

### 활성화

```bash
# backend/.env
LANCEDB_INDEX_TYPE=IVF_FLAT
```

### 서버 시작 시 로그 (첫 실행)

```
INFO: 벡터 인덱스 생성 시작: type=IVF_FLAT, partitions=503, rows=253768
INFO: 벡터 인덱스 생성 완료: IVF_FLAT
INFO: LanceDB 벡터 인덱스 생성 완료: IVF_FLAT
```

### 재시작 시 로그 (이미 존재)

```
INFO: 벡터 인덱스 이미 존재, 생성 스킵 (테이블: legal_chunks)
```

### 롤백

```bash
# backend/.env (빈 문자열 또는 제거)
LANCEDB_INDEX_TYPE=
```

---

## 5. 성능 영향 요약

| 지표 | Before (Brute-force) | After (IVF_FLAT) | 개선 |
|------|---------------------:|------------------:|-----:|
| Mean 검색 시간 | 90.86ms | 6.37ms | **14.3x** |
| Median 검색 시간 | 89.06ms | 5.95ms | **15.0x** |
| P95 검색 시간 | 114.20ms | 8.34ms | **13.7x** |
| Recall@10 | 100% | 100% | 동일 |
| 앱 시작 시간 추가 | - | +36초 (최초 1회) | 재시작 시 스킵 |

---

## 6. 벤치마크 재현

```bash
cd backend
uv run python scripts/benchmark_lancedb_search.py
```

벤치마크 스크립트는 서비스 코드와 독립적으로 유지됨. 인덱스 타입 추가/변경 시 재측정 가능.

---

## 7. 관련 파일

| 파일 | 설명 |
|------|------|
| `backend/scripts/benchmark_lancedb_search.py` | 벤치마크 스크립트 |
| `backend/app/core/config.py` | `LANCEDB_INDEX_TYPE` 설정 |
| `backend/app/tools/vectorstore/lancedb.py` | `create_vector_index()` 메서드 |
| `backend/app/main.py` | lifespan 인덱스 초기화 |
| `docs/architecture/lancedb_fts_guide.md` | LanceDB 아키텍처 가이드 |

---

## 8. 향후 고려사항

- 데이터 증가 시 (500K+ 청크) 벤치마크 재측정 필요
- `num_partitions` 튜닝: 현재 `sqrt(N)` 자동 계산이지만, 데이터 분포에 따라 조정 가능
- `nprobes` 파라미터: 검색 시 탐색할 파티션 수 (기본값 사용 중, 속도-정확도 트레이드오프)
- IVF_HNSW_SQ 재검토: 데이터 규모가 커지면 recall 90%도 허용될 수 있음
