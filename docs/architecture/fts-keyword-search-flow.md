# FTS 키워드 검색 전체 흐름 (BM25 + BMW)

현재 코드 기준 (`backend/app/services/rag/keyword_search.py`) 키워드 검색 파이프라인 정리.
pg_textsearch v0.5.1 기반 BM25 스코어링 + BMW(Block-Max WAND) 최적화.

---

## 사용자 쿼리: `"교통사고 손해배상 판례"`

### STEP 1. MeCab 토큰화 (명사만, 2자 이상)

```
MeCab("교통사고 손해배상 판례")
→ ["교통사고", "교통", "사고", "손해배상", "손해", "배상", "판례"]
```

userdic 복합어 인식 + decomposition_map 분해 적용.
명사만(NNG+NNP), 2자 이상 필터.

### STEP 2. 공백 구분 검색 쿼리 생성

```python
search_query = " ".join(tokens)
→ "교통사고 교통 사고 손해배상 손해 배상 판례"
```

### STEP 3. to_bm25query() 호출

```python
bm25_query = func.to_bm25query(search_query, "idx_fts_bm25")
```

인덱스명을 명시 전달하여 IDF 정확성 보장.
`text_config='simple'`이므로 PG 측 추가 토크나이징 없이 공백 분리만 수행.

### STEP 4. BM25 인덱스 <@> 연산 + BMW 최적화

```sql
SELECT source_id, data_type, title,
       search_text <@> to_bm25query('교통사고 교통 사고 손해배상 손해 배상 판례', 'idx_fts_bm25') AS rank
FROM fts_index
WHERE data_type = '판례'
ORDER BY search_text <@> to_bm25query(...) ASC
LIMIT 50;
```

**핵심 동작:**
- `<@>` 연산자: **음수** BM25 점수 반환 (PG는 ASC 인덱스 스캔만 지원)
- 더 작은(더 음수인) 값 = 더 높은 관련성
- `ORDER BY ... ASC LIMIT n` 패턴이 **BMW(Block-Max WAND) 최적화** 트리거
- BMW: 블록 단위 상한 점수로 기여하지 않는 블록을 스킵 → top-K만 스코어링

### STEP 5. BM25 스코어링 공식

```
score = Σ idf(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * dl/avgdl))
```

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| k1 | 1.2 | TF 포화도 (높을수록 반복 토큰 중시) |
| b | 0.75 | 문서 길이 정규화 (높을수록 긴 문서 패널티) |
| idf | 자동 | Lucene-style: log(1 + (N-df+0.5)/(df+0.5)) |

→ ts_rank 대비: IDF 가중치 + TF 포화 + 문서 길이 정규화 = **훨씬 정교한 랭킹**

### STEP 6. 음수 점수 → 양수 변환

```python
similarity = abs(float(row.rank))  # -13.59 → 13.59
```

### STEP 7. 반환 → pipeline.py에서 벡터 검색 결과와 RRF 병합

```python
{"id": "76396", "content": "", "metadata": {...}, "similarity": 13.59, "score_type": "bm25"}
```

`content`는 빈 문자열 — 이후 PostgreSQL 원본 테이블에서 보충.

---

## 한 줄 요약

```
쿼리 → MeCab 토큰화 → to_bm25query() → BM25 인덱스 <@> 연산 (BMW top-K) → abs(score) → RRF 병합
```

---

## 성능 벤치마크 (2026-03-01)

| 항목 | 이전 (ts_rank + GIN) | 현재 (BM25 + BMW) |
|------|---------------------|-------------------|
| 인덱스 타입 | GIN (역색인) | BM25 AM (pg_textsearch) |
| 스코어링 | ts_rank (TF only) | BM25 (IDF + TF 포화 + 길이 정규화) |
| 최적화 | 없음 (매칭 전체 스코어링) | BMW (Block-Max WAND, top-K만) |
| 쿼리 시간 (필터 없음) | ~15초 | **21ms** |
| 쿼리 시간 (판례 필터) | ~10초 | **13.5ms** |
| EXPLAIN 결과 | Parallel Seq Scan | Index Scan using idx_fts_bm25 |

## 관련 데이터 현황 (2026-03-01)

| 항목 | 값 |
|------|---|
| fts_index 전체 건수 | 425,209건 (search_text 적재) |
| BM25 인덱스 | idx_fts_bm25 (text_config='simple') |
| PG 확장 | pg_textsearch v0.5.1 |
| PostgreSQL 버전 | 17 |
| 평균 문서 길이 | 212.17 토큰 |

### data_type별 분포

| data_type | 건수 |
|-----------|---:|
| 특별행정심판 | 137,288 |
| 판례 | 92,055 |
| 위원회결정례 | 57,613 |
| 부처유권해석 | 37,455 |
| 행정심판례 | 34,254 |
| 헌재결정례 | 31,718 |
| 행정규칙 | 17,092 |
| 법령해석례 | 8,597 |
| 법령 | 5,548 |
| 조약 | 3,589 |
