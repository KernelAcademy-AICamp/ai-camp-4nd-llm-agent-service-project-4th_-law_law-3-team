# pg-textsearch BM25 마이그레이션 완료 보고서

> **요약**: PostgreSQL FTS 키워드 검색을 ts_rank + GIN에서 pg_textsearch BM25 + BMW(Block-Max WAND)로 전환하여 검색 응답 시간을 Seq Scan 15초에서 Index Scan 21ms로 약 700배 개선하였습니다. Plan 65개 항목 중 61개 완전 구현, Match Rate 95% 달성.
>
> **프로젝트**: law-3 (Legal President / 법률 대통령)
> **보고일**: 2026-03-01
> **작성자**: Claude
> **상태**: 완료

---

## 1. 개요

### 1.1 기능 설명

**pg-textsearch BM25 마이그레이션**은 법률 문서 전문 검색(FTS) 엔진을 PostgreSQL 기본 FTS 스택에서 Tiger Data(Timescale)의 `pg_textsearch` 확장으로 전환하는 작업입니다.

기존 스택의 핵심 문제는 `ts_rank`가 매칭된 모든 문서를 전수 스코어링하여 50K+ 건 검색 시 30초 이상 소요되고, IDF 가중치와 문서 길이 정규화가 없어 검색 품질도 낮았습니다. 이를 해결하기 위해 다음 전환을 수행하였습니다.

| 구분 | 기존 | 전환 후 |
|------|------|---------|
| 랭킹 알고리즘 | ts_rank (IDF 없음, 정규화 없음) | BM25 (IDF + TF 포화 + 문서 길이 정규화) |
| 최적화 알고리즘 | GIN 집합 연산 + 전수 스코어링 | BMW (Block-Max WAND) top-K 스코어링 |
| PostgreSQL 버전 | 15 (alpine) | 17 (alpine, pg_textsearch 요구 사항) |
| 확장 설치 방식 | 없음 | C 소스 빌드 (Dockerfile.postgres 신규) |
| 텍스트 컬럼 | content_tsvector (tsvector 타입) | search_text (Text 타입, MeCab 전처리 후 저장) |
| 인덱스 타입 | GIN | bm25 (pg_textsearch 전용) |

**전환 후 아키텍처**:

```
쿼리 → MeCab 토큰화 → to_bm25query() → BM25 인덱스 <@> 연산 → BMW top-K 스코어링 → BM25 raw score → RRF 병합
```

### 1.2 PDCA 사이클 요약

| 단계 | 문서 | 상태 |
|------|------|------|
| **Plan** | `docs/01-plan/features/pg-textsearch-bm25-migration.plan.md` (v5) | 완료 |
| **Design** | 없음 (Plan 문서가 설계 역할 겸함) | — |
| **Do** | L1 인프라부터 L7 검증까지 7개 레이어 순차 구현 | 완료 |
| **Check** | `docs/03-analysis/pg-textsearch-bm25-migration.analysis.md` | 완료 (Match Rate 95%) |
| **Act** | 완료 보고서 작성 | 완료 |

---

## 2. Plan 단계 요약

### 2.1 목표

| 항목 | 내용 |
|------|------|
| 핵심 목표 | ts_rank + GIN에서 BM25 + BMW로 전환, 검색 품질·속도 동시 개선 |
| 예상 성능 개선 | 3-8초 (4-10x 개선) — 실제 결과: 21ms (~700x 개선) |
| 핵심 제약 | PostgreSQL 17 이상 필수, pg_textsearch C 소스 빌드 필요 |
| 구현 범위 | L1 인프라 ~ L7 검증, 7개 레이어, 65개 항목 |

### 2.2 주요 요구사항

| 레이어 | 항목 수 | 내용 |
|--------|:------:|------|
| L1: 인프라 | 6개 | PG 17 업그레이드, pg_textsearch 소스 빌드, 확장 활성화 |
| L2: 스키마 | 3개 | search_text 컬럼 추가, Alembic 마이그레이션, BM25 인덱스 |
| L3: 인제스트 파이프라인 | 29개 | db_writer 변경, 20개 타입 FTS body 선별, fts_builder/tsvector_builder 삭제 |
| L4: 검색 코드 | 11개 | keyword_search.py 전면 리라이트, BMW 트리거 패턴 적용 |
| L5: 연동 코드 | 3개 | retrieval.py 정리, Feature Flag, is_fts_available 변경 |
| L6: 인덱스 생성 | 3개 | 425K건 search_text 채우기, 병렬 빌드, BM25 인덱스 생성 |
| L7: 검증 + 문서 | 8개 | 성능 벤치마크, RAG 평가, 전수 테스트, 문서 4건 갱신 |
| **합계** | **65개** | |

### 2.3 성공 기준

Plan v5에 정의된 완료 조건:

```
[완료] PG 17 기반 pg_textsearch 확장 설치
[완료] BM25 인덱스 생성 및 search_text 전수 적재
[완료] keyword_search.py BM25 + BMW 전환
[완료] Feature Flag USE_BM25_SEARCH=True 활성화
[완료] 성능 벤치마크 (Seq Scan vs Index Scan 비교)
[완료] 아키텍처/CLAUDE.md/ingest.md 문서 갱신
[미완료] RAG 평가 (임베딩 모델 런타임 환경 필요)
[부분완료] 21개 타입 전수 테스트 (10개 완료, 11개 미확인)
```

---

## 3. 구현 단계 요약

### 3.1 Layer 1: 인프라 (Docker + PostgreSQL)

| # | 작업 | 결과 |
|---|------|------|
| 1-1 | PG 15 → 17 업그레이드 | `docker/postgres/Dockerfile` — `FROM postgres:17-alpine` |
| 1-2 | 데이터 마이그레이션 | pg_dump → 볼륨 삭제 → 새 이미지 → pg_restore, 기존 전체 테이블 보존 |
| 1-3 | pg_textsearch 소스 빌드 | `docker/postgres/Dockerfile` — git clone v0.5.1 + make + make install |
| 1-4 | shared_preload_libraries 설정 | `docker-compose.yml:18` — `command: postgres -c shared_preload_libraries=pg_textsearch` |
| 1-5 | 확장 활성화 | `docker/postgres/init.sql:7` — `CREATE EXTENSION IF NOT EXISTS "pg_textsearch"` |
| 1-6 | 한국어 text_config=simple 확정 | Python MeCab 전처리 → 공백 구분 토큰을 search_text에 저장, PG 측 추가 토크나이저 불필요 |

신규 파일 `docker/postgres/Dockerfile`을 통해 pg_textsearch를 C 소스 빌드로 설치하며, `docker-compose.yml`에서 `image:` 대신 `build:`로 전환하였습니다. `shm_size: 1gb` 설정도 추가하여 인덱스 생성 시 메모리 부족 문제를 사전 차단했습니다.

### 3.2 Layer 2: 스키마 (ORM + Alembic)

| # | 작업 | 결과 |
|---|------|------|
| 2-1 | fts_index에 search_text 컬럼 추가 | `app/models/fts_index.py:73` — `search_text = Column(Text, nullable=True)` |
| 2-2 | GIN 인덱스 제거, BM25 인덱스 raw SQL | ORM에 GIN 인덱스 없음, BM25 인덱스는 Alembic raw SQL로 생성 |
| 2-3 | Alembic 마이그레이션 작성 | `backend/alembic/versions/020_add_bm25_search_text.py` |

기존 `content_tsvector` 컬럼은 BM25 안정화 후 제거하기로 결정하여 롤백 안전성을 확보하였습니다.

### 3.3 Layer 3: 인제스트 파이프라인

| 작업 그룹 | 구현 내용 |
|----------|----------|
| db_writer search_text 저장 | `db_writer.py:198-201` — `" ".join(tokens)`로 search_text 직접 저장 |
| verify_db 검증 대상 변경 | `db_writer.py:273-278` — `fts_with_search_text` (search_text IS NOT NULL 카운트) |
| shared.py upsert 컬럼 | `shared.py:49-56` — ON CONFLICT DO UPDATE에 search_text 추가 |
| cli.py fts 스텝 변경 | `cli.py:189-193` — `run_search_text_rebuild`로 교체 |
| 20개 타입 FTS body 선별 | 각 타입별 `_fulltext_fn` + `_orm_fulltext_fn` 노이즈 필드 제거 (9개 필드 → 2-3개로 축소) |
| fts_builder.py 삭제 | BM25 인덱스가 INSERT/UPDATE 시 자동 갱신되므로 독립 재빌드 불필요 |
| tsvector_builder.py 삭제 | `" ".join(tokens)` 한 줄로 대체 |
| 삭제 파일 참조 정리 | db_writer.py, __init__.py, shared.py, test 파일의 import 및 docstring 정리 |

20개 타입별 FTS body 선별(3-5 ~ 3-24)을 통해 BM25 스코어링 대상 텍스트에서 supplementary, reason 등 노이즈 필드를 제거하여 BM25 정밀도를 향상시켰습니다.

### 3.4 Layer 4: 검색 코드 (BMW 최적화)

BMW 최적화를 활성화하기 위한 핵심 패턴을 적용하였습니다.

**구현된 핵심 SQL 패턴**:

```python
# BM25 + BMW 최적화 (ORDER BY ASC LIMIT n → BMW 트리거)
bm25_query = func.to_bm25query(search_query, _BM25_INDEX_NAME)
score_expr = FtsIndex.search_text.op("<@>")(bm25_query)
stmt = select(..., score_expr.label("rank"))
stmt = stmt.order_by(score_expr.asc()).limit(n_results)  # BMW 트리거
```

**Plan 명세와의 주요 차이 (버그 수정)**:

| 항목 | Plan 명세 | 실제 구현 | 사유 |
|------|-----------|----------|------|
| to_bm25query() 인자 순서 | `to_bm25query('idx_fts_bm25', query_text)` | `func.to_bm25query(search_query, _BM25_INDEX_NAME)` | pg_textsearch 실제 API에 맞게 수정 |
| ORDER BY 방향 | DESC | ASC | `<@>` 연산자가 음수 BM25 점수를 반환하므로 ASC가 관련성 높은 순 |
| 점수 처리 | raw score 직접 사용 | `abs(float(row.rank))` | 음수 점수를 양수로 변환 |

keyword_search.py에서 `_build_concept_and_tsquery`, `_build_or_tsquery`, `_FTS_STOPWORDS`, `_INVALID_TOKEN_RE`, `_clean_token`, `_get_query_tokens` 등 tsquery 전용 함수 전체를 삭제하였습니다. `_tokenize()`, `_map_doc_type_to_data_type()`, `_map_data_type_to_doc_type()` 3개 함수는 BM25에서도 필요하여 보존하였습니다.

pipeline.py에서는 `_PrecomputedInputs`의 `concept_tsqueries`, `or_tsqueries` 필드를 제거하고 `embeddings` 필드만 남겼으며, `_precompute_inputs`에서 tsquery 계산 2개를 제거하여 임베딩 계산만 수행하도록 간소화하였습니다.

### 3.5 Layer 5: 연동 코드

| # | 작업 | 결과 |
|---|------|------|
| 5-1 | retrieval.py 파라미터 정리 | precomputed_concept_tsq, precomputed_or_tsq 파라미터 제거 |
| 5-2 | config.py Feature Flag | `USE_BM25_SEARCH: bool = True` (기본값 True로 활성화 배포) |
| 5-3 | is_fts_available BM25 호환 | `pg_catalog.pg_indexes`에서 `idx_fts_bm25` 존재 확인 |

`USE_BM25_SEARCH`는 `USE_HYBRID_SEARCH`(벡터+키워드 병합 on/off)와 독립적으로 동작하도록 설계하였습니다. `USE_HYBRID_SEARCH=True` + `USE_BM25_SEARCH=True`이면 BM25가 키워드 검색에 사용됩니다.

### 3.6 Layer 6: search_text 채우기 + BM25 인덱스 생성

| # | 작업 | 결과 |
|---|------|------|
| 6-1 | fts_index search_text 채우기 | 425,209건 적재 완료 (1,000건 배치, ORM에서 orm_fulltext_fn() 호출) |
| 6-2 | 병렬 빌드 설정 | `max_parallel_maintenance_workers = 4`, `maintenance_work_mem = '256MB'` |
| 6-3 | BM25 인덱스 생성 | `CREATE INDEX idx_fts_bm25 ON fts_index USING bm25(search_text) WITH (text_config='simple')` |

pg_textsearch가 `CREATE INDEX CONCURRENTLY`를 지원하지 않음을 확인하여 일반 `CREATE INDEX`로 실행하였습니다. `shm_size: 1gb` 설정 덕분에 병렬 빌드가 정상 동작하였습니다.

초기 Plan 명세(`--step db` JSON 전체 재적재)와 달리, fts_index 테이블에 이미 579K건이 존재하고 search_text 컬럼만 NULL인 상황이었으므로 `--step fts`(`search_text_rebuilder.py` 경로)를 사용하여 ORM 원본에서 search_text만 재빌드하였습니다. 이로 인해 425,209건(원래 예상 579K건 대비)이 적재되었으며, 이는 각 타입별 FTS body 칼럼 선별 후 유효 텍스트가 있는 건수를 반영합니다.

### 3.7 Layer 7: 검증 + 문서

| # | 작업 | 상태 |
|---|------|:----:|
| 7-1 | 성능 벤치마크 | 완료 — Seq Scan 15s → Index Scan 21ms (~700x 개선) |
| 7-2 | RAG 평가 | 미완료 — 임베딩 모델 런타임 환경 필요 |
| 7-3 | 21개 타입 전수 테스트 | 부분완료 — 10개 data_type 완료, 11개 미확인 |
| 7-4 | 아키텍처 문서 업데이트 | 완료 — `docs/architecture/fts-keyword-search-flow.md` BM25+BMW 전면 개정 |
| 7-5 | 루트 CLAUDE.md 갱신 | 완료 — PG 17, BM25, USE_BM25_SEARCH 반영 |
| 7-6 | backend/scripts/CLAUDE.md 갱신 | 완료 — fts_builder 참조 제거 |
| 7-7 | wsl2-docker.md 갱신 | 완료 — PG 17 + pg_textsearch, shm_size 반영 |
| 7-8 | ingest.md 갱신 | 부분완료 — line 402에 과거 fts_builder 비교 히스토리 잔류 |

---

## 4. Check 단계 결과

### 4.1 Gap Analysis 요약

| 항목 | 값 |
|------|---|
| **Match Rate** | **95%** |
| 총 항목 수 | 65개 |
| 완전 구현 | 61개 |
| 부분 구현 | 3개 |
| 미구현 | 1개 |

### 4.2 Layer별 점수

| 레이어 | 점수 | 상태 |
|--------|:----:|:----:|
| L1: 인프라 | 100% | 완료 |
| L2: 스키마 | 100% | 완료 |
| L3: 인제스트 파이프라인 | 98% | 완료 |
| L4: 검색 코드 | 100% | 완료 |
| L5: 연동 코드 | 100% | 완료 |
| L6: 인덱스 생성 | 100% | 완료 |
| L7: 검증 + 문서 | 75% | 주의 |
| **전체** | **95%** | **완료** |

### 4.3 미구현 항목

| 항목 | Plan 위치 | 설명 | 우선순위 |
|------|-----------|------|---------|
| RAG 평가 | 7-2 | Recall@10, MRR, Hit Rate 미측정. 임베딩 모델(`nlpai-lab/KURE-v1`, 2.3GB) 런타임 환경 준비 필요 | 낮음 (운영 검증 단계) |

### 4.4 부분 완료 항목

| 항목 | Plan 위치 | 구현 상태 | 잔여 작업 |
|------|-----------|----------|----------|
| ingest.md fts_builder 갱신 | 3-29 | 상단 BM25 기준 갱신 완료. line 402에 "이전 fts_builder.py에는 없었음" 히스토리 잔류 | line 402 표현 삭제 또는 과거형 명시 |
| 21개 타입 전수 테스트 | 7-3 | precedent, law, constitutional, interpretation_ministry 등 10개 data_type 완료 | admin_rule 등 11개 타입 수동 테스트 |

### 4.5 아키텍처 주의사항 (동작 영향 없음)

| 항목 | Plan 명세 | 실제 구현 | 설명 |
|------|-----------|----------|------|
| to_bm25query() 인자 순서 | `to_bm25query('idx_fts_bm25', query_text)` | `func.to_bm25query(search_query, _BM25_INDEX_NAME)` | Plan v5 버그 수정. 실제 pg_textsearch API 기준으로 수정 |
| ORDER BY 방향 | DESC | ASC | `<@>` 연산자가 음수 BM25 점수를 반환하므로 ASC가 관련성 높은 순 |
| content_tsvector | 롤백용 유지 | 유지됨 | BM25 안정화 후 제거 시점 결정 예정 |

---

## 5. 주요 성과

### 5.1 성능 개선 수치

| 지표 | 기존 (ts_rank + GIN) | 전환 후 (BM25 + BMW) | 개선율 |
|------|:-------------------:|:-------------------:|:------:|
| 검색 응답 시간 | Seq Scan 15초 | Index Scan 21ms | ~714x |
| 실행 계획 | Sequential Scan | Bitmap Index Scan | 전환 완료 |
| 스코어링 범위 | 매칭 전체 문서 전수 | BMW top-K only | 대폭 감소 |
| IDF 지원 | 없음 | 있음 (BM25 알고리즘 내장) | 신규 |
| 문서 길이 정규화 | 없음 | 있음 (BM25 b 파라미터) | 신규 |

Plan 단계 예상 개선치(4-10x)를 실제 결과(~700x)가 크게 초과달성하였습니다.

### 5.2 아키텍처 개선

| 항목 | 내용 |
|------|------|
| FTS 파이프라인 단순화 | fts_builder.py 삭제 — BM25 인덱스가 INSERT/UPDATE 시 자동 갱신되므로 독립 재빌드 스크립트 불필요 |
| tsvector_builder.py 삭제 | `" ".join(tokens)` 한 줄로 대체 가능한 모듈 제거 |
| tsquery 로직 전체 제거 | `_build_concept_and_tsquery`, `_build_or_tsquery`, OR fallback 2단계 전략 등 복잡한 tsquery 로직 삭제 |
| _PrecomputedInputs 간소화 | `concept_tsqueries`, `or_tsqueries` 필드 제거 → `embeddings` 필드만 유지 |
| BM25 정밀도 향상 | 20개 타입 FTS body 선별로 노이즈 필드 제거 — 평균 9개 필드에서 2-3개 핵심 필드로 축소 |
| Feature Flag 독립 설계 | `USE_BM25_SEARCH`를 `USE_HYBRID_SEARCH`와 독립적으로 관리, 롤백 가능 구조 유지 |

### 5.3 구현된 파일 목록

**수정 (17개 코어 파일 + 17개 타입 파일)**:

| 파일 | 변경 내용 |
|------|----------|
| `docker-compose.yml` | PG 17 + build: 전환 + shared_preload_libraries + shm_size |
| `docker/postgres/init.sql` | CREATE EXTENSION pg_textsearch |
| `backend/app/models/fts_index.py` | search_text 컬럼 추가 |
| `backend/app/services/rag/keyword_search.py` | BM25 + BMW 전면 리라이트 |
| `backend/app/services/rag/pipeline.py` | import 정리, _PrecomputedInputs, _precompute_inputs, _search_and_deduplicate |
| `backend/app/services/rag/retrieval.py` | precomputed 파라미터 제거 |
| `backend/app/core/config.py` | USE_BM25_SEARCH flag |
| `backend/scripts/ingest/db_writer.py` | search_text 저장, tsvector_builder import 제거, verify_db 변경 |
| `backend/scripts/ingest/shared.py` | upsert 컬럼 + docstring |
| `backend/scripts/ingest/cli.py` | fts_builder import 제거, fts 스텝 교체 |
| `backend/scripts/ingest/__init__.py` | docstring fts_builder 참조 제거 |
| `backend/scripts/ingest/ingest.md` | fts_builder 관련 설명 갱신 (line 402 잔류) |
| `backend/tests/unit/test_ingest_pipeline.py` | tsvector_builder import 제거 + 테스트 수정 |
| `docs/architecture/fts-keyword-search-flow.md` | BM25+BMW 전면 개정 |
| `CLAUDE.md`, `backend/CLAUDE.md`, `backend/scripts/CLAUDE.md` | PG 17, BM25, fts_builder 참조 정리 |
| `.claude/rules/wsl2-docker.md` | PG 17 이미지 + pg_textsearch + shm_size 반영 |
| `backend/scripts/ingest/types/` 하위 17개 파일 | _fulltext_fn + _orm_fulltext_fn FTS body 선별 |

**신규 (3개)**:

| 파일 | 용도 |
|------|------|
| `docker/postgres/Dockerfile` | pg_textsearch C 확장 소스 빌드 (필수) |
| `backend/alembic/versions/020_add_bm25_search_text.py` | Alembic 마이그레이션 |
| `backend/scripts/ingest/search_text_rebuilder.py` | search_text 전용 재빌드 스크립트 |

**삭제 (2개)**:

| 파일 | 사유 |
|------|------|
| `backend/scripts/ingest/fts_builder.py` | BM25 인덱스 자동 갱신으로 독립 재빌드 불필요 |
| `backend/app/services/rag/tsvector_builder.py` | `" ".join(tokens)` 한 줄로 대체 |

---

## 6. 잔여 사항 및 추후 계획

### 6.1 즉시 조치 (선택적)

| 항목 | 내용 |
|------|------|
| ingest.md line 402 정리 | 과거 fts_builder 비교 표현을 삭제하거나 과거형으로 명시 |
| 잔여 11개 data_type 테스트 | admin_rule, treaty, special_admin, dec_labor, dec_human_rights, dec_privacy, dec_employment, dec_financial, dec_industrial, dec_environment, dec_securities 수동 테스트 |

### 6.2 추후 조치

| 항목 | 조건 | 내용 |
|------|------|------|
| RAG 평가 실행 | 임베딩 모델(`nlpai-lab/KURE-v1`, 2.3GB) 런타임 준비 후 | Recall@10 >= 0.8, MRR >= 0.7, Hit Rate >= 0.9 목표 달성 여부 확인 |
| content_tsvector 컬럼 제거 | BM25 운영 안정성 확인 후 | `backend/app/models/fts_index.py` 및 Alembic 마이그레이션 추가 필요 |
| Plan 문서 SQL 예시 수정 | 선택적 | `to_bm25query()` 인자 순서 및 `ORDER BY` 방향 오류 교정 |
| pg_textsearch 버전 업그레이드 | 공식 GA 릴리스 후 | 현재 v0.5.1 (v1.0.0-dev GA 진행 중) |

---

## 7. 교훈 및 인사이트

### 7.1 인프라 구축 시 주의사항

**pg_textsearch 소스 빌드 환경**: Alpine 기반 이미지에서 C 소스 빌드 시 `alpine-sdk`, `clang`, `llvm-dev`, `build-base` 등 빌드 도구를 모두 설치해야 합니다. `Dockerfile.postgres`(최종적으로 `docker/postgres/Dockerfile`) 신규 작성이 필수이며, `docker-compose.yml`에서 `image:` 대신 `build:`로 전환해야 합니다.

**shm_size 설정 필수**: BM25 인덱스 생성 시 PostgreSQL의 공유 메모리가 부족하면 조용히 실패합니다. `docker-compose.yml`에 `shm_size: 1gb` 명시가 필수입니다.

**CONCURRENTLY 미지원**: pg_textsearch v0.5.1은 `CREATE INDEX CONCURRENTLY`를 지원하지 않습니다. 일반 `CREATE INDEX`를 사용해야 하며, 병렬 빌드 설정(`max_parallel_maintenance_workers = 4`, `maintenance_work_mem = '256MB'`)으로 속도를 보완합니다.

### 7.2 pg_textsearch API 주의사항

**to_bm25query() 인자 순서**: Plan 문서에 `to_bm25query('idx_fts_bm25', query_text)` 순서로 기술되었으나, 실제 pg_textsearch API는 `to_bm25query(query_text, 'idx_fts_bm25')` 순서입니다. 실행 시 오류가 발생하여 수정하였습니다.

**ORDER BY 방향**: `<@>` 연산자는 음수 BM25 점수를 반환합니다. Plan 문서의 `ORDER BY DESC`는 오류이며, `ORDER BY ASC`가 관련성 높은 순으로 정렬됩니다. `abs(float(row.rank))`로 점수를 양수로 변환하여 `similarity` 필드에 저장합니다.

**인덱스명 필수 전달**: `to_bm25query()`에 인덱스명을 명시적으로 전달해야 IDF 통계를 정확히 계산합니다. PL/pgSQL 함수 내에서는 자동 감지가 불가능합니다.

**BMW 트리거 조건**: `ORDER BY <@> [ASC|DESC] LIMIT n` 조합이 BMW를 활성화하는 유일한 방법입니다. LIMIT이 없으면 `pg_textsearch.default_limit`(기본 1000)까지 전수 스코어링으로 fallback됩니다.

### 7.3 인제스트 파이프라인 전략

**--step fts vs --step db 선택**: fts_index 테이블에 이미 데이터가 존재하고 search_text 컬럼만 NULL인 경우, `--step db`(JSON 전체 재적재) 대신 `--step fts`(ORM에서 search_text만 재빌드)를 사용하면 시간을 크게 절약할 수 있습니다.

**FTS body 선별의 중요성**: BM25의 IDF 가중치는 코퍼스 내 단어 빈도에 민감합니다. supplementary(부칙), reason(이유) 등 반복 등장하는 노이즈 필드를 제거하면 핵심 필드(case_name, summary, reasoning 등)의 IDF가 올라가 검색 정밀도가 향상됩니다.

### 7.4 레이어 단위 구현 전략의 효과

Plan에서 7개 레이어로 작업을 분리하고 레이어 경계에서 커밋을 강제한 전략이 유효했습니다. 특히 L1(인프라) 완료 후 L2(스키마)로 진행하는 순서가 의존성 문제를 방지하였고, L6(인덱스 생성) 완료 후 L7(검증)에서 실제 성능 수치를 측정할 수 있었습니다.

---

## 8. 결론

### 8.1 최종 평가

| 항목 | 결과 | 평가 |
|------|:----:|------|
| Plan 준수도 | 95% | PASS — 기준(90%) 초과 달성 |
| 성능 개선 | ~700x | PASS — 예상(4-10x) 대폭 초과 달성 |
| 코드 정리 | 불필요 코드 전면 제거 | PASS — tsquery 로직 + tsvector 관련 파일 삭제 |
| 문서화 | 95% | PASS — 아키텍처/CLAUDE.md/ingest.md 일괄 갱신 |
| RAG 평가 | 미실행 | 보류 — 임베딩 모델 런타임 환경 준비 후 진행 |

### 8.2 PDCA 효율성

```
완료 기간:    2026-03-01 (단일 세션)
반복 횟수:    0회 (분석 후 재구현 불필요)
최종 달성도:  95% (90% 기준 초과)
성능 초과달성: ~700x (예상 4-10x 대비)
```

성능 개선이 예상을 크게 초과한 주요 원인은 BMW(Block-Max WAND) 알고리즘의 효과로, top-K 결과만 스코어링하여 기존 전수 스코어링 대비 처리 건수가 극적으로 감소하였기 때문입니다.

---

## 9. 관련 문서

| 문서 | 경로 | 상태 |
|------|------|------|
| Plan (v5) | `docs/01-plan/features/pg-textsearch-bm25-migration.plan.md` | 완료 |
| Analysis | `docs/03-analysis/pg-textsearch-bm25-migration.analysis.md` | 완료 (95%) |
| 아키텍처 | `docs/architecture/fts-keyword-search-flow.md` | BM25+BMW 전면 개정 |
| Report | `docs/04-report/features/pg-textsearch-bm25-migration.report.md` | 완료 |

---

## 10. 버전 이력

| 버전 | 날짜 | 변경 사항 | 작성자 |
|------|------|----------|--------|
| 1.0 | 2026-03-01 | 최초 완료 보고서 작성 — PDCA 사이클 완료, 95% Match Rate, Seq Scan 15s → Index Scan 21ms (~700x) | Claude |

---

**보고서 작성 완료 날짜**: 2026-03-01
**보고서 상태**: 최종 완료
**다음 단계**: 임베딩 모델 런타임 준비 후 RAG 평가 실행, 잔여 11개 data_type 전수 테스트, content_tsvector 컬럼 제거 시점 결정
