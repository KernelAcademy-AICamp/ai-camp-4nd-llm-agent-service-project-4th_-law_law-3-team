# pg_textsearch BM25 마이그레이션 계획 (v4)

> **목표**: PostgreSQL FTS 키워드 검색을 `ts_rank` + GIN에서 `pg_textsearch` BM25 + BMW로 전환하여 검색 품질과 속도를 개선한다.
>
> **현재 문제**: 579K건 FTS 검색 시 `ts_rank`가 매칭된 전체 문서(50K+)를 스코어링 → 30초 소요. IDF 가중치 없음, 문서 길이 정규화 없음.
>
> **기대 효과**: BM25 랭킹(IDF + TF 포화 + 문서 길이 정규화) + BMW(Block-Max WAND)로 top-K만 스코어링 → 3-8초 (4-10x 개선)

## 배경

### pg_textsearch

- **제작**: Tiger Data (Timescale)
- **GitHub**: `timescale/pg_textsearch`
- **버전**: v1.0.0-dev (GA 작업 중), 최신 릴리스 태그 v0.5.1
- **요구**: PostgreSQL 17+
- **핵심 기능**: BM25 스코어링, BMW(Block-Max WAND) 알고리즘, 디스크 세그먼트, 압축

### 현재 아키텍처

```
쿼리 → MeCab 토큰화 → tsquery 생성 → GIN 역색인 집합연산 → ts_rank 전수 스코어링 → 수동 정규화 → RRF 병합
```

### 전환 후 아키텍처

```
쿼리 → MeCab 토큰화 → to_bm25query() → BM25 인덱스 <@> 연산 → BMW top-K 스코어링 → BM25 raw score → RRF 병합
```

---

## 레이어 1: 인프라 (Docker + PostgreSQL)

| # | 작업 | 파일 | 상세 |
|---|------|------|------|
| 1-1 | PG 15 → 17 업그레이드 | `docker-compose.yml` | `postgres:15-alpine` → `postgres:17-alpine` |
| 1-2 | 데이터 마이그레이션 | (CLI) | `pg_dump` → 볼륨 삭제 → 새 이미지 → `pg_restore`. 기존 579K건 + 21개 테이블 전체 |
| 1-3 | pg_textsearch 설치 | **신규** `Dockerfile.postgres` | C 확장이므로 소스 빌드 필수. `postgres:17-alpine` 기반으로 `git clone + make + make install`. `docker-compose.yml`의 `image:` → `build:` 전환 |
| 1-4 | shared_preload_libraries 설정 | `docker-compose.yml` 또는 `Dockerfile.postgres` | `command: postgres -c shared_preload_libraries=pg_textsearch` 또는 커스텀 `postgresql.conf` |
| 1-5 | 확장 활성화 | `docker/postgres/init.sql` | `CREATE EXTENSION IF NOT EXISTS pg_textsearch;` 추가 |
| 1-6 | 한국어 text_config | - | `text_config='simple'` 사용 확정. Python MeCab 전처리 → 공백 구분 토큰 텍스트를 `search_text`에 저장. PG 측 추가 토크나이저 불필요 |

---

## 레이어 2: 스키마 (ORM + Alembic)

| # | 작업 | 파일 | 상세 |
|---|------|------|------|
| 2-1 | fts_index에 원문 텍스트 컬럼 추가 | `backend/app/models/fts_index.py` | `search_text = Column(Text)` 추가 — BM25 인덱스 대상. 기존 `content_tsvector`는 롤백용으로 유지 또는 제거 |
| 2-2 | GIN 인덱스 → BM25 인덱스 교체 | `backend/app/models/fts_index.py` | `idx_fts_index_content_tsvector` (GIN) 제거, BM25 인덱스는 raw SQL로 생성 |
| 2-3 | Alembic 마이그레이션 작성 | `backend/alembic/versions/NNN_add_bm25_index.py` | `op.add_column('fts_index', Column('search_text', Text))` + raw SQL `CREATE INDEX ... USING bm25(search_text)` |

---

## 레이어 3: 인제스트 파이프라인

### 3-1. BM25 search_text 생성 방식 변경

| # | 작업 | 파일 | 상세 |
|---|------|------|------|
| 3-1 | db_writer: search_text 저장 | `db_writer.py` :193-205 | 현재: `tokens → tsvector_str → fts_meta["content_tsvector"]`. 변경: `fts_meta["search_text"] = " ".join(tokens)`. `build_tsvector_string()` 호출 제거, 인라인 처리 |
| 3-2 | db_writer: verify_db 검증 대상 변경 | `db_writer.py` :276-282 | `fts_with_tsvector` → `fts_with_search_text` (`FtsIndex.search_text.isnot(None)` 카운트) |
| 3-3 | shared.py: upsert 컬럼 업데이트 | `shared.py` :49-56 | `on_conflict_do_update`의 `set_`에 `search_text` 추가, `content_tsvector` 제거 |
| 3-4 | cli.py: fts 스텝 변경 | `cli.py` :44,54,191 | `fts_builder` import 제거. `fts` 스텝을 `db_writer` 기반 search_text 재생성으로 교체 (토크나이저/userdic 변경 후 사용) |

### 3-2. FTS body 칼럼 선별 (BM25 인덱스 텍스트 소스)

각 타입의 `fulltext_fn` / `orm_fulltext_fn`이 반환하는 텍스트가 곧 `search_text` 컬럼에 저장되어 BM25 인덱스의 스코어링 대상이 된다. 노이즈 필드를 제거하여 BM25 정밀도를 높인다.

> 상세 명세: `docs/01-plan/features/ingest-column-restructure.plan.md` 3.1절 참조

| # | 타입 | 변경 후 FTS body (search_text 소스) | 변경 |
|---|------|-------------------------------------|------|
| 3-5 | precedent | case_name, summary, reasoning | 9개 → 3개 |
| 3-6 | law | law_name, content | supplementary 제거 |
| 3-7 | admin_rule | admin_rule_name, content | supplementary 제거 |
| 3-8 | interpretation_ministry | case_name, inquiry, answer | 6개 → 3개 |
| 3-9 | constitutional | case_name, summary, reasoning | 9개 → 3개 |
| 3-10 | administration | case_name, claim, reason | 6개 → 3개 |
| 3-11 | legislation | case_name, inquiry, answer | reason 제거 |
| 3-12 | treaty | treaty_name_kr, content | 4개 → 2개 |
| 3-13 | special_admin | case_name, adjudication_summary(없으면 reason) | fallback 로직 |
| 3-14 | dec_labor | case_name, judgment_matter, judgment_summary | 6개 → 3개 |
| 3-15 | dec_human_rights | case_name, decision_summary, judgment_summary | 6개 → 3개 |
| 3-16 | dec_privacy | case_name, reason | **변경 없음** |
| 3-17 | dec_employment | case_name, claim, overview | 6개 → 3개 |
| 3-18 | dec_financial | case_name, action_reason, action_content | **변경 없음** |
| 3-19 | dec_industrial | case_label, case_major/mid/sub_category, issue, claim | 분류 추가 |
| 3-20 | dec_environment | case_name, party_claims, fact_investigation, case_overview | ruling/eval 제거 |
| 3-21 | dec_securities | case_name, action_reason | action_content 제거 |
| 3-22 | dec_civil_rights | case_name, complaint_flag, ruling | complaint_flag 추가 |
| 3-23 | dec_fair_trade | case_name, decision_summary, ruling | 5개 → 3개 |
| 3-24 | dec_media | case_name, ruling | **변경 없음** |

**수정 파일**: `backend/scripts/ingest/types/` 하위 17개 타입 파일의 `_fulltext_fn` + `_orm_fulltext_fn`

### 3-3. 삭제 대상

| 파일 | 사유 |
|------|------|
| `backend/scripts/ingest/fts_builder.py` | BM25 인덱스가 INSERT/UPDATE 시 자동 갱신되므로 독립 재빌드 불필요. `db_writer.py` 인제스트 경로로 통합 |
| `backend/app/services/rag/tsvector_builder.py` | `" ".join(tokens)` 한 줄로 대체 가능. 별도 모듈 불필요 |

### 3-4. 삭제 파일 참조 정리

fts_builder / tsvector_builder 삭제 시 깨지는 import와 참조를 정리한다.

| # | 파일 | 변경 |
|---|------|------|
| 3-25 | `db_writer.py` :25 | `from app.services.rag.tsvector_builder import build_tsvector_string` 제거 |
| 3-26 | `scripts/ingest/__init__.py` :11 | docstring에서 `fts_builder` 참조 제거 |
| 3-27 | `scripts/ingest/shared.py` :4 | docstring "db_writer / fts_builder 양쪽에서" → "db_writer에서" |
| 3-28 | `tests/unit/test_ingest_pipeline.py` :29 | `from app.services.rag.tsvector_builder import build_tsvector_string` 제거 + 관련 테스트 수정 |
| 3-29 | `scripts/ingest/ingest.md` | fts_builder 관련 설명 5곳 BM25 기준으로 갱신 |

---

## 레이어 4: 검색 코드 (BMW 최적화 활용)

### 핵심 원칙

1. **BMW 트리거**: `ORDER BY ... <@> ... DESC LIMIT n` 조합이 BMW(Block-Max WAND)를 트리거하는 **유일한 방법**. LIMIT 없으면 `pg_textsearch.default_limit`(기본 1000)까지 전수 스코어링으로 fallback
2. **IDF 정확성**: `to_bm25query('idx_fts_bm25', query_text)` 형태로 **인덱스명 필수 전달** — 해당 인덱스의 전체 코퍼스에서 IDF 통계를 정확히 계산. PL/pgSQL 함수 내에서는 자동 감지 불가
3. **정규화 위임**: BM25는 문서 길이 정규화(b 파라미터)가 알고리즘 내부에 포함 → 수동 정규화(`row.rank / max_rank`)는 BM25 점수를 왜곡하므로 제거

### keyword_search.py 보존/삭제 구분

| 구분 | 대상 | 사유 |
|------|------|------|
| **삭제** | `_build_concept_and_tsquery`, `_build_or_tsquery` | BM25가 자체 처리 |
| **삭제** | `_FTS_STOPWORDS`, `_INVALID_TOKEN_RE`, `_CONCEPT_AND_MIN_RESULTS` | BM25 불필요 |
| **삭제** | `_get_query_tokens` | 외부 참조 없음 (하위 호환용이었으나 미사용) |
| **삭제** | `_clean_token` | tsquery 전용 유틸, BM25 불필요 |
| **보존** | `_tokenize()` | BM25 쿼리 토큰화에 재사용 (MeCab 래퍼) |
| **보존** | `_map_doc_type_to_data_type()` | data_type 필터에 필요 |
| **보존** | `_map_data_type_to_doc_type()` | 결과 매핑에 필요 |
| **보존+수정** | `is_fts_available_sync()`, `is_fts_available()` | BM25 인덱스 존재 확인으로 변경 |

### 작업 목록

| # | 작업 | 파일 | 상세 |
|---|------|------|------|
| 4-1 | keyword_search.py 리라이트 | `keyword_search.py` | 위 보존/삭제 구분 적용. OR fallback 로직 전체 제거 — BM25가 부분 매칭을 자체 처리하므로 2단계 전략 불필요 |
| 4-2 | BM25 쿼리 — BMW 트리거 패턴 | 동 파일 `_execute_fts_query` 전면 교체 | `ORDER BY <@> DESC LIMIT n`으로 BMW 트리거. 아래 SQL 패턴 참조 |
| 4-3 | to_bm25query() — 인덱스명 명시 | 동 파일 | `to_bm25query('idx_fts_bm25', query_text)` — 인덱스명 필수 |
| 4-4 | 수동 정규화 제거 | 동 파일 :158-160 | `row.rank / max_rank` 제거 → `<@>` 연산자의 raw score를 `similarity`에 직접 사용 |
| 4-5 | score_type 변경 | 동 파일 :178 | `"fts_rank"` → `"bm25"` |
| 4-6 | search_by_keyword 시그니처 변경 | `keyword_search.py` :185-193 | `precomputed_concept_tsq`, `precomputed_or_tsq` 파라미터 제거. 원문 `query` 텍스트만으로 `to_bm25query()` 호출 |
| 4-7 | pipeline.py import 정리 | `pipeline.py` :20-23 | `_build_concept_and_tsquery`, `_build_or_tsquery` import 제거 |
| 4-8 | _PrecomputedInputs 정리 | `pipeline.py` :126-134 | `concept_tsqueries`, `or_tsqueries` 필드 제거. BM25는 원문 쿼리를 `to_bm25query()`에 직접 전달. `embeddings` 필드만 유지 |
| 4-9 | _precompute_inputs 간소화 | `pipeline.py` :472-501 | tsquery 계산 2개 (`_build_concept_and_tsquery`, `_build_or_tsquery`) 제거 → 임베딩 계산만 남음 |
| 4-10 | retrieval.py 호출 정리 | `retrieval.py` :637-641 | `precomputed_concept_tsq`, `precomputed_or_tsq` 전달 제거 |
| 4-11 | _search_and_deduplicate 정리 | `pipeline.py` :534-543 | `precomputed.concept_tsqueries`, `precomputed.or_tsqueries` 전달 코드 제거 |

### 변경 후 핵심 SQL 패턴

```python
# 현재 (ts_rank + @@ → 전수 스코어링, 30초)
tsquery_expr = func.to_tsquery("simple", tsquery_str)
rank_expr = func.ts_rank(FtsIndex.content_tsvector, tsquery_expr)
stmt = select(...).where(FtsIndex.content_tsvector.op("@@")(tsquery_expr))
stmt = stmt.order_by(rank_expr.desc()).limit(n_results)

# 변경 후 (BM25 <@> + ORDER BY LIMIT → BMW 최적화, 3-8초 예상)
bm25_query = func.to_bm25query("idx_fts_bm25", query_text)
score_expr = FtsIndex.search_text.op("<@>")(bm25_query)
stmt = select(..., score_expr.label("rank"))
stmt = stmt.order_by(score_expr.desc()).limit(n_results)  # ← BMW 트리거
```

---

## 레이어 5: 연동 코드

| # | 작업 | 파일 | 상세 |
|---|------|------|------|
| 5-1 | retrieval.py 파라미터 정리 | `retrieval.py` :594-598 | `precomputed_concept_tsq`, `precomputed_or_tsq` 파라미터 제거. `search_by_keyword()` 시그니처 변경에 맞춤 |
| 5-2 | config.py Feature Flag | `backend/app/core/config.py` | `USE_BM25_SEARCH: bool = False` 추가. 기존 `USE_HYBRID_SEARCH`(벡터+키워드 병합 on/off)와 독립. `USE_HYBRID_SEARCH=True` + `USE_BM25_SEARCH=True` → BM25, `USE_BM25_SEARCH=False` → 기존 ts_rank |
| 5-3 | is_fts_available 호환 | `keyword_search.py` :243-278 | `USE_BM25_SEARCH=True`일 때 `pg_catalog.pg_indexes`에서 `idx_fts_bm25` 존재 확인. `False`일 때 기존 로직 유지 |

---

## 레이어 6: search_text 채우기 + BM25 인덱스 생성

fts_index 테이블은 이미 579K행이 존재하고 `search_text` 컬럼만 NULL인 상태. JSON 소스 재적재(`--step db`) 없이 **`--step fts`로 ORM 테이블에서 읽어 search_text만 갱신**한다.

| # | 작업 | 상세 |
|---|------|------|
| 6-1 | fts_index 579K건에 search_text 채우기 | `uv run python -m scripts.ingest.cli --type all --step fts` (`--reset` 불필요 — 기존 579K행의 PK `(source_id, data_type)`가 동일하므로 `ON CONFLICT DO UPDATE`로 search_text만 제자리 갱신). `search_text_rebuilder.py`가 PostgreSQL ORM 원본에서 `orm_fulltext_fn()` → MeCab 토큰화 → `search_text` 컬럼 upsert. 1,000건 배치. JSON 재처리 없음 |
| 6-2 | 병렬 빌드 설정 | 인덱스 생성 전 실행: `SET max_parallel_maintenance_workers = 4;` + `SET maintenance_work_mem = '256MB';` (워커당 64MB 미만이면 **조용히 직렬 전환**되므로 최소 256MB 필수. 512MB면 더 빠름) |
| 6-3 | BM25 인덱스 생성 | 우선 `CREATE INDEX CONCURRENTLY` 시도 → 미지원 시 일반 `CREATE INDEX`로 fallback. `CREATE INDEX [CONCURRENTLY] idx_fts_bm25 ON fts_index USING bm25(search_text) WITH (text_config='simple')` |

> **`--step fts` vs `--step db`**: `--step db`는 JSON 소스 파일부터 파싱하여 ORM + FTS를 동시 적재하는 전체 재적재. `--step fts`는 이미 적재된 ORM 테이블에서 읽어 search_text만 재빌드하므로 훨씬 빠르다.
>
> **CONCURRENTLY 주의**: pg_textsearch README에서 명시적 지원을 확인하지 못함. 실행 시 `ERROR: ... does not support building indexes concurrently` 발생 가능 → 일반 `CREATE INDEX`로 전환. 579K건은 병렬 설정(6-2) 시 일반 모드로도 수 분 이내 완료 예상.

---

## 레이어 7: 검증 + 문서

| # | 작업 | 파일 | 상세 |
|---|------|------|------|
| 7-1 | 성능 벤치마크 | (스크립트) | ts_rank vs BM25 동일 쿼리셋 응답 시간 비교. BMW 트리거 확인 (`EXPLAIN ANALYZE`로 Block-Max WAND 사용 여부 검증) |
| 7-2 | RAG 평가 | `backend/evaluation/` | Recall@10, MRR, Hit Rate 비교 (기존 목표: Recall@10 ≥ 0.8, MRR ≥ 0.7, Hit Rate ≥ 0.9) |
| 7-3 | 21개 타입 전수 테스트 | (수동) | 각 data_type별 검색 결과 확인 |
| 7-4 | 아키텍처 문서 업데이트 | `docs/architecture/fts-keyword-search-flow.md` | BM25 + BMW 흐름으로 전면 개정 |
| 7-5 | CLAUDE.md 업데이트 | 루트 + `backend/CLAUDE.md` | FTS 관련 설명 갱신 (BM25, pg_textsearch, PG 17) |
| 7-6 | backend/scripts/CLAUDE.md | `backend/scripts/CLAUDE.md` | fts_builder 참조 제거, fts 스텝 설명 갱신 |
| 7-7 | docker 규칙 업데이트 | `.claude/rules/wsl2-docker.md` | PG 17 이미지 반영, Dockerfile.postgres 추가 |
| 7-8 | ingest.md 갱신 | `scripts/ingest/ingest.md` | fts_builder 관련 5곳 → BM25 search_text 기준으로 갱신 |

---

## 영향받는 파일 전체 목록

### 수정 (16개 + 17개 타입 파일)

| 파일 | 변경 내용 |
|------|----------|
| `docker-compose.yml` | PG 17 + `build:` 전환 + `shared_preload_libraries` |
| `docker/postgres/init.sql` | `CREATE EXTENSION pg_textsearch` |
| `backend/app/models/fts_index.py` | `search_text` 컬럼 + BM25 인덱스 |
| `backend/app/services/rag/keyword_search.py` | BM25 + BMW 리라이트 (보존 함수 4개 유지) |
| `backend/app/services/rag/pipeline.py` | import 정리, `_PrecomputedInputs`, `_precompute_inputs`, `_search_and_deduplicate` |
| `backend/app/services/rag/retrieval.py` | precomputed 파라미터 제거 |
| `backend/app/core/config.py` | `USE_BM25_SEARCH` flag |
| `backend/scripts/ingest/db_writer.py` | `search_text` 저장, `tsvector_builder` import 제거, `verify_db` 검증 대상 변경 |
| `backend/scripts/ingest/shared.py` | upsert 컬럼 + docstring |
| `backend/scripts/ingest/cli.py` | `fts_builder` import 제거, `fts` 스텝을 search_text 재생성으로 교체 |
| `backend/scripts/ingest/__init__.py` | docstring fts_builder 참조 제거 |
| `backend/scripts/ingest/ingest.md` | fts_builder 관련 설명 갱신 |
| `backend/tests/unit/test_ingest_pipeline.py` | `tsvector_builder` import 제거 + 테스트 수정 |
| `docs/architecture/fts-keyword-search-flow.md` | 아키텍처 문서 |
| `CLAUDE.md`, `backend/CLAUDE.md`, `backend/scripts/CLAUDE.md` | 프로젝트 문서 (PG 17, BM25, fts_builder 참조 정리) |
| `.claude/rules/wsl2-docker.md` | PG 17 이미지 반영 |
| `backend/scripts/ingest/types/` 하위 17개 파일 | `_fulltext_fn` + `_orm_fulltext_fn` FTS body 칼럼 선별 (BM25 search_text 소스) |

### 신규 (2개)

| 파일 | 용도 |
|------|------|
| `backend/alembic/versions/NNN_add_bm25_index.py` | Alembic 마이그레이션 |
| `Dockerfile.postgres` | pg_textsearch C 확장 소스 빌드 (필수) |

### 삭제 (2개)

| 파일 | 사유 |
|------|------|
| `backend/scripts/ingest/fts_builder.py` | BM25 인덱스 자동 갱신으로 독립 재빌드 불필요 |
| `backend/app/services/rag/tsvector_builder.py` | `" ".join(tokens)` 한 줄로 대체 |

---

## 미결 사항

| # | 항목 | 선택지 | 권장 |
|---|------|--------|------|
| ~~D-1~~ | ~~한국어 토큰화 방식~~ | ~~A: simple config + Python MeCab~~ | **확정: A** (1-6에서 결정) |
| ~~D-2~~ | ~~content_tsvector 컬럼 처리~~ | ~~A: 롤백용 유지~~ | **확정: A** (롤백 안전성 확보, BM25 안정화 후 추후 제거) |
| D-3 | CONCURRENTLY 지원 여부 | 실행 후 확인 | 시도 → fallback |

---

## 구현 규칙

- **레이어 단위 커밋**: 각 레이어 구현이 끝나면 반드시 중단하고 사용자에게 커밋 여부를 확인한 뒤, 다음 레이어로 진행한다.
- 레이어 내부 작업은 순차적으로 진행하되, 레이어 경계에서는 반드시 끊는다.

---

## 버전 이력

| 버전 | 날짜 | 변경 내용 |
|------|------|----------|
| v1 | 2026-03-01 | 초안 작성 (13개 수정, 2개 신규, 1개 롤백용 유지) |
| v2 | 2026-03-01 | fts_builder/tsvector_builder 삭제 반영 (11개 수정, 2개 삭제) |
| v3 | 2026-03-01 | BMW 최적화 상세 반영 (레이어 4: 11개 작업), 병렬 인덱스 빌드 설정 추가 (레이어 6: 3단계), CONCURRENTLY 시도→fallback 전략 |
| v4 | 2026-03-01 | 코드 교차 검증 반영: (1) 누락 파일 5개 추가 (cli.py, __init__.py, shared.py docstring, test_ingest_pipeline.py, verify_db), (2) pg_textsearch C 확장 설치 구체화 (Dockerfile.postgres 필수), (3) Feature Flag 관계 명시 (USE_BM25_SEARCH vs USE_HYBRID_SEARCH), (4) keyword_search.py 보존/삭제 함수 구분표 추가, (5) 삭제 파일 참조 정리 섹션 신설 (3-4), (6) CLAUDE.md 3곳 + ingest.md 문서 갱신 추가 |
| v5 | 2026-03-01 | 레이어 6-1 수정: `--step db` (JSON 전체 재적재) → `--step fts` (ORM에서 search_text만 재빌드). fts_index 테이블은 이미 579K행 존재, search_text 컬럼만 NULL이므로 `search_text_rebuilder.py` 경로가 적합 |
