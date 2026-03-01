# pg-textsearch-bm25-migration Gap Analysis

> **요약**: Plan 문서 65개 항목 중 62.5개 달성. Match Rate **95%**. 미구현은 RAG 평가(임베딩 모델 런타임 필요) 1건, 부분완료 2건.
>
> **분석일**: 2026-03-01
> **Plan 문서**: `docs/01-plan/features/pg-textsearch-bm25-migration.plan.md` (v5)
> **Design 문서**: 없음 (Plan 문서 기준으로 분석)

---

## Summary

| 항목 | 값 |
|------|---|
| Match Rate | **95%** |
| 총 항목 수 | 65개 |
| 완전 구현 | 61개 |
| 부분 구현 | 3개 |
| 미구현 | 1개 |

## 전체 점수

| 카테고리 | 점수 | 상태 |
|----------|:----:|:----:|
| L1: 인프라 | 100% | 완료 |
| L2: 스키마 | 100% | 완료 |
| L3: 인제스트 파이프라인 | 98% | 완료 |
| L4: 검색 코드 | 100% | 완료 |
| L5: 연동 코드 | 100% | 완료 |
| L6: 인덱스 생성 | 100% | 완료 |
| L7: 검증 + 문서 | 75% | 주의 |
| **전체** | **95%** | **완료** |

---

## Layer-by-Layer Analysis

### Layer 1: 인프라 (Docker + PostgreSQL) — 100%

| # | 항목 | 상태 | 비고 |
|---|------|:----:|------|
| 1-1 | PG 15→17 업그레이드 | 완료 | `docker/postgres/Dockerfile:3` — `FROM postgres:17-alpine` |
| 1-2 | 데이터 마이그레이션 | 완료 | Plan v5에서 실행 완료 기록 |
| 1-3 | pg_textsearch 설치 (소스 빌드) | 완료 | `docker/postgres/Dockerfile` — git clone v0.5.1 + make + make install |
| 1-4 | shared_preload_libraries 설정 | 완료 | `docker-compose.yml:18` — `command: postgres -c shared_preload_libraries=pg_textsearch` |
| 1-5 | 확장 활성화 | 완료 | `docker/postgres/init.sql:7` — `CREATE EXTENSION IF NOT EXISTS "pg_textsearch"` |
| 1-6 | 한국어 text_config=simple | 완료 | `scripts/create_bm25_index.py:40` — `TEXT_CONFIG = "simple"` |

### Layer 2: 스키마 (ORM + Alembic) — 100%

| # | 항목 | 상태 | 비고 |
|---|------|:----:|------|
| 2-1 | fts_index에 search_text 컬럼 추가 | 완료 | `app/models/fts_index.py:73` — `search_text = Column(Text, nullable=True)` |
| 2-2 | GIN 인덱스 제거, BM25 인덱스 raw SQL | 완료 | ORM에 GIN 없음, BM25는 Alembic raw SQL |
| 2-3 | Alembic 마이그레이션 작성 | 완료 | `alembic/versions/020_add_bm25_search_text.py` |

### Layer 3: 인제스트 파이프라인 — 98%

| # | 항목 | 상태 | 비고 |
|---|------|:----:|------|
| 3-1 | db_writer: search_text 저장 | 완료 | `db_writer.py:198-201` — `" ".join(tokens)` |
| 3-2 | verify_db 검증 대상 변경 | 완료 | `db_writer.py:273-278` — `fts_with_search_text` |
| 3-3 | shared.py upsert 컬럼 업데이트 | 완료 | `shared.py:49-56` — search_text upsert |
| 3-4 | cli.py fts 스텝 변경 | 완료 | `cli.py:189-193` — `run_search_text_rebuild` |
| 3-5~3-24 | 20개 타입 fulltext_fn | 완료 | 전수 확인, Plan 명세와 일치 |
| 3-25~3-28 | 삭제 파일 참조 정리 | 완료 | fts_builder/tsvector_builder import 없음 |
| 3-29 | ingest.md fts_builder 5곳 갱신 | **부분완료** | line 402에 과거 fts_builder 비교 히스토리 잔류 |
| - | fts_builder.py 삭제 | 완료 | 파일 없음 확인 |
| - | tsvector_builder.py 삭제 | 완료 | 파일 없음 확인 |

### Layer 4: 검색 코드 (BMW 최적화) — 100%

| # | 항목 | 상태 | 비고 |
|---|------|:----:|------|
| 4-1 | keyword_search.py 리라이트 | 완료 | 불필요 함수 전부 제거, 보존 4개 유지 |
| 4-2 | BM25 쿼리 — BMW 트리거 패턴 | 완료 | `ORDER BY score_expr.asc().limit(n)` |
| 4-3 | to_bm25query() 인덱스명 명시 | 완료 | `func.to_bm25query(search_query, _BM25_INDEX_NAME)` |
| 4-4 | 수동 정규화 제거 | 완료 | `abs(float(row.rank))` 직접 사용 |
| 4-5 | score_type → "bm25" | 완료 | `"score_type": "bm25"` |
| 4-6 | search_by_keyword 시그니처 변경 | 완료 | precomputed tsq 파라미터 제거 |
| 4-7 | pipeline.py import 정리 | 완료 | tsquery 함수 import 없음 |
| 4-8 | _PrecomputedInputs 정리 | 완료 | embeddings 필드만 존재 |
| 4-9 | _precompute_inputs 간소화 | 완료 | 임베딩 계산만 수행 |
| 4-10 | retrieval.py 호출 정리 | 완료 | tsquery 파라미터 없음 |
| 4-11 | _search_and_deduplicate 정리 | 완료 | embeddings만 전달 |

### Layer 5: 연동 코드 — 100%

| # | 항목 | 상태 | 비고 |
|---|------|:----:|------|
| 5-1 | retrieval.py 파라미터 정리 | 완료 | tsquery 파라미터 없음 |
| 5-2 | config.py Feature Flag | 완료 | `USE_BM25_SEARCH: bool = True` |
| 5-3 | is_fts_available BM25 호환 | 완료 | `idx_fts_bm25` pg_indexes 조회 |

### Layer 6: search_text 채우기 + BM25 인덱스 생성 — 100%

| # | 항목 | 상태 | 비고 |
|---|------|:----:|------|
| 6-1 | fts_index 425K건 search_text 채우기 | 완료 | 425,209건 적재 완료 |
| 6-2 | 병렬 빌드 설정 | 완료 | workers=4, 256MB |
| 6-3 | BM25 인덱스 생성 | 완료 | `USING bm25(search_text) WITH (text_config='simple')` |

### Layer 7: 검증 + 문서 — 75%

| # | 항목 | 상태 | 비고 |
|---|------|:----:|------|
| 7-1 | 성능 벤치마크 | 완료 | Seq Scan 15s → Index Scan 21ms (~700x 개선) |
| 7-2 | RAG 평가 | **미완료** | 임베딩 모델 런타임 필요 |
| 7-3 | 21개 타입 전수 테스트 | **부분완료** | 10개 data_type 완료, 11개 미확인 |
| 7-4 | 아키텍처 문서 업데이트 | 완료 | BM25+BMW 전면 개정 |
| 7-5 | 루트 CLAUDE.md 갱신 | 완료 | PG 17, BM25, USE_BM25_SEARCH 반영 |
| 7-6 | scripts/CLAUDE.md 갱신 | 완료 | fts_builder 참조 제거 |
| 7-7 | wsl2-docker.md 갱신 | 완료 | PG 17 + pg_textsearch, shm_size 반영 |
| 7-8 | ingest.md 갱신 | **부분완료** | line 402 과거 fts_builder 비교 잔류 |

---

## Gap List

### 미구현

| 항목 | Plan 위치 | 설명 | 우선순위 |
|------|-----------|------|---------|
| RAG 평가 | 7-2 | Recall@10, MRR, Hit Rate 미측정. 임베딩 모델(`nlpai-lab/KURE-v1`) 런타임 환경 필요. Plan v5에서도 "미실행"으로 명시 | 낮음 (운영 검증 단계) |

### 부분 완료

| 항목 | Plan 위치 | 구현 상태 | 잔여 작업 |
|------|-----------|----------|----------|
| ingest.md fts_builder 갱신 | 3-29 | 상단 BM25 기준 갱신. line 402에 "이전 fts_builder.py에는 없었음" 히스토리 잔류 | line 402 표현 삭제 또는 과거형 명시 |
| 21개 타입 전수 테스트 | 7-3 | 10개 data_type 테스트 완료. 11개 미확인 | admin_rule 등 11개 타입 수동 테스트 |

### 아키텍처 주의사항 (구현상 차이, 동작 영향 없음)

| 항목 | Plan 명세 | 실제 구현 | 설명 |
|------|-----------|----------|------|
| to_bm25query() 인자 순서 | `to_bm25query('idx_fts_bm25', query_text)` | `func.to_bm25query(search_query, _BM25_INDEX_NAME)` | Plan v5 버그 수정. 실제 pg_textsearch API에 맞게 수정됨. Plan SQL 예시 업데이트 권장 |
| ORDER BY 방향 | `ORDER BY DESC` | `ORDER BY ASC` | `<@>` 연산자가 음수 BM25 점수를 반환하므로 ASC가 관련성 높은 순. Plan v5에서 수정됨 |
| content_tsvector | 롤백용 유지 | 유지됨 | BM25 안정화 후 제거 시점 결정 필요 |

---

## Recommendations

### 즉시 조치 (선택적)

1. `backend/scripts/ingest/ingest.md` line 402 과거 fts_builder 비교 표현 정리
2. 잔여 11개 data_type BM25 검색 수동 테스트

### 추후 조치

3. 임베딩 모델 런타임 준비 후 RAG 평가 실행 (Recall@10 >= 0.8, MRR >= 0.7, Hit Rate >= 0.9)
4. Plan 문서의 `to_bm25query()` 인자 순서 및 `ORDER BY` 방향 SQL 예시 업데이트
5. `content_tsvector` 컬럼 제거 시점 결정

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2026-03-01 | 초안 작성 — Gap 분석 실행, Match Rate 95% |
