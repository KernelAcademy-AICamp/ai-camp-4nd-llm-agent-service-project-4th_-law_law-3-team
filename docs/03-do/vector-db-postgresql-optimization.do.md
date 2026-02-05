# Do: Vector DB + PostgreSQL 최적화

> Plan: `docs/01-plan/features/vector-db-postgresql-optimization.plan.md`
> Design: `docs/02-design/features/vector-db-postgresql-optimization.design.md`

---

## 현재 상태 (구현 시작점)

| 구성 요소 | 현재 상태 |
|----------|----------|
| Models | 7개 활성 (LawDocument, PrecedentDocument, Lawyer, TrialStatistics 등) |
| Alembic | 005번까지 완료 |
| Feature Flags | `USE_DB_LAWYERS` 1개만 존재 |
| DB Services | `lawyer_db_service.py`, `lawyer_stats_db_service.py` 2개 |
| RAG | retrieval, rerank, pipeline, query_rewrite, embedding |
| 한국어 NLP | 미설치 (mecab 없음) |

---

## 구현 체크리스트

### Phase 1: PostgreSQL 통합 + Tier 1 데이터 적재

#### 1-1. ORM 모델 5개 생성

| # | 파일 | 모델 | 소스 데이터 | 상태 |
|---|------|------|-----------|------|
| 1 | `backend/app/models/legislation_interpretation.py` | LegislationInterpretation | legislation_full.json (8,597건) | [ ] |
| 2 | `backend/app/models/legal_term.py` | LegalTerm | lawterms_full.json (37,169건) | [ ] |
| 3 | `backend/app/models/constitutional_decision.py` | ConstitutionalDecision | constitutional_full.json (36,781건) | [ ] |
| 4 | `backend/app/models/treaty.py` | Treaty | treaty-full.json (3,589건) | [ ] |
| 5 | `backend/app/models/law_amendment.py` | LawAmendment | lsStmd-full.json (1,683건) | [ ] |

**참고 패턴**: `backend/app/models/trial_statistics.py` (가장 최근 모델)

**동기화 필수**:
- `backend/app/models/__init__.py`에 신규 모델 import/export 추가
- `backend/alembic/env.py`에 신규 모델 import 추가

#### 1-2. Alembic 마이그레이션 5개

| # | 파일 | 내용 | 상태 |
|---|------|------|------|
| 1 | `006_add_legislation_interpretations.py` | 법령해석례 테이블 + 인덱스 | [ ] |
| 2 | `007_add_legal_terms.py` | 법률용어 테이블 + FTS 인덱스 (pg_trgm, GIN) | [ ] |
| 3 | `008_add_constitutional_decisions.py` | 헌재결정 테이블 + 인덱스 | [ ] |
| 4 | `009_add_treaties.py` | 조약 테이블 + 인덱스 | [ ] |
| 5 | `010_add_law_amendments.py` | 법령 제개정 연혁 테이블 + 인덱스 | [ ] |

**참고 패턴**: `backend/alembic/versions/005_add_trial_statistics_table.py`

**주의사항**:
- `007`에서 `pg_trgm` 확장 활성화 필요: `CREATE EXTENSION IF NOT EXISTS pg_trgm`
- `legal_terms` 테이블에 `definition_tsv tsvector` 컬럼 포함
- 각 마이그레이션의 `downgrade()`에서 인덱스 → 테이블 순서로 삭제

**실행**:
```bash
cd backend
uv run alembic upgrade head
```

#### 1-3. 데이터 로딩 스크립트

| 파일 | 내용 | 상태 |
|------|------|------|
| `backend/scripts/load_law_data.py` | 통합 데이터 로더 | [ ] |

**참고 패턴**: `backend/scripts/load_lawyers_data.py`

**구현 요구사항**:
- `--type` 플래그: `all`, `interpretations`, `terms`, `constitutional`, `treaties`, `amendments`
- `--verify` 플래그: 건수/무결성 검증
- `--reset` 플래그: 기존 데이터 삭제 후 재적재
- 배치 크기 1,000건, `ON CONFLICT DO UPDATE` 멱등성
- tqdm 진행바 + 배치별 로그
- HTML 태그 제거, 텍스트 정규화 포함

**실행**:
```bash
cd backend
uv run python scripts/load_law_data.py --type all
uv run python scripts/load_law_data.py --verify
```

#### 1-4. Feature Flag 추가

| 파일 | 변경 내용 | 상태 |
|------|----------|------|
| `backend/app/core/config.py` | 5개 Feature Flag 추가 | [ ] |

```python
# 추가할 플래그 (기본값 False)
USE_DB_LAWS: bool = False
USE_DB_PRECEDENTS: bool = False
USE_DB_INTERPRETATIONS: bool = False
USE_DB_TERMS: bool = False
USE_DB_CONSTITUTIONAL: bool = False
USE_DB_TREATIES: bool = False
```

#### 1-5. DB 서비스 함수 작성

| # | 파일 | 역할 | 상태 |
|---|------|------|------|
| 1 | `interpretation_db_service.py` | 법령해석례 조회/검색 | [ ] |
| 2 | `legal_terms_db_service.py` | 법률용어 조회/FTS 검색 | [ ] |
| 3 | `constitutional_db_service.py` | 헌재결정 조회/검색 | [ ] |
| 4 | `treaty_db_service.py` | 조약 조회/검색 | [ ] |
| 5 | `law_db_service.py` | 법령 PostgreSQL 조회 (기존→DB 전환) | [ ] |
| 6 | `precedent_db_service.py` | 판례 PostgreSQL 조회 (기존→DB 전환) | [ ] |

**참고 패턴**: `backend/app/services/service_function/lawyer_db_service.py`

**위치**: `backend/app/services/service_function/`

**함수 시그니처 규칙**:
```python
async def function_name_db(db: AsyncSession, ...) -> ReturnType:
```

#### 1-6. RAG retrieval.py에 PostgreSQL 원본 조회 통합

| 파일 | 변경 내용 | 상태 |
|------|----------|------|
| `backend/app/services/rag/retrieval.py` | `get_source_details()` 함수 추가 | [ ] |

`data_type`에 따라 적절한 PostgreSQL 테이블에서 원본을 조회하는 통합 함수.

---

### Phase 2: 신규 데이터 임베딩

> **전제조건**: Phase 1 완료 (PostgreSQL에 데이터 적재됨)
> **GPU 환경 필요**: RunPod 등 GPU 서버에서 실행

#### 2-1. 임베딩 스크립트 작성

| 파일 | 내용 | 상태 |
|------|------|------|
| `backend/scripts/embed_law_data.py` | 통합 임베딩 생성기 | [ ] |

**참고 패턴**: `backend/scripts/runpod_lancedb_embeddings.py`

**구현 요구사항**:
- `--type` 플래그: `all`, `interpretations`, `constitutional`, `treaties`, `amendments`
- `--stats` 플래그: 임베딩 현황 통계
- `StreamingEmbeddingProcessor` 상속 또는 동일 패턴 사용
- 데이터 유형별 Processor 클래스 (4개, 법률용어 제외)
- 배치 100건, gc.collect() 포함
- LanceDB `legal_chunks` 테이블에 `data_type` 필드로 구분

**임베딩 대상** (법률용어 제외):

| 데이터 | 임베딩 텍스트 | 예상 청크 |
|--------|-------------|----------|
| 해석례 | `[{안건명}] 질의: {질의요지}\n회답: {회답}\n이유: {이유}` | ~15,000 |
| 헌재결정 | `[{사건번호}] {결정요지}\n{이유}` | ~50,000 |
| 조약 | `[{조약명}] {내용}` | ~10,000 |
| 제개정이유 | `[{법령명} {제개정구분}] {이유}` | ~1,683 |

**실행** (GPU 환경):
```bash
cd backend
uv run --no-sync python scripts/embed_law_data.py --type all
uv run --no-sync python scripts/embed_law_data.py --stats
```

#### 2-2. 검색 파이프라인 data_type 필터 확장

| 파일 | 변경 내용 | 상태 |
|------|----------|------|
| `backend/app/services/rag/retrieval.py` | data_type 필터에 해석례/헌재결정/조약/제개정이유 추가 | [ ] |
| `backend/app/services/rag/pipeline.py` | 신규 data_type 통합 | [ ] |

#### 2-3. 검색 품질 검증

| 작업 | 내용 | 상태 |
|------|------|------|
| 평가 데이터셋 보완 | 해석례/헌재결정 관련 평가 질의 추가 | [ ] |
| evaluation runner 실행 | Recall@10, MRR, Hit Rate 측정 | [ ] |

**실행**:
```bash
cd backend
uv run python -m evaluation.runners.evaluation_runner \
    --dataset evaluation/datasets/eval_dataset_v1.json
```

---

### Phase 3: Neo4j 그래프 확장

> **전제조건**: Phase 1 완료

#### 3-1. 그래프 확장 스크립트

| 파일 | 내용 | 상태 |
|------|------|------|
| `backend/scripts/build_graph_extended.py` | Neo4j 그래프 확장 | [ ] |

**참고 패턴**: `backend/scripts/build_graph.py`

**신규 노드**: Interpretation (8,597), Treaty (3,589), ConstitutionalDecision (36,781)
**신규 관계**: INTERPRETS, REVIEWS, IMPLEMENTS, AMENDED_BY

**실행**:
```bash
cd backend
uv run python scripts/build_graph_extended.py
uv run python scripts/verify_graph.py
```

---

### Phase 4: 하이브리드 검색 (FTS + mecab)

> **전제조건**: Phase 1, 2 완료

#### 4-1. mecab 의존성 추가

| 파일 | 변경 내용 | 상태 |
|------|----------|------|
| `backend/pyproject.toml` | `mecab-python3` 의존성 추가 | [ ] |

```bash
cd backend
uv add mecab-python3
```

#### 4-2. mecab 토큰화 유틸리티

| 파일 | 내용 | 상태 |
|------|------|------|
| `backend/app/services/rag/tokenizer.py` | mecab 형태소 분석 + 토큰화 | [ ] |

**핵심 함수**:
```python
def tokenize_korean(text: str) -> str:
    """텍스트 → mecab 형태소 분석 → 조사/어미 제거 → 공백 구분 토큰 문자열"""
```

#### 4-3. FTS 마이그레이션

| 파일 | 내용 | 상태 |
|------|------|------|
| `backend/alembic/versions/011_add_fts_indices.py` | tsvector 컬럼 + GIN 인덱스 | [ ] |

**대상 테이블**: law_documents, precedent_documents, legislation_interpretations, constitutional_decisions, legal_terms

#### 4-4. tsvector 생성 스크립트

| 파일 | 내용 | 상태 |
|------|------|------|
| `backend/scripts/generate_tsvectors.py` | 기존 데이터 일괄 mecab 토큰화 → tsvector 갱신 | [ ] |

```bash
cd backend
uv run python scripts/generate_tsvectors.py --type all
```

#### 4-5. FTS 검색 서비스

| 파일 | 내용 | 상태 |
|------|------|------|
| `backend/app/services/rag/fts_search.py` | FTS 검색 (mecab 토큰화 → tsquery) | [ ] |

#### 4-6. 법률용어 FTS 전용 검색

| 파일 | 내용 | 상태 |
|------|------|------|
| `backend/app/services/service_function/legal_terms_db_service.py` | 3단계 검색 | [ ] |

```
1순위: term_name 정확 매칭 (WHERE term_name = ?)
2순위: definition_tsv FTS 검색 (WHERE definition_tsv @@ ?)
3순위: term_name pg_trgm 유사 매칭 (ORDER BY similarity())
```

#### 4-7. RRF 병합

| 파일 | 내용 | 상태 |
|------|------|------|
| `backend/app/services/rag/hybrid_search.py` | Reciprocal Rank Fusion | [ ] |

벡터 0.7 + FTS 0.3 가중치, 법률용어 컨텍스트 보강 포함.

---

## 의존성 설치

```bash
cd backend

# 기존 의존성 동기화
uv sync --dev

# Phase 4에서 추가
uv add mecab-python3
```

---

## 검증 명령어

```bash
cd backend

# Phase 1 검증
uv run alembic upgrade head                           # 마이그레이션 실행
uv run python scripts/load_law_data.py --type all     # 데이터 적재
uv run python scripts/load_law_data.py --verify       # 적재 검증

# Phase 2 검증 (GPU 환경)
uv run --no-sync python scripts/embed_law_data.py --type all
uv run --no-sync python scripts/embed_law_data.py --stats

# Phase 3 검증
uv run python scripts/build_graph_extended.py
uv run python scripts/verify_graph.py

# Phase 4 검증
uv run python scripts/generate_tsvectors.py --type all

# 정적 검증 (모든 Phase)
uv run ruff check backend/app/
npm run build  # frontend/ (API 변경 시)
```

---

## 구현 순서 요약

```
Phase 1 (PostgreSQL 통합)
  1-1 ORM 모델 5개
   → 1-2 Alembic 마이그레이션 5개
    → 1-3 데이터 로딩 스크립트
     → 1-4 Feature Flag
      → 1-5 DB 서비스 함수 6개
       → 1-6 RAG retrieval 통합

Phase 2 (임베딩) — Phase 1 완료 후, GPU 환경 필요
  2-1 임베딩 스크립트
   → 2-2 파이프라인 data_type 확장
    → 2-3 검색 품질 검증

Phase 3 (Neo4j) — Phase 1 완료 후
  3-1 그래프 확장 스크립트

Phase 4 (하이브리드 검색) — Phase 1, 2 완료 후
  4-1 mecab 의존성
   → 4-2 토큰화 유틸리티
    → 4-3 FTS 마이그레이션
     → 4-4 tsvector 생성
      → 4-5 FTS 검색 서비스
       → 4-6 법률용어 FTS 전용
        → 4-7 RRF 병합
```

---

## 문서 업데이트 대상

구현 완료 후 아래 문서 갱신 필요:

| 문서 | 갱신 내용 |
|------|----------|
| `CLAUDE.md` (루트) | 신규 테이블, 스크립트, Feature Flag |
| `backend/CLAUDE.md` | 신규 모델/서비스 반영 |
| `docs/vectordb_design.md` | 임베딩 스키마 확장 |
| `docs/DB_ARCHITECTURE.md` | 신규 테이블 구조 |
