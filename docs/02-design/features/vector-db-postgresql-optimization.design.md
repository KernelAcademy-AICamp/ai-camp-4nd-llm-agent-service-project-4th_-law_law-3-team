# Design: Vector DB + PostgreSQL 최적화

> Plan 참조: `docs/01-plan/features/vector-db-postgresql-optimization.plan.md`

---

## 1. 데이터 카탈로그

### 1.1 현재 적재 완료 데이터

| 데이터 | 건수 | LanceDB | PostgreSQL | Neo4j |
|--------|------|---------|------------|-------|
| 법령 (law_cleaned.json) | 5,841 | 118,922 청크 | law_documents 테이블 | Statute 노드 5,572 |
| 판례 (precedents_cleaned.json) | 65,107 | 134,846 청크 | precedent_documents 테이블 | Case 노드 65,107 |
| 변호사 (lawyers_with_coords.json) | 17,326 | - | lawyers 테이블 | - |
| 재판통계 | - | - | trial_statistics 테이블 | - |
| 법령약칭 (lsAbrv.json) | 2,642 | - | - | Alias 노드 69 |

### 1.2 신규 적재 대상 (`data/law_data/`)

68개 파일, 총 ~4.8GB, ~70만건을 **3개 티어**로 분류:

#### Tier 1: 핵심 (즉시 적재)

| 파일 | 건수 | 크기 | 유형 | 적재 대상 |
|------|------|------|------|----------|
| **legislation_full.json** | 8,597 | 76MB | 법령해석례 | PostgreSQL + LanceDB 임베딩 + Neo4j |
| **lawterms_full.json** | 37,169 | 19MB | 법률용어사전 | **PostgreSQL + FTS 전용** (임베딩 제외) |
| **constitutional_full.json** | 36,781 | 23MB | 헌법재판소 결정 | PostgreSQL + LanceDB 임베딩 + Neo4j |
| **treaty-full.json** | 3,589 | 57MB | 조약 | PostgreSQL + LanceDB 임베딩 + Neo4j |
| **lsStmd-full.json** | 1,683 | 141MB | 법령 제개정 연혁 | PostgreSQL + LanceDB 임베딩 |

**Tier 1 합계**: ~87,819건, ~316MB

> **법률용어 임베딩 제외 근거**: 정의 텍스트 평균 69자(중앙값 49자)로 너무 짧아
> 1024차원 벡터에 충분한 의미를 담기 어려움. 용어명 직접 조회(ILIKE),
> 정의 내 키워드 검색(FTS), 유사 용어 매칭(pg_trgm)으로 99% 사용 패턴 커버 가능.
> PostgreSQL FTS가 벡터 검색보다 정확하고 빠름.

#### Tier 2: 확장 (MVP 이후)

| 파일 | 건수 | 크기 | 유형 | 임베딩 대상 텍스트 |
|------|------|------|------|------------------|
| **administration_full.json** | 34,258 | 424MB | 법제처 법령해석례 | 질의요지 + 회답 + 이유 |
| **ftc-full.json** | 8,029 | 165MB | 공정거래위원회 심결 | 의결내용 |
| **nlrc-full.json** | 41,034 | 64MB | 노동위원회 조정 | 판정요지 |
| **nhrck-full.json** | 3,732 | 114MB | 국가인권위원회 | 판단요지 + 이유 |
| **ppc-full.json** | 3,889 | 21MB | 개인정보보호위원회 | 주문 + 이유 |
| **CgmExpc 상위 5개** | ~25,900 | ~59MB | 부처별 법령해석예규 | 질의요지 + 회답 |

**Tier 2 합계**: ~116,842건, ~847MB

#### Tier 3: 선택적 (특화 서비스)

| 파일 | 건수 | 크기 | 비고 |
|------|------|------|------|
| **ttSpecialDecc-full-002.json** | 137,801 | 2.2GB | 조세심판 (전문분야) |
| **local_rules-full-003.json** | 158,191 | 2.0GB | 자치법규 (지역별 서비스) |
| **administrative_rules_full.json** | 21,889 | 312MB | 행정규칙 (서식 위주) |
| **kmstSpecialDecc-full.json** | 13,848 | 151MB | 해양안전심판 |
| **school-full.json** | 5,259 | 73MB | 학교규칙 |
| 기타 위원회/예규 | ~20,000 | ~100MB | 소규모 데이터 |

**Tier 3 합계**: ~356,988건, ~4.8GB

---

## 2. DB별 적재 전략

### 2.1 PostgreSQL 스키마 설계

#### 신규 테이블 (Tier 1)

```sql
-- 법령해석례 (legislation_full.json)
CREATE TABLE legislation_interpretations (
    id SERIAL PRIMARY KEY,
    serial_number VARCHAR(20) UNIQUE NOT NULL,  -- 법령해석례일련번호
    title VARCHAR(500) NOT NULL,                -- 안건명
    interpretation_date DATE,                    -- 해석일자
    interpreting_agency VARCHAR(100),            -- 해석기관명
    requesting_agency VARCHAR(100),              -- 질의기관명
    related_laws TEXT[],                         -- 관계법령 (ARRAY)
    question_summary TEXT,                       -- 질의요지
    answer TEXT,                                 -- 회답
    reason TEXT,                                 -- 이유
    raw_data JSONB                               -- 원본 전체
);

CREATE INDEX idx_interp_date ON legislation_interpretations(interpretation_date);
CREATE INDEX idx_interp_agency ON legislation_interpretations(interpreting_agency);
CREATE INDEX idx_interp_related_laws ON legislation_interpretations USING GIN(related_laws);

-- 법률용어사전 (lawterms_full.json) — PostgreSQL + FTS 전용 (임베딩 제외)
CREATE TABLE legal_terms (
    id SERIAL PRIMARY KEY,
    term_code VARCHAR(20) UNIQUE,          -- 법령용어코드
    term_name VARCHAR(200) NOT NULL,       -- 법령용어명
    definition TEXT NOT NULL,              -- 법령용어정의
    definition_tsv tsvector,              -- FTS용 (mecab 토큰화 결과)
    source VARCHAR(200),                   -- 출처 법령명
    raw_data JSONB
);

CREATE INDEX idx_term_name ON legal_terms(term_name);
CREATE INDEX idx_term_name_trgm ON legal_terms USING GIN(term_name gin_trgm_ops);  -- 유사 용어 검색
CREATE INDEX idx_term_def_fts ON legal_terms USING GIN(definition_tsv);            -- 정의 FTS 검색

-- 헌법재판소 결정 (constitutional_full.json)
CREATE TABLE constitutional_decisions (
    id SERIAL PRIMARY KEY,
    serial_number VARCHAR(20) UNIQUE NOT NULL,
    case_number VARCHAR(100),              -- 사건번호
    case_name VARCHAR(500),                -- 사건명
    decision_date DATE,                    -- 선고일
    decision_type VARCHAR(50),             -- 결정유형 (위헌/합헌/각하 등)
    summary TEXT,                          -- 결정요지
    reason TEXT,                           -- 이유
    raw_data JSONB
);

CREATE INDEX idx_const_date ON constitutional_decisions(decision_date);
CREATE INDEX idx_const_type ON constitutional_decisions(decision_type);

-- 조약 (treaty-full.json)
CREATE TABLE treaties (
    id SERIAL PRIMARY KEY,
    treaty_id VARCHAR(20) UNIQUE NOT NULL,  -- 조약일련번호
    treaty_name VARCHAR(500) NOT NULL,      -- 조약명
    signing_date DATE,                      -- 서명일자
    effective_date DATE,                    -- 발효일자
    countries TEXT[],                       -- 체결대상국가
    ratification BOOLEAN,                   -- 국회비준동의여부
    content TEXT,                           -- 조약내용
    raw_data JSONB
);

CREATE INDEX idx_treaty_date ON treaties(effective_date);
CREATE INDEX idx_treaty_countries ON treaties USING GIN(countries);

-- 법령 제개정 연혁 (lsStmd-full.json)
CREATE TABLE law_amendments (
    id SERIAL PRIMARY KEY,
    law_id VARCHAR(20) NOT NULL,           -- 법령일련번호
    amendment_type VARCHAR(50),             -- 제개정구분 (제정/개정/폐지 등)
    promulgation_date DATE,                -- 공포일자
    enforcement_date DATE,                 -- 시행일자
    reason TEXT,                           -- 제개정이유
    raw_data JSONB,
    CONSTRAINT fk_law FOREIGN KEY (law_id)
        REFERENCES law_documents(law_id) ON DELETE SET NULL
);

CREATE INDEX idx_amendment_law ON law_amendments(law_id);
CREATE INDEX idx_amendment_date ON law_amendments(promulgation_date);
```

#### Feature Flag 설계

```python
# backend/app/core/config.py 추가
USE_DB_LAWS: bool = False            # 기존 법령 (이미 테이블 있음)
USE_DB_PRECEDENTS: bool = False      # 기존 판례 (이미 테이블 있음)
USE_DB_INTERPRETATIONS: bool = False  # 법령해석례
USE_DB_TERMS: bool = False           # 법률용어
USE_DB_CONSTITUTIONAL: bool = False  # 헌법재판소
USE_DB_TREATIES: bool = False        # 조약
```

#### Alembic 마이그레이션 계획

| 마이그레이션 | 내용 |
|------------|------|
| 006_add_legislation_interpretations.py | 법령해석례 테이블 |
| 007_add_legal_terms.py | 법률용어사전 테이블 |
| 008_add_constitutional_decisions.py | 헌법재판소 결정 테이블 |
| 009_add_treaties.py | 조약 테이블 |
| 010_add_law_amendments.py | 법령 제개정 연혁 테이블 |
| 011_add_fts_indices.py | FTS tsvector 컬럼 + GIN 인덱스 (모든 텍스트 테이블) |

### 2.2 LanceDB 임베딩 설계

#### 테이블 구조 확장

기존 `legal_chunks` 테이블의 `data_type` 필드를 확장:

| data_type 값 | 데이터 소스 | 임베딩 텍스트 |
|-------------|-----------|-------------|
| `법령` | law_cleaned.json | 조문 내용 (현행 유지) |
| `판례` | precedents_cleaned.json | 판시사항 + 판결요지 (현행 유지) |
| `해석례` | legislation_full.json | `[{안건명}] 질의: {질의요지}\n회답: {회답}\n이유: {이유}` |
| `헌재결정` | constitutional_full.json | `[{사건번호}] {결정요지}\n{이유}` |
| `조약` | treaty-full.json | `[{조약명}] {내용}` |
| `제개정이유` | lsStmd-full.json | `[{법령명} {제개정구분}] {이유}` |

> **참고**: 법률용어(`lawterms_full.json`)는 임베딩하지 않음.
> PostgreSQL FTS로 검색하며, RAG 파이프라인에서는 FTS 결과를 컨텍스트로 병합.

#### 청킹 전략 (데이터 유형별)

| 데이터 유형 | 청킹 방식 | max_chars | overlap | 예상 청크 수 |
|-----------|----------|-----------|---------|------------|
| 법령 | 조문 단위 (현행) | 800 토큰 | - | 118,922 (기존) |
| 판례 | 문자 기반 (현행) | 1,250자 | 125자 (10%) | 134,846 (기존) |
| 해석례 | 질의+회답+이유 통합 | 1,500자 | 150자 (10%) | ~15,000 |
| 헌재결정 | 문자 기반 | 1,500자 | 150자 | ~50,000 |
| 조약 | 문자 기반 | 1,500자 | 150자 | ~10,000 |
| 제개정이유 | 단일 청크 | - | - | ~1,683 |

**임베딩 총 예상**: 기존 253,768 + 신규 ~76,683 = **~330,451 청크**

> 법률용어 37,169건은 임베딩 제외 → 신규 청크 수 약 33% 감소

#### 추가 메타데이터 컬럼

```
기존 20컬럼 + 아래 컬럼 추가:

- interpreting_agency (VARCHAR) -- 해석기관명 (해석례용)
- decision_type (VARCHAR)       -- 결정유형 (헌재용)
- treaty_countries (VARCHAR)    -- 체결국 (조약용)
```

### 2.3 Neo4j 그래프 확장

#### 신규 노드

| Label | 소스 | 예상 수 | 속성 |
|-------|------|---------|------|
| Interpretation | legislation_full.json | 8,597 | serial_number, title, date |
| Treaty | treaty-full.json | 3,589 | treaty_id, name, effective_date |
| ConstitutionalDecision | constitutional_full.json | 36,781 | serial_number, case_number, decision_type |

#### 신규 관계

| 관계 | 설명 | 소스 필드 |
|------|------|----------|
| (Interpretation)-[:INTERPRETS]->(Statute) | 해석례→법령 참조 | related_laws |
| (ConstitutionalDecision)-[:REVIEWS]->(Statute) | 헌재결정→법령 위헌심사 | 관련법령 |
| (Treaty)-[:IMPLEMENTS]->(Statute) | 조약→국내법 연계 | 관련법령 |
| (Statute)-[:AMENDED_BY]->(Statute) | 법령 개정 관계 | lsStmd 제개정 |

#### 확장 후 그래프 규모 예상

| 항목 | 현재 | 확장 후 |
|------|------|---------|
| 노드 수 | ~70,748 | ~119,715 |
| 관계 수 | ~163,854 | ~220,000+ |

---

## 3. 데이터 로딩 파이프라인

### 3.1 통합 로딩 스크립트 설계

```
backend/scripts/
├── load_law_data.py              # 통합 데이터 로더 (신규)
│   ├── load_interpretations()     # 법령해석례 → PostgreSQL
│   ├── load_legal_terms()         # 법률용어 → PostgreSQL
│   ├── load_constitutional()      # 헌재결정 → PostgreSQL
│   ├── load_treaties()            # 조약 → PostgreSQL
│   └── load_amendments()          # 제개정연혁 → PostgreSQL
├── embed_law_data.py             # 통합 임베딩 생성기 (신규)
│   ├── EmbedInterpretations       # 해석례 임베딩
│   ├── EmbedConstitutional        # 헌재결정 임베딩
│   ├── EmbedTreaties              # 조약 임베딩
│   └── EmbedAmendments            # 제개정이유 임베딩
│   # 참고: 법률용어는 임베딩 제외 (PostgreSQL FTS 전용)
├── build_graph_extended.py       # Neo4j 그래프 확장 (신규)
│   ├── load_interpretation_nodes()
│   ├── load_treaty_nodes()
│   ├── load_constitutional_nodes()
│   └── create_new_relationships()
└── (기존 스크립트 유지)
    ├── runpod_lancedb_embeddings.py
    ├── load_lancedb_data.py
    ├── load_lawyers_data.py
    └── build_graph.py
```

### 3.2 로딩 순서

```
Step 1: PostgreSQL 테이블 생성
  $ uv run alembic upgrade head
      ↓
Step 2: 데이터 적재 (PostgreSQL)
  $ uv run python scripts/load_law_data.py --type all
  $ uv run python scripts/load_law_data.py --verify
      ↓
Step 3: 임베딩 생성 (LanceDB)
  $ uv run --no-sync python scripts/embed_law_data.py --type all
      ↓
Step 4: Neo4j 그래프 확장
  $ uv run python scripts/build_graph_extended.py
      ↓
Step 5: 검증
  $ uv run python scripts/load_law_data.py --verify
  $ uv run --no-sync python scripts/embed_law_data.py --stats
  $ uv run python scripts/verify_graph.py
```

### 3.3 배치 처리 규칙

| 규칙 | 값 |
|------|-----|
| 배치 크기 | 1,000건 (PostgreSQL), 100건 (임베딩) |
| 멱등성 | ON CONFLICT DO UPDATE (unique key) |
| 진행률 | tqdm 진행바 + 배치별 로그 |
| 트랜잭션 | 배치 단위 commit |
| GC | 매 배치 후 gc.collect() |
| 검증 | --verify 플래그로 건수/무결성 확인 |

---

## 4. 서비스 계층 설계

### 4.1 DB 서비스 함수

```
backend/app/services/service_function/
├── law_service.py                    # 기존 (JSON 기반)
├── law_db_service.py                 # 신규: PostgreSQL 기반 법령 조회
├── precedent_service.py              # 기존 (JSON 기반)
├── precedent_db_service.py           # 신규: PostgreSQL 기반 판례 조회
├── interpretation_db_service.py      # 신규: 법령해석례 조회
├── legal_terms_db_service.py         # 신규: 법률용어 조회
├── constitutional_db_service.py      # 신규: 헌재결정 조회
├── treaty_db_service.py              # 신규: 조약 조회
├── lawyer_service.py                 # 기존 (JSON)
├── lawyer_db_service.py              # 기존 (PostgreSQL)
├── lawyer_stats_service.py           # 기존
└── lawyer_stats_db_service.py        # 기존
```

### 4.2 RAG 파이프라인 통합

```python
# 현재: LanceDB → source_id → JSON 파일에서 원본 조회
# 변경: LanceDB → source_id + data_type → PostgreSQL 원본 조회

async def get_source_details(
    db: AsyncSession,
    source_id: str,
    data_type: str
) -> dict:
    """data_type에 따라 적절한 PostgreSQL 테이블에서 원본 조회"""
    table_map = {
        "법령": LawDocument,
        "판례": PrecedentDocument,
        "해석례": LegislationInterpretation,
        "헌재결정": ConstitutionalDecision,
        "조약": Treaty,
        "제개정이유": LawAmendment,
    }
    # 참고: 법률용어는 벡터 검색 대상이 아니므로 여기에 포함하지 않음
    # 법률용어는 FTS 전용 검색 경로(search_legal_terms)로 처리
    model = table_map.get(data_type)
    if not model:
        return {}

    pk_field = get_pk_field(model)  # 각 모델의 PK 필드명
    result = await db.execute(
        select(model).where(pk_field == source_id)
    )
    return result.scalar_one_or_none()
```

### 4.3 FTS 전략: 애플리케이션 레벨 mecab

PostgreSQL 내장 한국어 FTS의 한계(조사/어미 미분리)를 해결하기 위해
**Python mecab-python3로 형태소 분석 → 결과를 tsvector로 저장**하는 방식 채택.

```
저장 시 (데이터 적재/갱신):
  원본 텍스트 → Python mecab → 형태소 분리 → 조사/어미 제거 → tsvector로 저장

  예: "피고인은 손해배상 청구를 기각하였다"
       → mecab: 피고인/NNG + 은/JKS + 손해배상/NNG + 청구/NNG + 를/JKO + 기각/NNG + 하/XSV + 였/EP + 다/EF
       → 조사·어미 제거: "피고인 손해배상 청구 기각"
       → to_tsvector('simple', '피고인 손해배상 청구 기각')

검색 시:
  검색어 → Python mecab → 형태소 분리 → to_tsquery('simple', '...')
```

**pg_trgm 대신 mecab을 초기 도입하는 이유**:
- pg_trgm: "기각하였다"를 3글자씩 분리 → 노이즈 많음
- mecab: "기각하였다" → "기각" 정확 추출 → 법률 도메인에서 검색 품질 차이가 큼
- mecab-python3는 `pip install` 한 줄로 설치 가능 (PostgreSQL C 확장 빌드 불필요)

**FTS 적용 대상 테이블**:

| 테이블 | tsvector 컬럼 | 용도 |
|--------|-------------|------|
| legal_terms | definition_tsv | **FTS 전용 검색** (벡터 검색 없음) |
| law_documents | content_tsv | 벡터 검색 보완 (하이브리드) |
| precedent_documents | content_tsv | 벡터 검색 보완 (하이브리드) |
| legislation_interpretations | content_tsv | 벡터 검색 보완 (하이브리드) |
| constitutional_decisions | content_tsv | 벡터 검색 보완 (하이브리드) |

### 4.4 검색 흐름 (TO-BE)

```
사용자 쿼리
    │
    ▼
[Query Rewrite] ── LLM으로 검색 쿼리 최적화
    │
    ├─▶ [LanceDB Vector Search]
    │     data_type 필터: 법령, 판례, 해석례, 헌재결정, 조약
    │     top-K: 20
    │
    ├─▶ [PostgreSQL FTS] (mecab 토큰화)
    │     대상: law_documents, precedent_documents 등
    │     top-K: 20
    │
    ├─▶ [법률용어 FTS 검색] (별도 경로)
    │     1순위: term_name 정확 매칭
    │     2순위: definition_tsv FTS 검색
    │     3순위: term_name pg_trgm 유사 매칭
    │     → 매칭된 용어 정의를 컨텍스트에 추가
    │
    ▼
[Reciprocal Rank Fusion]
    vector_weight: 0.7, fts_weight: 0.3
    + 법률용어 정의 (컨텍스트 보강)
    │
    ▼
[Reranker] ── 기존 구현 활용
    top-K: 5~10
    │
    ▼
[PostgreSQL 원본 조회]
    source_id + data_type → 해당 테이블에서 상세 정보
    │
    ▼
[LLM 응답 생성]
    Solar API → 법률 상담 응답
```

---

## 5. 디스크/메모리 예상

### 5.1 PostgreSQL

| 테이블 | 예상 크기 |
|--------|----------|
| law_documents | 250 MB |
| precedent_documents | 600 MB |
| lawyers | 15 MB |
| trial_statistics | 1 MB |
| legislation_interpretations | 80 MB |
| legal_terms | 25 MB |
| constitutional_decisions | 30 MB |
| treaties | 60 MB |
| law_amendments | 150 MB |
| **합계** | **~1.2 GB** |

### 5.2 LanceDB

| 항목 | 값 |
|------|-----|
| 기존 청크 | 253,768 (1.6 GB) |
| 신규 청크 | ~76,683 (~0.5 GB) |
| **합계** | **~330,451 청크 (~2.1 GB)** |

> 법률용어 37,169건 임베딩 제외로 ~150MB 절감

### 5.3 Neo4j

| 항목 | 현재 | 확장 후 |
|------|------|---------|
| 노드 | ~70,748 | ~119,715 |
| 관계 | ~163,854 | ~220,000 |
| 디스크 | ~500 MB | ~800 MB |

### 5.4 총 디스크

| 구성 요소 | 크기 |
|----------|------|
| PostgreSQL | ~1.2 GB (FTS 인덱스 포함) |
| LanceDB | ~2.1 GB |
| Neo4j | ~800 MB |
| 임베딩 모델 | ~2.3 GB |
| **합계** | **~6.4 GB** |

Oracle Cloud Free Tier (200 GB) 내에서 충분히 수용 가능.

---

## 6. 구현 순서

### Phase 1: PostgreSQL 통합 + Tier 1 데이터 적재

```
1-1. ORM 모델 5개 생성
     └─ legislation_interpretation.py
     └─ legal_term.py
     └─ constitutional_decision.py
     └─ treaty.py
     └─ law_amendment.py

1-2. Alembic 마이그레이션 5개 작성 (006~010)

1-3. 데이터 로딩 스크립트 작성 (load_law_data.py)

1-4. Feature Flag 추가 (config.py)

1-5. DB 서비스 함수 작성 (5개)

1-6. 기존 law/precedent 서비스에 PostgreSQL 분기 추가

1-7. RAG retrieval.py에 PostgreSQL 원본 조회 통합
```

### Phase 2: 신규 데이터 임베딩

```
2-1. 임베딩 스크립트 작성 (embed_law_data.py)
     └─ StreamingEmbeddingProcessor 상속
     └─ 데이터 유형별 Processor 클래스

2-2. 임베딩 실행 (GPU 환경)
     └─ Tier 1 데이터: ~113,852 청크

2-3. 검색 파이프라인에 data_type 필터 추가

2-4. evaluation 모듈로 검색 품질 검증
```

### Phase 3: Neo4j 그래프 확장

```
3-1. build_graph_extended.py 작성
     └─ Interpretation, Treaty, ConstitutionalDecision 노드
     └─ INTERPRETS, REVIEWS, IMPLEMENTS 관계

3-2. 그래프 검증 (verify_graph.py 확장)
```

### Phase 4: 하이브리드 검색

```
4-1. mecab-python3 의존성 추가 (pyproject.toml)

4-2. mecab 토큰화 유틸리티 작성
     └─ backend/app/services/rag/tokenizer.py
     └─ 텍스트 → mecab 형태소 분석 → 조사/어미 제거 → 공백 구분 토큰 문자열

4-3. PostgreSQL FTS 마이그레이션 (011)
     └─ 대상 테이블에 tsvector 컬럼 + GIN 인덱스 추가

4-4. FTS tsvector 생성 스크립트 작성
     └─ 기존 데이터의 텍스트 → mecab 토큰화 → tsvector 컬럼 일괄 갱신

4-5. FTS 검색 서비스 작성
     └─ backend/app/services/rag/fts_search.py
     └─ 검색 쿼리 → mecab 토큰화 → tsquery 생성 → GIN 인덱스 검색

4-6. 법률용어 FTS 전용 검색 서비스
     └─ term_name 매칭 → definition FTS → pg_trgm 유사 매칭 (3단계)

4-7. RRF 병합 구현
     └─ backend/app/services/rag/hybrid_search.py
     └─ 벡터 0.7 + FTS 0.3 가중치 결합
```

---

## 7. 수정 대상 파일 상세

### 신규 파일

| 파일 | 목적 |
|------|------|
| `backend/app/models/legislation_interpretation.py` | ORM |
| `backend/app/models/legal_term.py` | ORM |
| `backend/app/models/constitutional_decision.py` | ORM |
| `backend/app/models/treaty.py` | ORM |
| `backend/app/models/law_amendment.py` | ORM |
| `backend/alembic/versions/006~010_*.py` | 마이그레이션 5개 |
| `backend/scripts/load_law_data.py` | 통합 데이터 로더 |
| `backend/scripts/embed_law_data.py` | 통합 임베딩 생성기 |
| `backend/scripts/build_graph_extended.py` | Neo4j 확장 |
| `backend/app/services/service_function/interpretation_db_service.py` | 서비스 |
| `backend/app/services/service_function/legal_terms_db_service.py` | 서비스 |
| `backend/app/services/service_function/constitutional_db_service.py` | 서비스 |
| `backend/app/services/service_function/treaty_db_service.py` | 서비스 |
| `backend/app/services/rag/tokenizer.py` | mecab 토큰화 유틸리티 |
| `backend/app/services/rag/fts_search.py` | FTS 검색 서비스 |
| `backend/app/services/rag/hybrid_search.py` | RRF 병합 |

### 수정 파일

| 파일 | 변경 내용 |
|------|----------|
| `backend/app/core/config.py` | Feature Flag 6개 추가 |
| `backend/app/models/__init__.py` | 신규 모델 import 추가 |
| `backend/alembic/env.py` | 신규 모델 import 추가 |
| `backend/app/services/rag/retrieval.py` | PostgreSQL 원본 조회 통합, data_type 필터 확장 |
| `backend/app/services/rag/pipeline.py` | 검색 파이프라인에 신규 data_type 통합 |
| `backend/app/multi_agent/agents/case_precedent_agent.py` | 해석례/헌재결정도 컨텍스트에 포함 |

### 문서 업데이트

| 문서 | 변경 내용 |
|------|----------|
| `CLAUDE.md` (루트) | 신규 데이터 카탈로그, 테이블 구조, 스크립트 사용법 |
| `backend/CLAUDE.md` | 신규 모델/서비스/스크립트 반영 |
| `docs/vectordb_design.md` | 임베딩 스키마 확장 반영 |
| `docs/DATA_CATALOG.md` | law_data/ 전체 데이터 카탈로그 |

---

## 8. 리스크 대응

| 리스크 | 대응 방안 |
|--------|----------|
| 임베딩 시간 (~77K 청크) | GPU 환경 필수, 배치 100건, 예상 1.5~2시간 |
| PostgreSQL 디스크 (1.2GB) | Oracle Free Tier 200GB 내 여유 충분 |
| 서비스 중단 없는 배포 | 별도 LanceDB 테이블에 생성 → 테이블명 스왑 |
| data_type 필터 성능 | LanceDB 인덱스 확인, 필요시 별도 테이블 분리 |
| 데이터 품질 (HTML 태그 등) | 로딩 시 HTML 태그 제거, 텍스트 정규화 |
| JSON 구조 불일치 | 각 파일별 스키마 검증 로직 포함 |
| mecab 사전에 없는 법률 용어 | mecab-ko-dic 기본 사전 사용, 미등록어는 원형 유지 |
| mecab 설치 환경 차이 (macOS/Linux) | Docker 환경 통일, pyproject.toml에 mecab-python3 추가 |
