# EDA 기반 DB 전환 설계서

> **작성 기준**: EDA Phase 1~6 + 노트북 05, 06 분석 결과 기반
>
> **기존 설계 관계**: `vectordb_design.md` (LanceDB), `DB_ARCHITECTURE.md` (PostgreSQL) 유지 및 확장

---

## 1. 개요

### 1.1 목적

11개 카테고리(47개 `[DONE]` JSON, ~4.6GB, ~526K 레코드)의 EDA 분석 결과를 바탕으로 3개 DB(LanceDB, Neo4j, PostgreSQL) 전환/확장 전략을 수립합니다.

### 1.2 참조 EDA

| Phase | 노트북 | 산출물 | 핵심 결과 |
|-------|--------|--------|----------|
| 1 | `01_inventory_schema.ipynb` | `phase1_inventory.json` | 47파일, 526K 레코드, ~4.6GB |
| 2 | `01_inventory_schema.ipynb` | `phase2_schema.json` | 11개 카테고리별 필드 구조 |
| 3 | `02_quality_text.ipynb` | `phase3_quality.json` | 텍스트 품질, null/empty율 |
| 4 | `02_quality_text.ipynb` | `phase4_text.json` | 텍스트 길이 분포 |
| 5 | `05_lancedb_embedding_strategy.ipynb` | `phase5_embedding_strategy.json` | 임베딩 전략 비교 |
| 6 | `06_neo4j_citation_analysis.ipynb` | `phase6_citation_analysis.json` | 인용/참조 필드 분석 |
| 7 | `04_projections_summary.ipynb` | `phase7_projections.json` | DB 용량 추정 |

### 1.3 기존 설계와의 관계

| 기존 설계 | 이 문서에서의 역할 |
|-----------|------------------|
| `vectordb_design.md` | 현행 LanceDB 설계 유지, `data_type` 확장 정의 |
| `DB_ARCHITECTURE.md` | 현행 PostgreSQL 설계 유지, 새 테이블 추가 정의 |
| `CLAUDE.md` (Neo4j 섹션) | 현행 그래프 스키마 유지, 노드/관계 확장 정의 |

---

## 2. EDA 결과 요약

### 2.1 데이터 규모

| 카테고리 | label | 파일 수 | 레코드 수 | 크기 |
|----------|-------|---------|----------|------|
| precedent | 판례 | 1 | 92,055 | 1,076 MB |
| law | 법령 | 1 | 5,548 | 355 MB |
| constitutional | 헌재결정례 | 1 | 31,718 | 278 MB |
| administration | 행정심판례 | 1 | 34,254 | 426 MB |
| special_tribunal | 특별행정심판 | 2 | 148,778 | 2,074 MB |
| legislation | 법령해석례 | 1 | 8,597 | 79 MB |
| committee | 위원회 결정문 | 10 | 56,802 | 393 MB |
| cgm_expc | 부처 해석례 | 27 | 37,496 | 82 MB |
| law_term | 법률용어사전 | 1 | 81,488 | 66 MB |
| treaty | 조약 | 1 | 3,589 | 57 MB |
| school | 행정규칙 | 1 | 5,258 | 57 MB |
| **합계** | | **47** | **~506K** | **~4.9 GB** |

### 2.2 요약 필드 커버리지 (노트북 05)

| 카테고리 | 요약 필드 | 존재율 | 비어있지 않은 비율 |
|----------|----------|--------|-----------------|
| precedent | 판례요약 | 98.7% | 98.7% |
| law | 법령 요약 | 100% | 100% |
| constitutional | 심판례요약 | 100% | 100% |
| administration | 심판례요약 | 100% | 100% |
| special_tribunal | 심판례요약 | 100% | 100% |
| legislation | 해석례요약 | 100% | 100% |
| committee | 결정문요약 | 100% | 100% |
| cgm_expc | 해석요약 | 100% | 100% |
| treaty | 조약요약 | 100% | 100% |
| school | 행정규칙요약 | 100% | 100% |
| **law_term** | **(없음)** | **0%** | **0%** |

### 2.3 인용/참조 필드 현황 (노트북 06)

| 카테고리 | 구조화 인용 필드 | 유효 데이터율 |
|----------|----------------|-------------|
| precedent | 참조조문 (81.3%), 참조판례 (44%) | **이미 Neo4j 구축** |
| constitutional | 참조조문 (14.2%), 참조판례 (13.1%), 심판대상조문 (10.6%) | 구조화 필드 있음 |
| cgm_expc | 관련법령 (100%) | **가장 풍부** |
| 기타 | (없음) | 텍스트 파싱 필요 |

---

## 3. LanceDB 임베딩 전략

### 3.1 결론: 요약문 중심 전략 채택

**근거** (노트북 05 분석 결과):

1. **10/11 카테고리에 요약 필드 존재** (커버리지 98.7~100%)
2. **요약문 평균 길이가 임베딩에 적합** (대부분 200~1000자, 청킹 임계값 1250자 이내)
3. **시나리오 A(요약만) vs B(전체) 비교**: B는 A 대비 5~50배 텍스트량이지만 검색 정밀도 기여는 제한적
4. **비용 효율**: 요약만 임베딩 시 청크 수가 전체 대비 1/5~1/10 수준

### 3.2 카테고리별 임베딩 대상 필드

| 카테고리 | 임베딩 대상 | 전략 | 비고 |
|----------|-----------|------|------|
| precedent | 판례요약 | 요약 단독 | 현행 유지 (판례내용 청킹) |
| law | 법령 요약 + 조문 | 요약 + 조문별 청킹 | 현행 유지 |
| constitutional | 심판례요약 | 요약 단독 | 신규 |
| administration | 심판례요약 | 요약 단독 | 신규 |
| special_tribunal | 심판례요약 | 요약 단독 | 신규 |
| legislation | 해석례요약 | 요약 단독 | 신규 |
| committee | 결정문요약 | 요약 단독 | 신규 |
| cgm_expc | 해석요약 | 요약 단독 | 신규 |
| treaty | 조약요약 | 요약 단독 | 신규 |
| school | 행정규칙요약 | 요약 단독 | 신규 |
| law_term | 법령용어정의 | text_field 사용 | 요약 없음, 정의 텍스트 임베딩 |

### 3.3 LanceDB 스키마 확장

현행 `legal_chunks` 테이블의 `data_type` 필드에 새 값 추가:

```
현행: "precedent" | "law"
확장: "precedent" | "law" | "constitutional" | "administration"
      | "special_tribunal" | "legislation" | "committee"
      | "cgm_expc" | "treaty" | "school" | "law_term"
```

메타데이터 필드는 기존 스키마 유지:
- `source_id`: 카테고리별 ID 필드
- `title`: 사건명/법령명/안건명
- `data_type`: 카테고리 키
- `chunk_index`, `total_chunks`: 청킹 정보

### 3.4 임베딩 용량 추정

| 시나리오 | 총 청크 수 | 벡터 용량 | 비고 |
|----------|----------|----------|------|
| 현행 (판례+법령) | ~254K | ~990 MB | 이미 구축 |
| A: 요약 중심 확장 | +~200K | +~780 MB | 9개 카테고리 추가 |
| B: 전체 텍스트 확장 | +~1.5M | +~5.8 GB | 과도한 비용 |

**권장**: 시나리오 A (요약 중심) - 현행 대비 약 1.8x 증가

---

## 4. Neo4j 그래프 확장 전략

### 4.1 확장 우선순위 (노트북 06 점수 기반)

| Tier | 카테고리 | 점수 | 근거 |
|------|---------|------|------|
| **Tier 1 (즉시)** | precedent | 7 | 이미 구축, 구조화 인용 81%+, 92K건 |
| **Tier 1 (즉시)** | cgm_expc | 5+ | 관련법령 100%, 37K건 |
| **Tier 2 (단기)** | constitutional | 4 | 구조화 필드 있음, 커버리지 낮음 |
| **Tier 2 (단기)** | legislation | 3 | 텍스트 파싱 가능, 8.6K건 |
| **Tier 3 (중기)** | administration | 2 | 텍스트 추출 필요, 34K건 |
| **Tier 3 (중기)** | committee | 2 | 텍스트 추출 필요, 57K건 |
| **Tier 3 (중기)** | special_tribunal | 1 | 대용량 149K건, 파싱 필요 |
| 제외 | law_term, treaty, school | 0 | 인용 데이터 부족 |

### 4.2 새 노드 타입 정의

```
현행 노드:
  (:Statute {id, name, type, abbreviation, citation_count})
  (:Case {id, case_number, name, summary})
  (:Alias {name, category})

확장 노드:
  (:Constitutional {id, case_number, name, summary, decision_date})
  (:AdminCase {id, case_number, name, summary, decision_date, tribunal})
  (:Interpretation {id, case_name, summary, agency, interpret_date})
  (:CommitteeDecision {id, title, summary, committee, decision_date})
```

### 4.3 새 관계 타입 정의

```
현행 관계:
  (Case)-[:CITES]->(Statute)
  (Case)-[:CITES_CASE]->(Case)
  (Statute)-[:HIERARCHY_OF]->(Statute)
  (Statute)-[:RELATED_TO]->(Statute)
  (Alias)-[:ALIAS_OF]->(Statute)

확장 관계 (Tier 1):
  (Interpretation)-[:INTERPRETS]->(Statute)     # cgm_expc 관련법령→Statute
  (Constitutional)-[:CITES]->(Statute)           # 헌재 참조조문→Statute
  (Constitutional)-[:CITES_CASE]->(Case)         # 헌재 참조판례→Case

확장 관계 (Tier 2):
  (AdminCase)-[:CITES]->(Statute)                # 행정심판 텍스트→Statute
  (CommitteeDecision)-[:CITES]->(Statute)        # 위원회 텍스트→Statute
```

### 4.4 그래프 규모 추정

| 항목 | 현행 | Tier 1 추가 | Tier 2 추가 | 합계 |
|------|------|-----------|-----------|------|
| 노드 | 70,748 | +37,496 | +74,569 | ~183K |
| 관계 | 163,854 | +~94K | +~50K | ~308K |

---

## 5. PostgreSQL 스키마 확장

### 5.1 카테고리별 테이블 매핑

| 카테고리 | 테이블명 | PK | 주요 컬럼 |
|----------|---------|-----|----------|
| constitutional | `constitutional_cases` | `serial_number` | case_number, case_name, decision_summary, ruling, reasoning |
| administration | `admin_cases` | `serial_number` | case_number, case_name, ruling, reasoning, tribunal |
| special_tribunal | `special_tribunal_cases` | `serial_number` | case_number, case_name, ruling, reasoning, vessel_type |
| legislation | `legislation_interpretations` | `serial_number` | case_name, query_summary, answer, reasoning, agency |
| committee | `committee_decisions` | `serial_number` | title, decision_summary, ruling, reasoning, committee |
| cgm_expc | `dept_interpretations` | `serial_number` | case_name, query_summary, answer, related_law, agency |
| treaty | `treaties` | `serial_number` | treaty_name_kr, treaty_name_en, content, summary |
| school | `admin_rules` | `rule_id` | rule_name, content, summary, department |

> **기존 테이블 유지**: `precedent_documents`, `law_documents`, `lawyers`, `legal_terms`, `trial_statistics`

### 5.2 Feature Flag 전략

```python
# backend/app/core/config.py
USE_DB_CONSTITUTIONAL: bool = False
USE_DB_ADMINISTRATION: bool = False
USE_DB_SPECIAL_TRIBUNAL: bool = False
USE_DB_LEGISLATION: bool = False
USE_DB_COMMITTEE: bool = False
USE_DB_CGM_EXPC: bool = False
USE_DB_TREATY: bool = False
USE_DB_SCHOOL: bool = False
```

각 feature flag는 `False` 기본값으로, `.env`에서 개별 활성화합니다.
JSON 파일 기반 서비스는 삭제하지 않아 즉시 롤백 가능합니다.

---

## 6. 데이터 로드 파이프라인

### 6.1 단계별 확장 플로우

```
Phase 1 (현행 완료)
  ├── precedent → PostgreSQL + LanceDB + Neo4j ✅
  └── law → PostgreSQL + LanceDB + Neo4j ✅

Phase 2 (Tier 1 - 즉시 확장)
  ├── constitutional → PostgreSQL + LanceDB + Neo4j
  └── cgm_expc → PostgreSQL + LanceDB + Neo4j

Phase 3 (Tier 2 - 단기 확장)
  ├── administration → PostgreSQL + LanceDB
  ├── legislation → PostgreSQL + LanceDB + Neo4j
  └── committee → PostgreSQL + LanceDB

Phase 4 (Tier 3 - 중기 확장)
  ├── special_tribunal → PostgreSQL + LanceDB
  ├── treaty → PostgreSQL + LanceDB
  └── school → PostgreSQL + LanceDB

Phase 5 (보조 데이터)
  └── law_term → PostgreSQL + LanceDB (선택)
```

### 6.2 각 Phase별 작업

| Phase | 작업 | 산출물 |
|-------|------|--------|
| 2 | ORM 모델, 마이그레이션, 로드 스크립트, 임베딩 스크립트, Neo4j 확장 | 2개 테이블, ~69K 임베딩, ~94K 관계 |
| 3 | ORM 모델, 마이그레이션, 로드 스크립트, 임베딩 스크립트 | 3개 테이블, ~100K 임베딩 |
| 4 | ORM 모델, 마이그레이션, 로드 스크립트, 임베딩 스크립트 | 3개 테이블, ~157K 임베딩 |
| 5 | 임베딩 스크립트 (선택) | ~82K 임베딩 |

---

## 7. 구현 우선순위 및 로드맵

### 7.1 우선순위 기준

1. **RAG 검색 품질 향상**: 법률 검색 범위 확장 기여도
2. **그래프 네트워크 효과**: 기존 관계와의 교차 참조 가능성
3. **구현 난이도**: 스키마 유사성, 파싱 복잡도
4. **데이터 규모**: 레코드 수, 용량

### 7.2 권장 로드맵

| 순서 | 카테고리 | DB | 예상 기간 | 이유 |
|------|---------|-----|----------|------|
| 1 | constitutional | PG + Lance + Neo4j | 1주 | 판례와 유사 스키마, 구조화 인용 |
| 2 | cgm_expc | PG + Lance + Neo4j | 1주 | 관련법령 100%, Neo4j 효과 높음 |
| 3 | administration | PG + Lance | 1주 | 34K건, 행정 분쟁 검색 확장 |
| 4 | legislation | PG + Lance + Neo4j | 1주 | 법령 해석 검색, 텍스트 인용 파싱 |
| 5 | committee | PG + Lance | 1주 | 57K건, 다수 위원회 통합 |
| 6 | special_tribunal | PG + Lance | 1주 | 149K건 (대용량), 해양 심판 특화 |
| 7 | treaty + school | PG + Lance | 1주 | 보조 데이터, 낮은 우선순위 |

---

## 8. 용량 추정 합산

| DB | 현행 | Phase 2 추가 | Phase 3 추가 | Phase 4 추가 | 최종 합계 |
|----|------|------------|------------|------------|----------|
| PostgreSQL | ~2.0 GB | +0.6 GB | +0.8 GB | +1.6 GB | ~5.0 GB |
| LanceDB | ~1.0 GB | +0.3 GB | +0.4 GB | +0.6 GB | ~2.3 GB |
| Neo4j | ~0.2 GB | +0.1 GB | +0.05 GB | - | ~0.35 GB |
| **합계** | **~3.2 GB** | **+1.0 GB** | **+1.25 GB** | **+2.2 GB** | **~7.65 GB** |

> 추정 기준: PostgreSQL 1.5x 오버헤드, LanceDB 1024dim float32 벡터 + 500B 메타, Neo4j 1KB/노드 + 0.5KB/엣지
