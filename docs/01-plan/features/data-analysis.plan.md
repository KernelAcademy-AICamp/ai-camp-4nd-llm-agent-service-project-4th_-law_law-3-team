# Data Analysis (EDA) Planning Document

> **Summary**: DB 구축 전 48개 [DONE] JSON 파일(~12GB)의 데이터 특성 파악을 위한 탐색적 분석(EDA)
>
> **Project**: law-3-team (법률 서비스 플랫폼)
> **Author**: Claude
> **Date**: 2026-02-11
> **Status**: Completed

---

## 1. Overview

### 1.1 Purpose

PostgreSQL, LanceDB, Neo4j 3개 DB 구축 전에 법률 데이터(48개 [DONE] JSON 파일, ~12GB)의 특성을 파악한다. 파일 인벤토리, 스키마 구조, 데이터 품질, 텍스트 분포, 시간 패턴, 데이터셋 간 관계, DB 볼륨 추정을 수행하여 DB 설계에 필요한 근거 데이터를 확보한다.

### 1.2 Background

- 법률 서비스 플랫폼의 핵심 데이터가 48개 JSON 파일(총 ~12GB)로 존재
- 판례(1.1GB), 법령(356MB), 헌재결정례(279MB), 행정심판례(427MB), 특별행정심판(2.05GB) 등 대용량 파일 포함
- 12개 카테고리로 분류 가능: precedent, law, constitutional, administration, special_tribunal, legislation, committee(10개), cgm_expc(28개), law_term, treaty, school
- DB 설계 시 테이블 구조, 인덱스 전략, 청킹 설정, 용량 추정 등에 EDA 결과가 직접 반영됨

### 1.3 Related Documents

- `docs/architecture/vectordb_design.md` - 벡터 DB 설계 문서
- `docs/architecture/DB_ARCHITECTURE.md` - DB 아키텍처 문서
- `docs/data/DATA_CATALOG.md` - 데이터 카탈로그

---

## 2. Scope

### 2.1 In Scope

- [x] 48개 [DONE] JSON 파일 인벤토리 (크기, 레코드 수, 루트 타입)
- [x] 카테고리별 스키마 발견 (필드명, 타입, 존재율)
- [x] 데이터 품질 분석 (null률, 중복 ID, 인코딩 이슈)
- [x] 텍스트 필드 길이 분포 + 청킹 전략 도출
- [x] 시간 분포 분석 (날짜 필드 파싱, 연도별 분포)
- [x] 데이터셋 간 참조/인용 관계 발견
- [x] 3개 DB별 볼륨 추정 (PostgreSQL, LanceDB, Neo4j)
- [x] Plotly 인터랙티브 시각화 (bar, scatter, heatmap, box, timeline, sankey, treemap, sunburst)

### 2.2 Out of Scope

- 실제 DB 마이그레이션 또는 데이터 로드
- 임베딩 생성 또는 벡터 검색 테스트
- 프론트엔드 대시보드 구현
- 데이터 정제/변환 (raw data 그대로 분석)

---

## 3. Requirements

### 3.1 Functional Requirements

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| FR-01 | 48개 [DONE] 파일의 크기, 레코드 수, 루트 타입 스캔 | High | Done |
| FR-02 | 12개 카테고리별 스키마 발견 (필드명, 타입, 존재율/null률) | High | Done |
| FR-03 | 카테고리별 데이터 품질 분석 (null/empty, 중복 ID, 인코딩) | High | Done |
| FR-04 | 주요 텍스트 필드 길이 분포 분석 (P25/P50/P75/P90/P99) | High | Done |
| FR-05 | 현재 청킹 설정(1,250자/800토큰) 대비 예상 청크 수 추정 | Medium | Done |
| FR-06 | 날짜 필드 파싱 및 시간 범위 분석 | Medium | Done |
| FR-07 | 판례 참조조문/참조판례 인용 패턴 분석 | Medium | Done |
| FR-08 | PostgreSQL 테이블별 행 수/평균 행 크기/총 용량 추정 | High | Done |
| FR-09 | LanceDB 카테고리별 청크 수/벡터 저장량 추정 | High | Done |
| FR-10 | Neo4j 노드/엣지 타입별 수량 추정 | Medium | Done |
| FR-11 | Plotly 인터랙티브 시각화 (bar, scatter, heatmap, box 등) | High | Done |
| FR-12 | 중간 결과를 JSON으로 저장 (phase1~phase7) | Medium | Done |

### 3.2 Non-Functional Requirements

| Category | Criteria | Measurement Method |
|----------|----------|-------------------|
| Memory Safety | 200MB 이상 파일은 ijson 스트리밍 처리 | `STREAMING_THRESHOLD_MB = 200` |
| Sampling | Reservoir sampling으로 메모리 O(k) 유지 | `reservoir_sample(k=10000)` |
| Reproducibility | 고정 시드(42) 사용 | `random.Random(seed=42)` |
| Interactivity | Plotly 차트로 줌/호버/필터 지원 | JupyterLab 환경 |

---

## 4. Success Criteria

### 4.1 Definition of Done

- [x] 공유 모듈 3개 작성 (`__init__.py`, `common.py`, `data_registry.py`)
- [x] Jupyter 노트북 4개 작성 (01~04)
- [x] dev 의존성 추가 (ijson, plotly, nbformat, tqdm, kaleido, jupyterlab)
- [x] `ruff check` 린트 통과 (공유 모듈)
- [x] `.gitignore`에 `eda_output/` 추가

### 4.2 Quality Criteria

- [x] 200MB 이상 파일은 스트리밍으로 처리
- [x] 모든 시각화가 Plotly 인터랙티브 차트
- [x] 노트북 간 의존성 명확 (phase N 결과 → phase N+1 로드)
- [x] 린트 에러 0개

---

## 5. Risks and Mitigation

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| 대용량 파일 메모리 초과 | High | Medium | ijson 스트리밍 + reservoir sampling |
| 날짜 형식 불일치 (YYYYMMDD vs YYYY-MM-DD) | Low | High | 다중 형식 파서 구현 |
| 일부 [DONE] 파일 대소문자 차이 | Low | Low | `name_lower` 비교 사용 |
| 카테고리 간 필드명 상이 | Medium | High | 카테고리별 레지스트리 매핑 |

---

## 6. Architecture Considerations

### 6.1 Project Level Selection

| Level | Characteristics | Recommended For | Selected |
|-------|-----------------|-----------------|:--------:|
| **Starter** | Simple structure | Static sites | |
| **Dynamic** | Feature-based modules, services layer | Web apps with backend | **V** |
| **Enterprise** | Strict layer separation, DI | High-traffic systems | |

### 6.2 Key Architectural Decisions

| Decision | Options | Selected | Rationale |
|----------|---------|----------|-----------|
| 시각화 라이브러리 | matplotlib / plotly / seaborn | plotly | 인터랙티브 (줌, 호버, 필터) |
| 대용량 파일 처리 | pandas / ijson / dask | ijson | 메모리 안전, 스트리밍 |
| 실행 환경 | Python 스크립트 / Jupyter | Jupyter (JupyterLab) | 셀 단위 탐색, 시각화 |
| 공유 코드 방식 | 각 노트북에 복사 / 공유 모듈 | 공유 모듈 (`scripts/eda/`) | 중복 제거, 유지보수 |
| 결과 저장 | 파일 / DB | JSON 파일 (`eda_output/`) | 간단, 노트북 간 공유 |

---

## 7. Implementation Plan

### 7.1 파일 구조

```
backend/
├── scripts/eda/
│   ├── __init__.py           # 패키지 초기화
│   ├── common.py             # 스트리밍, 샘플링, 파일 I/O (13 함수)
│   └── data_registry.py      # 48파일 → 12카테고리 매핑
├── notebooks/eda/
│   ├── 01_inventory_schema.ipynb    # 인벤토리 + 스키마
│   ├── 02_quality_text.ipynb        # 품질 + 텍스트
│   ├── 03_temporal_relationships.ipynb  # 시간 + 관계
│   └── 04_projections_summary.ipynb    # DB 추정 + 요약
└── eda_output/               # 중간 결과 (.gitignore)
```

### 7.2 구현 순서

1. 의존성 추가 (`pyproject.toml`) + `.gitignore` 업데이트
2. `scripts/eda/__init__.py` + `common.py` + `data_registry.py`
3. `01_inventory_schema.ipynb` (인벤토리 + 스키마)
4. `02_quality_text.ipynb` (품질 + 텍스트)
5. `03_temporal_relationships.ipynb` (시간 + 관계)
6. `04_projections_summary.ipynb` (추정 + 요약)
7. 린트 검증 (공유 모듈)

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2026-02-11 | Initial creation (post-implementation) | Claude |
