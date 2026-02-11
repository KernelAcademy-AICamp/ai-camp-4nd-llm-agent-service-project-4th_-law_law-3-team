# Data Analysis (EDA) Design Document

> **Summary**: 법률 데이터 EDA 노트북 + 공유 모듈의 구체적 구현 설계 - 스트리밍, 샘플링, 시각화, DB 추정
>
> **Project**: law-3-team (법률 서비스 플랫폼)
> **Author**: Claude
> **Date**: 2026-02-11
> **Status**: Completed
> **Plan Reference**: `docs/01-plan/features/data-analysis.plan.md` (v1.0)

---

## 1. Overview

### 1.1 Purpose

Plan 문서(FR-01~FR-12)에 정의된 EDA 요구사항을 **구체적인 코드 수준 설계**로 변환한다. 공유 모듈의 함수 인터페이스, 데이터 레지스트리 구조, 4개 노트북의 셀 구성과 시각화 사양을 명세한다.

### 1.2 Scope

| 산출물 | 파일 경로 | 대상 FR |
|--------|----------|---------|
| 공유 유틸리티 | `backend/scripts/eda/common.py` | FR-01, FR-12 |
| 데이터 레지스트리 | `backend/scripts/eda/data_registry.py` | FR-01 |
| 노트북 01: 인벤토리+스키마 | `backend/notebooks/eda/01_inventory_schema.ipynb` | FR-01, FR-02 |
| 노트북 02: 품질+텍스트 | `backend/notebooks/eda/02_quality_text.ipynb` | FR-03, FR-04, FR-05 |
| 노트북 03: 시간+관계 | `backend/notebooks/eda/03_temporal_relationships.ipynb` | FR-06, FR-07 |
| 노트북 04: DB추정+요약 | `backend/notebooks/eda/04_projections_summary.ipynb` | FR-08, FR-09, FR-10, FR-11 |

---

## 2. Architecture

### 2.1 모듈 의존관계

```
backend/scripts/eda/
├── __init__.py           # 패키지 docstring
├── common.py             # 핵심 유틸리티 (13 함수, 경로 상수)
│   ├── DATA_DIR          # law-3/data/
│   ├── OUTPUT_DIR        # backend/eda_output/
│   ├── stream_json()     # ijson 스트리밍
│   ├── load_json()       # json.load 전체 로드
│   ├── smart_load()      # 크기 기반 자동 선택
│   ├── reservoir_sample() # O(k) 메모리 샘플링
│   ├── save_result()     # eda_output/ JSON 저장
│   ├── load_result()     # 이전 단계 결과 로드
│   └── ...               # count, detect, infer 등
└── data_registry.py      # 48파일 → 12카테고리 매핑
    ├── CATEGORIES         # 카테고리별 메타데이터 dict
    ├── CGM_EXPC_AGENCIES  # 28개 부처 매핑
    ├── COMMITTEE_AGENCIES # 10개 위원회 매핑
    └── get_all_files()    # 전체 파일 리스트
```

### 2.2 노트북 실행 흐름

```
01_inventory_schema.ipynb
    │  → phase1_inventory.json (파일 인벤토리)
    │  → phase2_schema.json (스키마 정보)
    ▼
02_quality_text.ipynb
    │  ← phase1, phase2 로드
    │  → phase3_quality.json (품질 분석)
    │  → phase4_text.json (텍스트 분석)
    ▼
03_temporal_relationships.ipynb
    │  ← phase1, phase2 로드
    │  → phase5_temporal.json (시간 분포)
    │  → phase6_relationships.json (인용 관계)
    ▼
04_projections_summary.ipynb
    │  ← phase1~phase6 전체 로드
    │  → phase7_projections.json (DB 볼륨 추정)
    ▼
    종합 대시보드 (Plotly indicator cards)
```

### 2.3 대용량 파일 처리 전략

| 파일 크기 | 처리 방식 | 함수 |
|----------|----------|------|
| < 200MB | `json.load()` 전체 로드 | `load_json()` |
| >= 200MB | `ijson.items()` 스트리밍 | `stream_json()` |
| 자동 선택 | 크기 기반 분기 | `smart_load()` |
| 샘플링 | Reservoir sampling (O(k) 메모리) | `reservoir_sample(k)` |

---

## 3. Data Model

### 3.1 카테고리 레지스트리 (CATEGORIES)

```python
CATEGORIES: dict[str, dict[str, Any]] = {
    "precedent": {
        "label": "판례",
        "files": ["[DONE]precedents-4.json"],  # 1파일, ~1.1GB
        "id_field": "판례정보일련번호",
        "text_fields": ["판례내용", "판결요지", "판시사항", "이유"],
        "date_field": "선고일자",
        "streaming": True,
    },
    "law": {
        "label": "법령",
        "files": ["[DONE]law-2.json"],  # 1파일, ~356MB
        "id_field": "법령ID",
        "text_fields": ["조문"],
        "date_field": None,
        "streaming": True,
    },
    # ... 12개 카테고리
}
```

| 카테고리 | 파일 수 | streaming | 비고 |
|----------|---------|-----------|------|
| precedent | 1 | True | ~1.1GB |
| law | 1 | True | ~356MB |
| constitutional | 1 | True | ~279MB |
| administration | 1 | True | ~427MB |
| special_tribunal | 2 | True | ~2.05GB |
| legislation | 1 | False | ~79MB |
| committee | 10 | False | ~300MB (10개 위원회) |
| cgm_expc | 28 | False | ~100MB (28개 부처) |
| law_term | 1 | False | ~66MB |
| treaty | 1 | False | ~57MB |
| school | 1 | False | ~58MB |
| **합계** | **48** | | **~12GB** |

### 3.2 중간 결과 스키마 (eda_output/)

| 파일 | 내용 | 생성 노트북 |
|------|------|------------|
| `phase1_inventory.json` | `{category: {files: [{name, path, size_mb, record_count, root_type}]}}` | 01 |
| `phase2_schema.json` | `{category: {fields: {name: {types, null_rate, presence_rate}}}}` | 01 |
| `phase3_quality.json` | `{category: {null_rates, duplicate_ids, encoding_issues}}` | 02 |
| `phase4_text.json` | `{category: {field: {p25, p50, p75, p90, p99, mean, max}}}` | 02 |
| `phase5_temporal.json` | `{category: {min_date, max_date, year_distribution}}` | 03 |
| `phase6_relationships.json` | `{citation_stats, top_cited_laws, cross_references}` | 03 |
| `phase7_projections.json` | `{postgresql, lancedb, neo4j}` | 04 |

---

## 4. Module Design

### 4.1 `common.py` 함수 인터페이스

#### 경로 상수

```python
_THIS_DIR = Path(__file__).resolve().parent       # scripts/eda/
BACKEND_DIR = _THIS_DIR.parent.parent             # backend/
PROJECT_ROOT = BACKEND_DIR.parent                 # law-3/
DATA_DIR = PROJECT_ROOT / "data"                  # 원본 데이터
OUTPUT_DIR = BACKEND_DIR / "eda_output"           # 중간 결과
STREAMING_THRESHOLD_MB = 200                      # 스트리밍 기준
```

#### 핵심 함수

| 함수 | 시그니처 | 용도 |
|------|---------|------|
| `get_file_size_mb` | `(path: Path) -> float` | 파일 크기 (MB) |
| `stream_json` | `(path: Path) -> Generator[dict]` | ijson 스트리밍 |
| `load_json` | `(path: Path) -> list[dict]` | 전체 로드 |
| `smart_load` | `(path: Path, threshold_mb) -> list | Generator` | 크기 기반 자동 선택 |
| `reservoir_sample` | `(iterable, k=10000, seed=42) -> list[dict]` | Reservoir sampling |
| `count_records` | `(path: Path) -> int` | 스트리밍 카운트 |
| `count_records_fast` | `(path: Path) -> int` | 크기 기반 카운트 |
| `save_result` | `(name: str, data: Any) -> Path` | JSON 저장 |
| `load_result` | `(name: str) -> Any` | 결과 로드 |
| `discover_done_files` | `() -> list[dict]` | [DONE] 파일 탐색 |
| `get_sample` | `(path: Path, n=1000, seed=42) -> list[dict]` | 크기 기반 샘플 추출 |
| `detect_root_type` | `(path: Path) -> str` | JSON 루트 타입 ("array" | "object") |
| `infer_field_types` | `(records: list[dict]) -> dict` | 필드별 타입/null률 추론 |

### 4.2 `data_registry.py` 헬퍼 함수

| 함수 | 시그니처 | 용도 |
|------|---------|------|
| `get_all_files` | `() -> list[dict]` | 48파일 전체 리스트 `[{category, file, label}]` |
| `get_category_files` | `(category: str) -> list[str]` | 카테고리별 파일명 |
| `get_agency_name` | `(filename: str) -> str` | 파일명 → 기관명 변환 |
| `get_total_file_count` | `() -> int` | 전체 파일 수 (48) |

---

## 5. Notebook Specifications

### 5.1 노트북 01: `01_inventory_schema.ipynb`

**목표**: 전체 데이터 현황 파악 + 각 카테고리의 필드 구조 발견

| 셀 | 내용 | 시각화 |
|----|------|--------|
| 1 | 환경 설정 (imports, sys.path, plotly 테마) | - |
| 2 | 파일 인벤토리 스캔: 48개 파일 크기, 루트 타입 | - |
| 3 | 레코드 수 카운트: `count_records_fast()` (대용량은 스트리밍) | - |
| 4 | 카테고리별 파일 크기 수평 바 차트 | `px.bar(orientation='h')` |
| 5 | 카테고리별 레코드 수 바 차트 (로그 스케일) | `px.bar(log_y=True)` |
| 6 | 파일 크기 vs 레코드 수 버블 차트 | `px.scatter(size='size_mb')` |
| 7 | 스키마 발견: 카테고리별 1,000건 샘플 → `infer_field_types()` | - |
| 8 | 스키마 비교 테이블 | `go.Table()` |
| 9 | 카테고리별 필드 수 비교 바 차트 | `px.bar()` |
| 10 | 결과 저장 | `save_result("phase1_inventory", ...)` |

**출력**: `phase1_inventory.json`, `phase2_schema.json`

### 5.2 노트북 02: `02_quality_text.ipynb`

**목표**: 데이터 품질 이슈 발견 + 텍스트 길이 분포로 청킹 전략 도출

| 셀 | 내용 | 시각화 |
|----|------|--------|
| 1 | 환경 설정 + Phase 1 결과 로드 | - |
| 2 | 품질 분석: 카테고리별 5,000건 샘플 → null/empty/중복 ID | - |
| 3 | null률 히트맵 (카테고리 x 필드) | `px.imshow()` |
| 4 | 중복 ID 비율 바 차트 | `px.bar()` |
| 5 | 텍스트 길이 수집 (카테고리별 `text_fields` 기준) | - |
| 6 | 카테고리별 텍스트 길이 box plot | `px.box()` |
| 7 | 텍스트 길이 히스토그램 + 청킹 기준선 (1,250자 vline) | `px.histogram()` + `add_vline()` |
| 8 | 청킹 전략 권고 테이블 | `go.Table()` |
| 9 | 결과 저장 | `save_result("phase3_quality", ...)` |

**출력**: `phase3_quality.json`, `phase4_text.json`

### 5.3 노트북 03: `03_temporal_relationships.ipynb`

**목표**: 데이터 시간 범위 파악 + 데이터셋 간 참조/인용 관계 발견

| 셀 | 내용 | 시각화 |
|----|------|--------|
| 1 | 환경 설정 + Phase 1 결과 로드 | - |
| 2 | 날짜 파싱 유틸리티 (YYYYMMDD, YYYY.MM.DD, YYYY-MM-DD) | - |
| 3 | 카테고리별 시간 범위 수집 | - |
| 4 | 카테고리별 시간 범위 Gantt 차트 | `px.timeline()` |
| 5 | 연도별 레코드 수 누적 영역 차트 | `px.area(groupnorm='percent')` |
| 6 | 연도 x 카테고리 히트맵 | `px.imshow()` |
| 7 | 판례 참조조문/참조판례 인용 분석 | - |
| 8 | 데이터셋 간 인용 Sankey diagram | `go.Sankey()` |
| 9 | Neo4j 노드/엣지 타입 매핑 테이블 | `go.Table()` |
| 10 | 결과 저장 | `save_result("phase5_temporal", ...)` |

**출력**: `phase5_temporal.json`, `phase6_relationships.json`

### 5.4 노트북 04: `04_projections_summary.ipynb`

**목표**: 3개 DB별 용량 추정 + 전체 EDA 종합 요약

| 셀 | 내용 | 시각화 |
|----|------|--------|
| 1 | 환경 설정 + Phase 1-6 결과 전체 로드 | - |
| 2 | PostgreSQL 추정 (행 수, 평균 크기, 오버헤드 1.5x) | - |
| 3 | PostgreSQL 테이블별 용량 stacked bar | `px.bar(barmode='stack')` |
| 4 | LanceDB 추정 (청크 수, 1024차원 × 4바이트) | - |
| 5 | LanceDB 카테고리별 청크 수 treemap | `px.treemap()` |
| 6 | Neo4j 추정 (노드/엣지 타입별 수량) | - |
| 7 | Neo4j 구조 sunburst 차트 | `px.sunburst()` |
| 8 | 3개 DB 통합 용량 비교 바 차트 | `px.bar()` |
| 9 | 종합 대시보드 (6개 KPI indicator cards) | `go.Indicator()` |
| 10 | DB 구축 우선순위 권고 테이블 | `go.Table()` |
| 11 | 결과 저장 | `save_result("phase7_projections", ...)` |

**출력**: `phase7_projections.json`

---

## 6. Visualization Standards

### 6.1 Plotly 테마 설정

```python
import plotly.io as pio
pio.templates.default = "plotly_white"

COLORS = {
    "precedent": "#636EFA",
    "law": "#EF553B",
    "constitutional": "#00CC96",
    "administration": "#AB63FA",
    "special_tribunal": "#FFA15A",
    "legislation": "#19D3F3",
    "committee": "#FF6692",
    "cgm_expc": "#B6E880",
    "law_term": "#FF97FF",
    "treaty": "#FECB52",
    "school": "#72B7B2",
}
```

### 6.2 차트 타입별 사양

| 차트 타입 | plotly 함수 | 주요 파라미터 |
|----------|------------|--------------|
| 수평 바 | `px.bar(orientation='h')` | `hover_data`, `color` |
| 로그 스케일 바 | `px.bar(log_y=True)` | `text_auto=True` |
| 버블 스캐터 | `px.scatter(size=...)` | `hover_name`, `size_max=60` |
| 히트맵 | `px.imshow()` | `color_continuous_scale='RdYlGn_r'` |
| 박스 플롯 | `px.box()` | `points='outliers'` |
| 히스토그램 | `px.histogram()` | `nbins=50`, `add_vline()` |
| 타임라인 | `px.timeline()` | `color='category'` |
| 누적 영역 | `px.area()` | `groupnorm='percent'` |
| Sankey | `go.Sankey()` | `node`, `link` |
| Treemap | `px.treemap()` | `path`, `values` |
| Sunburst | `px.sunburst()` | `names`, `parents`, `values` |
| Indicator | `go.Indicator()` | `mode="number+delta"` |
| Table | `go.Table()` | `header`, `cells` |

---

## 7. DB Volume Estimation Parameters

### 7.1 PostgreSQL 추정 상수

```python
PG_OVERHEAD = 1.5       # 인덱스 + WAL 오버헤드 배율
PG_TEXT_RATIO = 0.6      # 텍스트 필드가 차지하는 비율
PG_HEADER_BYTES = 23     # 행 헤더 크기 (pg_catalog)
```

### 7.2 LanceDB 추정 상수

```python
VECTOR_DIM = 1024        # 임베딩 차원
BYTES_PER_FLOAT = 4      # float32
CHUNK_SIZE_CHARS = 1250  # 판례 청킹 기준
CHUNK_OVERLAP = 125      # 청킹 오버랩
LAW_MAX_TOKENS = 800     # 법령 청킹 토큰 기준
```

### 7.3 Neo4j 추정 기준

```python
# 현재 그래프 기준
CURRENT_NODES = {
    "Statute": 5572,
    "Case": 65107,
}
CURRENT_EDGES = {
    "HIERARCHY_OF": 3624,
    "CITES": 72414,
    "CITES_CASE": 87654,
    "RELATED_TO": 93,
}
```

---

## 8. Error Handling

### 8.1 노트북 에러 처리

| 상황 | 처리 방법 |
|------|----------|
| [DONE] 파일 없음 (data/ 비어있음) | `discover_done_files()` 빈 리스트 반환 + 경고 |
| 이전 단계 결과 없음 | `load_result()` → `FileNotFoundError` + 안내 메시지 |
| 대용량 파일 메모리 부족 | `smart_load()` 자동 스트리밍 전환 |
| 날짜 파싱 실패 | `try/except` → None 반환, 카운트 |
| JSON 파싱 에러 | `detect_root_type()` 선행 확인 |

### 8.2 데이터 품질 이슈 처리

| 이슈 | 처리 방법 |
|------|----------|
| null 값 | 카운트 + 비율 계산, 필터링하지 않음 |
| 빈 문자열 | null과 별도로 카운트 |
| 중복 ID | 탐지만 수행, 제거하지 않음 |
| 인코딩 깨짐 | 패턴 매칭으로 탐지, 보고만 |

---

## 9. Dependencies

### 9.1 추가 의존성 (`pyproject.toml` dev 섹션)

| Package | Version | Purpose |
|---------|---------|---------|
| `ijson` | >=3.2.0 | 대용량 JSON 스트리밍 파싱 |
| `plotly` | >=5.18.0 | 인터랙티브 시각화 |
| `nbformat` | >=5.9.0 | 노트북 포맷 유틸 |
| `tqdm` | >=4.66.0 | 진행률 표시 |
| `kaleido` | >=0.2.1 | plotly 이미지 export |
| `jupyterlab` | >=4.0.0 | 노트북 실행 환경 |

> `pandas`, `numpy`는 이미 main dependencies에 포함

### 9.2 양쪽 섹션 동기화

`[project.optional-dependencies]` dev + `[dependency-groups]` dev 양쪽에 동일하게 추가 (PEP 621 + PEP 735).

---

## 10. Coding Conventions

### 10.1 공유 모듈 규칙

| 규칙 | 적용 |
|------|------|
| 타입 힌트 | 모든 함수에 필수 (PEP 484) |
| Docstring | Google 스타일 (Args, Returns, Raises) |
| 상수 | UPPER_SNAKE_CASE |
| 함수명 | snake_case |
| import 순서 | 표준 → 서드파티 → 로컬 |
| 린트 | `ruff check` 통과 필수 |

### 10.2 노트북 규칙

| 규칙 | 적용 |
|------|------|
| 첫 셀 | 환경 설정 (imports, sys.path, plotly 테마) |
| 마지막 셀 | `save_result()` 호출 |
| 마크다운 | 각 분석 단계 전에 설명 셀 |
| 변수명 | snake_case, 의미 있는 이름 |
| 결과 공유 | `save_result()` / `load_result()` 사용 |

---

## 11. Execution Guide

```bash
cd backend

# 1. 의존성 설치
uv sync --dev

# 2. JupyterLab 실행
uv run jupyter lab

# 3. notebooks/eda/ 폴더에서 순서대로 실행
#    01 → 02 → 03 → 04

# 4. 린트 검증 (공유 모듈)
uv run ruff check backend/scripts/eda/
```

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2026-02-11 | Initial creation (post-implementation) | Claude |
