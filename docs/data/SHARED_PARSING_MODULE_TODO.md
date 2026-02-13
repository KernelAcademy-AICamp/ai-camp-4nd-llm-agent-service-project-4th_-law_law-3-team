# 공유 파싱 모듈 구현 TODO

> EDA 인사이트가 실제 적재 파이프라인에 자동 반영되도록 공유 파싱 모듈을 구축한다.
> 날짜 파싱 4곳 중복, 정규식 2곳 불일치, 카테고리 레지스트리 미활용 문제를 해결한다.

---

## 모듈 구조

```
backend/
├── app/core/
│   └── date_parser.py              # [신규] 날짜 파싱 (app/ + scripts/ 양쪽 사용)
│
└── scripts/parsing/                 # [신규] 스크립트 전용 공유 파싱 모듈
    ├── __init__.py                  # 공개 API re-export
    ├── regex.py                     # 법령 인용, 사건번호, 법령명 정규식
    ├── registry.py                  # 11개 카테고리 필드 레지스트리
    └── text_fields.py               # 카테고리별 텍스트 필드 추출
```

---

## Step 1: 날짜 파싱 모듈

- [ ] `app/core/date_parser.py` 생성
  - 출처: `notebooks/eda/03_temporal_relationships.ipynb` cell 3
  - `DANGI_OFFSET = 2333`, `MIN_VALID_YEAR = 1800`, `MAX_VALID_YEAR = 현재연도+1`
  - `normalize_year(year: int) -> int` — 3000~5000이면 단기→서기
  - `build_valid_date(year, month, day) -> date | None` — normalize + 유효성
  - `parse_date(value: str | None) -> date | None` — YYYYMMDD, YYYY-MM-DD, YYYY.M.D. + 단기 보정
  - `parse_datetime(value: str | None) -> datetime | None` — parse_date 래퍼 (EDA 호환)

## Step 2: 날짜 파싱 테스트

- [ ] `tests/unit/test_date_parser.py` 작성 (~20개)
  - 기본 형식 (YYYYMMDD, ISO, dotted)
  - 단기 보정 (4293→1960)
  - 엣지 케이스 (None, 빈문자열, 잘못된 월/일, 범위 밖 연도)
  - parse_datetime 래퍼

## Step 3: 공유 파싱 패키지 생성

- [ ] `scripts/parsing/__init__.py` — 공개 API re-export
- [ ] `scripts/parsing/regex.py` — `eda/common.py` 395-556행에서 이동
  - 정규식: `_CITATION_BRACKET_RE`, `_CITATION_PLAIN_RE`, `_LAW_NAME_RE`, `_STATUTE_NAME_PLAIN_RE`, `_CASE_NUMBER_RE`
  - 함수: `extract_citations()`, `extract_law_names()`, `extract_case_numbers()`, `extract_statute_names_plain()`
- [ ] `scripts/parsing/registry.py` — `eda/data_registry.py` 전체 이동
  - `CATEGORIES` (11개), `CGM_EXPC_AGENCIES` (27개), `COMMITTEE_AGENCIES` (10개)
  - `get_all_files()`, `get_category_files()`, `get_agency_name()`, `get_total_file_count()`
  - 노트북 03에서 `DATE_FIELD_CANDIDATES`, `resolve_date_field()` 추가
- [ ] `scripts/parsing/text_fields.py` — 레지스트리 기반 텍스트 추출
  - `get_text_fields(category) -> list[str]`
  - `get_id_field(category) -> str`
  - `get_date_field(category) -> str | None`
  - `extract_embedding_text(category, record) -> str`

## Step 4: 파싱 패키지 테스트

- [ ] `tests/unit/test_parsing_regex.py` (~15개) — 인용/사건번호/법령명 추출
- [ ] `tests/unit/test_parsing_registry.py` (~10개) — 카테고리 수, 파일 수, 에이전시명
- [ ] `tests/unit/test_parsing_text_fields.py` (~10개) — 텍스트 추출 결과

## Step 5: 기존 파일 마이그레이션

### 5a. EDA 래퍼 전환

- [ ] `scripts/eda/data_registry.py` → 내용 삭제, `from scripts.parsing.registry import *` re-export
  - 7개 노트북이 `from scripts.eda.data_registry import CATEGORIES` 사용 → 래퍼로 호환 유지

### 5b. EDA 정규식 제거

- [ ] `scripts/eda/common.py` 395-556행 삭제, 상단에 re-export 추가
  - 노트북에서 `from scripts.eda.common import extract_citations` → re-export로 호환 유지

### 5c. 노트북 날짜 함수 교체

- [ ] `notebooks/eda/03_temporal_relationships.ipynb` cell 3 → import로 교체
  - `from app.core.date_parser import parse_date, parse_datetime, normalize_year`
  - `from scripts.parsing.registry import DATE_FIELD_CANDIDATES, resolve_date_field`

### 5d. ORM 모델 _parse_date 위임 (4개)

- [ ] `app/models/precedent_document.py:212` → `app.core.date_parser.parse_date` 호출
- [ ] `app/models/law.py:159` → 동일
- [ ] `app/models/legal_reference.py:231` → 동일
- [ ] `app/models/legal_document.py:339` → 동일

### 5e. law_document 날짜 파싱 위임

- [ ] `app/models/law_document.py:143-149` → `from_json` 내 enforcement_date 파싱을 `parse_date` 호출로 교체

### 5f. Neo4j 그래프 정규식 교체

- [ ] `scripts/build_graph.py:244` `regex_statute` → `from scripts.parsing.regex import STATUTE_NAME_PLAIN_RE`
- [ ] `scripts/build_graph.py:298-302` `regex_case_number` → `from scripts.parsing.regex import CASE_NUMBER_RE`
- 주의: EDA 정규식이 공백 허용(`\s*`)하므로 결과 차이 가능 → Step 7에서 검증

### 5g. RunPod 임베딩 호환

- [ ] `scripts/runpod_lancedb_embeddings.py` → try/except 패턴 (독립 실행 보장)

## Step 6: 문서 업데이트

- [ ] `CLAUDE.md` (루트) — EDA 섹션에 `scripts/parsing/` 설명 추가
- [ ] `backend/scripts/CLAUDE.md` — 공유 파싱 모듈 섹션 추가
- [ ] `docs/data/DATA_CLEANING_RULES.md` — 단기 변환 적용 위치 체크리스트 업데이트 (적용됨 표시)

## Step 7: 검증

- [ ] `uv run ruff check .` — 린트
- [ ] `uv run mypy app/core/date_parser.py` — 타입
- [ ] `uv run pytest tests/unit/test_date_parser.py tests/unit/test_parsing_regex.py tests/unit/test_parsing_registry.py tests/unit/test_parsing_text_fields.py -v` — 신규 테스트
- [ ] `uv run pytest tests/unit/ -v` — 기존 테스트 회귀
- [ ] `NEO4J_PASSWORD=password uv run python scripts/verify_graph.py` — 그래프 정규식 변경 영향 확인

---

## 변경 파일 요약

| 구분 | 파일 | 변경 |
|------|------|------|
| 신규 | `app/core/date_parser.py` | 날짜 파싱 통합 |
| 신규 | `scripts/parsing/__init__.py` | 공개 API |
| 신규 | `scripts/parsing/regex.py` | 정규식 통합 |
| 신규 | `scripts/parsing/registry.py` | 카테고리 레지스트리 |
| 신규 | `scripts/parsing/text_fields.py` | 텍스트 추출 |
| 신규 | `tests/unit/test_date_parser.py` | 날짜 테스트 |
| 신규 | `tests/unit/test_parsing_regex.py` | 정규식 테스트 |
| 신규 | `tests/unit/test_parsing_registry.py` | 레지스트리 테스트 |
| 신규 | `tests/unit/test_parsing_text_fields.py` | 텍스트 추출 테스트 |
| 수정 | `scripts/eda/data_registry.py` | 래퍼로 전환 |
| 수정 | `scripts/eda/common.py` | 정규식 제거 + re-export |
| 수정 | `notebooks/eda/03_temporal_relationships.ipynb` | cell 3 import 교체 |
| 수정 | `app/models/precedent_document.py` | _parse_date 위임 |
| 수정 | `app/models/law.py` | _parse_date 위임 |
| 수정 | `app/models/legal_reference.py` | _parse_date 위임 |
| 수정 | `app/models/legal_document.py` | _parse_date 위임 |
| 수정 | `app/models/law_document.py` | enforcement_date 위임 |
| 수정 | `scripts/build_graph.py` | 정규식 import |
| 수정 | `scripts/runpod_lancedb_embeddings.py` | try/except |
| 수정 | 문서 2-3개 | CLAUDE.md 등 |

**총**: 신규 9파일, 수정 12-13파일

---

*생성일: 2026-02-13*
