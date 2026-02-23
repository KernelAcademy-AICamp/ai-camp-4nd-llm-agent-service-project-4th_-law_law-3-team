# 변호사시험 Markdown 데이터 활용 구현 계획

## 요약

목표는 `변호사시험` Markdown 원문을 반복 수집 가능한 형태로 정리하고, 이 프로젝트의 기존 JSON 기반 ingest 파이프라인에 연결할 수 있도록 준비하는 것입니다.

1차 범위는 구현 준비 단계로 제한합니다.

- 원문 보관 전략 확정
- 정규화 JSON 스키마 설계
- ingest용 평탄화 스키마 설계
- 파서/ORM/ingest 타입/마이그레이션/테스트 작업 계획 정의

## 배경 및 탐색 결과

- 현재 `backend/scripts/ingest`는 `IngestConfig` 기반 확장 구조를 사용하며 새 타입 추가가 가능함
- `backend/scripts/common/json_loader.py` 및 `backend/scripts/ingest/db_writer.py`는 JSON 파일/디렉토리 입력을 전제로 함
- 따라서 Markdown 원문은 ingest에 직접 넣기보다 `markdown -> normalized JSON -> ingest JSON` 전처리 레이어가 필요함
- RAG 검색은 `data_type` 확장이 가능하지만 실제 원문 조회는 `backend/app/services/rag/retrieval.py`의 registry 등록이 필요함

## 1차 범위 (확정)

- 포함:
  - 원문 markdown 저장 규칙 정리
  - 정규화/평탄화 스키마 설계
  - 전처리 파서 설계
  - 신규 ORM/ingest 타입/Alembic 작업 항목 정의
  - 테스트/검증 기준 정의
- 제외:
  - 실제 파서 구현
  - 실제 DB 마이그레이션 적용
  - RAG registry 연결
  - 챗봇/프론트 노출

## 디렉토리 및 명명 규칙

- 원문 markdown: `data/lawyer_exam_raw/`
- 정규화 JSON: `data/lawyer_exam_normalized/`
- ingest용 JSON: `data/lawyer_exam_ingest/`

ingest 타입명:
- `lawyer_exam`

검색/FTS `data_type_label`:
- `변호사시험`

## 저장 전략 (확정)

- 원문 markdown + 정규화 JSON을 함께 보관
- 원문은 재파싱/검증 기준(SSOT)으로 유지
- 정규화 JSON은 분석/후속 파생 생성의 기준으로 사용
- ingest JSON은 검색 파이프라인 투입용으로 사용

## 정규화 스키마 (계층형 JSON, 분석 중심)

파일 단위 = 시험 1회 + 과목/유형 문서 1개

권장 필드:

- `schema_version`
- `source_file`
- `source_path`
- `exam_type` (`lawyer_exam`)
- `exam_year`
- `exam_round`
- `subject_group` (예: `민사법`)
- `paper_type` (예: `기록형`)
- `title_raw`
- `language` (`ko`)
- `parsed_at`
- `full_text`
- `sections` (배열)
- `parse_warnings` (배열)

`sections[]` 권장 필드:

- `section_id` (예: `sec_001`)
- `heading`
- `heading_level`
- `order`
- `content_text`
- `content_markdown`
- `bullet_count`
- `char_count`
- `tags` (초기 빈 배열)

## ingest용 평탄화 스키마 (검색 중심)

레코드 단위:
- `document` 1건
- `section` N건

공통 필드:

- `record_id`
- `document_id`
- `unit_type` (`document` | `section`)
- `title`
- `content`
- `ai_summary` (1차는 `null`)
- `exam_year`
- `exam_round`
- `subject_group`
- `paper_type`
- `source_file`
- `ingest_version` (`v1`)

section 전용 필드:

- `section_heading`
- `section_order`

ID 규칙(길이 100자 이내 권장):

- 문서: `le_<year>_<round>_<subject>_<paper>_doc`
- 섹션: `le_<year>_<round>_<subject>_<paper>_s001`

## 구현 단계 계획

### 1. 데이터 레이아웃 정리

- `data/lawyer_exam_raw/`, `data/lawyer_exam_normalized/`, `data/lawyer_exam_ingest/` 사용
- 기존 수집 파일을 `lawyer_exam_raw`로 정리

### 2. Markdown 파서 추가 (`backend/scripts/parse_lawyer_exam_markdown.py`)

기능:
- 제목/과목 줄에서 메타데이터 추출
- Markdown 헤더 기준 섹션 분할
- 정규화 JSON 생성
- ingest용 평탄화 JSON 생성
- deterministic `record_id` 생성

예외 처리:
- 메타 파싱 실패 시 `parse_warnings`에 기록
- 헤더 없는 문서는 단일 섹션으로 fallback
- 원문 줄바꿈은 1차에서 최대한 보존

### 3. ORM 모델 추가 (`lawyer_exam_documents`)

권장 컬럼:
- `record_id` (unique)
- `document_id`
- `unit_type`
- `title`
- `content`
- `ai_summary`
- `exam_year`
- `exam_round`
- `subject_group`
- `paper_type`
- `section_heading`
- `section_order`
- `source_file`
- `created_at`, `updated_at`

### 4. Alembic 마이그레이션 추가

- `lawyer_exam_documents` 테이블 생성
- 주요 인덱스 추가:
  - `record_id` (unique)
  - `document_id`
  - `unit_type`
  - `exam_year`
  - `exam_round`
  - `subject_group`
  - `paper_type`

### 5. ingest 타입 추가 (`backend/scripts/ingest/types/lawyer_exam.py`)

`IngestConfig` 기준:
- `name="lawyer_exam"`
- `data_type_label="변호사시험"`
- `id_field="record_id"`
- `summary_field="content"` (1차)
- `title_field="title"`

구현 함수:
- `orm_factory_fn`
- `vector_metadata_fn`
- `fulltext_fn`
- `fts_metadata_fn`
- ORM 기반 FTS 함수 2개

### 6. `sources.yaml` 등록

예시:
- `lawyer_exam: lawyer_exam_ingest/lawyer_exam_sections_v1.json`

### 7. 테스트 보강

신규:
- `backend/tests/unit/test_lawyer_exam_parser.py`

기존 영향:
- `backend/tests/unit/test_ingest_pipeline.py`의 등록 타입 수 증가 반영

## 후속 단계 (2차)

- `backend/app/services/rag/retrieval.py`의 `DOCUMENT_TABLE_REGISTRY`에 `변호사시험` 추가
- RAG/챗봇에서 `변호사시험` 데이터 조회 연결
- 문항 번호/소문항 단위 파싱 확장
- 정답/해설 데이터가 추가되면 스키마 확장

## 테스트 및 검증 기준

1. 샘플 markdown에서 메타(`exam_year`, `exam_round`, `subject_group`, `paper_type`) 추출 가능
2. 섹션이 1개 이상 생성됨
3. ingest용 `document + section` 레코드 생성됨
4. `record_id`가 deterministic하고 중복 없음
5. 향후 ingest 타입 등록 시 `list_configs()`에 `lawyer_exam` 포함

## 가정 및 기본값

- 수집되는 Markdown 형식은 크게 바뀌지 않음 (헤더/제목 구조는 유사)
- 1차에서는 문항 번호 파싱을 하지 않고 섹션 단위까지만 구조화
- `ai_summary`는 생성하지 않음 (`null`)
- `.claude`/기존 ingest 규칙을 재사용하고 새 타입만 추가하는 방향을 기본으로 함
