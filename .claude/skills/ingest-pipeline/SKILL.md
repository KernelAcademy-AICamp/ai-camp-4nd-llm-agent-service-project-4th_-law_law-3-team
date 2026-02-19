---
name: ingest-pipeline
description: Config-driven 인제스트 파이프라인. 19개 데이터 타입의 DB 적재, 벡터 임베딩, FTS 빌드, ANN 인덱스 일괄 처리. 데이터 임베딩, 벡터 DB 구축, 데이터 적재 시 사용.
---

# 인제스트 파이프라인 스킬

Config-driven 인제스트 파이프라인. 19개 데이터 타입의 DB 적재, 벡터 임베딩, FTS 빌드, ANN 인덱스를 일괄 처리합니다.

> **중요**: 벡터 DB 임베딩 작업은 이 파이프라인(`scripts/ingest/`)이 기준입니다.
> `scripts/runpod_lancedb_embeddings.py`, `scripts/colab_lancedb_embeddings.py`는 노트북용 **thin wrapper** (내부적으로 이 파이프라인 호출)입니다.
> 기존 독자 구현 스크립트(`local_lancedb_embeddings.py`, `create_lancedb_embeddings.py` 등)는 **삭제**되었습니다.

## 개요

| 항목 | 내용 |
|------|------|
| CLI 진입점 | `cd backend && uv run python -m scripts.ingest.cli` |
| 데이터 타입 | 19개 (법령, 판례, 행정규칙, 헌재결정례, 조약, 위원회 결정문 등) |
| 총 건수 | ~423,924건 |
| 설정 방식 | `scripts/ingest/types/` 하위에 타입별 설정 파일 |
| 소스 경로 | `scripts/ingest/sources.yaml` (YAML 중앙 관리) |

## CLI 사용법

```bash
cd backend

# 전체 타입 × 전체 파이프라인 (최초 적재)
uv run python -m scripts.ingest.cli --type all --step all --reset

# 특정 타입만
uv run python -m scripts.ingest.cli --type precedent --step all --reset

# 단계별 실행
uv run python -m scripts.ingest.cli --type precedent --step db       # PostgreSQL + FTS
uv run python -m scripts.ingest.cli --type precedent --step vector   # LanceDB 벡터
uv run python -m scripts.ingest.cli --type precedent --step fts      # FTS만 재빌드
uv run python -m scripts.ingest.cli --type precedent --step index    # ANN 인덱스만

# 통계 / 검증
uv run python -m scripts.ingest.cli --type all --stats
uv run python -m scripts.ingest.cli --type all --verify
```

### 주요 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--type` | `all` 또는 19개 타입명 | (필수) |
| `--step` | `all`, `db`, `vector`, `fts`, `index` | `all` |
| `--reset` | 기존 데이터 삭제 후 재실행 | `false` |
| `--source` | 커스텀 JSON 소스 (단일 타입만) | 타입별 기본 |
| `--batch-size` | 배치 크기 | DB: 1000, 벡터: 자동 |
| `--device` | `cuda`, `mps`, `cpu` | 자동 감지 |
| `--profile` | `desktop`, `laptop`, `mac`, `cpu` | 자동 감지 |
| `--no-cache` | 임베딩 캐시 비활성화 | `false` |

## 단계별 설명

| 단계 | 설명 | 의존성 |
|------|------|--------|
| `db` | JSON → PostgreSQL ORM + FTS tsvector 동시 적재 | PostgreSQL, Alembic, MeCab |
| `vector` | JSON → LanceDB 벡터 임베딩 (1문서=1벡터, 요약문 기반) | 임베딩 모델, PyTorch |
| `fts` | PostgreSQL ORM에서 읽어 tsvector만 재빌드 | PostgreSQL, MeCab |
| `index` | LanceDB ANN 인덱스 재빌드 (IVF_FLAT) | `vector` 완료 |
| `all` | `db` → `vector` → `index` 순차 실행 | 전체 |

## 핵심 파일 경로

```
scripts/ingest/
├── cli.py              # CLI 진입점
├── config.py           # IngestConfig dataclass + 레지스트리
├── sources.yaml        # 19개 타입 소스 경로 (YAML)
├── db_writer.py        # PostgreSQL + FTS 적재
├── vector_writer.py    # LanceDB 벡터 임베딩
├── fts_builder.py      # FTS tsvector 재빌드
├── shared.py           # 공유 유틸 (토크나이저, FTS 배치)
├── ingest.md           # 19개 타입 저장 구조 상세 문서
└── types/              # 데이터 타입별 설정 (19개)
    ├── __init__.py     # 자동 등록
    ├── _template.py    # 신규 타입 템플릿
    ├── law.py          # 법령
    ├── precedent.py    # 판례
    └── ...             # 나머지 17개 타입

scripts/embedding_common/  # 임베딩 공통 모듈 (store, model, cache 등)
```

## 19개 데이터 타입

| `--type` | 데이터 | 건수 | 소스 형식 |
|----------|--------|------|----------|
| `law` | 법령 | 5,548 | 단일 JSON |
| `precedent` | 판례 | 92,055 | 단일 JSON |
| `admin_rule` | 행정규칙 | 5,258 | 단일 JSON |
| `constitutional` | 헌재결정례 | 31,718 | 단일 JSON |
| `administration` | 행정심판례 | 34,254 | 단일 JSON |
| `legislation` | 법령해석례 | 8,597 | 단일 JSON |
| `treaty` | 조약 | 3,589 | 단일 JSON |
| `interpretation_ministry` | 부처해석례 | 37,325 | 디렉토리 (28개) |
| `special_admin_appeal` | 특별행정심판례 | 148,778 | 디렉토리 (2개) |
| `dec_privacy` | 개인정보보호위원회 | 1,448 | 개별 JSON |
| `dec_employment` | 고용보험심사위원회 | 118 | 개별 JSON |
| `dec_fair_trade` | 공정거래위원회 | ~7,728 | 개별 JSON |
| `dec_human_rights` | 국가인권위원회 | 3,721 | 개별 JSON |
| `dec_civil_rights` | 국민권익위원회 | 635 | 개별 JSON |
| `dec_financial` | 금융위원회 | 662 | 개별 JSON |
| `dec_labor` | 노동위원회 | 40,714 | 개별 JSON |
| `dec_industrial` | 산업재해보상보험재심사위원회 | 782 | 개별 JSON |
| `dec_environment` | 중앙환경분쟁조정위원회 | 358 | 개별 JSON |
| `dec_securities` | 증권선물위원회 | 636 | 개별 JSON |

## reset 동작

- `--step db --reset`: 해당 타입의 ORM 레코드 + FTS 레코드 삭제 후 재적재
- `--step vector --reset`: 해당 타입의 LanceDB 레코드만 삭제 후 재임베딩 (다른 타입 데이터 보존)
- `--step fts --reset`: 해당 타입의 FTS tsvector만 재빌드

## 새 타입 추가 패턴

1. ORM 모델 생성 (`app/models/ingest/`)
2. `app/models/ingest/__init__.py` + `app/models/__init__.py`에 import 추가
3. `alembic/env.py`에 import 추가
4. Alembic 마이그레이션 작성
5. `scripts/ingest/sources.yaml`에 소스 경로 등록
6. `scripts/ingest/types/new_type.py` 생성 (`_template.py` 복사)
7. `data/`에 JSON 소스 배치

## 주의사항

- `scripts/runpod_lancedb_embeddings.py`, `scripts/colab_lancedb_embeddings.py`는 노트북용 thin wrapper (ingest 파이프라인 호출)
- `--type all --reset`은 각 타입별로 해당 타입 레코드만 삭제 (전체 테이블 DROP 아님)
- 벡터 단계는 `uv run --no-sync` 불필요 (CLI가 자동 처리)
- ANN 인덱스(`--step index`)는 전체 LanceDB 대상이므로 마지막에 1회만 실행
