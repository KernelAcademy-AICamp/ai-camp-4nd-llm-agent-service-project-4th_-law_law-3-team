# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start (새 환경 세팅)

### 전제 조건

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (패키지 매니저)
- Docker & Docker Compose

### 1. 의존성 설치

```bash
cd backend
uv sync --dev
```

### 2. 환경 변수 설정

```bash
cp .env.example .env
# .env 파일 편집 (API 키 등 설정)
```

### 3. PostgreSQL 실행

```bash
# 프로젝트 루트에서
docker-compose up -d postgres

# 연결 확인
docker logs law-platform-db
```

### 4. DB 마이그레이션

```bash
cd backend
uv run alembic upgrade head
```

### 5. 데이터 로드 (PostgreSQL)

```bash
# data/ 폴더에 법령/판례 JSON 파일 필요
# - data/law_cleaned.json
# - data/precedents_cleaned.json

uv run python scripts/load_lancedb_data.py --type all
```

### 6. LanceDB 데이터

```bash
# 옵션 A: 기존 lancedb_data/ 폴더 복사 (권장 - 빠름)
# 다른 팀원에게서 lancedb_data.zip 받아서 압축 해제

# 옵션 B: 임베딩 직접 생성 (GPU 필요, 시간 오래 걸림)
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv run --no-sync python scripts/runpod_lancedb_embeddings.py --type all
```

### 7. 임베딩 모델 다운로드 ⚠️ 중요

검색 API를 사용하려면 **반드시 임베딩 모델을 먼저 다운로드**해야 합니다.

```bash
# 모델 다운로드 (약 2.3GB, 네트워크 상태에 따라 시간 소요)
uv run python scripts/download_models.py

# 캐시 상태만 확인
uv run python scripts/download_models.py --check
```

> **참고**: 서버 시작 시 모델이 없으면 경고만 표시하고 서버는 실행됩니다.
> 단, 검색 API 호출 시 503 에러가 반환됩니다.

### 8. 서버 실행

```bash
uv run uvicorn app.main:app --reload
# http://localhost:8000/docs 에서 API 문서 확인
```

### 9. 데이터 확인 (선택)

```bash
# PostgreSQL 데이터 확인
uv run python tests/integration/test_postgresql_data.py

# LanceDB 검색 테스트
uv run python tests/integration/test_lancedb_search.py
```

---

## Commands

```bash
uv sync                              # 의존성 설치
uv sync --dev                        # 개발 의존성 포함 (pytest, ruff, mypy)
uv run uvicorn app.main:app --reload # 개발 서버 실행
uv run pytest                        # 전체 테스트
uv run pytest tests/test_file.py::test_name  # 단일 테스트
uv run ruff check .                  # 린트
uv run ruff check . --fix            # 린트 자동 수정
uv run mypy .                        # 타입 체크
uv add <package>                     # 패키지 추가
uv add --dev <package>               # 개발 패키지 추가
```

## Architecture

### 모듈 자동 등록

`app/core/registry.py`의 `ModuleRegistry`가 서버 시작 시 `app/modules/` 폴더를 스캔하여 각 모듈의 라우터를 자동 등록합니다.

모듈이 등록되려면:
1. `app/modules/<module_name>/` 폴더 존재
2. `router/__init__.py`에 `router = APIRouter()` 정의

### 모듈 구조

```
app/modules/<module_name>/
├── __init__.py
├── router/
│   └── __init__.py    # router = APIRouter() 필수
├── service/
│   └── __init__.py    # 비즈니스 로직
├── schema/
│   └── __init__.py    # Pydantic 모델 (request/response)
└── model/
    └── __init__.py    # SQLAlchemy 모델
```

### API 경로 규칙

모듈명 `snake_case` → API 경로 `/api/kebab-case`
- `lawyer_finder` → `/api/lawyer-finder`
- `small_claims` → `/api/small-claims`

### 설정

`app/core/config.py`에서 pydantic-settings 사용. `.env` 파일에서 환경변수 로드.

```python
from app.core.config import settings
settings.DATABASE_URL
settings.ENABLED_MODULES  # 빈 리스트면 모든 모듈 활성화
settings.VECTOR_DB        # lancedb | chroma | qdrant
```

### 주요 환경 변수 (`.env`)

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `DATABASE_URL` | PostgreSQL 연결 문자열 | - |
| `VECTOR_DB` | 벡터 DB 선택 | `lancedb` |
| `LANCEDB_URI` | LanceDB 저장 경로 | `./lancedb_data` |
| `LANCEDB_TABLE_NAME` | LanceDB 테이블명 | `legal_chunks` |
| `USE_LOCAL_EMBEDDING` | 로컬 임베딩 사용 | `true` |
| `LOCAL_EMBEDDING_MODEL` | 임베딩 모델 | `nlpai-lab/KURE-v1` |
| `UPSTAGE_API_KEY` | Solar API 키 | - |
| `UPSTAGE_MODEL` | Solar 모델명 | `solar-pro3-260126` |

자세한 설정은 `.env.example` 참조.

## Conventions

- 라우터 함수는 `async def` 사용
- Pydantic v2 문법 사용 (`model_validator`, `field_validator`)
- 타입 힌트 필수 (mypy strict 모드)
- ruff 린터 규칙: E, F, I, N, W

## Tests

### 테스트 폴더 구조

```
backend/tests/
├── __init__.py
├── conftest.py                      # 공통 픽스처
├── integration/                     # 통합 테스트 (DB, 외부 서비스 연동)
│   ├── __init__.py
│   ├── test_lancedb_search.py       # LanceDB 벡터 검색 테스트
│   ├── test_postgresql_data.py      # PostgreSQL 데이터 확인
│   ├── test_neo4j_graph.py          # Neo4j 그래프 검증 테스트 (27개)
│   ├── test_evaluation_runner.py    # 평가 실행기 테스트
│   └── test_evaluation_search.py    # 평가 검색 테스트
├── unit/                            # 단위 테스트 (개별 함수/클래스)
│   ├── __init__.py
│   ├── test_evaluation_metrics.py   # 메트릭 계산 테스트 (31개)
│   ├── test_evaluation_schemas.py   # 스키마 검증 테스트 (21개)
│   └── test_evaluation_dataset_builder.py  # 데이터셋 빌더 테스트 (14개)
└── e2e/                             # E2E 테스트 (API 엔드포인트)
    └── __init__.py
```

### 테스트 실행

```bash
# 전체 테스트
uv run pytest

# 특정 폴더 테스트
uv run pytest tests/integration/
uv run pytest tests/unit/

# 특정 파일 테스트
uv run pytest tests/integration/test_lancedb_search.py

# 특정 테스트 함수
uv run pytest tests/integration/test_lancedb_search.py::test_lancedb_search

# 직접 실행 (pytest 없이)
uv run python tests/integration/test_lancedb_search.py
uv run python tests/integration/test_postgresql_data.py

# 상세 출력
uv run pytest -v tests/integration/

# 마커별 테스트 (pyproject.toml 참조)
uv run pytest -m "not slow"              # 느린 테스트 제외
uv run pytest -m "not requires_lancedb"  # LanceDB 없이 실행
uv run pytest -m "not requires_postgres" # PostgreSQL 없이 실행
```

### 테스트 작성 규칙

1. **파일명**: `test_*.py` 또는 `*_test.py`
2. **함수명**: `test_` 접두사 필수
3. **경로 설정**: 프로젝트 루트 import 필요 시
   ```python
   import sys
   from pathlib import Path
   PROJECT_ROOT = Path(__file__).parent.parent.parent
   sys.path.insert(0, str(PROJECT_ROOT))
   ```
4. **테스트 유형별 폴더**:
   - `integration/` - DB, 외부 API 연동 테스트
   - `unit/` - 순수 함수, 클래스 테스트
   - `e2e/` - FastAPI 엔드포인트 테스트

## PostgreSQL

### 실행 (Docker)

```bash
# 프로젝트 루트에서 실행
cd ..
docker-compose up -d postgres

# 상태 확인
docker ps
docker logs law-platform-db
```

### .env 설정

```bash
# backend/.env
DATABASE_URL=postgresql://lawuser:lawpassword@localhost:5432/lawdb
```

### 마이그레이션 (Alembic)

```bash
cd backend

# 마이그레이션 실행 (테이블 생성)
uv run alembic upgrade head

# 마이그레이션 상태 확인
uv run alembic current

# 새 마이그레이션 생성
uv run alembic revision -m "add_new_table"

# 롤백
uv run alembic downgrade -1
```

### 데이터 로드

```bash
# 법령 데이터 로드 (data/law_cleaned.json → PostgreSQL)
uv run python scripts/load_lancedb_data.py --type law

# 판례 데이터 로드 (data/precedents_cleaned.json → PostgreSQL)
uv run python scripts/load_lancedb_data.py --type precedent

# 전체 로드 (법령 + 판례)
uv run python scripts/load_lancedb_data.py --type all

# 기존 데이터 삭제 후 재로드
uv run python scripts/load_lancedb_data.py --type all --reset
```

### 모델 파일 위치

```
app/models/
├── __init__.py
├── law_document.py        # 법령 원본 (LanceDB 연동)
├── precedent_document.py  # 판례 원본 (LanceDB 연동)
├── legal_document.py      # 법률 문서 (일반)
└── legal_reference.py     # 참조 정보
```

### 테이블 구조

| 테이블 | 설명 | 주요 컬럼 |
|--------|------|-----------|
| `law_documents` | 법령 원본 | law_id, law_name, content, raw_data |
| `precedent_documents` | 판례 원본 | serial_number, case_name, ruling, reasoning |

### 데이터 조회 예시

```python
from sqlalchemy import select
from app.common.database import async_session_factory
from app.models.precedent_document import PrecedentDocument

async with async_session_factory() as session:
    # serial_number로 조회 (LanceDB source_id와 매핑)
    result = await session.execute(
        select(PrecedentDocument).where(
            PrecedentDocument.serial_number == "76396"
        )
    )
    precedent = result.scalar_one_or_none()
    print(precedent.ruling)  # 주문
    print(precedent.reasoning)  # 판결요지
```

## Vector DB (LanceDB)

### 임베딩 스크립트

```bash
# PyTorch CUDA 설치 (환경에 맞게 선택)
uv pip install --reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 임베딩 생성 (--no-sync 필수: torch 버전 유지)
uv run --no-sync python scripts/runpod_lancedb_embeddings.py --type precedent
uv run --no-sync python scripts/runpod_lancedb_embeddings.py --type law
uv run --no-sync python scripts/runpod_lancedb_embeddings.py --type all --reset

# 통계 확인
uv run --no-sync python scripts/runpod_lancedb_embeddings.py --stats
```

### 저장 위치

```
backend/
├── lancedb_data/           # LanceDB 데이터
│   └── legal_chunks.lance/ # 법령 + 판례 통합 테이블
└── scripts/
    ├── runpod_lancedb_embeddings.py  # 메인 임베딩 스크립트
    └── CLAUDE.md                      # 스크립트 상세 가이드
```

### 핵심 클래스

```python
# 통합 임베딩 프로세서
from scripts.runpod_lancedb_embeddings import (
    StreamingEmbeddingProcessor,  # 추상 베이스
    LawEmbeddingProcessor,        # 법령 임베딩
    PrecedentEmbeddingProcessor,  # 판례 임베딩
    EmbeddingCache,               # 임베딩 캐싱
    EmbeddingQualityChecker,      # 품질 검증
)

# 사용 예시
processor = PrecedentEmbeddingProcessor()
stats = processor.run("data.json", reset=True, batch_size=100)
```

### 임베딩 캐싱

```python
from scripts.runpod_lancedb_embeddings import EmbeddingCache, create_embeddings

cache = EmbeddingCache("./embedding_cache")
embedding = cache.get_or_compute("텍스트", create_embeddings)
print(cache.get_stats())  # {'hits': 10, 'misses': 5, 'hit_rate': '66.7%'}
```

### 품질 검증

```python
from scripts.runpod_lancedb_embeddings import EmbeddingQualityChecker

checker = EmbeddingQualityChecker()
report = checker.quick_test()  # 법률 도메인 기본 테스트
# Quality: GOOD (separation > 0.2)
```

### 검색 예시

```python
import lancedb
from scripts.runpod_lancedb_embeddings import get_embedding_model

model = get_embedding_model('cuda')
query_vector = model.encode('손해배상 책임')

db = lancedb.connect('./lancedb_data')
table = db.open_table('legal_chunks')

results = table.search(query_vector).metric('cosine').limit(10).to_pandas()
```

### 주의사항

1. **torch는 pyproject.toml에 없음** - 환경별로 수동 설치
2. **--no-sync 필수** - `uv run --no-sync`로 실행
3. **GPU 자동 감지** - VRAM에 따라 batch_size 자동 설정

## Embedding Model (임베딩 모델)

검색 API는 쿼리를 벡터로 변환하기 위해 **임베딩 모델**이 필요합니다.

### 모델 정보

| 항목 | 값 |
|------|-----|
| 모델명 | `nlpai-lab/KURE-v1` |
| 크기 | 약 2.3GB |
| 차원 | 1024 |
| 캐시 경로 | `backend/data/models/` |

### 모델 다운로드

```bash
cd backend

# 모델 다운로드 (네트워크 상태에 따라 시간 소요)
uv run python scripts/download_models.py

# 캐시 상태만 확인
uv run python scripts/download_models.py --check

# 재다운로드 (기존 캐시 무시)
uv run python scripts/download_models.py --force

# 다른 모델 다운로드
uv run python scripts/download_models.py --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

### 서버 동작 방식

| 상태 | 서버 시작 | 검색 API |
|------|----------|----------|
| 모델 캐시됨 | ✓ 정상 | ✓ 정상 |
| 모델 미캐시 | ⚠️ 경고 후 시작 | ❌ 503 에러 |

**서버 시작 시 경고 예시** (모델 미캐시):
```
============================================================
[WARNING] 임베딩 모델이 캐시되지 않았습니다.
모델명: nlpai-lab/KURE-v1
검색 API 사용 전 먼저 모델을 다운로드해주세요:
  uv run python scripts/download_models.py
============================================================
```

### 관련 코드

| 파일 | 설명 |
|------|------|
| `app/common/chat_service.py` | `check_embedding_model_availability()`, `get_local_model()` |
| `scripts/download_models.py` | 모델 다운로드 CLI |
| `app/main.py` | lifespan에서 시작 시 체크 |

### 환경 변수

```bash
# backend/.env
USE_LOCAL_EMBEDDING=true              # 로컬 임베딩 사용 (기본값: true)
LOCAL_EMBEDDING_MODEL=nlpai-lab/KURE-v1  # 임베딩 모델명
```

> **팁**: `USE_LOCAL_EMBEDDING=false`로 설정하면 OpenAI 임베딩을 사용하며,
> 이 경우 로컬 모델 다운로드가 필요 없습니다 (단, `OPENAI_API_KEY` 필요).

## Graph DB (Neo4j)

법령 계급(시행령→법률), 판례 인용 관계를 Neo4j 그래프로 저장합니다.

### 실행 (Docker)

```bash
# 프로젝트 루트에서 실행
cd ..
docker compose up -d neo4j

# 상태 확인
docker ps
docker logs neo4j-law-graph

# Neo4j Browser 접속
# http://localhost:7474 (neo4j / password)
```

### .env 설정

```bash
# backend/.env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

### 그래프 구축

```bash
cd backend

# 전체 그래프 구축 (초기 1회)
uv run python scripts/build_graph.py

# 구축 내용:
# - Statute 노드 (법령): 5,572개
# - Case 노드 (판례): 65,107개
# - Alias 노드 (비공식 약칭): 69개
# - HIERARCHY_OF 관계 (법령 계급): 3,624개
# - CITES 관계 (판례→법령): 72,414개
# - CITES_CASE 관계 (판례→판례): 87,654개
# - RELATED_TO 관계 (법령→법령): 93개
# - ALIAS_OF 관계 (약칭→법령): 69개
```

### 검증

```bash
# CLI 검증 (통계, 샘플 경로)
uv run python scripts/verify_graph.py

# Gradio UI 검증 (약칭 통합 검색 지원)
uv run python scripts/verify_gradio.py
# → http://localhost:7860

# 테스트 실행 (30개 테스트)
uv run pytest tests/integration/test_neo4j_graph.py -v
```

### 그래프 스키마

```
노드 (Nodes):
- Statute: id, name, type, promulgation_date, abbreviation, citation_count
- Case: id, case_number, name, summary
- Alias: name, category (비공식 약칭)

관계 (Relationships):
- (Statute)-[:HIERARCHY_OF]->(Statute)  # 시행령 → 법률
- (Case)-[:CITES]->(Statute)             # 판례 → 법령 인용
- (Case)-[:CITES_CASE]->(Case)           # 판례 → 판례 인용
- (Statute)-[:RELATED_TO]->(Statute)     # 법령 → 법령 관련
- (Alias)-[:ALIAS_OF]->(Statute)         # 비공식 약칭 → 법령
```

### 약칭 검색

세 가지 방식으로 법령을 검색할 수 있습니다:
- **정식 법령명**: `민사소송법`, `도로교통법`
- **공식 약칭** (`lsAbrv.json`): `119법`, `특정범죄가중법`
- **비공식 약칭** (`informal_abbreviations.json`): `민소법`, `도교법`, `특가법`

```cypher
-- 통합 검색 (정식명/공식약칭/비공식약칭)
OPTIONAL MATCH (s1:Statute {name: $query})
OPTIONAL MATCH (s2:Statute {abbreviation: $query})
OPTIONAL MATCH (a:Alias {name: $query})-[:ALIAS_OF]->(s3:Statute)
WITH coalesce(s1, s2, s3) as s WHERE s IS NOT NULL
RETURN s
```

### 성능 최적화

인덱스 및 Full-text 검색이 적용되어 있습니다:

| 인덱스 | 용도 |
|--------|------|
| `Statute.id`, `Statute.name` | 기본 조회 |
| `Statute.abbreviation` | 공식 약칭 검색 |
| `Case.id`, `Case.case_number`, `Case.name` | 판례 조회 |
| `Alias.name` | 비공식 약칭 검색 |
| `ft_statute_search` | 법령 Full-text 검색 |
| `ft_case_search` | 판례 Full-text 검색 |
| `ft_alias_search` | 약칭 Full-text 검색 |

**성능 벤치마크:**

| 쿼리 | 응답 시간 |
|------|-----------|
| ID/이름 조회 | 3-14ms |
| 약칭 통합 검색 | 9ms |
| 계급 탐색 | 17-33ms |
| 인용 법령 TOP 10 | 13ms |
| 유사 판례 검색 | 35ms |

### Cypher 쿼리 예시

```cypher
-- 특정 법령의 상하위 계급 조회
MATCH (s:Statute {name: '도로교통법'})
OPTIONAL MATCH (s)-[:HIERARCHY_OF]->(upper)
OPTIONAL MATCH (lower)-[:HIERARCHY_OF]->(s)
RETURN s.name, collect(upper.name) as 상위법, collect(lower.name) as 하위법

-- 비공식 약칭으로 법령 검색
MATCH (a:Alias {name: '민소법'})-[:ALIAS_OF]->(s:Statute)
RETURN s.name, s.abbreviation

-- Full-text 법령 검색
CALL db.index.fulltext.queryNodes("ft_statute_search", "도로교통")
YIELD node RETURN node LIMIT 10

-- Full-text 판례 검색
CALL db.index.fulltext.queryNodes("ft_case_search", "손해배상")
YIELD node RETURN node LIMIT 10

-- 가장 많이 인용된 법령 TOP 10 (최적화: citation_count 사용)
MATCH (s:Statute) WHERE s.citation_count > 0
RETURN s.name, s.citation_count
ORDER BY s.citation_count DESC LIMIT 10

-- 같은 법령을 인용한 유사 판례
MATCH (c1:Case {id: $id})-[:CITES]->(s:Statute)<-[:CITES]-(c2:Case)
WHERE c1 <> c2
RETURN c2.case_number, count(s) as common
ORDER BY common DESC LIMIT 10
```

### 스크립트 파일

| 파일 | 설명 |
|------|------|
| `scripts/build_graph.py` | 그래프 구축 (법령, 판례, 관계) |
| `scripts/verify_graph.py` | CLI 검증 (통계, 샘플) |
| `scripts/verify_gradio.py` | Gradio UI 검증 |

### Python API 사용

```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "password")
)

with driver.session() as session:
    # 법령 계급 조회
    result = session.run("""
        MATCH (s:Statute {name: $name})-[:HIERARCHY_OF]->(upper)
        RETURN upper.name
    """, name="도로교통법 시행령")

    for record in result:
        print(record["upper.name"])

driver.close()
```

### 활용 시나리오

| 시나리오 | 관계 | 설명 |
|----------|------|------|
| RAG 컨텍스트 보강 | HIERARCHY_OF, CITES, RELATED_TO | 검색 결과에 관련 법령/판례 추가 |
| 법령 탐색 UI | HIERARCHY_OF, RELATED_TO | 계급도 시각화, 관련 법령 탐색 |
| 판례 추천 | CITES_CASE, CITES | 유사 판례 찾기 (같은 법령 인용) |
