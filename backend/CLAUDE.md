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

### 2-1. 환경 검증 (선택, 권장)

```bash
# 필수 조건 자동 점검 (Python, MeCab, Docker, 디스크 등)
uv run python scripts/check_environment.py

# 특정 범위만: --step db | vector
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

### 5. data/ JSON 파일 준비

```bash
# data/ 폴더는 .gitignore에 포함 → git clone만으로는 받을 수 없음
# Google Drive에서 복원 (rclone + rclone.conf 필요, 루트 CLAUDE.md "사전 준비" 참조)
# ⚠ rclone 직접 실행 금지 → 경로 매핑 스크립트 사용
./scripts/sync_data_from_gdrive.sh
```

### 6. 데이터 로드 (PostgreSQL)

```bash
# data/ingest_source/ 폴더에 법령/판례 JSON 파일 필요
# - data/ingest_source/law_v3.json
# - data/ingest_source/precedents_v2.json
# (sources.yaml에서 경로 관리)

uv run python -m scripts.ingest.cli --type all --step db
```

### 6-1. BM25 인덱스 생성 (키워드 검색)

```bash
# search_text 데이터가 Step 6에서 함께 적재됨 (--step db 시 자동)
# 적재 확인 후 BM25 인덱스 생성
uv run python scripts/create_bm25_index.py

# 인덱스 상태만 확인
uv run python scripts/create_bm25_index.py --check
```

> **순서 중요**: search_text 데이터가 먼저 적재되어야 BM25 인덱스 생성이 가능합니다.
> `--step db`로 데이터를 로드하면 search_text가 자동으로 함께 적재됩니다.
> 토크나이저(MeCab userdic) 변경 후에는 search_text만 재빌드할 수 있습니다:
> `uv run python -m scripts.ingest.cli --type all --step fts`

### 6-2. law_articles 데이터 로드 (조문 단위 RAG 컨텍스트)

```bash
# law_v3.json에서 조문 단위로 분리하여 law_articles 테이블에 적재
uv run python scripts/load_law_articles_data.py

# 검증
uv run python scripts/load_law_articles_data.py --verify
```

> law_articles 테이블은 RAG 검색 시 벡터 매칭된 조문만 선별적으로 LLM 컨텍스트에 포함시키기 위해 사용됩니다.

### 7. LanceDB 데이터

```bash
# 옵션 A: 기존 lancedb_data/ 폴더 복사 (권장 - 빠름)
# 다른 팀원에게서 lancedb_data.zip 받아서 압축 해제

# 옵션 B: 임베딩 직접 생성 (GPU 필요, 시간 오래 걸림)
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv run --no-sync python scripts/runpod_lancedb_embeddings.py --type all
```

### 8. 임베딩/리랭커 모델 다운로드 ⚠️ 중요

검색 API를 사용하려면 **임베딩 모델과 리랭커 모델을 먼저 다운로드**해야 합니다.

```bash
# 전체 모델 다운로드 (임베딩 ~2.3GB + 리랭커 ~2.1GB)
uv run python scripts/download_models.py

# 캐시 상태만 확인
uv run python scripts/download_models.py --check

# 임베딩 또는 리랭커만 다운로드
uv run python scripts/download_models.py --embedding-only
uv run python scripts/download_models.py --reranker-only
```

> **참고**: 서버 시작 시 모델이 없으면 경고만 표시하고 서버는 실행됩니다.
> 단, 검색 API 호출 시 503 에러가 반환됩니다.

### 9. 서버 실행

```bash
uv run uvicorn app.main:app --reload
# http://localhost:8000/docs 에서 API 문서 확인
```

### 10. 데이터 확인 (선택)

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

### 전체 폴더 구조

```
app/
├── api/router/          # 통합 API 라우터
│   ├── chat.py          # /api/chat, /api/chat/stream 엔드포인트
│   └── chat_conversations.py  # /api/chat/conversations 대화 목록/조회
├── core/                # 핵심 인프라
│   ├── config.py        # 환경 설정 (pydantic-settings)
│   ├── database.py      # SQLAlchemy 연결
│   ├── registry.py      # 모듈 자동 등록
│   ├── errors.py        # 공통 예외 클래스
│   ├── context.py       # 요청 컨텍스트
│   ├── session.py       # 쿠키 기반 세션 관리 (HttpOnly session_token)
│   ├── logging.py       # 로깅 유틸리티
│   ├── policies/        # 법률 안전정책
│   └── state/           # 세션 저장소
├── multi_agent/         # LangGraph 멀티 에이전트 시스템
│   ├── graph.py         # LangGraph StateGraph 빌드/컴파일
│   ├── nodes.py         # router_node + 에이전트 노드 함수
│   ├── router.py        # RulesRouter, AgentType, UserRole, ROLE_AGENTS
│   ├── state.py         # ChatState TypedDict (emotion, conversation_id, case_id 포함), 변환 함수
│   ├── agents/          # 에이전트 구현 (BaseChatAgent 상속)
│   │   ├── base_chat.py              # 베이스 클래스
│   │   ├── legal_search_agent.py     # 판례/법령 RAG 검색 (Focus+Supplementary 병렬)
│   │   ├── lawyer_finder_agent.py    # 변호사 찾기
│   │   ├── small_claims_agent.py     # 소액소송
│   │   ├── storyboard_agent.py       # 사건 타임라인
│   │   ├── lawyer_stats_agent.py     # 변호사 통계
│   │   ├── law_study_agent.py        # 로스쿨 학습
│   │   ├── mock_trial_agent.py       # 모의 법정
│   │   └── workspace_agent.py       # 워크스페이스 (사건 관리, 태그 수집)
│   ├── services/        # 에이전트 공통 서비스
│   │   └── tagger.py                 # 태그 추출 (LLM 기반, 에이전트별 힌트, 사용자+응답 통합)
│   ├── subgraphs/       # 서브그래프
│   │   ├── small_claims.py           # 소액소송 서브그래프
│   │   ├── storyboard.py            # 스토리보드 서브그래프 (태그 수집, 타임라인 생성)
│   │   ├── mock_trial.py             # 모의 법정 서브그래프 (MockTrialState, 그래프 빌드)
│   │   ├── mock_trial_utils.py       # 모의 법정 헬퍼 함수
│   │   ├── mock_trial_criminal.py    # 모의 법정 형사 노드
│   │   ├── mock_trial_civil.py       # 모의 법정 민사 노드
│   │   ├── mock_trial_agents.py      # 모의 법정 에이전트 (generate→tuple[str, str], _parse_emotion)
│   │   └── mock_trial_prompts.py     # 모의 법정 프롬프트 (EMOTION_TAG_INSTRUCTION 포함)
│   └── schemas/         # 스키마
│       ├── plan.py      # AgentPlan, AgentResult
│       ├── tag.py       # TaggedItem (6 tag types, extraction_source)
│       └── messages.py  # 메시지 타입 (ChatResponse.emotion 포함)
├── services/            # 비즈니스 로직
│   ├── rag/
│   │   ├── embedding.py  # 임베딩 모델
│   │   ├── retrieval.py  # 벡터 검색 (테이블 레지스트리/문서 조회는 아래 파일로 분리)
│   │   ├── table_registry.py  # 테이블 레지스트리 (TableConfig, DOCUMENT_TABLE_REGISTRY)
│   │   ├── document_fetcher.py  # 문서 콘텐츠 배치 조회 (fetch_document_contents, fetch_ai_summaries)
│   │   ├── rerank.py     # 리랭킹
│   │   ├── query_rewrite.py  # 쿼리 리라이팅
│   │   ├── keyword_search.py  # BM25 키워드 검색 (pg_textsearch)
│   │   ├── pipeline.py   # 검색 파이프라인 (동기 + async)
│   │   ├── format_utils.py     # LLM 컨텍스트 + 프론트엔드 소스 포맷팅
│   │   ├── onnx_session.py       # ONNX 세션 싱글턴 관리
│   │   └── onnx_quality_gate.py  # ONNX 품질 게이트 (PyTorch 비교)
│   ├── workspace/       # 워크스페이스 서비스
│   │   ├── workspace_case_service.py  # 사건 CRUD
│   │   ├── chat_persistence.py        # 대화 저장/조회
│   │   ├── timeline_engine.py         # 타임라인 자동 추출/재구성
│   │   ├── conversation_classifier.py # 대화 자동 분류 (에이전트 판별)
│   │   ├── structured_summarizer.py   # 구조화된 대화 요약
│   │   ├── activity_logger.py         # 활동 로그 기록
│   │   └── cleanup.py                 # 고아 데이터 정리
│   └── service_function/ # 통합 서비스 함수
│       ├── lawyer_service.py       # 변호사 검색/클러스터링
│       ├── lawyer_stats_service.py # 변호사 통계
│       ├── precedent_service.py    # 판례 조회
│       ├── law_service.py          # 법령 조회
│       ├── small_claims_service.py # 소액소송 가이드
│       └── mock_trial_service.py  # 모의 법정 서비스
├── tools/               # 외부 도구 클라이언트
│   ├── llm/             # LLM (Solar, OpenAI)
│   ├── vectorstore/     # LanceDB, Chroma, Qdrant
│   ├── graph/           # PgGraphService (PostgreSQL Recursive CTE)
│   └── geo/             # 거리 계산
├── modules/             # 독립 API 모듈 (자동 등록)
│   ├── case_precedent/
│   ├── law_study/
│   ├── lawyer_finder/
│   ├── lawyer_stats/
│   ├── mock_trial/
│   ├── small_claims/
│   ├── storyboard/
│   └── workspace/
├── models/              # SQLAlchemy ORM 모델
│   ├── __init__.py
│   ├── law_document.py
│   ├── precedent_document.py
│   ├── legal_document.py
│   ├── legal_reference.py
│   ├── lawyer.py
│   ├── legal_term.py          # 법률 용어 사전
│   ├── trial_statistics.py
│   ├── fts_index.py           # FTS 전문 검색 인덱스
│   ├── law.py
│   ├── chat_conversation.py   # 채팅 대화 (session_token 기반)
│   ├── workspace_case.py      # 워크스페이스 사건 (6개 테이블)
│   └── ingest/                # 인제스트 원본 테이블 (21개)
│       ├── admin_rule_document.py
│       ├── constitutional_document.py
│       ├── administration_document.py
│       ├── legislation_document.py
│       ├── treaty_document.py
│       ├── interpretation_ministry_document.py
│       ├── special_admin_appeal_document.py
│       ├── local_ordinance_document.py  # 자치법규 (160,276건)
│       └── dec_*_document.py  # 위원회 결정례 (10개)
└── common/              # (deprecated) 레거시 코드
    └── chat_service.py  # → services/rag/로 이전됨
```

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
├── schema/
│   └── __init__.py    # Pydantic 모델 (request/response)
└── model/
    └── __init__.py    # SQLAlchemy 모델
```

> **참고:** 비즈니스 로직은 `app/services/service_function/`에 위치합니다.
> 모듈의 `router/`는 서비스 함수를 import하여 사용합니다.

### API 경로 규칙

모듈명 `snake_case` → API 경로 `/api/kebab-case`
- `lawyer_finder` → `/api/lawyer-finder`
- `small_claims` → `/api/small-claims`

### Multi-Agent 시스템 (LangGraph)

```
START → router_node ──(Command)──→ legal_search_node ───→ END
                      ├──────────→ lawyer_finder_node ──→ END
                      ├──────────→ small_claims_subgraph ─→ END
                      ├──────────→ storyboard_node ─────→ END
                      ├──────────→ lawyer_stats_node ───→ END
                      ├──────────→ law_study_node ──────→ END
                      ├──────────→ mock_trial_subgraph ─→ END
                      ├──────────→ workspace_node ─────→ END
                      └──────────→ simple_chat_node ────→ END
```

**에이전트 목록:**
| 에이전트 | 역할 | 노드 | RAG | LLM |
|---------|------|------|-----|-----|
| `LegalSearchAgent` | 판례/법령 RAG 검색 (Focus+Supplementary 병렬), 체계도 NAVIGATE 액션 | `legal_search_node` | ✅ | ✅ |
| `LawyerFinderAgent` | 위치 기반 변호사 추천 (동 단위 지원) | `lawyer_finder_node` | ❌ | ❌ |
| `SmallClaimsAgent` | 소액소송 단계별 가이드 | `small_claims_subgraph` | ✅ | ❌ |
| `StoryboardAgent` | 사건 타임라인 생성 | `storyboard_node` | ❌ | ✅ |
| `LawyerStatsAgent` | 변호사 통계 안내 | `lawyer_stats_node` | ❌ | ❌ |
| `LawStudyAgent` | 로스쿨 학습 가이드 | `law_study_node` | ✅ | ✅ |
| `MockTrialAgent` | 모의 법정 시뮬레이션 | `mock_trial_subgraph` | ❌ | ✅ |
| `WorkspaceAgent` | 사건 관리, 태그 수집, 타임라인 | `workspace_node` | ❌ | ✅ |
| `SimpleChatAgent` | 일반 LLM 채팅 (폴백) | `simple_chat_node` | ❌ | ✅ |

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
| `LANCEDB_INDEX_TYPE` | 벡터 인덱스 타입 (빈 문자열: brute-force) | `""` |
| `LANCEDB_MODE` | LanceDB 모드 (local: 임베디드, remote: 마이크로서비스) | `local` |
| `LANCEDB_SERVICE_URL` | remote 모드 시 마이크로서비스 URL | `http://localhost:8100` |
| `LANCEDB_SERVICE_TIMEOUT` | remote 모드 HTTP timeout (초) | `30.0` |
| `USE_LOCAL_EMBEDDING` | 로컬 임베딩 사용 | `true` |
| `LOCAL_EMBEDDING_MODEL` | 임베딩 모델 | `nlpai-lab/KURE-v1` |
| `UPSTAGE_API_KEY` | Solar API 키 | - |
| `UPSTAGE_MODEL` | Solar 모델명 | `solar-pro3-260126` |
| `USE_DB_LAWYERS` | 변호사 데이터 소스 (true: PostgreSQL, false: JSON) | `false` |
| `USE_LEGAL_TERM_DICT` | 법률 용어 사전 사용 (true: MeCab 토큰 보강) | `true` |
| `LANGCHAIN_TRACING_V2` | LangSmith 트레이싱 활성화 | `false` |
| `LANGCHAIN_PROJECT` | LangSmith 프로젝트명 | `law-platform` |
| `LANGCHAIN_API_KEY` | LangSmith API 키 | `""` |
| `USE_ONNX_EMBEDDING` | ONNX 임베딩 사용 (쿼리 + 인제스트 배치, CUDA 자동 감지) | `false` |
| `ONNX_EMBEDDING_VARIANT` | ONNX 임베딩 variant (`ort-opt`, `ort-opt-qdq`, `onnx-fp16`) | `ort-opt` |
| `USE_ONNX_RERANKER` | ONNX 리랭커 사용 | `false` |
| `ONNX_RERANKER_VARIANT` | ONNX 리랭커 variant | `ort-opt` |
| `ONNX_INTRA_OP_THREADS` | ORT 스레드 수 (0=자동, 4=Mac ARM P코어) | `0` |
| `ONNX_QUALITY_GATE_ENABLED` | ONNX 품질 게이트 활성화 (PyTorch 대비 cosine/pearson 검증) | `true` |
| `ONNX_QUALITY_GATE_FALLBACK` | 품질 미달 시 자동 PyTorch 폴백 | `true` |
| `ONNX_INFERENCE_TIMEOUT_SECONDS` | ONNX 추론 타임아웃 (초) | `30.0` |

> **ONNX Variant (임베딩)**: `ort-opt` (FP32 무손실, cosine 1.0), `ort-opt-qdq` (INT8, cosine 0.999, 23% 빠름), `onnx-fp16` (FP16, cosine 1.0).
> **ONNX Variant (리랭커)**: `ort-opt` (FP32 무손실) | `ort-opt-qdq` (INT8, 4 FP32, Pearson 0.9999, 3.52x) | `ort-opt-qdq-6fp32` (INT8, 6 FP32, Pearson 0.9994, Spearman 0.993).
> **ONNX EP**: `onnxruntime-gpu` 설치 시 CUDA 자동 감지, 미설치 시 CPU fallback. 인제스트 배치 임베딩도 지원.
> ONNX 모델 빌드 및 RAG 비교 테스트 가이드: `scripts/CLAUDE.md`의 "ONNX 최적화 모델 빌드 + RAG 테스트 환경 구축" 참조.
> **리랭커 ONNX variant 테스트**: `docs/04-report/features/reranker-onnx-variant-test-guide.md` (다운로드, .env 설정, 수동 추론, 트러블슈팅).

자세한 설정은 `.env.example` 참조.

## Services

### RAG 서비스 (`app/services/rag/`)

```python
from app.services.rag.retrieval import get_retrieval_service, create_query_embedding

# 검색 서비스 (동기)
service = get_retrieval_service()
results = service.search(
    query="손해배상 판례",
    n_results=5,
    doc_type="precedent"  # "precedent" | "law"
)

# 임베딩 생성
embedding = create_query_embedding("검색 쿼리")
```

```python
# 파이프라인 (async, 내부 병렬화)
from app.services.rag.pipeline import search_with_pipeline_async, PipelineConfig

config = PipelineConfig(
    n_results=15, doc_type="precedent",
    enable_rerank=True, rerank_top_k=5,
)
result = await search_with_pipeline_async(query, config)
# result.documents, result.metrics, result.rewritten_query
```

**async 파이프라인 내부 병렬화:**
- 다중 리라이팅 쿼리 → `asyncio.gather` 병렬 검색
- 각 검색 내 벡터 + FTS → `asyncio.gather` 병렬 실행
- 요약문/원문 조회 → data_type별 `asyncio.gather` 병렬
- 리랭킹 → `asyncio.to_thread` (CPU-bound)

### 통합 서비스 함수 (`app/services/service_function/`)

```python
# 변호사 검색/클러스터링
from app.services.service_function.lawyer_service import (
    find_nearby_lawyers,
    search_lawyers,
    get_clusters,
    load_lawyers_data,
)

# 변호사 통계
from app.services.service_function.lawyer_stats_service import (
    calculate_overview,
    calculate_by_region,
    calculate_density_by_region,
    calculate_by_specialty,
    calculate_cross_analysis,
)
```

## Modules

### lawyer_stats (변호사 통계)

**경로:** `app/modules/lawyer_stats/`

**서비스 함수:** `app/services/service_function/lawyer_stats_service.py`
- `calculate_overview()` - 전체 현황 요약
- `calculate_by_region()` - 지역별 변호사 수 집계
- `calculate_density_by_region(year, include_change)` - 인구 대비 밀도 계산
- `calculate_by_specialty()` - 전문분야별 통계
- `calculate_cross_analysis()` - 지역×전문분야 교차 분석

**인구 데이터:** `../data/population.json` (프로젝트 루트)
- 출처: KOSIS e지방지표 (주민등록인구, 추계인구)
- `current` - 현재 인구 (2025.12 기준)
- `2030/2035/2040` - 추계인구
- 업데이트: `python scripts/update_population.py` (CSV → JSON 변환)

**API 엔드포인트:**
- `GET /overview` - 전체 현황 요약
- `GET /by-region` - 지역별 변호사 수
- `GET /density-by-region?year=current&include_change=false` - 밀도 조회
- `GET /by-specialty` - 전문분야별 통계
- `GET /cross-analysis` - 지역×전문분야 교차 분석

### lawyer_finder (변호사 찾기)

**경로:** `app/modules/lawyer_finder/`

**서비스 함수:** `app/services/service_function/lawyer_service.py`
- `load_lawyers_data()` - 변호사 데이터 로드 (캐싱)
- `find_nearby_lawyers()` - 반경 내 변호사 검색
- `search_lawyers()` - 조건 기반 검색
- `get_clusters()` - 지도 클러스터링
- `get_categories()` / `get_specialties_by_category()` - 전문분야 분류
- `build_dong_coords_cache()` - 동 이름별 중심점 좌표 캐시 (`@lru_cache`, 변호사 주소에서 동 추출, 최소 3명 이상)

**API 엔드포인트:**
- `GET /nearby` - 반경 내 변호사 검색
- `GET /search` - 조건 기반 검색
- `GET /clusters` - 클러스터 데이터
- `GET /specialties` - 전문분야 목록
- `GET /{lawyer_id}` - 변호사 상세 정보

## Conventions

- 라우터 함수는 `async def` 사용
- Pydantic v2 문법 사용 (`model_validator`, `field_validator`)
- 타입 힌트 필수 (mypy strict 모드)
- ruff 린터 규칙: E, F, I, N, W

### 보안

- **Rate Limiting**: 모든 모듈 라우터의 POST 엔드포인트에 `@limiter.limit(AI_RATE_LIMIT)` 적용 (`app/core/rate_limit.py`)
- **입력 길이 제한**: `ChatRequest.message` max_length=10,000, `history` max_length=50 (`multi_agent/schemas/messages.py`)
- **Prompt Injection 방어**: `/api/chat`, `/api/chat/stream` 엔드포인트에서 `check_input_safety()` 호출 (`core/policies/legal_safety.py`)

## Tests

### 테스트 폴더 구조

```
backend/tests/
├── __init__.py
├── conftest.py                      # 공통 픽스처
├── integration/                     # 통합 테스트 (DB, 외부 서비스 연동)
│   ├── __init__.py
│   ├── test_lancedb_search.py       # LanceDB 벡터 검색 테스트
│   ├── test_lancedb_integration.py  # LanceDB E2E + FTS + 하이브리드 (15개)
│   ├── test_postgresql_data.py      # PostgreSQL 데이터 확인
│   ├── test_evaluation_runner.py    # 평가 실행기 테스트
│   └── test_evaluation_search.py    # 평가 검색 테스트
├── unit/                            # 단위 테스트 (개별 함수/클래스)
│   ├── __init__.py
│   ├── test_vectorstore_schema.py   # 스키마 v2 단위 테스트 (15개)
│   ├── test_lancedb_store.py        # LanceDBStore CRUD 테스트 (21개)
│   ├── test_mecab_tokenizer.py      # MeCab 토크나이저 테스트 (19개, 보강 포함)
│   ├── test_legal_term_dict.py      # 법률 용어 사전 테스트 (18개)
│   ├── test_evaluation_metrics.py   # 메트릭 계산 테스트 (31개)
│   ├── test_evaluation_schemas.py   # 스키마 검증 테스트 (21개)
│   ├── test_evaluation_dataset_builder.py  # 데이터셋 빌더 테스트 (14개)
│   ├── test_storyboard_kakao_parser.py    # 카카오톡 파서 테스트 (16개)
│   ├── test_storyboard_file_validation.py # 파일 검증 게이트 테스트 (29개)
│   └── test_storyboard_helpers.py         # 스토리보드 헬퍼 테스트 (28개)
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
uv run pytest -m "not requires_mecab"    # MeCab 없이 실행
uv run pytest -m "not requires_fts"      # FTS 없이 실행
uv run pytest -m "not requires_openai"   # OpenAI API 없이 실행

# LanceDB 벡터 DB 테스트만 실행
uv run pytest tests/unit/test_vectorstore_schema.py tests/unit/test_lancedb_store.py tests/unit/test_mecab_tokenizer.py tests/integration/test_lancedb_integration.py -v
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
# 법령 데이터 로드 (data/law_v3.json → PostgreSQL)
uv run python -m scripts.ingest.cli --type law --step db

# 판례 데이터 로드 (data/precedents_v2.json → PostgreSQL)
uv run python -m scripts.ingest.cli --type precedent --step db

# 전체 로드 (법령 + 판례)
uv run python -m scripts.ingest.cli --type all --step db

# 기존 데이터 삭제 후 재로드
uv run python -m scripts.ingest.cli --type all --step db --reset
```

### 모델 파일 위치

```
app/models/
├── __init__.py
├── law_document.py        # 법령 원본 (LanceDB 연동)
├── precedent_document.py  # 판례 원본 (LanceDB 연동)
├── legal_document.py      # 법률 문서 (일반)
├── legal_reference.py     # 참조 정보
├── lawyer.py              # 변호사 정보
├── legal_term.py          # 법률 용어 사전 (MeCab 보강용)
├── trial_statistics.py    # 재판 통계
├── statute_hierarchy.py   # 법령 계급 관계 (child→parent)
├── statute_alias.py       # 법령 약칭
├── statute_relation.py    # 법령 관련 관계
├── case_statute_citation.py # 판례→법령 인용
├── case_case_citation.py  # 판례→판례 인용
├── chat_conversation.py   # 채팅 대화
└── workspace_case.py      # 워크스페이스 사건 (6개 테이블)
```

### 테이블 구조

| 테이블 | 설명 | 주요 컬럼 |
|--------|------|-----------|
| `law_documents` | 법령 원본 | law_id, law_name, content, ai_summary |
| `precedent_documents` | 판례 원본 | serial_number, case_name, ruling, reasoning |
| `lawyers` | 변호사 정보 (17,326건) | name, address, specialties(ARRAY), latitude, longitude, region |
| `legal_terms` | 법률 용어 사전 (~72,700건) | term(UNIQUE), definition, source_code, source_count, term_length, is_korean_only |
| `trial_statistics` | 재판 통계 | category, court_name, court_type, parent_court, year, case_count |
| `local_ordinance_documents` | 자치법규 원본 (160,276건) | ordinance_id, ordinance_name, local_government, overall_summary, content |
| `fts_index` | BM25 전문 검색 인덱스 (425,209건) | **PK: (source_id, data_type)**, title, date, search_text (MeCab 명사 공백 구분), BM25 인덱스 `idx_fts_bm25` (pg_textsearch). dec_* source_id는 `{name}:{serial_number}` 형식 |
| `statute_hierarchy` | 법령 계급 관계 | child_id(FK→law_documents), parent_id(FK→law_documents), relation_type |
| `statute_aliases` | 법령 약칭 | law_id(FK→law_documents), alias, alias_type |
| `statute_relations` | 법령 관련 관계 | source_id(FK), target_id(FK), relation_type, weight |
| `case_statute_citations` | 판례→법령 인용 | case_id(FK→precedent_documents), statute_id(FK→law_documents) |
| `case_case_citations` | 판례→판례 인용 | citing_id(FK→precedent_documents), cited_id(FK→precedent_documents) |
| `chat_conversations` | 채팅 대화 | id(UUID), session_token, title, last_agent, case_id(FK→workspace_cases), message_count |
| `workspace_cases` | 워크스페이스 사건 | id(UUID), session_token, case_name, case_type, status(open/closed/archived) |
| `workspace_tagged_items` | 태그 수집 항목 | id(UUID), case_id(FK), type, label, value, confidence, source_conversation_id |
| `workspace_timeline_items` | 타임라인 이벤트 | id(UUID), case_id(FK), date, title, description, confidence, source_type |
| `workspace_activity_logs` | 활동 로그 | id(UUID), case_id(FK), action, detail |
| `workspace_structured_summaries` | 구조화 요약 | id(UUID), case_id(FK), conversation_id(FK), category, content |

### 변호사 데이터 (lawyers 테이블)

`USE_DB_LAWYERS=true` 설정 시 JSON 대신 PostgreSQL에서 변호사 데이터를 조회합니다.

```bash
# 마이그레이션 + 데이터 로드
uv run alembic upgrade head
uv run python scripts/load_lawyers_data.py

# 검증
uv run python scripts/load_lawyers_data.py --verify
```

**주요 인덱스:**
- `idx_lawyers_coords`: (latitude, longitude) B-tree - 바운딩 박스 검색
- `idx_lawyers_specialties`: specialties GIN - ARRAY 포함/교집합 연산
- `idx_lawyers_region`: region B-tree - 통계 GROUP BY

**서비스 파일:**
- `app/services/service_function/lawyer_db_service.py` - 검색/클러스터링
- `app/services/service_function/lawyer_stats_db_service.py` - 통계 계산

### 법률 용어 사전 (legal_terms 테이블)

`USE_LEGAL_TERM_DICT=true` 설정 시 앱 시작(lifespan)에서 PostgreSQL → 메모리(frozenset)로 법률 용어를 로드하여 MeCab 토크나이징을 보강합니다.

```bash
# 마이그레이션 + 데이터 로드
uv run alembic upgrade head
uv run python scripts/load_legal_terms_data.py
uv run python scripts/load_legal_terms_data.py --verify  # 검증
```

**동작 원리:**
1. `legal_terms` 테이블에서 한글 전용 + 2-15자 용어 로드 → `frozenset` (O(1) lookup)
2. MeCab 기본 형태소 분석 결과에 법률 복합명사를 추가 토큰으로 삽입
3. 예: "손해배상청구" → MeCab ["손해","배상","청구"] + 사전 ["손해배상","손해배상청구"]

**데이터 현황:**
- 총 엔트리: ~72,700개 (lawterms_v1.json 기반, 평탄화 + 역추출 포함)
- userdic 적재: ~37,366개 (한글 전용 + 혼합 + 괄호 변형 추출)
- 유효 커버리지: ≥99% (구조적 제외 35,234개 차감 기준)
- 상세 분석: `docs/tokenizer/USERDIC_COVERAGE_ANALYSIS.md`

**주요 인덱스:**
- `idx_legal_terms_term`: term UNIQUE - 용어 조회
- `idx_legal_terms_source_code`: source_code B-tree - 사전유형별 필터
- `idx_legal_terms_length`: term_length B-tree - 길이 필터
- `idx_legal_terms_korean`: is_korean_only B-tree - 한글 전용 필터
- `idx_legal_terms_priority`: priority B-tree - 우선순위 정렬

**관련 파일:**
- `app/models/legal_term.py` - ORM 모델
- `app/tools/vectorstore/legal_term_dict.py` - 메모리 사전 (frozenset 기반)
- `app/tools/vectorstore/mecab_tokenizer.py` - MeCab 보강 토크나이저
- `alembic/versions/006_add_legal_terms_table.py` - 마이그레이션
- `scripts/load_legal_terms_data.py` - 데이터 로드 스크립트

**롤백:** `USE_LEGAL_TERM_DICT=false`로 설정하면 기존 MeCab 동작 100% 유지

### MeCab userdic (사용자 사전)

법률 복합명사를 MeCab이 직접 인식하도록 userdic에 등록합니다.

```bash
# userdic 빌드
uv run python scripts/build_mecab_userdic.py           # DB에서 빌드
uv run python scripts/build_mecab_userdic.py --from-json  # JSON fallback
uv run python scripts/build_mecab_userdic.py --verify   # 빌드 후 검증
uv run python scripts/build_mecab_userdic.py --dry-run  # 통계만
```

**출력 파일:**
- `data/mecab_userdic/legal_terms.csv` - userdic 소스 CSV
- `data/mecab_userdic/legal_terms.dic` - 컴파일된 MeCab 바이너리 사전
- `data/mecab_userdic/decomposition_map.json` - 복합어→서브 토큰 분해맵

**FTS 명사 필터링:**
- `_FTS_POS_TAGS = frozenset({"NNG", "NNP"})` — 일반명사 + 고유명사만 허용 (내부 상수)
- `_MIN_TOKEN_LENGTH = 2` — 1자 명사("시", "때" 등) 노이즈 제거
- `morphs()` 호출 시 항상 명사 필터 + 2자 이상 필터 적용 (옵션 없음)
- FTS 생성(`db_writer`, `search_text_rebuilder`)과 검색(`keyword_search`) 양쪽에 적용
- MeCab 미설치 시 에러 발생 (silent fallback 없음)

**수동 용어 추가 (`scripts/manual_terms.json`):**
- MeCab이 오분석하는 법률 용어를 수동으로 userdic에 추가하는 JSON 파일
- VV+ETN(동사 활용형): 괴롭힘, 파면 / Compound 분리 방지: 임대차, 부당해고, 채무불이행 등
- 빌드 시 자동 병합: `build_mecab_userdic.py`가 DB 용어 + manual_terms.json을 합산

**시스템 요구:**
- `mecab`, `libmecab-dev`, `mecab-ko-dic` 시스템 패키지
- `mecab-dict-index`: `/usr/lib/mecab/mecab-dict-index`

### 데이터 조회 예시

```python
from sqlalchemy import select
from app.core.database import async_session_factory
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

### 서비스 사용 예시

```python
# 판례 서비스
from app.services.cases.precedent_service import get_precedent_service

service = get_precedent_service()
details = service.get_details(["76396", "76397"])

# RAG 검색 서비스
from app.services.rag.retrieval import get_retrieval_service

retrieval = get_retrieval_service()
results = retrieval.search("손해배상 판례", n_results=5, doc_type="precedent")
```

## Vector DB (LanceDB)

### 임베딩 생성 (ingest 파이프라인 - 메인)

```bash
# PyTorch CUDA 설치 (환경에 맞게 선택)
uv pip install --reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 임베딩 생성 (--no-sync 필수: torch 버전 유지)
uv run --no-sync python -m scripts.ingest.cli --type precedent --step vector
uv run --no-sync python -m scripts.ingest.cli --type law --step vector
uv run --no-sync python -m scripts.ingest.cli --type all --step vector --reset

# ONNX INT8 벡터 적재 (CPU 최적화, ONNX 모델 미존재 시 자동 변환)
uv run --no-sync python -m scripts.ingest.cli --step onnx-export              # 명시적 ONNX 변환
uv run --no-sync python -m scripts.ingest.cli --type all --step vector --backend onnx-int8  # INT8 적재

# 통계 확인
uv run --no-sync python -m scripts.ingest.cli --type all --stats
```

### 노트북용 thin wrapper

RunPod/Colab 노트북에서는 하위 호환 API를 유지하는 thin wrapper를 사용합니다.
내부적으로 ingest 파이프라인의 `run_vector_ingest()`를 호출합니다.

```python
# RunPod/Colab 노트북에서 사용
from scripts.runpod_lancedb_embeddings import (
    run_law_embedding,          # 법령 임베딩
    run_precedent_embedding,    # 판례 임베딩
    show_stats,                 # 통계 출력
    split_precedents,           # 판례 분할
    split_laws,                 # 법령 분할
    EmbeddingQualityChecker,    # 품질 검증
    EmbeddingCache,             # 임베딩 캐시
)
```

### 저장 위치

```
backend/
├── lancedb_data/           # LanceDB 데이터
│   ├── legal_chunks.lance/              # 21개 타입 통합 테이블 (법령 포함, summary_type/article_number 컬럼)
│   └── local_ordinance_chunks.lance/   # 자치법규 전용 테이블 (전체요약 + 조문요약)
└── scripts/
    ├── ingest/                         # 메인 인제스트 파이프라인
    ├── embedding_common/               # 공통 임베딩 모듈
    ├── runpod_lancedb_embeddings.py    # RunPod thin wrapper
    └── colab_lancedb_embeddings.py     # Colab thin wrapper
```

### 품질 검증

```python
from scripts.embedding_common.quality import EmbeddingQualityChecker

checker = EmbeddingQualityChecker()
report = checker.quick_test()  # 법률 도메인 기본 테스트
# Quality: GOOD (separation > 0.2)
```

### 검색 예시

```python
import lancedb
from scripts.embedding_common.model import get_embedding_model

model = get_embedding_model()
query_vector = model.encode('손해배상 책임')

db = lancedb.connect('./lancedb_data')
table = db.open_table('legal_chunks')

results = table.search(query_vector).metric('cosine').limit(10).to_pandas()
```

### 주의사항

1. **torch는 pyproject.toml에 없음** - 환경별로 수동 설치
2. **--no-sync 필수** - `uv run --no-sync`로 실행
3. **GPU 자동 감지** - VRAM에 따라 batch_size 자동 설정

## Embedding / Reranker Model (임베딩 · 리랭커 모델)

검색 API는 **임베딩 모델**(쿼리→벡터)과 **리랭커 모델**(문서 재정렬)이 필요합니다.
두 모델 모두 `backend/data/models/`에 캐시됩니다.

### 모델 정보

| 용도 | 모델명 | 크기 | 비고 |
|------|--------|------|------|
| 임베딩 | `nlpai-lab/KURE-v1` | ~2.3GB | 1024차원, SentenceTransformer |
| 리랭커 | `dragonkue/bge-reranker-v2-m3-ko` | ~2.1GB | CrossEncoder, Sigmoid |

### 모델 다운로드

```bash
cd backend

# 전체 모델 다운로드 (임베딩 + 리랭커)
uv run python scripts/download_models.py

# 캐시 상태만 확인
uv run python scripts/download_models.py --check

# 재다운로드 (기존 캐시 무시)
uv run python scripts/download_models.py --force

# 임베딩 또는 리랭커만 다운로드
uv run python scripts/download_models.py --embedding-only
uv run python scripts/download_models.py --reranker-only

# 특정 모델만 다운로드 (SentenceTransformer)
uv run python scripts/download_models.py --model nlpai-lab/KURE-v1
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
| `app/services/rag/embedding.py` | 임베딩 모델 로드 (`cache_folder` → `data/models/`) |
| `app/services/rag/rerank.py` | 리랭커 모델 로드 (`cache_folder` → `data/models/`) |
| `app/core/errors.py` | `EmbeddingModelNotFoundError` 예외 클래스 |
| `scripts/download_models.py` | 모델 다운로드 CLI (임베딩 + 리랭커) |
| `app/main.py` | lifespan에서 시작 시 체크 |

### 환경 변수

```bash
# backend/.env
USE_LOCAL_EMBEDDING=true              # 로컬 임베딩 사용 (기본값: true)
LOCAL_EMBEDDING_MODEL=nlpai-lab/KURE-v1  # 임베딩 모델명
```

> **팁**: `USE_LOCAL_EMBEDDING=false`로 설정하면 OpenAI 임베딩을 사용하며,
> 이 경우 로컬 모델 다운로드가 필요 없습니다 (단, `OPENAI_API_KEY` 필요).

## Graph (PostgreSQL Recursive CTE)

법령 계급(시행령→법률), 판례 인용 관계를 PostgreSQL 테이블 + Recursive CTE로 처리합니다.
(이전 Neo4j에서 이관 완료)

### 관련 테이블

| 테이블 | 설명 |
|--------|------|
| `statute_hierarchy` | 법령 계급 관계 (child→parent) |
| `statute_aliases` | 법령 약칭 |
| `statute_relations` | 법령 관련 관계 |
| `case_statute_citations` | 판례→법령 인용 |
| `case_case_citations` | 판례→판례 인용 |

### 서비스

`app/tools/graph/pg_graph_service.py` — `PgGraphService`

```python
from app.tools.graph import get_pg_graph_service

pg = get_pg_graph_service()
results = await pg.search_statutes("도로교통법", limit=5)
hierarchy = await pg.get_statute_hierarchy("law_id")
graph = await pg.get_statute_graph(center_id, depth=2, limit=100)
```

## Trial Statistics DB (재판 통계)

법원별/카테고리별/연도별 사건 처리 건수를 PostgreSQL `trial_statistics` 테이블에 저장합니다.

### 테이블 구조

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `category` | VARCHAR(50) NOT NULL | 사건 카테고리 (민사_본안_단독, 형사_공판 등) |
| `court_name` | VARCHAR(100) NOT NULL | 법원명 (서울중앙지방법원, 고양지원 등) |
| `court_type` | VARCHAR(20) NOT NULL | 법원 유형 (main: 본원, branch: 지원) |
| `parent_court` | VARCHAR(100) NULL | 지원의 상위 본원명 (본원은 NULL) |
| `year` | INTEGER NOT NULL | 연도 (2015~2024) |
| `case_count` | INTEGER NOT NULL | 사건 처리 건수 |
| UNIQUE | (category, court_name, year) | 중복 방지 제약조건 |

### 카테고리 매핑

| CSV 파일 | category |
|----------|----------|
| 제2항_민사_민사본안_단독_제1심 | `민사_본안_단독` |
| 제2항_민사_민사본안_합의_제1심 | `민사_본안_합의` |
| 제3항_가사_가사소송_제1심 | `가사` |
| 제4항_행정_행정소송_제1심 | `행정` |
| 제6항_형사_형사공판_제1심 | `형사_공판` |
| 제6항_형사_약식명령 | `형사_약식` |
| 제7항_소년보호_소년보호 | `소년보호` |
| 제8항_가정보호_가정보호 | `가정보호` |

### 저장 규칙
- 소계/합계 행: 저장하지 않음 (쿼리로 SUM 계산)
- 보정값: 저장하지 않음 (원본값만 사용)
- 평균 열: 저장하지 않음 (AVG로 계산)

### 관련 파일

| 파일 | 설명 |
|------|------|
| `app/models/trial_statistics.py` | TrialStatistics ORM 모델 |
| `alembic/versions/005_add_trial_statistics_table.py` | 마이그레이션 |

## EDA (탐색적 데이터 분석)

법률 데이터 48개 JSON 파일(~4.9GB)의 품질, 구조, 관계를 분석하는 노트북 + 공유 모듈입니다.

```
scripts/common/             # 범용 공통 모듈 (JSON, DB, 로깅, 배치, 인용)
scripts/eda/
├── common.py           # EDA 전용 유틸 + common/ re-export (하위 호환)
└── data_registry.py    # 11개 카테고리, 48개 파일 레지스트리

notebooks/eda/
├── 01_inventory_schema.ipynb  ~ 07_citation_recovery.ipynb  # 7개 노트북

eda_output/              # 분석 결과 JSON (phase1~phase7)
```

상세: `scripts/CLAUDE.md` "탐색적 데이터 분석 (EDA)" 섹션 참조
