# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

법률 서비스 플랫폼 - 모듈형 아키텍처로 기능을 유연하게 추가/삭제할 수 있는 법률 서비스 플랫폼입니다.

## Primary Languages

| 언어 | 용도 | 타입 검증 |
|------|------|----------|
| **TypeScript** | Frontend (Next.js), 프론트엔드 컴포넌트 | `npm run build` (tsc + ESLint) |
| **Python** | Backend (FastAPI), 멀티에이전트, RAG | `ruff check` + `mypy` |

양쪽 모두 **타입 어노테이션 필수**. 타입 미지정 코드 작성 금지.

## Long-Running Tasks

보안 감사, 대규모 리팩토링, 아키텍처 분석 등 장시간 작업 시:

1. 작업 시작 전 **단계별 체크리스트**를 TaskCreate로 생성
2. 각 단계 완료 시 **중간 결과를 파일로 저장** (세션 중단 시 유실 방지)
3. 3개 이상 파일을 수정하는 작업은 반드시 단계별 추적

### 세션 범위 원칙

- **하나의 세션에 하나의 목적**을 유지
- 디버깅 도중 보안 감사 등 관련 없는 작업을 혼합하지 않음
- 목적이 다르면 별도 세션으로 분리

## Claude Code Hooks 설정 (팀원 필수)

프로젝트에 `Edit`/`Write` 후 자동 린트 검증 훅이 포함되어 있습니다.
훅 스크립트(`.claude/hooks/post-write-verify.sh`)는 git에 포함되지만, 활성화 설정은 **각자 로컬에서** 해야 합니다.

### 설정 방법

`.claude/settings.local.json` 파일에 아래 내용을 추가하세요 (파일이 없으면 새로 생성):

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "\"$CLAUDE_PROJECT_DIR\"/.claude/hooks/post-write-verify.sh",
            "timeout": 30
          }
        ]
      }
    ]
  }
}
```

### 동작 방식

| 파일 타입 | 검증 도구 | 실패 시 |
|-----------|----------|---------|
| `*.py` (backend/) | `uv run ruff check` | 수정 후 재검증 |
| `*.ts, *.tsx` (frontend/) | `npx tsc --noEmit` | 수정 후 재검증 |

### 참고
- `.claude/settings.local.json`은 `.gitignore`에 포함되어 있으므로 커밋되지 않습니다
- 훅이 실패하면 Claude가 자동으로 코드를 수정하고 재검증합니다
- `jq`가 설치되어 있어야 합니다 (`brew install jq`)

## Commands

### Backend (uv + FastAPI)
```bash
cd backend
uv sync                              # 의존성 설치
uv sync --dev                        # 개발 의존성 포함
uv run uvicorn app.main:app --reload # 서버 실행 (localhost:8000)
uv run pytest                        # 테스트 실행
uv run pytest tests/test_file.py -k test_name  # 단일 테스트
uv run ruff check .                  # 린트
uv run mypy .                        # 타입 체크
```

### Frontend (Next.js)
```bash
cd frontend
npm install
npm run dev      # 개발 서버 (localhost:3000)
npm run build    # 빌드
npm run lint     # 린트
```

### Module Management
```bash
# 새 모듈 추가 (Backend + Frontend 자동 생성)
python3 scripts/add_module.py <module_name> "<description>"

# 모듈 삭제
python3 scripts/add_module.py remove <module_name>
```

## Architecture

### 모듈 자동 등록 시스템

**Backend**: `backend/app/core/registry.py`의 `ModuleRegistry`가 `backend/app/modules/` 폴더를 스캔하여 자동으로 라우터 등록. 각 모듈의 `router/__init__.py`에 `router = APIRouter()` 정의 필요.

**Frontend**: `frontend/src/lib/modules.ts`에서 모듈 정의 후 `getEnabledModules()`로 활성화된 모듈만 표시.

### 모듈 구조

Backend 모듈 (`backend/app/modules/<module_name>/`):
- `router/` - API 라우터 (필수: `router` 변수)
- `schema/` - Pydantic 스키마
- `model/` - DB 모델

> **Note**: 비즈니스 로직은 `app/services/service_function/`에 통합 관리됩니다.
> 모듈의 router는 services를 import하여 사용합니다.

Frontend 모듈:
- `src/app/<module-name>/page.tsx` - 페이지
- `src/features/<module-name>/services/` - API 서비스
- `src/features/<module-name>/components/` - 컴포넌트
- `src/lib/api.ts` - endpoints 정의
- `src/lib/modules.ts` - 모듈 메타데이터

### 모듈 비활성화

Backend: `.env`에서 `ENABLED_MODULES=["module1","module2"]` (빈 배열이면 모두 활성화)
Frontend: `modules.ts`에서 `enabled: false`

### API 경로 규칙

Backend 모듈명 `snake_case` → API 경로 `/api/kebab-case`
예: `lawyer_finder` → `/api/lawyer-finder`

## Key Files

- `backend/app/main.py` - FastAPI 앱 진입점
- `backend/app/core/config.py` - 환경 설정 (pydantic-settings)
- `backend/app/core/database.py` - DB 연결 (SQLAlchemy)
- `backend/app/core/registry.py` - 모듈 자동 등록
- `frontend/src/lib/modules.ts` - 프론트엔드 모듈 정의
- `frontend/src/lib/api.ts` - API 클라이언트 및 endpoints
- `scripts/add_module.py` - 모듈 생성 스크립트

## Backend Architecture

### 폴더 구조

```
backend/app/
├── api/router/          # 통합 API (채팅 등)
│   └── chat.py          # /api/chat 엔드포인트
├── core/                # 핵심 인프라
│   ├── config.py        # 환경 설정
│   ├── database.py      # DB 연결
│   ├── errors.py        # 공통 예외
│   ├── context.py       # 요청 컨텍스트
│   ├── policies/        # 법률 안전정책
│   └── state/           # 세션 저장소
├── multi_agent/         # LangGraph 멀티 에이전트 시스템
│   ├── graph.py         # LangGraph StateGraph 빌드/컴파일
│   ├── nodes.py         # router_node + 에이전트 노드 함수
│   ├── router.py        # RulesRouter, AgentType, UserRole, ROLE_AGENTS
│   ├── state.py         # ChatState TypedDict, 변환 함수
│   ├── agents/          # 에이전트 구현체 (BaseChatAgent 상속)
│   ├── subgraphs/       # 서브그래프 (small_claims)
│   └── schemas/         # 스키마 (AgentPlan, AgentResult)
├── services/            # 비즈니스 로직 서비스
│   ├── rag/             # RAG 검색 (retrieval, rerank, pipeline)
│   └── service_function/ # 통합 서비스 함수
│       ├── lawyer_service.py       # 변호사 검색/클러스터링
│       ├── lawyer_stats_service.py # 변호사 통계
│       ├── precedent_service.py    # 판례 조회
│       ├── law_service.py          # 법령 조회
│       └── small_claims_service.py # 소액소송 가이드
├── tools/               # 외부 도구 클라이언트
│   ├── llm/             # LLM 클라이언트 (Solar)
│   ├── vectorstore/     # 벡터 DB (LanceDB)
│   ├── graph/           # Neo4j 그래프 서비스
│   └── geo/             # 거리 계산
├── modules/             # 독립 API 모듈 (자동 등록)
│   ├── case_precedent/
│   ├── lawyer_finder/
│   ├── lawyer_stats/
│   └── small_claims/
└── models/              # ORM 모델
```

### Multi-Agent 시스템 (LangGraph)

```
START → router_node ──(Command)──→ legal_search_node ───→ END
                      ├──────────→ lawyer_finder_node ──→ END
                      ├──────────→ small_claims_subgraph ─→ END
                      ├──────────→ storyboard_node ─────→ END
                      ├──────────→ lawyer_stats_node ───→ END
                      ├──────────→ law_study_node ──────→ END
                      └──────────→ simple_chat_node ────→ END
```

**에이전트 목록:**
| 에이전트 | 역할 | 노드 | RAG 사용 |
|---------|------|------|---------|
| `LegalSearchAgent` | 판례/법령 RAG 검색 (search_focus로 분기) | `legal_search_node` | ✅ |
| `LawyerFinderAgent` | 변호사 찾기 페이지 이동 | `lawyer_finder_node` | ❌ |
| `SmallClaimsAgent` | 소액소송 단계별 가이드 (서브그래프) | `small_claims_subgraph` | ✅ |
| `StoryboardAgent` | 사건 타임라인 LLM 생성 | `storyboard_node` | ❌ |
| `LawyerStatsAgent` | 변호사 통계 대시보드 안내 | `lawyer_stats_node` | ❌ |
| `LawStudyAgent` | 로스쿨 학습 가이드 (RAG+LLM) | `law_study_node` | ✅ |
| `SimpleChatAgent` | 일반 LLM 채팅 (폴백) | `simple_chat_node` | ❌ |

### 통합 채팅 API

```bash
# 새 통합 채팅 API
POST /api/chat
{
  "message": "손해배상 판례 알려줘",
  "history": [],
  "session_data": {}
}
```

프론트엔드 `ChatWidget`은 `/api/chat` 사용 (기존 `/api/multi-agent/chat` 대체)

## Vector DB (LanceDB)

법령/판례 임베딩 데이터를 LanceDB에 저장합니다.

### 임베딩 스크립트
```bash
cd backend

# PyTorch CUDA 설치 (GPU 사용 시)
uv pip install --reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 임베딩 생성 (--no-sync 필수)
uv run --no-sync python scripts/runpod_lancedb_embeddings.py --type all --reset

# 통계 확인
uv run --no-sync python scripts/runpod_lancedb_embeddings.py --stats
```

### 저장 위치
- `backend/lancedb_data/` - LanceDB 데이터
- 테이블: `legal_chunks` (법령 + 판례 통합)

### 데이터 현황
| 타입 | 원본 건수 | 임베딩 청크 |
|------|-----------|-------------|
| 판례 | 65,107건 | 134,846개 |
| 법령 | 5,841건 | 118,922개 |

### 관련 문서
- `docs/vectordb_design.md` - 벡터 DB 설계
- `backend/scripts/CLAUDE.md` - 임베딩 스크립트 가이드
- `docs/EMBEDDING_DEV_LOG_20260129.md` - 개발 로그

## Embedding Model (임베딩 모델)

검색 API 사용 전 임베딩 모델(약 2.3GB)을 먼저 다운로드해야 합니다.

```bash
cd backend

# 모델 다운로드
uv run python scripts/download_models.py

# 캐시 상태 확인
uv run python scripts/download_models.py --check
```

| 항목 | 값 |
|------|-----|
| 모델명 | `nlpai-lab/KURE-v1` |
| 크기 | 약 2.3GB |
| 캐시 경로 | `backend/data/models/` |

> **참고**: 서버는 모델 없이도 시작되지만, 검색 API 호출 시 503 에러가 반환됩니다.
> 상세 내용은 `backend/CLAUDE.md`의 "Embedding Model" 섹션 참조.

## RAG 평가 시스템

RAG 챗봇 평가 데이터셋 생성 및 Gradio 분석 UI 제공

### 빠른 시작
```bash
cd backend

# Gradio UI 실행
uv run python -m evaluation
# → http://localhost:7860 접속

# Solar 자동 질문 생성
uv run python -m evaluation.tools.solar_generator --count 30

# 평가 실행
uv run python -m evaluation.runners.evaluation_runner \
    --dataset evaluation/datasets/eval_dataset_v1.json

# 데이터셋 검증
uv run python -m evaluation.tools.validate_dataset eval_dataset_v1.json
```

### 성능 목표
| 지표 | 목표값 |
|------|--------|
| Recall@5 | ≥ 0.7 |
| Recall@10 | ≥ 0.8 |
| MRR | ≥ 0.7 |
| Hit Rate | ≥ 0.9 |
| NDCG@10 | ≥ 0.75 |

### 관련 문서
- `backend/evaluation/CLAUDE.md` - 평가 시스템 상세 가이드

## Graph DB (Neo4j)

법령 계급, 판례 인용 관계를 Neo4j 그래프로 저장합니다.

### 빠른 시작
```bash
# Neo4j 컨테이너 실행
docker compose up -d neo4j

# 그래프 구축 (초기 1회)
cd backend
uv run python scripts/build_graph.py

# 검증
NEO4J_PASSWORD=password uv run python scripts/verify_graph.py

# Gradio UI 검증
NEO4J_PASSWORD=password uv run python scripts/verify_gradio.py
# → http://localhost:7860
```

### 그래프 스키마

**노드 (Nodes)**
| Label | 설명 | 개수 |
|-------|------|------|
| Statute | 법령 | 5,572 |
| Case | 판례 | 65,107 |

**관계 (Relationships)**
| Type | 설명 | 개수 |
|------|------|------|
| HIERARCHY_OF | 법령 계급 (시행령→법률) | 3,624 |
| CITES | 판례→법령 인용 | 72,414 |
| CITES_CASE | 판례→판례 인용 | 87,654 |
| RELATED_TO | 법령→법령 관련 | 93 |

### 환경 변수
```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

### 테스트
```bash
NEO4J_PASSWORD=password uv run python tests/integration/test_neo4j_graph.py
```

### 활용 시나리오
1. **RAG 컨텍스트 보강** - 검색된 법령/판례의 관련 정보 추가
2. **법령 탐색 UI** - 법령 계급도 시각화, 인용 네트워크
3. **판례 추천** - 유사 판례 찾기 (같은 법령 인용, 인용 관계)

### 관련 문서
- `.claude/skills/neo4j-graph-construction/SKILL.md` - 그래프 구축 스킬

## Lawyer DB (PostgreSQL)

변호사 데이터(17,326건)를 PostgreSQL `lawyers` 테이블에 저장하고, DB 기반으로 검색/통계를 수행합니다.

### 활성화

```bash
# backend/.env
USE_DB_LAWYERS=true

# 마이그레이션 + 데이터 로드
cd backend
uv run alembic upgrade head
uv run python scripts/load_lawyers_data.py
uv run python scripts/load_lawyers_data.py --verify  # 검증
```

### 롤백

`USE_DB_LAWYERS=false`로 설정하면 즉시 JSON 파일 모드로 복귀합니다.
JSON 파일(`data/lawyers_with_coords.json`)은 변경하지 않으므로 데이터 손실 없음.

### 관련 파일

| 파일 | 설명 |
|------|------|
| `backend/app/models/lawyer.py` | Lawyer ORM 모델 |
| `backend/alembic/versions/004_add_lawyers_table.py` | 마이그레이션 |
| `backend/scripts/load_lawyers_data.py` | 데이터 로드 스크립트 |
| `backend/app/services/service_function/lawyer_db_service.py` | DB 기반 검색 서비스 |
| `backend/app/services/service_function/lawyer_stats_db_service.py` | DB 기반 통계 서비스 |
| `backend/tests/integration/test_lawyer_db.py` | 통합 테스트 |

### 관련 스킬
- `.claude/skills/postgresql-migration/SKILL.md` - PostgreSQL 마이그레이션 패턴
- `.claude/skills/spatial-query-patterns/SKILL.md` - 위치 기반 검색 패턴

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
| `backend/app/models/trial_statistics.py` | TrialStatistics ORM 모델 |
| `backend/alembic/versions/005_add_trial_statistics_table.py` | 마이그레이션 |

## Modules

### lawyer-stats (변호사 통계 대시보드)

지역별·전문분야별 변호사 분포 및 시장 분석 대시보드

**주요 기능:**
- 지역별 변호사 현황 (시/도 → 시/군/구 드릴다운)
- 인구 대비 밀도 분석 (변호사 수 / 인구 × 10만명)
- 향후 예측 모드 (2030/2035/2040년 추계인구 기반)
- 전문분야별 변호사 분포
- 지역×전문분야 교차 분석 히트맵

**API 엔드포인트:**
- `GET /api/lawyer-stats/overview` - 전체 현황 요약
- `GET /api/lawyer-stats/by-region` - 지역별 변호사 수
- `GET /api/lawyer-stats/density-by-region` - 지역별 밀도 (year, include_change 파라미터)
- `GET /api/lawyer-stats/by-specialty` - 전문분야별 통계
- `GET /api/lawyer-stats/cross-analysis` - 지역×전문분야 교차 분석
- `GET /api/lawyer-stats/region/{region}/specialties` - 특정 지역 전문분야 상세

**프론트엔드 컴포넌트:**
- `RegionGeoMap` - 대한민국 시군구 지도 시각화 (TopoJSON)
- `RegionDetailList` - 지역 상세 목록 및 예측 상세 뷰
- `CrossAnalysisHeatmap` - 지역×전문분야 히트맵
- `SpecialtyBarChart` - 전문분야별 바 차트
- `StickyTabNav` - 스크롤 연동 탭 네비게이션
