# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Language

모든 응답, 분석 결과, 보고서, 스킬 출력은 **한국어**로 작성합니다.
코드 주석과 docstring은 기존 스타일(한국어)을 유지합니다.
커밋 메시지는 기존 규칙(타입은 영어, 본문은 한국어)을 따릅니다.

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

## Claude Code Hooks

`Edit`/`Write` 후 자동 린트 검증 훅 포함. 활성화 설정은 각자 로컬에서 수행.
→ 상세: `.claude/hooks/README.md`

## Git Branch Strategy

- 기본 브랜치: `main` (프로덕션), `dev` (개발 통합)
- feature 브랜치는 **`dev` 기반**으로 생성 (`main` 기반 X)
- 워크플로우: `dev` → `feature/xxx` → PR → `dev` → `main`
- PR 타이틀: `git-convention.md`의 커밋 타입 규칙 준수 (GitHub Actions로 자동 검증)

## Commands

| 영역 | 핵심 명령어 |
|------|-----------|
| Backend | `cd backend && uv sync --dev && uv run uvicorn app.main:app --reload` |
| Frontend | `cd frontend && npm install && npm run dev` |
| 모듈 추가 | `python3 scripts/add_module.py <module_name> "<description>"` |

→ 상세: `backend/CLAUDE.md`, `frontend/CLAUDE.md`

## Architecture

### 모듈 자동 등록

**Backend**: `backend/app/core/registry.py`의 `ModuleRegistry`가 `backend/app/modules/` 폴더를 스캔하여 자동으로 라우터 등록. 각 모듈의 `router/__init__.py`에 `router = APIRouter()` 정의 필요.

**Frontend**: `frontend/src/lib/modules.ts`에서 모듈 정의 후 `getEnabledModules()`로 활성화된 모듈만 표시.

### API 경로 규칙

Backend 모듈명 `snake_case` → API 경로 `/api/kebab-case` (예: `lawyer_finder` → `/api/lawyer-finder`)

### 모듈 비활성화

Backend: `.env`에서 `ENABLED_MODULES=["module1","module2"]` (빈 배열이면 모두 활성화)
Frontend: `modules.ts`에서 `enabled: false`

## Key Files

- `backend/app/main.py` - FastAPI 앱 진입점
- `backend/app/core/config.py` - 환경 설정 (pydantic-settings)
- `backend/app/core/database.py` - DB 연결 (SQLAlchemy)
- `backend/app/core/registry.py` - 모듈 자동 등록
- `frontend/src/lib/modules.ts` - 프론트엔드 모듈 정의
- `frontend/src/lib/api.ts` - API 클라이언트 및 endpoints
- `scripts/add_module.py` - 모듈 생성 스크립트

## 모듈 매핑

| Backend 모듈 | API 경로 | Frontend ID | api.ts key |
|-------------|---------|-------------|-----------|
| `case_precedent` | `/api/case-precedent` | `case-precedent` | `casePrecedent` |
| `lawyer_finder` | `/api/lawyer-finder` | `lawyer-finder` | `lawyerFinder` |
| `lawyer_stats` | `/api/lawyer-stats` | `lawyer-stats` | `lawyerStat` |
| `small_claims` | `/api/small-claims` | `small-claims` | `smallClaims` |
| `storyboard` | `/api/storyboard` | `storyboard` | `storyboard` |
| `law_study` | `/api/law-study` | `law-study` | `lawStudy` |
| `mock_trial` | `/api/mock-trial` | `mock-trial` | `mockTrial` |

## Backend Architecture

통합 채팅 API (`POST /api/chat`)와 8개 에이전트 기반 멀티에이전트 시스템 (LangGraph).
→ 상세: `backend/CLAUDE.md`

## DB / 인프라 요약

| 인프라 | 용도 | Feature Flag | 상세 문서 |
|--------|------|-------------|----------|
| **PostgreSQL** | 변호사(17,326건), 법률용어(72,700건), 재판통계, 법령/판례 원본, 인제스트 21개 타입 원본 | `USE_DB_LAWYERS`, `USE_LEGAL_TERM_DICT` | `backend/CLAUDE.md` |
| **LanceDB** | 법령+판례 등 21개 타입 벡터 임베딩 `legal_chunks`(656,532, `summary_type`/`article_number` 포함), 자치법규 `local_ordinance_chunks`(~240만), FTS | `LANCEDB_MODE`, `LANCEDB_INDEX_TYPE` | `backend/CLAUDE.md` |
| **Neo4j** | 법령 계급, 판례 인용 그래프 (5,572 법령 + 65,107 판례) | - | `backend/CLAUDE.md` |
| **MeCab userdic** | 법률 복합명사 사전 (37,366+ 엔트리, `scripts/manual_terms.json` 수동 보강 포함) | - | `backend/CLAUDE.md` |
| **Google Drive (ONNX)** | ONNX 최적화 모델 (리랭커/임베딩 FP32, QDQ INT8) | `USE_ONNX_RERANKER`, `ONNX_RERANKER_VARIANT` | `docs/operations/backup-restore.md` |

임베딩 모델: `nlpai-lab/KURE-v1` (2.3GB) + 리랭커: `dragonkue/bge-reranker-v2-m3-ko` (2.1GB), `uv run python scripts/download_models.py`로 다운로드 필요.

## RAG 평가 시스템

RAG 검색 품질 평가. 목표: Recall@10 ≥ 0.8, MRR ≥ 0.7, Hit Rate ≥ 0.9.
→ 상세: `backend/evaluation/CLAUDE.md`

## DB 백업 / 복원

PostgreSQL, Neo4j, LanceDB → Google Drive (rclone).
→ 상세: `docs/operations/backup-restore.md`

## 상세 참조 인덱스

| 문서 | 설명 |
|------|------|
| `backend/CLAUDE.md` | Backend 아키텍처, 폴더 구조, 에이전트, DB, 벡터 DB, 환경변수 |
| `frontend/CLAUDE.md` | Frontend 컴포넌트, 모듈 시스템, API 프록시, 스타일 |
| `backend/scripts/CLAUDE.md` | 임베딩, EDA, 데이터 로드 스크립트 |
| `backend/evaluation/CLAUDE.md` | RAG 평가 시스템 상세 |
| `docs/operations/backup-restore.md` | DB 백업/복원 상세 (rclone) |
| `docs/operations/wsl2-docker-guide.md` | WSL2 Docker 상세 명령어 |
| `.claude/hooks/README.md` | Hooks 설정 상세 (JSON 예시) |
| `.claude/skills/CATALOG.md` | 스킬/에이전트/규칙 카탈로그 |
