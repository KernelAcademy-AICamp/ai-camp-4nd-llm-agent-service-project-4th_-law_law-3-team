# AGENTS.md

법률 서비스 플랫폼 - AI 에이전트/자동화 도구용 간결한 프로젝트 가이드.

## 프로젝트 개요

모듈형 법률 서비스 플랫폼. Backend(FastAPI + Python), Frontend(Next.js + TypeScript).
8개 AI 에이전트 기반 멀티에이전트 시스템 (LangGraph).

## 빌드 & 테스트

```bash
# Backend
cd backend
uv sync --dev                        # 의존성 설치
uv run ruff check backend/app/       # 린트
uv run mypy backend/app/             # 타입 체크
uv run pytest                        # 테스트

# Frontend
cd frontend
npm install                          # 의존성 설치
npm run build                        # 빌드 (tsc + ESLint)
npm run lint                         # 린트
```

## 코드 스타일

| 규칙 | Python | TypeScript |
|------|--------|-----------|
| 변수/함수 | `snake_case` | `camelCase` |
| 클래스/컴포넌트 | `PascalCase` | `PascalCase` |
| 상수 | `UPPER_SNAKE_CASE` | `UPPER_SNAKE_CASE` |
| 린트 | `ruff check` | ESLint |
| 포맷 | `ruff format` | Prettier |
| 타입 체크 | `mypy` | `tsc --noEmit` |

- 타입 어노테이션 필수 (양쪽 모두)
- `any` 타입 사용 금지 (TypeScript)
- 디버그 코드 커밋 금지 (`print`, `console.log`)
- 하드코딩된 비밀번호/API 키 금지

## 주요 디렉토리

```
backend/app/
├── core/           # config, database, registry
├── multi_agent/    # LangGraph 에이전트 시스템
├── services/       # RAG, 비즈니스 로직
├── modules/        # 독립 API 모듈 (9개)
├── models/         # SQLAlchemy ORM
└── tools/          # LLM, VectorStore, Neo4j

frontend/src/
├── app/            # Next.js App Router
├── features/       # 모듈별 컴포넌트/훅/타입
├── components/     # 공통 UI 컴포넌트
└── lib/            # modules.ts, api.ts
```

## 필수 환경변수

| 변수 | 설명 |
|------|------|
| `DATABASE_URL` | PostgreSQL 연결 문자열 |
| `UPSTAGE_API_KEY` | Solar LLM API 키 |
| `NEO4J_URI` | Neo4j bolt 연결 |
| `USE_LOCAL_EMBEDDING` | 로컬 임베딩 모델 사용 (기본: true) |
| `USE_PG_GRAPH` | PostgreSQL 그래프 사용 (Neo4j 대체, 기본: false) |

## API 경로 규칙

Backend 모듈명 `snake_case` → API 경로 `/api/kebab-case`
- `lawyer_finder` → `/api/lawyer-finder`
- `mock_trial` → `/api/mock-trial`

## 커밋 메시지

```
<타입>(<범위>): <제목>   # 타입: feat, fix, docs, refactor, test 등
```
타입은 영어, 본문은 한국어. 72자 이내.
