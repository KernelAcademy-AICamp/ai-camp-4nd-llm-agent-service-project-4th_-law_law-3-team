# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

법률 서비스 플랫폼 - 모듈형 아키텍처로 기능을 유연하게 추가/삭제할 수 있는 법률 서비스 플랫폼입니다.

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
- `service/` - 비즈니스 로직
- `schema/` - Pydantic 스키마
- `model/` - DB 모델

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
- `backend/app/core/registry.py` - 모듈 자동 등록
- `frontend/src/lib/modules.ts` - 프론트엔드 모듈 정의
- `frontend/src/lib/api.ts` - API 클라이언트 및 endpoints
- `scripts/add_module.py` - 모듈 생성 스크립트

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
