# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
```

## Conventions

- 라우터 함수는 `async def` 사용
- Pydantic v2 문법 사용 (`model_validator`, `field_validator`)
- 타입 힌트 필수 (mypy strict 모드)
- ruff 린터 규칙: E, F, I, N, W
