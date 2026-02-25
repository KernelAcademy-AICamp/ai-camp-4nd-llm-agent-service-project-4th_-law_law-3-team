---
name: security-authentication
description: |
  FastAPI 백엔드 + Next.js 프론트엔드의 보안/인증 구현 가이드.
  API Key 인증 강화, JWT Bearer 토큰, Rate Limiting 전략, 입력 검증, Prompt Injection 방어.
  인증/보안 관련 코드 작성, API 보호, 미들웨어 구현, 프론트엔드 인증 통합 시 반드시 사용.
  보안 감사 결과 대응, 인증 미들웨어 추가, Rate Limiting 설정 변경 시에도 사용.
---

# Security & Authentication

FastAPI + Next.js 법률 서비스 플랫폼의 보안/인증 구현 패턴.

## 현재 인증 아키텍처

```
클라이언트 → Next.js 프록시 (/api/*) → FastAPI 미들웨어 → 라우터
                                         ├─ verify_api_key (전역)
                                         ├─ Rate Limiting (slowapi)
                                         └─ 공개 경로 제외 (/health, /docs)
```

### 핵심 파일

| 파일 | 역할 |
|------|------|
| `backend/app/core/auth.py` | API Key 검증 미들웨어 |
| `backend/app/core/rate_limit.py` | Rate Limiting 설정 |
| `backend/app/core/config.py` | 보안 환경변수 정의 |
| `backend/app/main.py` | 미들웨어 등록, CORS, 전역 인증 |
| `frontend/src/lib/api.ts` | Axios 인스턴스 (인증 헤더) |

---

## 1. API Key 인증

### 현재 구현

`backend/app/core/auth.py`에서 `X-API-Key` 헤더 검증:
- `API_KEY=""` (기본값) → 인증 비활성화 (개발 모드)
- `API_KEY="xxx"` → 모든 엔드포인트에 전역 적용
- 공개 경로: `/health`, `/docs`, `/openapi.json`, `/redoc`

### 강화 패턴

```python
# backend/app/core/auth.py 패턴
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    if not settings.API_KEY:
        return "dev-mode"  # 개발 모드
    if not api_key or api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return api_key
```

### 프론트엔드 API Key 전달

```typescript
// frontend/src/lib/api.ts 패턴
const api = axios.create({
  baseURL: '/api',
  headers: { 'Content-Type': 'application/json' },
  timeout: 60000,
})

// 프록시 환경에서는 Next.js API Route에서 서버 측 주입
// 직접 연결 시 인터셉터로 추가
api.interceptors.request.use((config) => {
  const apiKey = process.env.NEXT_PUBLIC_API_KEY
  if (apiKey) {
    config.headers['X-API-Key'] = apiKey
  }
  return config
})
```

### 환경변수 체크리스트

| 변수 | 기본값 | 프로덕션 필수 |
|------|--------|-------------|
| `API_KEY` | `""` (비활성화) | 강력한 랜덤 키 설정 |
| `CORS_ORIGINS` | `["http://localhost:3000"]` | 프로덕션 도메인만 |
| `ENVIRONMENT` | `"development"` | `"production"` |
| `DEBUG` | `False` | `False` 유지 |

---

## 2. JWT Bearer 토큰 (확장 시)

프로젝트에 `python-jose[cryptography]`, `passlib[bcrypt]` 이미 설치됨 (현재 미사용).

### 구현 패턴

```python
# backend/app/core/jwt.py 패턴
from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

SECRET_KEY = settings.JWT_SECRET_KEY  # .env에서 로드
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

### JWT 환경변수

| 변수 | 설명 | 생성 |
|------|------|------|
| `JWT_SECRET_KEY` | 서명 키 | `openssl rand -hex 32` |
| `JWT_ALGORITHM` | 알고리즘 | `HS256` (기본) |
| `JWT_EXPIRE_MINUTES` | 만료 시간 | `30` (기본) |

---

## 3. Rate Limiting

### 현재 구현

`backend/app/core/rate_limit.py`에서 slowapi 기반:

| 설정 | 기본값 | 적용 대상 |
|------|--------|----------|
| `RATE_LIMIT_PER_MINUTE` | 30 | 일반 API |
| `RATE_LIMIT_AI_PER_MINUTE` | 10 | `/api/chat`, `/api/chat/stream` |
| `RATE_LIMIT_STORAGE_URI` | `"memory://"` | 스토리지 (프로덕션: Redis) |

### 모듈별 Rate Limiting 패턴

```python
# 라우터에 엔드포인트별 제한 추가
from app.core.rate_limit import limiter, DEFAULT_RATE_LIMIT, AI_RATE_LIMIT

@router.get("/nearby")
@limiter.limit(DEFAULT_RATE_LIMIT)  # 30/분
async def get_nearby_lawyers(request: Request, ...):
    ...

@router.post("/chat")
@limiter.limit(AI_RATE_LIMIT)  # 10/분
async def chat(request: Request, ...):
    ...
```

### 프로덕션 Rate Limiting (Redis)

```bash
# .env (프로덕션)
RATE_LIMIT_STORAGE_URI=redis://redis:6379
```

```toml
# pyproject.toml에 추가 필요
[project.dependencies]
redis = ">=5.0.0"
```

### 엔드포인트별 권장 제한

| 엔드포인트 | 제한 | 이유 |
|-----------|------|------|
| `POST /api/chat` | 10/분 | LLM 비용 |
| `POST /api/chat/stream` | 10/분 | LLM 비용 |
| `GET /api/lawyer-finder/*` | 30/분 | DB 조회 |
| `GET /api/case-precedent/*` | 20/분 | 벡터 검색 비용 |
| `GET /api/lawyer-stats/*` | 60/분 | 경량 통계 |
| `GET /health` | 제한 없음 | 모니터링 |

---

## 4. 입력 검증 & Prompt Injection 방어

### 메시지 길이 제한

```python
# backend/app/api/schema/chat.py 패턴
from pydantic import BaseModel, Field, field_validator

MAX_MESSAGE_LENGTH = 5000  # 자

class ChatRequest(BaseModel):
    message: str = Field(..., max_length=MAX_MESSAGE_LENGTH)
    thread_id: str | None = None

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("메시지가 비어있습니다")
        return stripped
```

### Prompt Injection 기본 방어

```python
# 시스템 프롬프트와 사용자 입력 분리
messages = [
    ("system", system_prompt),      # 시스템 지시사항 (변경 불가)
    *history_messages,               # 대화 히스토리
    ("user", user_message),          # 사용자 입력 (격리)
]

# 사용자 입력을 시스템 프롬프트에 직접 삽입하지 않음
# BAD:  system_prompt = f"...{user_input}..."
# GOOD: messages = [("system", prompt), ("user", user_input)]
```

### 입력 정제 패턴

```python
import re

DANGEROUS_PATTERNS = [
    r"ignore\s+(previous|above|all)\s+instructions",
    r"you\s+are\s+now\s+",
    r"system\s*:\s*",
    r"<\|.*?\|>",
]

def sanitize_input(text: str) -> str:
    """위험 패턴 감지 (차단이 아닌 로깅 + 경고)"""
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            logger.warning(f"Suspicious input detected: {pattern}")
    return text.strip()[:MAX_MESSAGE_LENGTH]
```

---

## 5. CORS 설정

### 현재 설정 (`main.py`)

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,  # 기본: ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)
```

### 프로덕션 체크리스트

- [ ] `CORS_ORIGINS`에 프로덕션 도메인만 포함
- [ ] `allow_origins=["*"]` 절대 금지
- [ ] `allow_credentials=True` 시 와일드카드 불가 (브라우저 차단)
- [ ] Vercel 도메인 + 커스텀 도메인 모두 포함

---

## 6. 보안 헤더

### Nginx 보안 헤더 (프로덕션)

```nginx
# docker/nginx/conf.d/api.conf.template에 추가
add_header X-Content-Type-Options "nosniff" always;
add_header X-Frame-Options "DENY" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy "strict-origin-when-cross-origin" always;
add_header Content-Security-Policy "default-src 'self'" always;
```

### FastAPI 보안 헤더 미들웨어

```python
from starlette.middleware.base import BaseHTTPMiddleware

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        return response
```

---

## 7. 환경별 보안 설정 매트릭스

| 설정 | 개발 | Docker | 프로덕션 |
|------|------|--------|---------|
| `API_KEY` | `""` (비활성화) | 설정 | 강력한 키 |
| `DEBUG` | `True` 가능 | `False` | `False` |
| `CORS_ORIGINS` | `localhost:3000` | 내부망 | 도메인만 |
| Rate Limit 스토리지 | `memory://` | `memory://` | `redis://` |
| HTTPS | 불필요 | 선택 | 필수 |
| `/docs` 접근 | 허용 | 허용 | 차단 권장 |

---

## 8. 보안 점검 체크리스트

### 배포 전 필수

- [ ] `API_KEY` 설정됨 (빈 문자열 아님)
- [ ] `CORS_ORIGINS` 프로덕션 도메인만
- [ ] `DEBUG=False`
- [ ] Rate Limiting 활성화
- [ ] 메시지 길이 제한 적용
- [ ] DB 비밀번호 변경 (`change_me_to_a_strong_password` 아님)
- [ ] `.env` 파일 `.gitignore` 포함 확인

### 정기 점검

- [ ] 의존성 보안 업데이트 (`uv run pip-audit`)
- [ ] Rate Limiting 로그 모니터링
- [ ] API Key 로테이션 (분기별)
- [ ] Prompt Injection 시도 로그 확인
