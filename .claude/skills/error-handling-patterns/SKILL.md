---
name: error-handling-patterns
description: Backend(FastAPI HTTPException)와 Frontend(Error Boundary, try-catch) 에러 처리 표준화 패턴. 에러 응답 형식 통일, 사용자 친화적 에러 메시지, 로깅 전략 포함. 에러 처리 구현, API 에러 응답 설계, 예외 처리 리뷰 시 사용.
---

# Error Handling Patterns

Backend(FastAPI)와 Frontend(Next.js) 에러 처리 표준화 가이드.

## 1. 에러 응답 형식 통일

### 1.1 표준 에러 응답 스키마

```python
# backend/app/core/errors.py
from pydantic import BaseModel
from typing import Optional, Any

class ErrorResponse(BaseModel):
    """표준 에러 응답 형식"""
    error: str                    # 에러 코드 (예: "VALIDATION_ERROR")
    message: str                  # 사용자 친화적 메시지
    detail: Optional[Any] = None  # 상세 정보 (개발용)

    class Config:
        json_schema_extra = {
            "example": {
                "error": "NOT_FOUND",
                "message": "요청한 리소스를 찾을 수 없습니다.",
                "detail": {"resource": "lawyer", "id": 123}
            }
        }
```

### 1.2 에러 코드 정의

```python
# backend/app/core/errors.py
from enum import Enum

class ErrorCode(str, Enum):
    # 400 Bad Request
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INVALID_INPUT = "INVALID_INPUT"

    # 401 Unauthorized
    UNAUTHORIZED = "UNAUTHORIZED"
    INVALID_TOKEN = "INVALID_TOKEN"

    # 403 Forbidden
    FORBIDDEN = "FORBIDDEN"

    # 404 Not Found
    NOT_FOUND = "NOT_FOUND"
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"

    # 409 Conflict
    CONFLICT = "CONFLICT"
    ALREADY_EXISTS = "ALREADY_EXISTS"

    # 422 Unprocessable Entity
    UNPROCESSABLE = "UNPROCESSABLE"

    # 500 Internal Server Error
    INTERNAL_ERROR = "INTERNAL_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"

    # 503 Service Unavailable
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    MODEL_NOT_LOADED = "MODEL_NOT_LOADED"
```

## 2. Backend (FastAPI) 에러 처리

### 2.1 커스텀 예외 클래스

```python
# backend/app/core/errors.py
from fastapi import HTTPException, status

class AppException(HTTPException):
    """애플리케이션 기본 예외"""
    def __init__(
        self,
        status_code: int,
        error: str,
        message: str,
        detail: Any = None,
    ):
        super().__init__(
            status_code=status_code,
            detail={
                "error": error,
                "message": message,
                "detail": detail,
            }
        )

class NotFoundError(AppException):
    """리소스 없음 (404)"""
    def __init__(self, resource: str, identifier: Any = None):
        detail = {"resource": resource}
        if identifier:
            detail["id"] = identifier
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            error=ErrorCode.NOT_FOUND,
            message=f"요청한 {resource}을(를) 찾을 수 없습니다.",
            detail=detail,
        )

class ValidationError(AppException):
    """입력 검증 실패 (400)"""
    def __init__(self, message: str, detail: Any = None):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            error=ErrorCode.VALIDATION_ERROR,
            message=message,
            detail=detail,
        )

class ServiceUnavailableError(AppException):
    """서비스 이용 불가 (503)"""
    def __init__(self, service: str, message: str = None):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error=ErrorCode.SERVICE_UNAVAILABLE,
            message=message or f"{service} 서비스를 일시적으로 이용할 수 없습니다.",
            detail={"service": service},
        )

class ExternalServiceError(AppException):
    """외부 서비스 오류 (502)"""
    def __init__(self, service: str, original_error: str = None):
        super().__init__(
            status_code=status.HTTP_502_BAD_GATEWAY,
            error=ErrorCode.EXTERNAL_SERVICE_ERROR,
            message=f"외부 서비스({service}) 연동 중 오류가 발생했습니다.",
            detail={"service": service, "original_error": original_error},
        )
```

### 2.2 전역 예외 핸들러

```python
# backend/app/main.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from app.core.errors import ErrorCode, AppException
import logging

logger = logging.getLogger(__name__)

app = FastAPI()

@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    """애플리케이션 예외 핸들러"""
    logger.warning(f"AppException: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.detail,
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Pydantic 검증 에러 핸들러"""
    errors = exc.errors()
    logger.warning(f"ValidationError: {errors}")
    return JSONResponse(
        status_code=422,
        content={
            "error": ErrorCode.VALIDATION_ERROR,
            "message": "입력 데이터가 올바르지 않습니다.",
            "detail": errors,
        },
    )

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """처리되지 않은 예외 핸들러"""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": ErrorCode.INTERNAL_ERROR,
            "message": "서버 내부 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            "detail": None,  # 프로덕션에서는 상세 정보 숨김
        },
    )
```

### 2.3 라우터에서 예외 사용

```python
# backend/app/modules/lawyer_finder/router/__init__.py
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.errors import NotFoundError, ValidationError

router = APIRouter()

@router.get("/lawyers/{lawyer_id}")
async def get_lawyer(lawyer_id: int, db: AsyncSession = Depends(get_db)):
    lawyer = await get_lawyer_by_id(db, lawyer_id)
    if not lawyer:
        raise NotFoundError(resource="변호사", identifier=lawyer_id)
    return lawyer

@router.get("/lawyers/nearby")
async def get_nearby_lawyers(
    latitude: float,
    longitude: float,
    db: AsyncSession = Depends(get_db),
):
    if not (-90 <= latitude <= 90):
        raise ValidationError(
            message="위도는 -90 ~ 90 범위여야 합니다.",
            detail={"field": "latitude", "value": latitude}
        )
    # ...
```

## 3. Frontend (Next.js) 에러 처리

### 3.1 API 에러 타입 정의

```typescript
// frontend/src/types/error.ts
export interface ApiError {
  error: string;
  message: string;
  detail?: unknown;
}

export function isApiError(error: unknown): error is ApiError {
  return (
    typeof error === 'object' &&
    error !== null &&
    'error' in error &&
    'message' in error
  );
}

export class ApiException extends Error {
  constructor(
    public readonly error: string,
    message: string,
    public readonly detail?: unknown,
    public readonly status?: number,
  ) {
    super(message);
    this.name = 'ApiException';
  }

  static fromResponse(data: ApiError, status?: number): ApiException {
    return new ApiException(data.error, data.message, data.detail, status);
  }
}
```

### 3.2 API 클라이언트 에러 처리

```typescript
// frontend/src/lib/api.ts
import { ApiError, ApiException, isApiError } from '@/types/error';

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  async request<T>(
    endpoint: string,
    options: RequestInit = {},
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;

    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
      });

      const data = await response.json();

      if (!response.ok) {
        if (isApiError(data)) {
          throw ApiException.fromResponse(data, response.status);
        }
        throw new Error(`HTTP Error: ${response.status}`);
      }

      return data as T;
    } catch (error) {
      if (error instanceof ApiException) {
        throw error;
      }

      // 네트워크 에러 등
      throw new ApiException(
        'NETWORK_ERROR',
        '네트워크 연결을 확인해주세요.',
        { originalError: String(error) },
      );
    }
  }

  async get<T>(endpoint: string): Promise<T> {
    return this.request<T>(endpoint, { method: 'GET' });
  }

  async post<T>(endpoint: string, body: unknown): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'POST',
      body: JSON.stringify(body),
    });
  }
}

export const api = new ApiClient('/api');
```

### 3.3 React Error Boundary

```typescript
// frontend/src/components/ErrorBoundary.tsx
'use client';

import { Component, ErrorInfo, ReactNode } from 'react';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
}

interface State {
  hasError: boolean;
  error?: Error;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    this.props.onError?.(error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback || <DefaultErrorFallback error={this.state.error} />;
    }

    return this.props.children;
  }
}

function DefaultErrorFallback({ error }: { error?: Error }) {
  return (
    <div className="p-6 bg-red-50 border border-red-200 rounded-lg">
      <h2 className="text-lg font-semibold text-red-800 mb-2">
        문제가 발생했습니다
      </h2>
      <p className="text-red-600 mb-4">
        {error?.message || '알 수 없는 오류가 발생했습니다.'}
      </p>
      <button
        onClick={() => window.location.reload()}
        className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
      >
        페이지 새로고침
      </button>
    </div>
  );
}
```

### 3.4 Next.js App Router error.tsx

```typescript
// frontend/src/app/error.tsx
'use client';

import { useEffect } from 'react';

interface ErrorPageProps {
  error: Error & { digest?: string };
  reset: () => void;
}

export default function ErrorPage({ error, reset }: ErrorPageProps) {
  useEffect(() => {
    // 에러 로깅 서비스에 전송
    console.error('Page Error:', error);
  }, [error]);

  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <div className="max-w-md w-full bg-white rounded-lg shadow-lg p-6">
        <h1 className="text-2xl font-bold text-gray-900 mb-4">
          오류가 발생했습니다
        </h1>
        <p className="text-gray-600 mb-6">
          페이지를 불러오는 중 문제가 발생했습니다.
          잠시 후 다시 시도해주세요.
        </p>
        <div className="flex gap-4">
          <button
            onClick={reset}
            className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            다시 시도
          </button>
          <button
            onClick={() => window.location.href = '/'}
            className="flex-1 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
          >
            홈으로
          </button>
        </div>
      </div>
    </div>
  );
}
```

### 3.5 Next.js not-found.tsx

```typescript
// frontend/src/app/not-found.tsx
import Link from 'next/link';

export default function NotFound() {
  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <div className="text-center">
        <h1 className="text-6xl font-bold text-gray-300 mb-4">404</h1>
        <h2 className="text-2xl font-semibold text-gray-900 mb-4">
          페이지를 찾을 수 없습니다
        </h2>
        <p className="text-gray-600 mb-8">
          요청하신 페이지가 존재하지 않거나 이동되었을 수 있습니다.
        </p>
        <Link
          href="/"
          className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          홈으로 돌아가기
        </Link>
      </div>
    </div>
  );
}
```

## 4. 컴포넌트 에러 처리 패턴

### 4.1 커스텀 훅에서 에러 처리

```typescript
// frontend/src/hooks/useApi.ts
import { useState, useCallback } from 'react';
import { ApiException } from '@/types/error';

interface UseApiState<T> {
  data: T | null;
  loading: boolean;
  error: ApiException | null;
}

export function useApi<T>() {
  const [state, setState] = useState<UseApiState<T>>({
    data: null,
    loading: false,
    error: null,
  });

  const execute = useCallback(async (apiCall: () => Promise<T>) => {
    setState({ data: null, loading: true, error: null });

    try {
      const data = await apiCall();
      setState({ data, loading: false, error: null });
      return data;
    } catch (error) {
      const apiError = error instanceof ApiException
        ? error
        : new ApiException('UNKNOWN_ERROR', '알 수 없는 오류가 발생했습니다.');
      setState({ data: null, loading: false, error: apiError });
      throw apiError;
    }
  }, []);

  const reset = useCallback(() => {
    setState({ data: null, loading: false, error: null });
  }, []);

  return { ...state, execute, reset };
}
```

### 4.2 에러 표시 컴포넌트

```typescript
// frontend/src/components/ErrorMessage.tsx
import { ApiException } from '@/types/error';

interface ErrorMessageProps {
  error: ApiException | Error | null;
  onRetry?: () => void;
}

export function ErrorMessage({ error, onRetry }: ErrorMessageProps) {
  if (!error) return null;

  const message = error instanceof ApiException
    ? error.message
    : '오류가 발생했습니다.';

  return (
    <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
      <div className="flex items-start gap-3">
        <svg
          className="w-5 h-5 text-red-500 mt-0.5"
          fill="currentColor"
          viewBox="0 0 20 20"
        >
          <path
            fillRule="evenodd"
            d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
            clipRule="evenodd"
          />
        </svg>
        <div className="flex-1">
          <p className="text-red-800">{message}</p>
          {onRetry && (
            <button
              onClick={onRetry}
              className="mt-2 text-sm text-red-600 hover:text-red-800 underline"
            >
              다시 시도
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
```

## 5. 로깅 전략

### 5.1 Backend 로깅

```python
# backend/app/core/logging.py
import logging
import sys
from typing import Any

def setup_logging(level: str = "INFO") -> None:
    """로깅 설정"""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

def log_error(
    logger: logging.Logger,
    error: Exception,
    context: dict[str, Any] | None = None,
) -> None:
    """에러 로깅 헬퍼"""
    extra = {"error_type": type(error).__name__}
    if context:
        extra.update(context)
    logger.error(f"{error}", extra=extra, exc_info=True)
```

### 5.2 에러 레벨 가이드

| 레벨 | 사용 시점 | 예시 |
|------|----------|------|
| `DEBUG` | 개발 중 디버깅 | 함수 진입/종료, 변수 값 |
| `INFO` | 정상 동작 기록 | 요청 처리 완료, 작업 시작 |
| `WARNING` | 예상된 문제 | 입력 검증 실패, 리소스 없음 |
| `ERROR` | 예상치 못한 문제 | 외부 서비스 실패, DB 오류 |
| `CRITICAL` | 시스템 중단 위험 | 필수 서비스 불가, 데이터 손상 |

## 6. 에러 처리 체크리스트

### Backend

- [ ] 모든 커스텀 예외가 `AppException` 상속
- [ ] 에러 응답 형식이 `ErrorResponse` 스키마 준수
- [ ] 전역 예외 핸들러 등록 (`app.exception_handler`)
- [ ] 외부 서비스 호출에 try-except 적용
- [ ] 민감한 정보가 에러 응답에 포함되지 않음
- [ ] 에러 로깅이 적절한 레벨로 기록됨

### Frontend

- [ ] API 클라이언트에서 에러 응답 파싱
- [ ] Error Boundary로 컴포넌트 에러 캐치
- [ ] `error.tsx`, `not-found.tsx` 페이지 구현
- [ ] 사용자 친화적 에러 메시지 표시
- [ ] 재시도 버튼 제공 (가능한 경우)
- [ ] 에러 상태에서 로딩 스피너 제거
