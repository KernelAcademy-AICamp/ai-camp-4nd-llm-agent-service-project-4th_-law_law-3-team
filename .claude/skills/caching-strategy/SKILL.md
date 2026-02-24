---
name: caching-strategy
description: |
  FastAPI 백엔드 + Next.js 프론트엔드의 다계층 캐싱 전략 가이드.
  인메모리 캐시, Redis, HTTP 캐싱 헤더, 벡터 검색 결과 캐싱, 클라이언트 데이터 페칭.
  API 응답 속도 개선, 캐시 추가/수정, 성능 최적화, 반복 쿼리 최적화 시 반드시 사용.
  LanceDB 검색 캐싱, SWR/React Query 도입, ISR/SSG 설정, Cache-Control 헤더 설정 시에도 사용.
---

# Caching Strategy

법률 서비스 플랫폼의 다계층 캐싱 아키텍처 및 구현 패턴.

## 현재 캐싱 현황

```
┌─────────────────────────────────────────────────┐
│ 클라이언트 (브라우저)                              │
│  [캐싱 없음] ← HTTP 기본 동작만                  │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────┐
│ Next.js (프록시)                                  │
│  [캐싱 없음] ← ISR/SSG 미설정                    │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────┐
│ FastAPI 백엔드                                    │
│  [✅] @lru_cache(1) ← 모델 로드 (임베딩, 리랭커) │
│  [✅] 임베딩 캐시 ← MD5 디스크+메모리 2계층      │
│  [✅] SQLAlchemy 연결 풀                          │
│  [❌] 벡터 검색 결과 캐싱 없음                    │
│  [❌] API 응답 캐싱 없음                          │
│  [❌] HTTP Cache-Control 미설정                   │
└─────────────────────────────────────────────────┘
```

---

## 1. 백엔드 캐싱 레이어

### Layer 1: 모델/리소스 로드 캐시 (현재 구현)

```python
# 현재 패턴 - @lru_cache(maxsize=1) 싱글톤
from functools import lru_cache

@lru_cache(maxsize=1)
def get_local_model() -> SentenceTransformer:
    """임베딩 모델 1회 로드 후 재사용"""
    return SentenceTransformer(model_name, cache_folder=str(MODEL_CACHE_DIR))

@lru_cache(maxsize=1)
def _load_reranker_model() -> CrossEncoder:
    """리랭커 모델 1회 로드 후 재사용"""
    return CrossEncoder(model_name, cache_folder=str(MODEL_CACHE_DIR))
```

**적용 대상:**
- 임베딩 모델 (KURE-v1, ~2.3GB)
- 리랭커 모델 (bge-reranker, ~2.1GB)
- 변호사 JSON 데이터 (lawyer_service.py)

### Layer 2: 임베딩 벡터 캐시 (현재 구현)

```python
# scripts/embedding_common/cache.py
# MD5 기반 2계층 캐시 (메모리 + 디스크)
cache = EmbeddingCache("./embedding_cache")
embedding = cache.get_or_compute("텍스트", model.encode)
stats = cache.get_stats()  # {'hits': 150, 'misses': 50, 'hit_rate': '75.0%'}
```

**캐시 구조:**
```
embedding_cache/
├── a1/a1b2c3d4...json  # MD5 해시 첫 2자로 디렉토리 분류
├── b2/...
└── stats.json
```

### Layer 3: 벡터 검색 결과 캐시 (미구현 - 추가 권장)

동일 쿼리의 LanceDB 검색 결과를 캐싱하여 반복 검색 비용 절감.

```python
# backend/app/services/rag/search_cache.py 패턴
from functools import lru_cache
from hashlib import md5
import time

class SearchResultCache:
    """벡터 검색 결과 인메모리 캐시 (TTL 기반)"""

    def __init__(self, max_size: int = 500, ttl_seconds: int = 300):
        self._cache: dict[str, tuple[float, list]] = {}
        self._max_size = max_size
        self._ttl = ttl_seconds

    def _make_key(self, query: str, doc_type: str | None, top_k: int) -> str:
        raw = f"{query}:{doc_type}:{top_k}"
        return md5(raw.encode()).hexdigest()

    def get(self, query: str, doc_type: str | None, top_k: int) -> list | None:
        key = self._make_key(query, doc_type, top_k)
        if key in self._cache:
            timestamp, results = self._cache[key]
            if time.time() - timestamp < self._ttl:
                return results
            del self._cache[key]
        return None

    def set(self, query: str, doc_type: str | None, top_k: int, results: list) -> None:
        if len(self._cache) >= self._max_size:
            self._evict_oldest()
        key = self._make_key(query, doc_type, top_k)
        self._cache[key] = (time.time(), results)

    def _evict_oldest(self) -> None:
        oldest_key = min(self._cache, key=lambda k: self._cache[k][0])
        del self._cache[oldest_key]

# 모듈 수준 싱글톤
search_cache = SearchResultCache(max_size=500, ttl_seconds=300)
```

**사용 예시:**

```python
# services/rag/retrieval.py에 캐시 적용
from app.services.rag.search_cache import search_cache

async def search_with_cache(query: str, config: PipelineConfig) -> list:
    cached = search_cache.get(query, config.doc_type, config.n_results)
    if cached is not None:
        return cached

    results = await search_vectors(query, config)
    search_cache.set(query, config.doc_type, config.n_results, results)
    return results
```

**설계 고려사항:**

| 파라미터 | 권장값 | 근거 |
|---------|--------|------|
| `max_size` | 500 | 메모리 부담 적정 수준 |
| `ttl_seconds` | 300 (5분) | 법령/판례 데이터 변경 빈도 낮음 |
| 캐시 키 | query + doc_type + top_k | 검색 파라미터 조합 |
| 무효화 | TTL 만료 | 데이터 업데이트 빈도 고려 |

---

## 2. HTTP 캐싱 헤더

### API 엔드포인트별 캐싱 전략

| 엔드포인트 | Cache-Control | 이유 |
|-----------|---------------|------|
| `GET /api/lawyer-stats/*` | `max-age=3600` (1시간) | 통계 데이터 변경 드묾 |
| `GET /api/lawyer-finder/nearby` | `max-age=300` (5분) | 위치 기반, 적당한 신선도 |
| `GET /api/case-precedent/search` | `no-cache` | 검색 결과 개인화 |
| `POST /api/chat` | `no-store` | 대화 내용 캐싱 금지 |
| `POST /api/chat/stream` | `no-store` | SSE 스트리밍 |
| `GET /health` | `no-cache` | 실시간 상태 |

### FastAPI 캐시 헤더 구현

```python
from fastapi import Response

@router.get("/stats/overview")
async def get_stats_overview(response: Response):
    response.headers["Cache-Control"] = "public, max-age=3600"
    return {"data": ...}

@router.get("/nearby")
async def get_nearby(response: Response):
    response.headers["Cache-Control"] = "public, max-age=300"
    return {"data": ...}

@router.post("/chat")
async def chat():
    # POST는 기본적으로 캐싱 안 됨 (추가 헤더 불필요)
    return {"data": ...}
```

### 캐시 헤더 미들웨어 (일괄 적용)

```python
from starlette.middleware.base import BaseHTTPMiddleware

CACHE_RULES: dict[str, str] = {
    "/api/lawyer-stats": "public, max-age=3600",
    "/api/lawyer-finder": "public, max-age=300",
}
NO_CACHE_PREFIXES = ["/api/chat", "/api/storyboard"]

class CacheControlMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        path = request.url.path

        for prefix in NO_CACHE_PREFIXES:
            if path.startswith(prefix):
                response.headers["Cache-Control"] = "no-store"
                return response

        for prefix, directive in CACHE_RULES.items():
            if path.startswith(prefix):
                response.headers["Cache-Control"] = directive
                return response

        return response
```

---

## 3. 프론트엔드 캐싱

### SWR / React Query 도입

현재 Axios만 사용 (캐싱 없음). SWR 또는 TanStack Query 추가 권장.

```typescript
// SWR 예시
import useSWR from 'swr'

const fetcher = (url: string) => api.get(url).then(r => r.data)

function LawyerStats() {
  const { data, error, isLoading } = useSWR('/lawyer-stats/overview', fetcher, {
    revalidateOnFocus: false,    // 탭 포커스 시 재요청 안 함
    dedupingInterval: 60000,     // 1분 내 중복 요청 방지
    refreshInterval: 300000,     // 5분마다 자동 갱신
  })
  // ...
}
```

### 엔드포인트별 SWR 설정 권장

| 엔드포인트 | dedupingInterval | refreshInterval | revalidateOnFocus |
|-----------|-----------------|-----------------|-------------------|
| lawyer-stats | 60s | 300s | false |
| lawyer-finder/nearby | 30s | 60s | true |
| case-precedent/search | 0 (없음) | 0 (없음) | false |
| chat | 해당 없음 (POST) | 해당 없음 | 해당 없음 |

### Axios 응답 인터셉터 캐시

SWR 도입 전 간단한 GET 캐싱:

```typescript
// frontend/src/lib/api-cache.ts 패턴
const cache = new Map<string, { data: unknown; timestamp: number }>()
const DEFAULT_TTL = 60_000  // 1분

api.interceptors.response.use((response) => {
  if (response.config.method === 'get') {
    const key = response.config.url || ''
    cache.set(key, { data: response.data, timestamp: Date.now() })
  }
  return response
})

api.interceptors.request.use((config) => {
  if (config.method === 'get') {
    const key = config.url || ''
    const cached = cache.get(key)
    if (cached && Date.now() - cached.timestamp < DEFAULT_TTL) {
      return Promise.reject({
        __CACHE_HIT__: true,
        data: cached.data,
      })
    }
  }
  return config
})
```

---

## 4. Next.js 캐싱

### ISR (Incremental Static Regeneration)

정적 콘텐츠 (법률 용어 사전, 통계 페이지 등)에 적용:

```typescript
// app/law-study/page.tsx 패턴
export const revalidate = 3600  // 1시간마다 재생성

export default async function LawStudyPage() {
  const data = await fetch(`${process.env.API_URL}/api/law-study/topics`, {
    next: { revalidate: 3600 },
  })
  return <TopicList data={data} />
}
```

### 적용 가능한 페이지

| 페이지 | revalidate | 이유 |
|--------|-----------|------|
| 법률 용어 사전 | 86400 (1일) | 데이터 변경 드묾 |
| 변호사 통계 요약 | 3600 (1시간) | 통계 갱신 주기 |
| 소액소송 가이드 | 86400 (1일) | 정적 콘텐츠 |
| 채팅 페이지 | 0 (동적) | 실시간 대화 |
| 변호사 검색 | 0 (동적) | 위치 기반 |

---

## 5. Redis 통합 (프로덕션)

### 용도

| 기능 | 현재 | Redis 적용 후 |
|------|------|-------------|
| Rate Limiting | `memory://` (단일 프로세스) | `redis://` (분산) |
| 세션 캐시 | 없음 | Redis 기반 세션 |
| 검색 결과 캐시 | 인메모리 (프로세스 내) | Redis (공유) |
| 응답 캐시 | 없음 | Redis 기반 |

### 설치 및 설정

```toml
# pyproject.toml에 추가
[project.dependencies]
redis = ">=5.0.0"
```

```bash
# .env (프로덕션)
REDIS_URL=redis://redis:6379/0
RATE_LIMIT_STORAGE_URI=redis://redis:6379/1
```

### docker-compose에 Redis 추가

```yaml
services:
  redis:
    image: redis:7-alpine
    container_name: law-platform-redis
    ports:
      - "127.0.0.1:6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3
    restart: unless-stopped

volumes:
  redis_data:
```

### Redis 캐시 유틸리티

```python
# backend/app/core/cache.py 패턴
import json
from redis.asyncio import Redis

class RedisCache:
    def __init__(self, redis: Redis, prefix: str = "cache"):
        self._redis = redis
        self._prefix = prefix

    async def get(self, key: str) -> dict | None:
        data = await self._redis.get(f"{self._prefix}:{key}")
        return json.loads(data) if data else None

    async def set(self, key: str, value: dict, ttl: int = 300) -> None:
        await self._redis.setex(
            f"{self._prefix}:{key}",
            ttl,
            json.dumps(value, ensure_ascii=False),
        )

    async def delete(self, key: str) -> None:
        await self._redis.delete(f"{self._prefix}:{key}")
```

---

## 6. 캐시 무효화 전략

### TTL 기반 (권장)

| 데이터 유형 | TTL | 이유 |
|-----------|-----|------|
| 변호사 통계 | 1시간 | 일별 갱신 |
| 벡터 검색 결과 | 5분 | 인덱스 갱신 빈도 낮음 |
| 법률 용어 | 24시간 | 거의 변경 없음 |
| 변호사 검색 | 5분 | 위치 기반 신선도 |

### 이벤트 기반 (데이터 갱신 시)

```python
# 데이터 로드 스크립트 실행 후 캐시 무효화
async def invalidate_after_data_load(cache: RedisCache, prefix: str) -> None:
    """데이터 적재 후 관련 캐시 삭제"""
    keys = await cache._redis.keys(f"{prefix}:*")
    if keys:
        await cache._redis.delete(*keys)
```

---

## 7. 캐싱 레이어 선택 가이드

```
질문: 이 데이터를 캐싱해야 하는가?

1. 자주 변경되는가?
   ├─ Yes (실시간): 캐싱하지 않음 (채팅, 스트리밍)
   └─ No: 다음 질문

2. 여러 사용자가 같은 결과를 보는가?
   ├─ Yes (공유): HTTP Cache-Control + CDN
   └─ No (개인화): 사용자별 캐시

3. 계산 비용이 높은가?
   ├─ Yes (벡터 검색, LLM): 백엔드 인메모리 또는 Redis
   └─ No (DB 조회): HTTP 캐시 헤더만

4. 프로세스 간 공유 필요?
   ├─ Yes (멀티워커): Redis
   └─ No (싱글 프로세스): @lru_cache 또는 dict
```

---

## 8. 성능 모니터링

### 캐시 히트율 로깅

```python
import logging
logger = logging.getLogger("cache")

class MonitoredCache:
    def __init__(self):
        self._hits = 0
        self._misses = 0

    def get(self, key: str):
        result = self._inner_get(key)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1

        total = self._hits + self._misses
        if total % 100 == 0:  # 100건마다 로깅
            rate = self._hits / total * 100
            logger.info(f"Cache hit rate: {rate:.1f}% ({self._hits}/{total})")

        return result
```

### 권장 히트율 목표

| 캐시 레이어 | 목표 히트율 | 미달 시 조치 |
|-----------|-----------|-------------|
| 모델 로드 | 100% | 로드 실패 디버깅 |
| 임베딩 캐시 | >70% | 캐시 크기 확인 |
| 검색 결과 | >30% | TTL 연장 또는 max_size 증가 |
| HTTP 캐시 | >50% | Cache-Control 헤더 확인 |

---

## 9. 구현 우선순위

| 순위 | 작업 | 효과 | 난이도 |
|------|------|------|--------|
| 1 | HTTP Cache-Control 헤더 | 브라우저 캐싱 활성화 | 낮음 |
| 2 | 벡터 검색 결과 캐시 | 반복 쿼리 5분 절약 | 중간 |
| 3 | SWR 도입 (프론트엔드) | UX 개선 + 네트워크 절감 | 중간 |
| 4 | ISR 정적 페이지 | 초기 로딩 속도 | 낮음 |
| 5 | Redis 통합 (프로덕션) | 분산 캐싱 + Rate Limit | 높음 |
