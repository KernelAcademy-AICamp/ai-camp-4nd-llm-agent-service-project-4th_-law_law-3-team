# Content Marketing 검색 소스 미작동 수정 기획서

> **Summary**: 콘텐츠 마케팅 모듈에서 YouTube API, Newsdata.io, NewsAPI.org, Google CSE 4개 소스가 API 키 설정에도 불구하고 검색에 활용되지 않는 문제를 수정. 키워드 수집 플로우 확장, safe_fetch() 에러 가시성 개선, sources_used 하드코딩 제거, 보안 필터링 레이어 추가, 프론트엔드 소스 상태 표시를 포함.
>
> **Project**: law-3 (Legal President)
> **Author**: PM (Product Manager)
> **Date**: 2026-02-27
> **Status**: Review (v2.0) -- Red Team + External Consultant 피드백 반영
> **Root Cause Analysis**: `docs/03-analysis/content-marketing-source-fix.analysis.md`
> **Red Team Review**: `docs/03-analysis/content-marketing-source-fix.redteam.md`
> **External Consultant Review**: `docs/03-analysis/content-marketing-source-fix.consulting.md`
> **QA Strategy**: `docs/03-analysis/content-marketing-source-fix.qa-strategy.md`

---

## 1. 문제 정의

### 1.1 현상

키워드 검색(`POST /api/content-marketing/keywords/collect`)과 뉴스 검색(`POST /api/content-marketing/keywords/{id}/news`) 시 **Tavily, Naver, Perplexity만 결과에 반영**되고, YouTube API, Newsdata.io, NewsAPI.org, Google CSE(Custom Search Engine)는 API 키를 `.env`에 정상 설정했음에도 검색 결과에 나타나지 않음.

### 1.2 사용자 영향

| 영향 항목 | 설명 |
|----------|------|
| 검색 다양성 저하 | 7개 소스 중 3개만 활용. 법률 관련 YouTube 영상, 해외 뉴스(NewsData/NewsAPI), Google 뉴스 결과가 완전히 누락 |
| 투명성 부재 | 어떤 소스가 성공/실패했는지 사용자에게 전혀 표시되지 않아 "검색이 빈약하다"고 느낌 |
| API 키 낭비 | 사용자가 유료 API 키를 발급했으나 키워드 수집 단계에서 아예 호출되지 않음 |
| 캐시 오염 | 캐시 히트 시 `sources_used`가 항상 `["tavily", "naver"]`로 하드코딩되어 실제 데이터와 불일치 |

### 1.3 근본 원인 요약 (분석 보고서 참조)

| # | 근본 원인 | 위치 | 심각도 |
|---|----------|------|--------|
| RC-1 | **키워드 수집에서 4개 소스 미사용** (설계 제한) | `collector.py:collect_community_keywords()` line 272-285 | **P0 (Critical)** |
| RC-2 | **safe_fetch() 무음 실패** (Graceful Degradation 과도) | `sources/__init__.py:safe_fetch()` line 60-70 | **P1 (High)** |
| RC-3 | **sources_used 하드코딩** (캐시 히트 시) | `content_marketing_service.py` line 310, 367-369 | **P0 (Critical)** |
| RC-4 | **뉴스 검색에서도 4개 소스가 결과 미반환 가능성** | `collector.py:search_news_for_keyword()` + 각 소스 | **P1 (High)** |

---

## 2. 기존 기획과의 관계

### 2.1 관련 기획 문서

| 문서 | 설명 | 본 기획과의 관계 |
|------|------|----------------|
| `content-marketing.plan.md` (v0.1) | 초기 콘텐츠 마케팅 기획 | v0.1에서 Google Trends, YouTube를 Phase 2로 분류. 현재 Phase 2가 미완 |
| `content-marketing-v2.plan.md` (v2.0) | 페르소나 + 스코어링 고도화 | 소스 수집 확장에 대한 언급 없음 (스코어링/페르소나에 집중) |
| `content-marketing-keyword-flow.plan.md` (v1.0) | 키워드 플로우 리팩토링 | collect_community_keywords()가 Tavily+Naver만 호출하도록 설계됨 (본 기획이 수정) |

### 2.2 기존 설계 의도와 변경 필요성

기존 `content-marketing-keyword-flow.plan.md`는 키워드 수집 단계(Step 1)를 **"커뮤니티 인기글 + 키워드 추출"** 용도로 설계하여 Tavily + Naver만 사용했다. 이는 다음 전제에 기반:

> "키워드 수집은 커뮤니티 소스만으로 충분하고, 뉴스 소스는 Step 2(뉴스 검색)에서 사용한다."

그러나 실제 운영에서 다음 문제가 발견됨:

1. **Step 1에서 YouTube/Google 트렌드 키워드가 누락** -- YouTube에서 화제인 법률 이슈, Google 검색 트렌드가 키워드 추출에 반영되지 않아 커버리지 부족
2. **Step 2에서도 4개 소스가 silent failure** -- API 키 설정 문제, 타임아웃, 응답 형식 오류 등이 safe_fetch()에 의해 무음 처리되어 원인 파악 불가
3. **사용자가 소스 작동 현황을 알 수 없음** -- 프론트엔드에 소스 상태 표시가 없어 "왜 검색 결과가 빈약한지" 인지 불가

---

## 3. 외부 검증 피드백 반영 요약 (v2.0 신규)

### 3.1 Red Team + External Consultant 피드백 판정표

#### Red Team 피드백

| # | 제안 | 등급 | 판정 | 사유 |
|---|------|------|------|------|
| RT-1 | 에러 메시지 내 API Key 포함 가능성 → 필터링 레이어 필수 | Critical | **수용 (Must)** | sanitize_error_message() 추가로 API 키 패턴 마스킹 |
| RT-2 | SSRF: sanitize_keyword 검증 강화 | High | **부분 수용 (Should)** | 기존 sanitize_keyword() 블랙리스트 패턴 보강 |
| RT-3 | 리소스 고갈 DoS: 커넥션 풀 고갈 | Medium | **부분 수용 (Should)** | asyncio.Semaphore(5)로 동시 호출 수 제한 |
| RT-4 | sources_failed에 기술 스택 정보 노출 | Medium | **수용 (Must)** | 에러 메시지 추상화 (내부 라이브러리명 미노출) |
| RT-5 | Circuit Breaker 패턴 | - | **차기** | 안정화 후 도입 |
| RT-6 | Connection Pooling (httpx 싱글톤) | - | **부분 수용 (Should)** | httpx.AsyncClient 재사용 검토 |
| RT-7 | Redis 분산 캐시 | - | **차기** | 인프라 변경, 현재 범위 초과 |
| RT-8 | Celery/RabbitMQ 비동기 워커 | - | **거부** | 현재 아키텍처(FastAPI async + SSE)로 충분 |

#### External Consultant 피드백

| # | 제안 | 판정 | 사유 |
|---|------|------|------|
| EC-1 | 기획-QA 인터페이스 불일치 해소 | **수용 (Must)** | safe_fetch 유지 + safe_fetch_with_status 추가형 확정 |
| EC-2 | 성능 목표 단일화 (15초) | **수용 (Must)** | 기획 15초 기준 확정, QA 20초는 경고 기준 |
| EC-3 | 소스별 가중치/신뢰도 스코어 | **차기** | 데이터 축적 후 도입 |
| EC-4 | 실패 소스에 조치 CTA | **부분 수용 (Should)** | FR-05에 간단한 CTA 추가 |
| EC-5 | 기능 테스트 + 품질 KPI 테스트 분리 | **수용 (Should)** | 성공 기준에 2트랙 분리 명시 |
| EC-6 | 비즈니스 상품 구조화 | **거부** | 현재 단계(버그 수정)에서 범위 초과 |
| EC-7~9 | OpenTelemetry/OpenLineage/LangGraph HITL | **차기** | 인프라/아키텍처 영역 |

### 3.2 v2.0 반영 변경사항 요약

| 변경 | 근거 | 영향 |
|------|------|------|
| **[Must] 보안 필터링 레이어 추가** (FR-08 신규) | RT-1 (Critical) | safe_fetch_with_status()의 error_message를 sanitize_error_message()로 필터링 |
| **[Must] sources_failed 에러 메시지 추상화** (FR-09 신규) | RT-4 (Medium) | 프론트엔드에 전달하는 에러 메시지에서 기술 스택 정보 제거 |
| **[Must] 인터페이스 합의 확정** | EC-1 | safe_fetch() 유지 + safe_fetch_with_status() 추가 (파괴적 변경 금지) |
| **[Must] 성능 목표 단일화** | EC-2 | 15초 이내 (기획/QA 통일) |
| **[Should] 동시 호출 수 제한** | RT-3 | asyncio.Semaphore(5) |
| **[Should] sanitize_keyword 보강** | RT-2 | URL 패턴, 특수문자 시퀀스 필터링 추가 |
| **[Should] 실패 소스 CTA** | EC-4 | 프론트엔드 소스 상태 UI에 간단한 조치 안내 포함 |
| **[Should] 테스트 2트랙 분리** | EC-5 | 기능 정상화 테스트 + 품질 KPI 테스트 분리 |

---

## 4. 요구사항

### 4.1 기능 요구사항 (Functional Requirements)

| ID | 요구사항 | 우선순위 | 근거 |
|----|---------|---------|------|
| **FR-01** | 키워드 수집 시 Tavily + Naver뿐 아니라, YouTube/Google/NewsData/NewsAPI 중 API 키가 설정된 소스도 **선택적으로** 키워드 추출 입력에 포함 | **Must** | RC-1 해결 |
| **FR-02** | `safe_fetch_with_status()` 메서드를 별도 추가하여, 실패 시 에러 정보(소스명, 에러 유형, 타임스탬프)를 수집. 기존 `safe_fetch()`는 그대로 유지 (하위호환) | **Must** | RC-2 해결, EC-1 인터페이스 합의 |
| **FR-03** | 캐시 히트 시 `sources_used` 하드코딩 제거. 캐시 저장 시 원본 `sources_used`를 함께 저장하고 캐시 히트 시 그대로 반환 | **Must** | RC-3 해결 |
| **FR-04** | 뉴스 검색(`search_news_for_keyword`) 응답에 `sources_failed` 필드 추가 | **Must** | RC-4 해결 |
| **FR-05** | 프론트엔드에 소스 상태 표시 UI 추가 (활성/비활성/실패/결과 수 배지). 실패 소스에는 간단한 조치 안내("API 키 확인", "할당량 초과") 포함 | **Should** | 사용자 투명성, EC-4 CTA |
| **FR-06** | 소스별 health check API 엔드포인트 제공 (`GET /api/content-marketing/sources/status`) | **Could** | 운영 편의 |
| **FR-07** | SSE 스트리밍(`collect_keywords_stream`)에서도 `sources_used` 하드코딩 제거 | **Must** | RC-3의 SSE 경로 |
| **FR-08** | `safe_fetch_with_status()`의 error_message에 API 키/토큰 패턴이 포함되지 않도록 `sanitize_error_message()` 필터링 레이어 적용 | **Must** | RT-1 (Critical) 보안 |
| **FR-09** | `sources_failed` 필드의 에러 메시지를 사용자 친화적으로 추상화. 내부 라이브러리명, HTTP 클라이언트명, 스택트레이스 미노출 | **Must** | RT-4 (Medium) 보안 |

### 4.2 비기능 요구사항 (Non-Functional Requirements)

| 카테고리 | 기준 | 측정 방법 |
|---------|------|----------|
| **성능** | 키워드 수집(Step 1) 소요 시간 **15초 이내** 유지 (추가 소스 병렬). 이는 기획서-QA 통일 목표 (EC-2 반영) | API 응답 시간 |
| 안정성 | 추가된 소스 중 하나 이상 실패해도 기존 Tavily+Naver 결과는 정상 반환 (graceful degradation 유지) | E2E 테스트 |
| 하위호환 | 기존 API 응답 스키마에 필드 추가만 허용. 기존 필드 제거/변경 금지. `safe_fetch()` 시그니처 변경 금지 | 스키마 비교 |
| 비용 | 키워드 수집 시 추가 소스는 최소 호출 (뉴스 소스 본격 사용은 Step 2에서) | API 호출 횟수 로깅 |
| **보안** | 에러 메시지/로그에 API 키, 인증 토큰 미포함. sources_failed 응답에 내부 기술 스택 정보 미노출 (RT-1, RT-4 반영) | 보안 스캔, grep 패턴 검사 |
| **동시성** | 외부 소스 병렬 호출 시 동시 커넥션 수를 최대 5개로 제한 (RT-3 반영) | asyncio.Semaphore 적용 확인 |

---

## 5. 수정 범위 (MoSCoW)

### 5.1 Must (현재 이터레이션 필수)

| 변경 대상 | 파일 경로 | 변경 유형 | 설명 |
|----------|----------|----------|------|
| **키워드 수집 확장** | `backend/app/tools/trend/collector.py` | 수정 | `collect_community_keywords()`에 YouTube/Google/NewsData/NewsAPI 중 가용 소스 병렬 수집 추가. 기존 Tavily+Naver를 Primary, 나머지를 Secondary로 분류하여 병렬 호출 |
| **safe_fetch_with_status() 추가** | `backend/app/tools/trend/sources/__init__.py` | 수정 | 별도 `safe_fetch_with_status()` 메서드 추가. **기존 `safe_fetch()` 시그니처 변경 금지** (EC-1 인터페이스 합의) |
| **보안 필터링 레이어** | `backend/app/tools/trend/sources/__init__.py` | 수정 | `sanitize_error_message()` 함수 추가: API 키 패턴, 인증 토큰, 내부 라이브러리명 마스킹 (FR-08, FR-09) |
| **sources_used 하드코딩 제거** | `backend/app/services/service_function/content_marketing_service.py` | 수정 | line 310, 367-369의 `sources_used=["tavily", "naver"]` 하드코딩을 캐시된 실제 값으로 교체 |
| **캐시 구조 확장** | `backend/app/services/service_function/content_marketing_service.py` | 수정 | 캐시 저장 시 `sources_used`, `sources_failed` 정보를 함께 저장 |
| **응답 스키마 확장** | `backend/app/modules/content_marketing/schema/__init__.py` | 수정 | `KeywordCollectResponse`와 `KeywordNewsResponse`에 `sources_failed` 필드 추가 |
| **SSE 스트리밍 수정** | `backend/app/services/service_function/content_marketing_service.py` | 수정 | `collect_keywords_stream()` 내 `sources_used` 하드코딩 제거 |

### 5.2 Should (시간 허용 시 포함)

| 변경 대상 | 파일 경로 | 변경 유형 | 설명 |
|----------|----------|----------|------|
| **프론트엔드 소스 상태 UI** | `frontend/src/features/content-marketing/components/KeywordCollector.tsx` | 수정 | 소스별 상태 배지 (성공/실패/미설정) 표시. 실패 소스에 조치 CTA 포함 (EC-4) |
| **프론트엔드 뉴스 소스 상태** | `frontend/src/features/content-marketing/components/KeywordNewsList.tsx` | 수정 | 뉴스 검색 시 사용된/실패한 소스 표시 |
| **프론트엔드 타입 확장** | `frontend/src/features/content-marketing/types/index.ts` | 수정 | `SourceFailInfo` 타입 추가 |
| **동시 호출 수 제한** | `backend/app/tools/trend/collector.py` | 수정 | asyncio.Semaphore(5)로 외부 소스 동시 호출 수 제한 (RT-3) |
| **sanitize_keyword 보강** | `backend/app/tools/trend/keyword_blacklist.py` | 수정 | URL 패턴, 특수문자 시퀀스, 과도한 길이 필터링 추가 (RT-2) |
| **httpx.AsyncClient 재사용** | `backend/app/tools/trend/sources/__init__.py` | 수정 | 매 요청 생성 대신 싱글톤 클라이언트 재사용 검토 (RT-6) |

### 5.3 Could (다음 이터레이션 고려)

| 변경 대상 | 설명 |
|----------|------|
| **소스 health check API** | `GET /api/content-marketing/sources/status` -- 각 소스의 API 키 설정 여부, 최근 호출 성공/실패 이력 반환 |
| **소스 선택 토글 UI** | 사용자가 프론트엔드에서 특정 소스를 활성/비활성할 수 있는 토글 |
| **소스별 API 할당량 모니터링** | YouTube(10,000 units/day), Google CSE(100/day) 등의 잔여 할당량 표시 |
| **SLO 재정의 (P50/P95)** | 15초 단일 목표에서 P50/P95 분리 정의 (EC-10) |
| **Circuit Breaker 패턴** | 지속적 실패 소스 자동 차단/복구 (RT-5) |
| **소스별 가중치/신뢰도 스코어** | 데이터 축적 후 소스별 품질 가중치 도입 (EC-3) |

### 5.4 Won't (이번 범위 제외)

| 항목 | 제외 사유 |
|------|----------|
| 새로운 검색 소스 추가 (Bing, DuckDuckGo 등) | 기존 7개 소스 정상화가 우선 |
| 소스별 과금 관리 대시보드 | 운영 인프라 영역, 별도 기획 필요 |
| 기존 `/trends` API (5단계 파이프라인) 수정 | 하위호환 유지, 이번 범위는 키워드 플로우만 |
| 소스 API 키 자동 발급/갱신 | 사용자 수동 관리 유지 |
| Redis 분산 캐시 도입 | 인프라 변경, 인메모리 캐시로 당분간 충분 (RT-7 차기) |
| Celery/RabbitMQ 비동기 워커 | 현재 아키텍처(FastAPI async + SSE)로 충분 (RT-8 거부) |
| 비즈니스 상품 구조화 (Starter/Pro/Enterprise) | 버그 수정 단계에서 수익화 전략은 범위 초과 (EC-6 거부) |
| OpenTelemetry/OpenLineage | 운영 인프라 영역 (EC-7, EC-8 차기) |

---

## 6. 상세 수정 설계

### 6.1 RC-1 수정: 키워드 수집 소스 확장

#### 현재 (AS-IS)

```python
# collector.py:collect_community_keywords() line 272-285
tasks = [self._community_source.safe_fetch("사건 사고 논란 이슈", config)]
if naver_source:
    tasks.append(naver_source.safe_fetch("사건 사고 논란 법률", naver_config))
# YouTube, Google, NewsData, NewsAPI -- 호출하지 않음
```

#### 변경 (TO-BE)

```python
# collector.py:collect_community_keywords()
# Primary 소스: Tavily + Naver (키워드 추출 핵심 입력)
tasks = [self._community_source.safe_fetch("사건 사고 논란 이슈", config)]
if naver_source:
    tasks.append(naver_source.safe_fetch("사건 사고 논란 법률", naver_config))

# Secondary 소스: YouTube/Google/NewsData/NewsAPI (보완적 커버리지)
# - 가용한 소스만 추가 호출
# - 별도 타임아웃 적용 (느린 소스가 Primary 결과를 지연시키지 않도록)
# - "법률 사건 사고" 키워드로 경량 검색 (max_results 줄임)
# - Semaphore로 동시 호출 수 제한 (RT-3)
secondary_sources = [
    s for s in self._news_sources
    if s.is_available and s.name not in (TrendSource.NAVER,)
]
if secondary_sources:
    secondary_config = SourceConfig(
        time_range=time_range,
        max_results=max_keywords,  # Primary보다 적게
        search_query="법률 사건 사고 이슈",
    )
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_SOURCES)  # 기본값 5
    for source in secondary_sources:
        tasks.append(
            _fetch_with_semaphore(
                semaphore,
                source.safe_fetch_with_status("법률 사건 사고 이슈", secondary_config),
                timeout=SOURCE_TIMEOUTS.get(source.name.value, 10),
            )
        )
```

#### 설계 원칙

1. **Primary/Secondary 분리**: Tavily+Naver는 필수, 나머지는 보완적. Secondary 실패 시에도 Primary 결과로 키워드 추출 진행
2. **비용 최소화**: Secondary 소스는 `max_results`를 줄여 호출 (키워드 추출 보완 용도이므로 대량 수집 불필요)
3. **타임아웃 격리**: `asyncio.wait_for`로 Secondary 소스에 개별 타임아웃 적용. 느린 소스 1개가 전체를 지연시키지 않음
4. **결과 병합**: Primary + Secondary 결과를 병합 후 기존 `extract_with_scores()`에 입력. 키워드 추출기가 더 풍부한 맥락을 가짐
5. **동시성 제한** (v2.0 신규): `asyncio.Semaphore(5)`로 동시 외부 호출 수를 제한하여 커넥션 풀 고갈 방지 (RT-3)

### 6.2 RC-2 수정: safe_fetch() 에러 가시성 개선

#### 현재 (AS-IS)

```python
# sources/__init__.py
async def safe_fetch(self, query, config):
    try:
        return await self.fetch(query, config)
    except Exception:
        logger.warning("소스 %s 수집 실패, 건너뜀", self.name.value)
        return []  # 에러 정보 소실
```

#### 변경 (TO-BE)

**인터페이스 합의 (EC-1 확정)**: 기존 `safe_fetch()` 시그니처는 변경하지 않는다. 별도 `safe_fetch_with_status()` 메서드를 추가한다. 신규 코드에서는 `safe_fetch_with_status()`를 사용하고, 기존 호출부는 그대로 유지한다.

```python
# --- 보안 필터링 함수 (FR-08, FR-09 신규) ---

# API 키 패턴 (마스킹 대상)
_SECRET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(AIza[0-9A-Za-z_-]{35})"),           # Google API Key
    re.compile(r"(sk-[a-zA-Z0-9]{20,})"),              # OpenAI-style key
    re.compile(r"(Bearer\s+[a-zA-Z0-9._~+/=-]{20,})"), # Bearer token
    re.compile(r"(api[_-]?key[=:]\s*\S+)", re.IGNORECASE),
    re.compile(r"(token[=:]\s*\S+)", re.IGNORECASE),
]

# 내부 기술 스택 정보 (추상화 대상)
_TECH_STACK_KEYWORDS = [
    "httpx", "aiohttp", "requests", "urllib3", "asyncio",
    "Traceback", "File \"", "line ", "ModuleNotFoundError",
]

def sanitize_error_message(raw_message: str) -> str:
    """에러 메시지에서 민감 정보 및 기술 스택 정보를 제거한다.

    - API 키/토큰 패턴을 [REDACTED]로 마스킹
    - 내부 라이브러리명/스택트레이스를 일반적 에러 유형으로 치환
    """
    sanitized = raw_message
    for pattern in _SECRET_PATTERNS:
        sanitized = pattern.sub("[REDACTED]", sanitized)

    for keyword in _TECH_STACK_KEYWORDS:
        if keyword.lower() in sanitized.lower():
            # 기술 스택이 포함된 경우 일반화된 메시지로 대체
            return _classify_user_facing_message(raw_message)

    return sanitized[:200]  # 최대 200자


def _classify_user_facing_message(raw_message: str) -> str:
    """내부 에러를 사용자 친화적 메시지로 변환한다."""
    lower = raw_message.lower()
    if "timeout" in lower:
        return "응답 시간 초과"
    if "401" in lower or "403" in lower or "unauthorized" in lower:
        return "인증 오류 (API 키 확인 필요)"
    if "429" in lower or "quota" in lower or "rate" in lower:
        return "API 할당량 초과"
    if "404" in lower:
        return "API 엔드포인트를 찾을 수 없음"
    if "500" in lower or "502" in lower or "503" in lower:
        return "외부 서비스 일시적 오류"
    return "소스 연결 오류"


@dataclass
class SourceFetchResult:
    """소스 수집 결과 + 에러 정보"""
    items: list[RawTrendItem]
    source_name: str
    is_success: bool
    error_type: str | None = None   # "timeout", "auth", "rate_limit", "network", "parse", "unknown"
    error_message: str | None = None  # sanitize_error_message() 적용 후 값
    fetched_at: datetime | None = None

async def safe_fetch_with_status(
    self, query: str | None, config: SourceConfig,
) -> SourceFetchResult:
    """에러 정보를 포함한 수집 (기존 safe_fetch 대체용)"""
    try:
        items = await self.fetch(query, config)
        return SourceFetchResult(
            items=items,
            source_name=self.name.value,
            is_success=True,
            fetched_at=datetime.now(tz=timezone.utc),
        )
    except asyncio.TimeoutError:
        logger.warning("소스 %s 타임아웃", self.name.value)
        return SourceFetchResult(
            items=[], source_name=self.name.value,
            is_success=False, error_type="timeout",
            error_message="응답 시간 초과",
        )
    except Exception as exc:
        error_type = _classify_error(exc)
        # 내부 로그에는 상세 정보 기록 (운영자용)
        logger.warning("소스 %s 수집 실패 (%s): %s", self.name.value, error_type, str(exc)[:200])
        # 외부 전달용은 sanitize 적용 (FR-08)
        safe_message = sanitize_error_message(str(exc)[:500])
        return SourceFetchResult(
            items=[], source_name=self.name.value,
            is_success=False, error_type=error_type,
            error_message=safe_message,
        )
```

**기존 `safe_fetch()`는 그대로 유지** (하위호환). 신규 코드에서는 `safe_fetch_with_status()`를 사용.

### 6.3 RC-3 수정: sources_used 하드코딩 제거

#### 현재 (AS-IS)

```python
# content_marketing_service.py line 310 (캐시 히트)
sources_used=["tavily", "naver"],  # 하드코딩!

# content_marketing_service.py line 367-369 (SSE 캐시 히트)
sources_used=["tavily", "naver"],  # 하드코딩!
```

#### 변경 (TO-BE)

```python
# 캐시 저장 시 sources_used도 함께 저장
_keyword_cache[cache_key] = (
    scored_keywords,
    now,
    actual_sources_used,    # 실제 사용된 소스 목록
    sources_failed_info,    # 실패한 소스 정보
)

# 캐시 히트 시 저장된 sources_used 반환
cached_keywords, cached_at, cached_sources_used, cached_sources_failed = cached
return KeywordCollectResponse(
    keywords=[_scored_to_item(kw) for kw in cached_keywords],
    total_count=len(cached_keywords),
    collected_at=cached_at,
    sources_used=cached_sources_used,      # 캐시된 실제 값
    sources_failed=cached_sources_failed,  # 캐시된 실패 정보
    cache_hit=True,
    prompt_version="1.0",
)
```

### 6.4 RC-4 수정: 뉴스 검색 소스 실패 정보 포함

#### 변경

`search_news_for_keyword()` 내부에서 `safe_fetch_with_status()`를 사용하여 소스별 성공/실패를 추적하고, 반환값에 `sources_failed` 정보를 포함.

### 6.5 스키마 확장

```python
# schema/__init__.py에 추가

class SourceFailInfo(BaseModel):
    """소스 실패 정보 (사용자 친화적 메시지만 포함)"""
    source_name: str          # "youtube", "google_trends", etc.
    error_type: str           # "timeout", "auth", "rate_limit", "network", "parse", "unknown"
    error_message: str | None = None  # sanitize된 메시지 (기술 스택 미노출, FR-09)

# KeywordCollectResponse 필드 추가
class KeywordCollectResponse(BaseModel):
    # ... 기존 필드 유지 ...
    sources_failed: list[SourceFailInfo] = Field(default_factory=list)  # NEW

# KeywordNewsResponse 필드 추가
class KeywordNewsResponse(BaseModel):
    # ... 기존 필드 유지 ...
    sources_failed: list[SourceFailInfo] = Field(default_factory=list)  # NEW
```

### 6.6 프론트엔드 소스 상태 UI (Should)

#### 키워드 수집 결과 하단에 소스 상태 표시

```
+------------------------------------------------------------+
| 검색 소스 현황                                               |
|                                                             |
| [Tavily] 성공 12건   [Naver] 성공 8건   [Perplexity] 성공 5건 |
| [YouTube] 타임아웃 → API 키 확인   [Google] 미설정            |
| [NewsData] 성공 3건  [NewsAPI] 할당초과 → 유료 전환 안내      |
+------------------------------------------------------------+
```

- 성공 소스: 녹색 배지 + 결과 건수
- 실패 소스: 빨간/노란 배지 + 에러 유형 + **조치 안내 CTA** (EC-4 반영)
  - 타임아웃 → "재시도" 안내
  - 인증 오류 → "API 키 확인" 안내
  - 할당량 초과 → "할당량 초과, 내일 재시도" 안내
- 미설정 소스: 회색 배지 + "API 키 미설정"

---

## 7. 수정 우선순위 및 구현 순서

### 7.1 우선순위 분류

| 순위 | 근본 원인 | 수정 난이도 | 영향 범위 | 구현 순서 |
|------|----------|-----------|----------|----------|
| **P0** | RC-3: sources_used 하드코딩 | Low | 서비스 레이어 | **1단계** (즉시 수정 가능, 가장 빠른 효과) |
| **P0** | RC-1: 키워드 수집 소스 확장 | Medium | collector.py | **2단계** (핵심 문제) |
| **P1** | RC-2: safe_fetch() 에러 가시성 + 보안 필터링 | Medium | sources/ + collector.py + service | **3단계** (보안 필터링 레이어 포함, RT-1/RT-4) |
| **P1** | RC-4: 뉴스 검색 소스 실패 추적 | Low | collector.py + schema | **4단계** (3단계 완료 후) |
| P2 | 프론트엔드 소스 상태 UI | Medium | frontend 3개 파일 | **5단계** |

### 7.2 구현 단계

#### 1단계: sources_used 하드코딩 제거 (P0, 즉시)

- `content_marketing_service.py` 내 2개 하드코딩 지점 수정
- 캐시 저장/조회 구조 확장
- **영향**: 서비스 레이어만 변경, 다른 모듈 영향 없음
- **예상 소요**: 0.5일

#### 2단계: 키워드 수집 소스 확장 (P0)

- `collector.py:collect_community_keywords()`에 Secondary 소스 병렬 수집 추가
- `asyncio.Semaphore(5)` 동시성 제한 적용 (RT-3)
- SOURCE_TIMEOUTS 상수를 공통으로 사용하도록 리팩토링
- **영향**: collector.py 수정, 기존 메서드 시그니처 변경 없음
- **예상 소요**: 1일

#### 3단계: safe_fetch_with_status() + 보안 필터링 (P1)

- `sources/__init__.py`에 `SourceFetchResult` 데이터클래스 추가
- `safe_fetch_with_status()` 메서드 추가 (기존 `safe_fetch()` 유지, **EC-1 인터페이스 합의**)
- `sanitize_error_message()` 보안 필터링 함수 추가 (FR-08, FR-09)
  - API 키 패턴 마스킹 (`_SECRET_PATTERNS`)
  - 기술 스택 정보 추상화 (`_TECH_STACK_KEYWORDS`)
- `collector.py`의 키워드 수집/뉴스 검색에서 신규 메서드 활용
- **영향**: sources/ 인터페이스 확장 (추가만, 변경 없음)
- **예상 소요**: 1.5일 (보안 필터링 추가로 +0.5일)

#### 4단계: 스키마 확장 + 뉴스 검색 수정 (P1)

- `SourceFailInfo` Pydantic 모델 추가 (sanitize된 메시지만 포함)
- `KeywordCollectResponse`, `KeywordNewsResponse`에 `sources_failed` 필드 추가
- `search_news_for_keyword()`에서 `safe_fetch_with_status()` 활용
- Frontend 타입 확장 (`types/index.ts`)
- **영향**: 스키마 필드 추가 (기존 필드 변경 없음, 하위호환)
- **예상 소요**: 0.5일

#### 5단계: 프론트엔드 소스 상태 UI (P2)

- `KeywordCollector.tsx`에 소스 상태 배지 컴포넌트 추가 (조치 CTA 포함, EC-4)
- `KeywordNewsList.tsx`에 뉴스 소스 상태 표시
- **예상 소요**: 1일

#### 총 예상 소요: 4.5일

| 단계 | 소요 | 누적 | 우선순위 | v2.0 변경 |
|------|------|------|---------|----------|
| 1단계: 하드코딩 제거 | 0.5일 | 0.5일 | P0 | - |
| 2단계: 소스 확장 | 1일 | 1.5일 | P0 | Semaphore 추가 |
| 3단계: 에러 가시성 + 보안 | 1.5일 | 3일 | P1 | **보안 필터링 레이어 추가 (+0.5일)** |
| 4단계: 스키마 + 뉴스 | 0.5일 | 3.5일 | P1 | - |
| 5단계: 프론트엔드 | 1일 | 4.5일 | P2 | CTA 추가 |

---

## 8. 리스크 평가

### 8.1 수정 시 기존 기능 영향

| 리스크 | 영향 | 가능성 | 완화 방안 |
|--------|------|--------|----------|
| Secondary 소스 추가로 키워드 수집 시간 증가 | Medium | Medium | asyncio.wait_for로 개별 타임아웃, Primary 결과는 Secondary 완료 전에도 사용 가능하도록 설계 |
| safe_fetch_with_status() 도입으로 기존 safe_fetch() 호출부 혼란 | Low | Low | 기존 safe_fetch()는 그대로 유지, 신규 메서드만 점진적 전환 (EC-1 합의) |
| 캐시 구조 변경으로 기존 캐시 무효화 | Low | Medium | 인메모리 캐시이므로 서버 재시작 시 자동 초기화. 코드 내 캐시 조회 시 backward-compatible 처리 (튜플 길이 체크) |
| sources_failed 필드 추가로 프론트엔드 빌드 실패 | Low | Low | 필드 기본값 `[]`이므로 기존 프론트엔드 코드는 무시 가능. TypeScript 타입만 확장 |
| YouTube API 일일 할당량(10,000 units) 빠른 소진 | Medium | Medium | 키워드 수집 시 max_results를 낮게 설정 (10건), Rate Limiter 기존 유지 (1시간 5회) |
| Google CSE 무료 일일 100건 제한 | Medium | High | 키워드 수집에서는 건수 최소화, 주요 사용은 Step 2(뉴스 검색)로 유도 |
| NewsData.io 48시간 timeframe 제한으로 빈 결과 반환 | Low | Medium | time_range 파라미터를 소스에 맞게 변환. 빈 결과는 정상 처리 (실패와 구분) |
| NewsAPI.org 무료 플랜 /v2/everything 접근 제한 | Medium | High | 무료 플랜 제약을 에러 메시지에 명시. 유료 전환 안내 제공 |
| **[v2.0 신규] 병렬 호출 시 커넥션 풀 고갈 (RT-3)** | Medium | Low | asyncio.Semaphore(5)로 동시 호출 수 제한. httpx.AsyncClient 재사용 검토 |
| **[v2.0 신규] sanitize_error_message()가 유효한 에러 정보까지 과도하게 마스킹** | Low | Medium | 내부 로그에는 원본 에러 유지, sanitize는 외부 전달용에만 적용. 필터링 패턴은 단위 테스트로 검증 |

### 8.2 하위호환성 보장

| 항목 | 보장 방법 |
|------|----------|
| API 응답 스키마 | 기존 필드 100% 유지. `sources_failed` 필드만 추가 (기본값 `[]`) |
| `safe_fetch()` 메서드 | **기존 메서드 시그니처 변경 금지** (EC-1 확정). 새 메서드(`safe_fetch_with_status`)를 별도 추가 |
| 기존 `/trends` API | 변경 없음 (키워드 플로우 API만 수정) |
| 프론트엔드 기존 컴포넌트 | TypeScript 타입에 optional 필드 추가. 기존 로직 변경 없음 |
| 캐시 구조 | 튜플 길이 체크로 이전/이후 캐시 모두 호환 |

### 8.3 보안 고려사항 (v2.0 신규)

| 위협 | 대응 | 검증 방법 |
|------|------|----------|
| API 키가 에러 메시지에 포함 (RT-1) | sanitize_error_message()로 정규식 기반 마스킹 | grep 패턴 스캔 + 단위 테스트 |
| 기술 스택 정보 노출 (RT-4) | _TECH_STACK_KEYWORDS 기반 추상화 | sources_failed 응답 필드 수동 검증 |
| SSRF via 키워드 입력 (RT-2) | sanitize_keyword() 패턴 보강 | 블랙리스트 테스트 |
| 커넥션 풀 고갈 (RT-3) | Semaphore(5) 동시성 제한 | 부하 테스트 |

---

## 9. 성공 기준

### 9.1 정량적 기준 -- Track 1: 기능 정상화 (EC-5 반영)

| 지표 | 목표 | 측정 방법 |
|------|------|----------|
| 활성 소스 수 | API 키가 설정된 모든 소스(최대 7개)가 키워드 수집/뉴스 검색에 참여 | `sources_used` 필드 확인 |
| sources_used 정확성 | 캐시 히트 시에도 실제 수집에 사용된 소스 목록이 정확히 반환 | API 응답 검증 |
| sources_failed 가시성 | 실패한 소스의 에러 유형이 API 응답에 포함 | `sources_failed` 필드 비어있지 않은 경우 에러 유형 명시 확인 |
| 키워드 수집 시간 | **15초 이내** (기획-QA 통일 목표, EC-2 확정) | API 응답 시간 측정 |
| 정적 검증 통과 | `ruff check`, `mypy`, `npm run build` 모두 통과 | CI 파이프라인 |
| **보안 검증** (v2.0) | sources_failed 에러 메시지에 API 키, 인증 토큰, 기술 스택 정보 미포함 | grep 패턴 스캔 + 단위 테스트 |
| **인터페이스 호환성** (v2.0) | 기존 safe_fetch() 시그니처 변경 없음. 기존 호출부 정상 동작 | 회귀 테스트 |

### 9.2 정량적 기준 -- Track 2: 품질 KPI (EC-5 반영, v2.0 신규)

| 지표 | 목표 | 측정 방법 | 비고 |
|------|------|----------|------|
| 키워드 법률 관련성 | 추출 키워드 중 법률 관련 비율 70% 이상 유지 | 수동 샘플링 (20건) | Secondary 소스 추가로 노이즈 증가 여부 모니터링 |
| 소스 실패 투명성 | 실패 소스의 에러 유형이 사용자에게 이해 가능한 형태 | UX 검토 | "응답 시간 초과", "인증 오류" 등 |
| 캐시 히트율 | 기존 캐시 히트율 유지 또는 개선 | 로그 분석 | 캐시 구조 변경 영향 모니터링 |

### 9.3 정성적 기준

- 사용자가 프론트엔드에서 어떤 소스가 작동 중이고 어떤 소스가 실패했는지 한눈에 파악 가능
- 운영자가 API 응답의 `sources_failed` 필드를 통해 소스별 문제를 즉시 진단 가능 (**단, 민감 정보는 미노출**)
- 새로운 검색 소스를 추가할 때 `_news_sources` 리스트에만 추가하면 자동으로 키워드 수집/뉴스 검색에 참여하는 확장 가능한 구조
- 실패 소스에 대해 사용자가 취할 수 있는 조치가 UI에 안내됨 (EC-4)

---

## 10. 에이전트 팀 검토 요청

본 기획서에 대해 다음 파트별 검토가 필요합니다.

| 파트 | 검토 항목 |
|------|----------|
| **백엔드 개발자** | collector.py Primary/Secondary 분리 구현, safe_fetch_with_status() 설계, sanitize_error_message() 보안 필터링, Semaphore 동시성 제한, 캐시 구조 변경 |
| **프론트엔드 개발자** | 소스 상태 UI 컴포넌트 설계, 실패 소스 CTA, TypeScript 타입 확장 |
| **QA 엔지니어** | 소스별 실패 시나리오 테스트 계획, 캐시 히트/미스 시 sources_used 정확성 검증, **보안 검증 (API 키 미노출 확인)**, **Track 1/Track 2 테스트 분리 계획** (EC-5) |
| **AI/ML 엔지니어** | Secondary 소스 추가가 키워드 추출 LLM 품질에 미치는 영향 (입력 노이즈 vs 커버리지 개선) |

### QA 전략 동기화 사항 (v2.0)

기획서와 QA 전략 간 다음 사항을 동기화해야 합니다:

| 항목 | 기획서 (v2.0) 기준 | QA 전략 업데이트 필요 |
|------|-------------------|---------------------|
| safe_fetch 인터페이스 | safe_fetch() 유지 + safe_fetch_with_status() 추가 (파괴형 변경 금지) | QC-03의 `([], error_string)` 튜플 반환 → `SourceFetchResult` 반환으로 변경 |
| 성능 목표 | 15초 이내 (통일) | QC-08의 20초 → 15초로 변경 |
| 보안 검증 | QC-10을 Must로 승격 | Should → Must로 승격, sanitize_error_message() 단위 테스트 추가 |

---

## 11. 관련 파일 목록

### 11.1 수정 대상 파일 (Backend)

| 파일 | 변경 유형 |
|------|----------|
| `/Users/mac/Downloads/project-4th/law-3/backend/app/tools/trend/collector.py` | 수정 (키워드 수집 확장 + Semaphore) |
| `/Users/mac/Downloads/project-4th/law-3/backend/app/tools/trend/sources/__init__.py` | 수정 (SourceFetchResult + safe_fetch_with_status + sanitize_error_message) |
| `/Users/mac/Downloads/project-4th/law-3/backend/app/services/service_function/content_marketing_service.py` | 수정 (sources_used 하드코딩 제거 + 캐시 구조) |
| `/Users/mac/Downloads/project-4th/law-3/backend/app/modules/content_marketing/schema/__init__.py` | 수정 (SourceFailInfo + 응답 필드 추가) |

### 11.2 수정 대상 파일 (Frontend)

| 파일 | 변경 유형 |
|------|----------|
| `/Users/mac/Downloads/project-4th/law-3/frontend/src/features/content-marketing/types/index.ts` | 수정 (SourceFailInfo 타입 추가) |
| `/Users/mac/Downloads/project-4th/law-3/frontend/src/features/content-marketing/components/KeywordCollector.tsx` | 수정 (소스 상태 배지 + CTA) |
| `/Users/mac/Downloads/project-4th/law-3/frontend/src/features/content-marketing/components/KeywordNewsList.tsx` | 수정 (뉴스 소스 상태) |

### 11.3 보강 대상 파일 (v2.0 신규)

| 파일 | 변경 유형 |
|------|----------|
| `/Users/mac/Downloads/project-4th/law-3/backend/app/tools/trend/keyword_blacklist.py` | 수정 (sanitize_keyword 패턴 보강, RT-2) |

### 11.4 참조 파일 (변경 없음)

| 파일 | 참조 사유 |
|------|----------|
| `/Users/mac/Downloads/project-4th/law-3/backend/app/tools/trend/sources/youtube_source.py` | is_available 로직 확인 |
| `/Users/mac/Downloads/project-4th/law-3/backend/app/tools/trend/sources/google_source.py` | is_available 로직 확인 (CSE_API_KEY + CSE_ID 필요) |
| `/Users/mac/Downloads/project-4th/law-3/backend/app/tools/trend/sources/newsdata_source.py` | is_available 로직 확인 |
| `/Users/mac/Downloads/project-4th/law-3/backend/app/tools/trend/sources/newsapi_source.py` | is_available 로직 확인 |
| `/Users/mac/Downloads/project-4th/law-3/backend/app/core/config.py` | API 키 환경변수 확인 |

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2026-02-27 | 초안 작성 -- 4가지 근본 원인 분석 기반 기획 수정안 수립. MoSCoW 우선순위, 5단계 구현 계획, 리스크 평가, 성공 기준 정의 | PM (Product Manager) |
| 2.0 | 2026-02-27 | Red Team + External Consultant 피드백 반영. **(1)** FR-08/FR-09 보안 필터링 레이어 추가 (RT-1 Critical, RT-4 Medium). **(2)** 인터페이스 합의 확정: safe_fetch() 유지 + safe_fetch_with_status() 추가형 (EC-1). **(3)** 성능 목표 15초로 단일화 (EC-2). **(4)** sources_failed 에러 메시지 추상화 (기술 스택 미노출). **(5)** asyncio.Semaphore(5) 동시성 제한 추가 (RT-3). **(6)** 프론트엔드 실패 소스 CTA 추가 (EC-4). **(7)** 성공 기준을 Track 1(기능 정상화) + Track 2(품질 KPI)로 분리 (EC-5). **(8)** QA 전략 동기화 사항 명시. **(9)** 총 예상 소요 4일 → 4.5일 (보안 필터링 +0.5일). **(10)** 피드백 판정표 전문 포함 (섹션 3). | PM (Product Manager) |
