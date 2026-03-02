# QA 전략 보고서: 콘텐츠 마케팅 검색 소스 수정 검증

> **작성자**: QA 엔지니어
> **작성일**: 2026-02-27
> **대상 기능**: Content Marketing 모듈 — 7개 검색 소스 정상화 수정
> **참조 분석**: `docs/03-analysis/content-marketing-source-fix.analysis.md`

---

## 1. 테스트 범위 및 목표

### 1.1 검증 대상 파일

| 파일 | 수정 내용 |
|------|---------|
| `backend/app/tools/trend/collector.py` | `collect_community_keywords()` 소스 확장, `search_news_for_keyword()` 에러 정보 반환 |
| `backend/app/tools/trend/sources/__init__.py` | `safe_fetch()` 반환 타입 변경 (`(items, error_info)` 튜플) |
| `backend/app/services/service_function/content_marketing_service.py` | 캐시 히트 시 `sources_used` 하드코딩 제거 |

### 1.2 품질 목표

| 지표 | 목표 기준 | 미달 시 조치 |
|------|---------|-----------|
| 소스 활성화 정확성 | API 키 설정 소스 100% 활성화 | 즉시 수정 차단 |
| `sources_used` 정합성 | 캐시 히트/미스 모두 실제 소스 반영 | 즉시 수정 차단 |
| `sources_failed` 에러 전달 | 실패 소스 1개 이상 발생 시 필드 포함 | 즉시 수정 차단 |
| 병렬 호출 성능 | 7개 소스 병렬 호출 시 기존 대비 +2초 이내 | 성능 재설계 권고 |
| 에러 격리 | 1개 소스 실패가 전체 결과에 영향 없음 | 즉시 수정 차단 |

---

## 2. 테스트 전략

### 2.1 계층별 테스트 접근

```
E2E 테스트 (프론트 → 백엔드 → 외부 API)
    └── 통합 테스트 (백엔드 API 엔드포인트)
            └── 단위 테스트 (소스 어댑터 / 서비스 함수)
```

### 2.2 단위 테스트 (소스 어댑터별 Mock)

각 소스 어댑터의 `is_available`, `fetch()` 메서드를 Mock API 응답으로 검증한다.

#### 테스트 파일 위치

```
backend/tests/unit/
└── content_marketing/
    ├── test_youtube_source.py
    ├── test_newsdata_source.py
    ├── test_newsapi_source.py
    ├── test_google_source.py
    └── test_safe_fetch_error_propagation.py
```

#### 2.2.1 YouTubeSource 단위 테스트

```python
# test_youtube_source.py
import pytest
from unittest.mock import AsyncMock, patch

class TestYouTubeSourceAvailability:
    """is_available 프로퍼티 검증"""

    def test_is_available_when_api_key_set(self):
        """YOUTUBE_API_KEY 설정 시 is_available=True"""
        with patch("app.core.config.settings") as mock_settings:
            mock_settings.YOUTUBE_API_KEY = "test-key"
            from app.tools.trend.sources.youtube_source import YouTubeSource
            source = YouTubeSource()
            assert source.is_available is True

    def test_is_not_available_when_api_key_empty(self):
        """YOUTUBE_API_KEY 미설정 시 is_available=False"""
        with patch("app.core.config.settings") as mock_settings:
            mock_settings.YOUTUBE_API_KEY = ""
            from app.tools.trend.sources.youtube_source import YouTubeSource
            source = YouTubeSource()
            assert source.is_available is False


class TestYouTubeSourceFetch:
    """fetch() 정상 응답 및 에러 처리 검증"""

    @pytest.mark.asyncio
    async def test_fetch_returns_raw_trend_items(self):
        """정상 응답 시 RawTrendItem 리스트 반환"""
        mock_response = {
            "items": [
                {
                    "id": {"videoId": "abc123"},
                    "snippet": {
                        "title": "법률 뉴스 영상",
                        "description": "테스트 설명",
                        "publishedAt": "2026-02-27T00:00:00Z",
                    }
                }
            ]
        }
        # httpx.AsyncClient.get 모킹
        # ...

    @pytest.mark.asyncio
    async def test_fetch_raises_on_http_error(self):
        """4xx/5xx HTTP 오류 시 예외 발생 (safe_fetch가 잡아야 함)"""
        # httpx.HTTPStatusError 발생 검증
        ...

    @pytest.mark.asyncio
    async def test_fetch_quota_exceeded_error(self):
        """일일 할당량 초과 응답(403) 처리"""
        # 403 Forbidden → HTTPStatusError 검증
        ...
```

#### 2.2.2 NewsDataSource 단위 테스트

```python
class TestNewsDataSourceFetch:

    @pytest.mark.asyncio
    async def test_fetch_with_timeframe_48h(self):
        """time_range=48h 시 timeframe=48 파라미터 전달 확인"""
        # SourceConfig(time_range="48h") → params["timeframe"] == 48
        ...

    @pytest.mark.asyncio
    async def test_fetch_filters_empty_title_or_link(self):
        """title/link 비어있는 기사 필터링 확인"""
        mock_response = {
            "results": [
                {"title": "", "link": "http://example.com", "description": "테스트"},
                {"title": "유효 기사", "link": "", "description": "테스트"},
                {"title": "정상 기사", "link": "http://example.com/valid", "description": "설명"},
            ]
        }
        # 결과: 정상 기사 1건만 반환
        ...
```

#### 2.2.3 NewsAPISource 단위 테스트

```python
class TestNewsAPISourceFetch:

    @pytest.mark.asyncio
    async def test_fetch_skips_removed_articles(self):
        """title="[Removed]" 기사 필터링 확인"""
        # 정상 기사만 반환
        ...

    @pytest.mark.asyncio
    async def test_fetch_handles_non_ok_status(self):
        """status != "ok" 응답 시 빈 리스트 반환"""
        mock_response = {"status": "error", "message": "apiKey missing"}
        # 결과: [] 반환 (예외 발생 없음)
        ...
```

#### 2.2.4 GoogleSource 단위 테스트

```python
class TestGoogleSourceAvailability:

    def test_is_available_requires_both_api_key_and_cse_id(self):
        """GOOGLE_CSE_API_KEY와 GOOGLE_CSE_ID 모두 있어야 is_available=True"""
        # API_KEY만 있는 경우 → False
        # CSE_ID만 있는 경우 → False
        # 둘 다 있는 경우 → True
        ...
```

#### 2.2.5 safe_fetch() 에러 정보 전달 테스트 (핵심)

```python
class TestSafeFetchErrorPropagation:
    """수정 후 safe_fetch() 에러 정보 전달 검증"""

    @pytest.mark.asyncio
    async def test_safe_fetch_returns_error_info_on_failure(self):
        """fetch() 실패 시 error_info가 None이 아닌 문자열 반환 (수정 후 동작)"""
        source = MockFailingSource()  # fetch()가 항상 예외 발생하는 Mock
        items, error_info = await source.safe_fetch("test", SourceConfig())
        assert items == []
        assert error_info is not None
        assert "MockFailingSource" in error_info or isinstance(error_info, str)

    @pytest.mark.asyncio
    async def test_safe_fetch_returns_none_error_info_on_success(self):
        """fetch() 성공 시 error_info=None 반환"""
        source = MockSuccessSource()
        items, error_info = await source.safe_fetch("test", SourceConfig())
        assert len(items) > 0
        assert error_info is None
```

### 2.3 통합 테스트 (실제 API 호출 불가 환경 → Mock 서버)

```python
# backend/tests/integration/test_source_availability.py

class TestSourceAvailabilityIntegration:
    """환경 변수 기반 소스 활성화 통합 테스트"""

    def test_all_seven_sources_instantiated(self):
        """TrendCollector 초기화 시 7개 소스 모두 등록"""
        collector = TrendCollector()
        all_sources = collector._all_sources
        source_names = {s.name for s in all_sources}
        expected = {
            TrendSource.TAVILY,
            TrendSource.NAVER,
            TrendSource.YOUTUBE,
            TrendSource.PERPLEXITY,
            TrendSource.GOOGLE_TRENDS,
            TrendSource.NEWSDATA,
            TrendSource.NEWSAPI,
        }
        assert source_names == expected

    def test_sources_activated_by_api_key(self, monkeypatch):
        """API 키 설정 시 해당 소스 활성화"""
        monkeypatch.setenv("YOUTUBE_API_KEY", "fake-youtube-key")
        monkeypatch.setenv("NEWSDATA_API_KEY", "fake-newsdata-key")
        monkeypatch.setenv("NEWSAPI_API_KEY", "fake-newsapi-key")
        monkeypatch.setenv("GOOGLE_CSE_API_KEY", "fake-google-key")
        monkeypatch.setenv("GOOGLE_CSE_ID", "fake-cse-id")

        # settings 재로드 후 TrendCollector 재생성
        collector = TrendCollector()
        available = collector._get_available_sources()
        available_names = {s.name.value for s in available}

        assert "youtube" in available_names
        assert "newsdata" in available_names
        assert "newsapi" in available_names
        assert "google_trends" in available_names


class TestSourcesUsedAccuracy:
    """sources_used 정합성 통합 테스트"""

    @pytest.mark.asyncio
    async def test_sources_used_reflects_actual_results(self):
        """뉴스 검색 시 실제 결과를 반환한 소스만 sources_used에 포함"""
        # Naver만 Mock 결과 반환, 나머지는 빈 리스트 반환
        # sources_used == ["naver"]
        ...

    @pytest.mark.asyncio
    async def test_cache_hit_sources_used_not_hardcoded(self):
        """캐시 히트 시 sources_used가 ['tavily', 'naver'] 하드코딩 아닌 실제 값"""
        # 수정 후: 캐시 저장 시 sources_used도 함께 저장
        # 캐시 히트 응답의 sources_used == 원래 수집 시 sources_used
        ...
```

### 2.4 E2E 테스트 (프론트엔드 → 백엔드 → 외부 API)

#### 2.4.1 프론트엔드 → 백엔드 API 계약 테스트

```bash
# 키워드 수집 API 응답 구조 검증
curl -s -X POST http://localhost:8000/api/content-marketing/keywords/collect \
  -H "Content-Type: application/json" \
  -d '{"time_range": "48h", "max_keywords": 5}' | jq '{
    sources_used: .sources_used,
    cache_hit: .cache_hit,
    total_count: .total_count
  }'

# 기대 응답 (4개 소스 활성화 후):
# {
#   "sources_used": ["tavily", "naver", "youtube", ...],  ← 하드코딩 없음
#   "cache_hit": false,
#   "total_count": 5
# }
```

```bash
# 뉴스 검색 API 응답 구조 검증 (sources_failed 필드 확인)
curl -s -X POST http://localhost:8000/api/content-marketing/keywords/{keyword_id}/news \
  -H "Content-Type: application/json" \
  -d '{"max_results": 10}' | jq '{
    sources_used: .sources_used,
    sources_failed: .sources_failed
  }'

# 기대 응답:
# {
#   "sources_used": ["naver"],
#   "sources_failed": [{"source": "youtube", "error": "API quota exceeded"}]
# }
```

#### 2.4.2 E2E 시나리오 흐름

```
시나리오 1: 4개 소스 모두 정상 (Happy Path)
  1. 환경: 7개 API 키 모두 설정
  2. 실행: 키워드 수집 → 뉴스 검색
  3. 검증: sources_used에 7개 소스명 포함

시나리오 2: 일부 소스 실패 (Partial Failure)
  1. 환경: YouTube API 할당량 초과 Mock
  2. 실행: 키워드 수집 → 뉴스 검색
  3. 검증:
     - sources_failed에 "youtube" 포함
     - 나머지 소스 결과는 정상 반환

시나리오 3: 캐시 히트 정합성
  1. 실행: 동일 요청 2회 반복
  2. 검증:
     - 2번째 응답: cache_hit=true
     - 2번째 sources_used == 1번째 sources_used (하드코딩 아님)

시나리오 4: 전체 소스 실패 (Complete Failure)
  1. 환경: Tavily API 키 미설정 + 나머지 모두 실패 Mock
  2. 실행: 뉴스 검색
  3. 검증: TrendSourceError 발생 또는 빈 결과 + sources_failed 포함
```

---

## 3. 품질 기준 (합격/불합격 판정 기준)

### 3.1 필수 통과 기준 (Must Pass — 미통과 시 배포 차단)

| ID | 기준 | 검증 방법 |
|----|------|---------|
| QC-01 | `YOUTUBE_API_KEY` 설정 시 `YouTubeSource.is_available == True` | 단위 테스트 |
| QC-02 | `GOOGLE_CSE_API_KEY` + `GOOGLE_CSE_ID` 모두 설정 시만 `GoogleSource.is_available == True` | 단위 테스트 |
| QC-03 | `safe_fetch()` 실패 시 `([], error_string)` 반환 (예외 누출 없음) | 단위 테스트 |
| QC-04 | 캐시 히트 시 `sources_used` 필드가 `["tavily", "naver"]` 아닌 실제 값 | 통합 테스트 |
| QC-05 | `sources_failed` 필드가 API 응답 스키마에 포함 (`KeywordNewsResponse`) | API 계약 테스트 |
| QC-06 | 1개 소스 실패가 다른 소스 결과에 영향 없음 (에러 격리) | 통합 테스트 |
| QC-07 | 7개 소스 모두 `TrendCollector._all_sources`에 등록 확인 | 통합 테스트 |

### 3.2 권고 기준 (Should Pass — 미통과 시 경고)

| ID | 기준 | 검증 방법 |
|----|------|---------|
| QC-08 | 7개 소스 병렬 호출 시 전체 응답 시간 20초 이내 | 성능 테스트 |
| QC-09 | 소스별 타임아웃이 `source_timeouts` 딕셔너리 기준 적용 | 단위 테스트 |
| QC-10 | `sources_failed` 에러 메시지에 API 키 값 미포함 (보안) | 보안 스캔 |
| QC-11 | 로그에 실패 소스명 + 원인 기록 (`logger.warning(...)`) | 로그 분석 |

---

## 4. Pre-mortem 분석 (사전 실패 지점 식별)

### 4.1 고위험 실패 시나리오

#### 실패 시나리오 A: 외부 API 할당량 초과

| 항목 | 내용 |
|------|------|
| **발생 조건** | YouTube (10,000 units/day), Google CSE (100건/day), NewsAPI (100건/day), NewsData (200건/day) 무료 한도 초과 |
| **증상** | `safe_fetch()` 호출 → HTTP 429/403 → 빈 리스트 반환 → `sources_used`에서 제외 |
| **현재 처리** | 예외 삼킴, 로그만 출력 (사용자에게 불투명) |
| **수정 후 처리** | `sources_failed` 필드에 소스명 + 에러 코드 포함 |
| **테스트** | Mock 429 응답 → `sources_failed` 필드 검증 |
| **위험도** | 높음 — 비용 없이 실패 원인 파악 불가 |

#### 실패 시나리오 B: 네트워크 타임아웃

| 항목 | 내용 |
|------|------|
| **발생 조건** | 외부 API 응답 지연 (YouTube 10초, Google 10초, NewsData 15초, NewsAPI 10초) |
| **증상** | `asyncio.wait_for()` 타임아웃 → `TimeoutError` → 빈 리스트 |
| **현재 처리** | `search_news_for_keyword()`의 `_fetch_with_timeout()` 처리 중 |
| **위험 사항** | `collect_community_keywords()`에는 개별 소스 타임아웃 없음 |
| **권고** | 커뮤니티 수집 메서드에도 소스별 타임아웃 추가 필요 |
| **위험도** | 중간 — 전체 요청 지연 가능성 |

#### 실패 시나리오 C: API 응답 구조 변경

| 항목 | 내용 |
|------|------|
| **발생 조건** | 외부 API의 응답 JSON 구조가 변경되는 경우 (예: `items` → `data`) |
| **영향 소스** | YouTube, Newsdata.io, NewsAPI.org, Google CSE |
| **증상** | `data.get("items", [])` 등 방어 코드가 있어 빈 리스트 반환 |
| **감지 방법** | `sources_used`에서 해당 소스 사라짐 (실패 원인은 로그 확인 필요) |
| **위험도** | 낮음 — 방어 코드 존재, 단 탐지 지연 가능 |

#### 실패 시나리오 D: 병렬 호출 수 증가로 성능 저하

| 항목 | 내용 |
|------|------|
| **발생 조건** | `collect_community_keywords()`에 4개 소스 추가 시 Tavily(15초) + 나머지 병렬 |
| **현재 처리** | Tavily + Naver 2개만 병렬 → 최대 15초 |
| **수정 후** | 최대 7개 소스 병렬 → 가장 느린 소스가 병목 |
| **위험 사항** | 뉴스 소스(YouTube, Google 등)는 커뮤니티 데이터와 성격이 달라 품질 저하 가능 |
| **권고** | Stage 1(커뮤니티)은 Tavily + Naver 유지, Stage 3(뉴스 검색)에서만 4개 추가 사용 |
| **위험도** | 중간 — 설계 의도와 품질 간 트레이드오프 |

#### 실패 시나리오 E: 캐시 히트 시 sources_used 불일치

| 항목 | 내용 |
|------|------|
| **발생 조건** | 캐시 저장 시 sources_used 미포함 → 캐시 히트 시 하드코딩 반환 |
| **현재 코드** | `content_marketing_service.py` line 310: `sources_used=["tavily", "naver"]` 하드코딩 |
| **수정 방향** | 캐시 저장 시 `sources_used`도 함께 저장, 캐시 히트 시 저장된 값 반환 |
| **테스트** | 동일 요청 2회 → 두 번째 응답의 sources_used == 첫 번째 응답의 sources_used |
| **위험도** | 높음 — 사용자에게 잘못된 정보 제공 |

### 4.2 의존성 리스크

| 의존성 | 리스크 | 완화 방안 |
|--------|--------|---------|
| YouTube Data API v3 | 일일 10,000 units 무료 제한 (검색 1회 = 100 units → 100회/day) | 결과 캐싱, 할당량 초과 시 명확한 에러 메시지 |
| Google CSE | 무료 100건/day 매우 제한적 | 프리미엄 플랜 검토 또는 낮은 우선순위 소스로 분류 |
| NewsAPI.org | 무료 플랜 `/v2/everything` 100건/day, 1개월 이전 데이터 불가 | 유료 플랜 또는 대체 소스 검토 |
| NewsData.io | 무료 200건/day, 48시간 timeframe 고정 | 제한 고지 UI 추가 |

---

## 5. 보안 사전 검증

### 5.1 API 키 노출 위험 검토

#### 검증 대상: 에러 메시지 내 키 포함 여부

```python
# 위험 패턴 (수정 전 가능성)
logger.warning("소스 youtube 수집 실패: API key invalid: %s", settings.YOUTUBE_API_KEY)
# → 로그에 API 키 노출 가능

# 안전 패턴 (수정 후 필수)
logger.warning("소스 youtube 수집 실패: HTTP 403 (API key invalid)")
# → 키 값 미포함
```

#### 보안 스캔 체크리스트

```bash
# 에러 메시지/로그에 API 키 패턴 포함 여부 확인
grep -rn "settings\.YOUTUBE_API_KEY\|settings\.NEWSDATA_API_KEY\|settings\.NEWSAPI_API_KEY\|settings\.GOOGLE_CSE_API_KEY" \
  backend/app/tools/trend/sources/ \
  backend/app/services/service_function/content_marketing_service.py

# 기대 결과: is_available 프로퍼티와 params 구성에만 사용, logger 호출에는 미포함
```

#### `sources_failed` 응답 필드 보안 검증

```python
# 허용되는 에러 정보 (보안 안전)
{
    "sources_failed": [
        {"source": "youtube", "error": "HTTP 403: quota exceeded"},
        {"source": "newsapi", "error": "timeout after 10s"}
    ]
}

# 허용 안 되는 에러 정보 (키 포함 금지)
{
    "sources_failed": [
        {"source": "youtube", "error": "API key AIza...xyz is invalid"}  # 금지
    ]
}
```

### 5.2 입력값 검증 (인젝션 방지)

#### 검색 키워드 sanitize 경로 검증

```python
# search_news_for_keyword() 진입점 (collector.py line 318-319)
sanitized = sanitize_keyword(keyword)
if not sanitized:
    logger.warning("키워드 Sanitize 실패: %s", keyword[:30])
    return [], []
```

검증 항목:
- `sanitize_keyword()`가 SQL 특수문자, HTML 태그, 긴 문자열을 필터링하는가?
- 각 소스의 `params`에 sanitize된 값만 전달되는가?
- HTTP 파라미터 인코딩은 `httpx` 라이브러리가 자동 처리하는가? (확인 필요)

#### 보안 검증 명령어

```bash
# keyword_blacklist.py의 sanitize 로직 확인
grep -n "sanitize_keyword\|blacklist" backend/app/tools/trend/keyword_blacklist.py
```

---

## 6. Zero Script QA — Docker 로그 기반 런타임 검증

### 6.1 검증 환경 준비

```bash
# 서버 시작 (Docker Compose 또는 로컬)
cd backend && uv run uvicorn app.main:app --reload --log-level debug 2>&1 | \
  grep -E "소스|source|fetch|WARNING|ERROR"
```

### 6.2 모니터링 포인트

수정 후 다음 로그 패턴을 실시간 모니터링한다.

| 로그 패턴 | 의미 | 기대 동작 |
|----------|------|---------|
| `소스 youtube 수집 실패` | YouTube 실패 | `sources_failed`에 youtube 포함 |
| `YouTube 수집 완료: N건` | YouTube 성공 | `sources_used`에 youtube 포함 |
| `소스 X 수집 실패, 건너뜀` | 기존 WARNING 로그 | 수정 후 에러 정보도 함께 출력 |
| `키워드 뉴스 검색 'X': N건 (소스 M개)` | 뉴스 검색 완료 | M이 활성화 소스 수와 일치 |

### 6.3 E2E 로그 추적 시나리오

```bash
# 시나리오 1: YouTube API 정상 동작 확인 (API 키 설정 환경)
curl -X POST http://localhost:8000/api/content-marketing/keywords/{keyword_id}/news \
  -H "Content-Type: application/json" -d '{"max_results": 5}'

# 로그 기대:
# INFO     YouTube 수집 완료: 3건
# INFO     키워드 뉴스 검색 'X': 10건 (소스 4개)

# 시나리오 2: 타임아웃 발생 (네트워크 느린 환경 시뮬레이션)
# → "소스 youtube 타임아웃 (10s)" 로그 확인
# → sources_failed에 youtube 포함 확인
```

---

## 7. 테스트 실행 계획

### 7.1 수정 전 베이스라인 측정

```bash
cd backend

# 현재 단위 테스트 통과 상태 확인
uv run pytest tests/unit/ -v --tb=short 2>&1 | tail -20

# 정적 분석
uv run ruff check app/tools/trend/
uv run mypy app/tools/trend/ --ignore-missing-imports
```

### 7.2 수정 후 검증 순서

```bash
# 1단계: 정적 검증 (필수)
uv run ruff check app/tools/trend/ app/services/service_function/content_marketing_service.py
uv run mypy app/tools/trend/ app/services/service_function/content_marketing_service.py

# 2단계: 단위 테스트
uv run pytest tests/unit/content_marketing/ -v

# 3단계: 통합 테스트 (Mock API 환경)
uv run pytest tests/integration/test_source_availability.py -v

# 4단계: API 계약 테스트
uv run uvicorn app.main:app &
curl -s http://localhost:8000/api/content-marketing/keywords/collect \
  -X POST -H "Content-Type: application/json" -d '{}' | jq .sources_used
```

### 7.3 합격 기준 요약

| 단계 | 합격 기준 |
|------|---------|
| 정적 검증 | ruff 에러 0개, mypy 에러 0개 |
| 단위 테스트 | QC-01 ~ QC-07 모두 통과 |
| 통합 테스트 | QC-04 (sources_used 정합성), QC-05 (sources_failed 필드) 통과 |
| 보안 검증 | API 키 로그/응답 미포함 확인 |
| 성능 기준 | QC-08 통과 (20초 이내) |

---

## 8. 갭 분석 (수정 전 현재 상태)

### 8.1 설계 vs 현재 구현 갭

| 항목 | 설계 의도 | 현재 구현 | 갭 |
|------|----------|---------|-----|
| 7개 소스 활성화 | API 키 설정 소스 자동 활성화 | Tavily + Naver만 키워드 수집에 사용 | 4개 소스 미활용 |
| 에러 가시성 | 실패 소스 + 원인 API 응답 포함 | 에러 삼킴, WARNING 로그만 출력 | `sources_failed` 필드 없음 |
| sources_used 정합성 | 캐시 히트 시에도 실제 소스 반영 | 하드코딩 `["tavily", "naver"]` | 캐시 히트 시 부정확 |
| 소스별 타임아웃 | 뉴스 검색에 소스별 차등 타임아웃 | `search_news_for_keyword()`에만 적용 | `collect_community_keywords()`에 미적용 |

### 8.2 Match Rate 기준 평가

현재 구현의 설계 대비 Match Rate 추정:

| 검증 영역 | 가중치 | 현재 점수 |
|---------|-------|---------|
| 소스 활성화 정확성 | 30% | 40% (3/7 소스만 활용) |
| 에러 가시성 | 25% | 20% (로그만 있음) |
| sources_used 정합성 | 25% | 30% (캐시 미스 시만 정확) |
| 성능 및 타임아웃 처리 | 20% | 70% (뉴스 검색에는 적용됨) |
| **전체 Match Rate** | **100%** | **약 38%** |

38%는 최소 합격 기준(90%)에 크게 미달 → 수정 필수.

---

## 9. 권고 사항

### 9.1 즉시 수정 필요 (Critical)

1. `safe_fetch()` 반환 타입을 `(list[RawTrendItem], str | None)` 튜플로 변경, 실패 시 에러 문자열 포함
2. `KeywordNewsResponse` 스키마에 `sources_failed: list[SourceFailureInfo]` 필드 추가
3. 캐시 히트 시 `sources_used` 하드코딩 제거 — 캐시 저장 시 `sources_used` 함께 보존

### 9.2 설계 결정 필요 (팀 논의)

4. **키워드 수집(`collect_community_keywords()`)에 뉴스 소스(YouTube 등) 포함 여부**: 커뮤니티 인기글 수집과 뉴스 기사 수집은 성격이 다름. Stage 1은 Tavily + Naver 유지하고 Stage 3 뉴스 검색에서만 4개 추가 소스 활용하는 현재 설계가 더 적합할 수 있음.

5. **Google CSE 무료 한도(100건/day) 대응**: 현 무료 플랜으로는 실질적 활용 제한. 우선순위 낮은 선택적 소스로 분류 권고.

### 9.3 향후 개선 (Medium)

6. 소스별 health check 엔드포인트 (`GET /api/content-marketing/sources/health`) 추가
7. `collect_community_keywords()` 내 소스별 타임아웃 적용
8. 프론트엔드 소스 상태 배지 UI (활성/비활성/실패) 추가

---

## 10. 체크리스트 (구현팀 전달용)

### 백엔드 수정 체크리스트

- [ ] `BaseTrendSource.safe_fetch()` 반환 타입 `tuple[list[RawTrendItem], str | None]`로 변경
- [ ] `TrendCollector._collect_news()` 에러 정보 수집 후 `sources_failed` 빌드
- [ ] `TrendCollector.search_news_for_keyword()` `sources_failed` 반환 포함
- [ ] `KeywordNewsResponse` 스키마에 `sources_failed` 필드 추가 (`frontend/src/features/content-marketing/types/`와 동기화)
- [ ] `content_marketing_service.collect_keywords()` 캐시 히트 시 `sources_used` 하드코딩 제거
- [ ] `content_marketing_service.collect_keywords_stream()` 동일 수정
- [ ] 에러 메시지에 API 키 값 미포함 확인
- [ ] `ruff check` + `mypy` 통과

### 프론트엔드 동기화 체크리스트

- [ ] `frontend/src/features/content-marketing/types/index.ts` — `KeywordNewsResponse`에 `sources_failed` 필드 추가
- [ ] `frontend/src/features/content-marketing/components/KeywordNewsList.tsx` — `sources_failed` 표시 UI (옵션)

### 테스트 체크리스트

- [ ] `tests/unit/content_marketing/test_youtube_source.py` 작성
- [ ] `tests/unit/content_marketing/test_newsdata_source.py` 작성
- [ ] `tests/unit/content_marketing/test_newsapi_source.py` 작성
- [ ] `tests/unit/content_marketing/test_google_source.py` 작성
- [ ] `tests/unit/content_marketing/test_safe_fetch_error_propagation.py` 작성
- [ ] `tests/integration/test_source_availability.py` 작성
- [ ] QC-01 ~ QC-07 모두 통과 확인
