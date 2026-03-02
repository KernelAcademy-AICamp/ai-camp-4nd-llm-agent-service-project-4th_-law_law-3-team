# Legal News Pipeline (법률 뉴스 수집/요약 파이프라인) Design Document

> **Summary**: 법률신문 + 네이버 법 관련 기사를 매일 자동 수집하여 정제, PII 마스킹, AI 요약, PostgreSQL 저장, LanceDB 청킹/임베딩까지 수행하는 배치 파이프라인의 상세 설계
>
> **Project**: law-3 (Legal President / 법률 대통령)
> **Version**: 0.3.0
> **Author**: Claude (PDCA Agent Team)
> **Date**: 2026-02-26
> **Status**: Reviewed (3중 검증 완료 + 보류 항목 전량 반영)
> **Planning Doc**: [legal-news-pipeline.plan.md](../../01-plan/features/legal-news-pipeline.plan.md) (v0.3)

---

## 1. Overview

### 1.1 Design Goals

1. **기존 아키텍처 일관성**: `app/tools/` 하위에 독립 파이프라인 패키지로 구성하여, 기존 `trend/`, `llm/`, `vectorstore/` 패턴과 동일한 구조 유지
2. **소스 확장성**: `BaseNewsSource` ABC를 상속하여 새 뉴스 소스를 추가할 수 있는 Strategy 패턴 적용
3. **기존 인프라 재활용**: `get_chat_model()` (LLM), `create_query_embedding()` (임베딩), `InMemoryRateLimiter` (속도 제한), `async_session_factory` (DB) 등 기존 코드 직접 재사용
4. **파이프라인 단계 분리**: 수집→정제→PII→요약→저장→청킹을 각각 독립 클래스로 분리하여 단계별 테스트/재처리 가능
5. **안전한 운영**: PII 마스킹, Cross-Reference 검증, LLM fallback, 부분 실패 허용 등 운영 안정성 확보
6. **v0.3.0 — 프론트엔드 연동**: `app/modules/legal_news` API 모듈로 뉴스 조회/검색 엔드포인트 제공
7. **v0.3.0 — 파이프라인 관측성**: OpenTelemetry trace/span + OpenLineage 계보 이벤트
8. **v0.3.0 — 비동기 오케스트레이션**: TaskIQ 기반 소스별 병렬 수집 + Dead-letter queue
9. **v0.3.0 — 검색 품질 고도화**: LanceDB 하이브리드 검색(Vector+FTS) + 리랭커 적용
10. **v0.3.0 — Fuzzy 중복 제거**: SimHash 기반 유사 기사 탐지 (Stage 4)

### 1.2 Design Principles

- **Single Responsibility**: 각 파이프라인 단계(Source, Cleaner, PIIFilter, Summarizer, Chunker)는 독립 클래스
- **Open/Closed**: `BaseNewsSource` 추상 클래스 → 새 소스 추가 시 기존 코드 무수정
- **Dependency Inversion**: LLM 클라이언트는 `get_chat_model()` 팩토리로 추상화, DB는 `async_session_factory`로 DI
- **Fail-Safe**: 한 기사/소스 실패 시 나머지 계속 진행, 실패 건은 리포트에 기록
- **기존 패턴 100% 준수**: ORM 모델은 `app/models/`, 서비스는 `app/services/service_function/`, 설정은 `app/core/config.py`

---

## 2. Architecture

### 2.1 Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    News Pipeline Orchestrator                            │
│                  (news_pipeline_service.py)                              │
│                                                                         │
│  ┌──── Phase 1: Collect ─────────────────────────────────────────────┐ │
│  │                                                                    │ │
│  │  ┌──────────────────┐    ┌──────────────────┐                    │ │
│  │  │  LawtimesSource   │    │  NaverNewsSource  │  ... (확장 가능)  │ │
│  │  │  (RSS/HTML 크롤)  │    │  (검색 API)       │                    │ │
│  │  └────────┬─────────┘    └────────┬─────────┘                    │ │
│  │           └──────────┬───────────┘                                │ │
│  │                      ▼                                            │ │
│  │           ┌─────────────────────┐                                │ │
│  │           │   Deduplicator       │                                │ │
│  │           │   (3단계 중복 제거)   │                                │ │
│  │           └──────────┬──────────┘                                │ │
│  └──────────────────────┼────────────────────────────────────────────┘ │
│                         ▼                                              │
│  ┌──── Phase 2: Process ─────────────────────────────────────────────┐ │
│  │                                                                    │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌────────────────────────┐   │ │
│  │  │  Cleaner     │→│  PIIFilter   │→│  Summarizer             │   │ │
│  │  │  (HTML 정제)  │  │  (마스킹)    │  │  (Solar-Pro2 + fallback)│   │ │
│  │  └─────────────┘  └─────────────┘  └──────────┬─────────────┘   │ │
│  │                                                │                  │ │
│  │                                     ┌──────────▼─────────────┐   │ │
│  │                                     │  ReferenceValidator     │   │ │
│  │                                     │  (법령/판례 DB 대조)     │   │ │
│  │                                     └──────────┬─────────────┘   │ │
│  └────────────────────────────────────────────────┼──────────────────┘ │
│                                                   ▼                    │
│  ┌──── Phase 3: Store ───────────────────────────────────────────────┐ │
│  │                                                                    │ │
│  │  ┌───────────────────────┐    ┌───────────────────────────────┐  │ │
│  │  │  PostgreSQL Writer     │    │  Chunker + LanceDB Writer     │  │ │
│  │  │  (news_articles 테이블) │    │  (news_chunks 테이블)          │  │ │
│  │  └───────────────────────┘    └───────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌──── Phase 4: Report ──────────────────────────────────────────────┐ │
│  │  PipelineReporter (일일 운영 리포트 생성)                           │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘

외부 의존성:
┌────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐
│ Solar-Pro2  │  │ PostgreSQL   │  │ LanceDB      │  │ Naver API   │
│ (Upstage)   │  │ (news_       │  │ (news_       │  │ (검색)      │
│             │  │  articles)   │  │  chunks)     │  │             │
└────────────┘  └──────────────┘  └──────────────┘  └─────────────┘

v0.3.0 추가 컴포넌트:
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐
│ TaskIQ       │  │ OpenTelemetry│  │ OpenLineage  │  │ Redis       │
│ (비동기 큐)   │  │ (Trace/Span) │  │ (계보 이벤트) │  │ (DLQ+Broker)│
└──────────────┘  └──────────────┘  └──────────────┘  └─────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  API Layer (v0.3.0): app/modules/legal_news                      │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────────────┐  │
│  │ GET /list       │  │ GET /{id}       │  │ GET /search       │  │
│  │ (뉴스 목록)     │  │ (뉴스 상세)     │  │ (하이브리드 검색)  │  │
│  └────────────────┘  └────────────────┘  └───────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 Class Diagram

```
                        ┌──────────────────────┐
                        │  NewsPipelineService  │
                        │  (오케스트레이터)       │
                        └──────────┬───────────┘
                                   │ uses
          ┌────────────┬───────────┼────────────┬──────────────┐
          ▼            ▼           ▼            ▼              ▼
  ┌───────────┐ ┌───────────┐ ┌────────┐ ┌──────────┐ ┌────────────┐
  │SourceMgr  │ │ Cleaner   │ │PIIFilter│ │Summarizer│ │   Chunker  │
  └─────┬─────┘ └───────────┘ └────────┘ └──────────┘ └────────────┘
        │ manages
  ┌─────┴──────────────────┐
  │                         │
  ▼                         ▼
┌──────────────┐  ┌──────────────────┐
│BaseNewsSource│  │  Deduplicator    │
│  (ABC)       │  └──────────────────┘
└──────┬───────┘
       │ implements
  ┌────┴────────────────┐
  ▼                     ▼
┌──────────────┐  ┌──────────────────┐
│LawtimesSource│  │NaverNewsSource   │
└──────────────┘  └──────────────────┘

보조:
┌─────────────────────┐  ┌───────────────────┐  ┌────────────────────┐
│ ReferenceValidator   │  │ PipelineReporter  │  │ NewsPipelineConfig │
└─────────────────────┘  └───────────────────┘  └────────────────────┘
```

---

## 3. Detailed Interface Design

### 3.1 Internal Data Models (`models.py`)

```python
# app/tools/news_pipeline/models.py
"""뉴스 파이프라인 내부 데이터 모델"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class NewsSourceType(str, Enum):
    """뉴스 소스 유형"""
    LAWTIMES = "lawtimes"
    NAVER = "naver"


@dataclass
class RawArticle:
    """수집 단계 출력: 원시 기사 데이터"""
    url: str
    title: str
    raw_html: str                       # 원문 HTML
    source: NewsSourceType
    publisher: str                      # 매체명
    published_at: datetime | None
    author: str | None = None
    section: str | None = None
    tags: list[str] = field(default_factory=list)
    raw_metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class CleanedArticle:
    """정제 단계 출력: 정제된 기사"""
    url: str
    title: str
    cleaned_text: str                   # HTML 제거, 정규화된 본문
    content_hash: str                   # SHA256(cleaned_text)
    source: NewsSourceType
    publisher: str
    published_at: datetime | None
    author: str | None = None
    section: str | None = None
    tags: list[str] = field(default_factory=list)
    char_count: int = 0                 # 본문 글자 수


@dataclass
class ArticleSummary:
    """요약 결과"""
    one_liner: str                      # 한줄 요지
    issues: list[str]                   # 주요 쟁점 (3~5개)
    laws: list[str]                     # 언급 법령
    cases: list[str]                    # 언급 판례
    institutions: list[str]             # 언급 기관
    implications: list[str]             # 시사점 (1~3개)
    law_ids: list[int] = field(default_factory=list)   # v0.2.0: 검증된 법령 DB ID
    case_ids: list[int] = field(default_factory=list)  # v0.2.0: 검증된 판례 DB ID


@dataclass
class ProcessedArticle:
    """최종 처리 완료 기사 (DB 저장 직전)"""
    doc_id: str                         # SHA256(url)
    url: str
    title: str
    cleaned_text: str
    content_hash: str
    source: NewsSourceType
    publisher: str
    published_at: datetime | None
    collected_at: datetime
    author: str | None
    section: str | None
    tags: list[str]
    summary: ArticleSummary
    disclaimer: str = "본 문서는 기사 요약이며 법령/판례 원문이 아닙니다"
    schema_version: str = "1.0"


@dataclass
class PipelineResult:
    """파이프라인 실행 결과 (리포트용)"""
    run_id: str                         # UUID
    started_at: datetime
    finished_at: datetime | None = None
    source_stats: dict[str, SourceStat] = field(default_factory=dict)
    total_collected: int = 0
    total_deduplicated: int = 0         # 중복 제거된 수
    total_cleaned: int = 0
    total_summarized: int = 0
    total_stored: int = 0
    total_chunked: int = 0
    errors: list[PipelineError] = field(default_factory=list)


@dataclass
class SourceStat:
    """소스별 통계"""
    collected: int = 0
    deduplicated: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class PipelineError:
    """파이프라인 에러 기록"""
    stage: str                          # "collect" | "clean" | "summarize" | "store" | "chunk"
    article_url: str | None
    error_type: str
    error_message: str
    timestamp: datetime = field(default_factory=datetime.now)
```

### 3.2 Configuration (`config.py`)

```python
# app/tools/news_pipeline/config.py
"""뉴스 파이프라인 설정"""

from __future__ import annotations

from dataclasses import dataclass, field
from app.core.config import settings


@dataclass(frozen=True)
class NewsPipelineConfig:
    """파이프라인 설정 (환경 변수에서 로드)"""

    # 활성화
    enabled: bool = True

    # 요약 LLM
    summary_provider: str = "upstage"
    summary_model: str = "solar-pro2"
    summary_temperature: float = 0.3
    fallback_provider: str = "openai"

    # 수집
    min_article_length: int = 500       # 최소 본문 길이 (자)
    max_articles_per_run: int = 200     # 1회 최대 수집 수
    rate_limit_crawl: float = 2.0       # 크롤링 요청 간격 (초)

    # 소스 활성화
    lawtimes_enabled: bool = True
    naver_enabled: bool = True

    # 청킹
    chunk_size: int = 1500              # 청크 크기 (자)
    chunk_overlap: int = 200            # 청크 겹침 (자)

    # LanceDB
    lancedb_table: str = "news_chunks"

    # 보관
    retention_days: int = 90

    # 네이버 키워드
    naver_keywords: list[str] = field(default_factory=lambda: [
        "법원 판결", "대법원", "헌법재판소",
        "검찰 기소", "로펌", "변호사",
        "민사소송", "형사재판", "행정소송",
        "노동법", "세무 판결", "공정거래",
    ])

    @classmethod
    def from_settings(cls) -> NewsPipelineConfig:
        """settings에서 환경 변수를 읽어 설정 생성"""
        return cls(
            enabled=getattr(settings, "NEWS_PIPELINE_ENABLED", True),
            summary_provider=getattr(settings, "NEWS_PIPELINE_SUMMARY_PROVIDER", "upstage"),
            summary_model=getattr(settings, "NEWS_PIPELINE_SUMMARY_MODEL", "solar-pro2"),
            min_article_length=getattr(settings, "NEWS_PIPELINE_MIN_ARTICLE_LENGTH", 500),
            max_articles_per_run=getattr(settings, "NEWS_PIPELINE_MAX_ARTICLES_PER_RUN", 200),
            rate_limit_crawl=getattr(settings, "NEWS_PIPELINE_RATE_LIMIT_CRAWL", 2.0),
            lawtimes_enabled=getattr(settings, "NEWS_PIPELINE_LAWTIMES_ENABLED", True),
            naver_enabled=getattr(settings, "NEWS_PIPELINE_NAVER_ENABLED", True),
            chunk_size=getattr(settings, "NEWS_PIPELINE_CHUNK_SIZE", 1500),
            chunk_overlap=getattr(settings, "NEWS_PIPELINE_CHUNK_OVERLAP", 200),
            lancedb_table=getattr(settings, "NEWS_PIPELINE_LANCEDB_TABLE", "news_chunks"),
            retention_days=getattr(settings, "NEWS_PIPELINE_RETENTION_DAYS", 90),
            fallback_provider=getattr(settings, "NEWS_PIPELINE_LLM_FALLBACK_PROVIDER", "openai"),
        )
```

### 3.3 SSRF Guard (`ssrf_guard.py`)

```python
# app/tools/news_pipeline/ssrf_guard.py
"""SSRF 방어: httpx 요청 전 내부망 IP 차단"""

from __future__ import annotations

import ipaddress
import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# 차단 대상 IP 대역 (내부망, 로컬호스트, 링크-로컬)
_BLOCKED_NETWORKS: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = [
    ipaddress.IPv4Network("10.0.0.0/8"),
    ipaddress.IPv4Network("172.16.0.0/12"),
    ipaddress.IPv4Network("192.168.0.0/16"),
    ipaddress.IPv4Network("127.0.0.0/8"),
    ipaddress.IPv4Network("169.254.0.0/16"),
    ipaddress.IPv6Network("::1/128"),
    ipaddress.IPv6Network("fc00::/7"),
    ipaddress.IPv6Network("fe80::/10"),
]


def validate_url(url: str) -> bool:
    """URL이 외부 접근 가능한지 검증. 내부망 IP면 False 반환."""
    import socket

    parsed = urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        return False

    try:
        addr_info = socket.getaddrinfo(hostname, None)
        for _, _, _, _, sockaddr in addr_info:
            ip = ipaddress.ip_address(sockaddr[0])
            for network in _BLOCKED_NETWORKS:
                if ip in network:
                    logger.warning("SSRF 차단: %s → %s (내부망)", url, ip)
                    return False
    except (socket.gaierror, ValueError):
        logger.warning("DNS 확인 실패: %s", hostname)
        return False

    return True
```

> **v0.2.0 추가 (Red Team 피드백)**: httpx 클라이언트에서 내부망 IP 요청 차단. 모든 소스의 `fetch()` 호출 전 `validate_url()` 검증.

### 3.4 Exception Classes (`exceptions.py`)

```python
# app/tools/news_pipeline/exceptions.py
"""뉴스 파이프라인 예외 클래스"""


class NewsPipelineError(Exception):
    """파이프라인 기본 예외"""


class SourceFetchError(NewsPipelineError):
    """소스 수집 실패"""
    def __init__(self, source: str, message: str) -> None:
        self.source = source
        super().__init__(f"[{source}] 수집 실패: {message}")


class ArticleCleanError(NewsPipelineError):
    """기사 정제 실패"""


class SummaryGenerationError(NewsPipelineError):
    """요약 생성 실패"""
    def __init__(self, url: str, message: str) -> None:
        self.url = url
        super().__init__(f"요약 실패 [{url}]: {message}")


class StorageError(NewsPipelineError):
    """저장 실패"""


class ChunkingError(NewsPipelineError):
    """청킹 실패"""
```

### 3.5 Base News Source (`sources/__init__.py`)

```python
# app/tools/news_pipeline/sources/__init__.py
"""뉴스 소스 추상 인터페이스"""

from __future__ import annotations

import abc
from datetime import date

from app.tools.news_pipeline.models import RawArticle, NewsSourceType


class BaseNewsSource(abc.ABC):
    """뉴스 소스 추상 베이스 클래스

    새 뉴스 소스 추가 시 이 클래스를 상속:
        class NewSource(BaseNewsSource):
            @property
            def source_type(self) -> NewsSourceType: ...
            @property
            def is_available(self) -> bool: ...
            async def fetch(self, target_date: date) -> list[RawArticle]: ...
    """

    @property
    @abc.abstractmethod
    def source_type(self) -> NewsSourceType:
        """소스 유형 반환"""

    @property
    @abc.abstractmethod
    def is_available(self) -> bool:
        """소스 사용 가능 여부 (API 키, 설정 확인)"""

    @abc.abstractmethod
    async def fetch(self, target_date: date) -> list[RawArticle]:
        """지정 날짜의 기사를 수집하여 반환

        Args:
            target_date: 수집 대상 날짜

        Returns:
            수집된 원시 기사 목록

        Raises:
            SourceFetchError: 수집 실패 시
        """

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(type={self.source_type.value})>"
```

### 3.6 Lawtimes Source (`sources/lawtimes_source.py`)

```python
# app/tools/news_pipeline/sources/lawtimes_source.py
"""법률신문 크롤러"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import date, datetime
from typing import Any
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup

from app.tools.news_pipeline.config import NewsPipelineConfig
from app.tools.news_pipeline.exceptions import SourceFetchError
from app.tools.news_pipeline.models import NewsSourceType, RawArticle
from app.tools.news_pipeline.sources import BaseNewsSource

logger = logging.getLogger(__name__)

LAWTIMES_BASE_URL = "https://www.lawtimes.co.kr"
LAWTIMES_RSS_URL = "https://www.lawtimes.co.kr/rss"  # RSS 피드 URL (확인 필요)
USER_AGENT = "LegalNewsBot/1.0 (legal-president; +contact@example.com)"

# 크롤링 대상 섹션
TARGET_SECTIONS: list[str] = [
    "사회", "법조", "판결", "입법", "해외법조", "법률신문 뉴스",
]

# v0.2.0: Selector config (HTML 크롤링 구조 변경 대비)
# 사이트 구조 변경 시 이 dict만 수정하면 됨
SELECTOR_CONFIG: dict[str, str] = {
    "article_list": "div.news-list .article-item",     # 목록 페이지 기사 항목
    "article_title": "h2.article-title a",              # 제목 링크
    "article_date": "span.date",                        # 발행일
    "body_content": "div.article-body, div#article-body",  # 본문 영역 (CSS selector)
    "body_exclude": "div.ad-area, div.related-news",    # 본문에서 제외할 영역
}


class LawtimesSource(BaseNewsSource):
    """법률신문 (lawtimes.co.kr) 크롤러

    수집 전략:
    1. RSS 피드가 존재하면 RSS 우선 사용
    2. RSS 불가 시 HTML 크롤링 fallback
    3. robots.txt 준수, Rate Limit 적용
    """

    def __init__(self, config: NewsPipelineConfig) -> None:
        self._config = config
        self._robot_parser: RobotFileParser | None = None

    @property
    def source_type(self) -> NewsSourceType:
        return NewsSourceType.LAWTIMES

    @property
    def is_available(self) -> bool:
        return self._config.lawtimes_enabled

    async def fetch(self, target_date: date) -> list[RawArticle]:
        """법률신문 기사 수집"""
        if not self.is_available:
            return []

        await self._check_robots_txt()

        articles: list[RawArticle] = []
        try:
            # RSS 시도 → 실패 시 HTML fallback
            articles = await self._fetch_via_rss(target_date)
            if not articles:
                articles = await self._fetch_via_html(target_date)
        except Exception as exc:
            raise SourceFetchError("lawtimes", str(exc)) from exc

        logger.info("법률신문 수집 완료: %d건 (대상: %s)", len(articles), target_date)
        return articles

    async def _check_robots_txt(self) -> None:
        """robots.txt 확인 및 캐싱"""
        if self._robot_parser is not None:
            return
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{LAWTIMES_BASE_URL}/robots.txt",
                    headers={"User-Agent": USER_AGENT},
                )
                self._robot_parser = RobotFileParser()
                self._robot_parser.parse(resp.text.splitlines())
        except Exception:
            logger.warning("robots.txt 로드 실패, 기본 정책 적용")
            self._robot_parser = RobotFileParser()

    def _can_fetch(self, url: str) -> bool:
        """robots.txt 기반 접근 허용 여부"""
        if self._robot_parser is None:
            return True
        return self._robot_parser.can_fetch(USER_AGENT, url)

    async def _fetch_via_rss(self, target_date: date) -> list[RawArticle]:
        """RSS 피드로 기사 수집 (구현 상세는 Do 단계)"""
        # feedparser를 사용하여 RSS 파싱
        # 대상 날짜 필터링
        # RawArticle 변환
        ...  # Do 단계에서 구현

    async def _fetch_via_html(self, target_date: date) -> list[RawArticle]:
        """HTML 크롤링으로 기사 수집 (RSS 불가 시 fallback)"""
        # 섹션별 목록 페이지 크롤링
        # 개별 기사 페이지 본문 추출
        # Rate Limit 적용 (config.rate_limit_crawl 간격)
        ...  # Do 단계에서 구현

    async def _fetch_article_body(
        self, url: str, client: httpx.AsyncClient
    ) -> str | None:
        """개별 기사 본문 HTML 추출"""
        if not self._can_fetch(url):
            logger.warning("robots.txt 차단: %s", url)
            return None

        await asyncio.sleep(self._config.rate_limit_crawl)
        resp = await client.get(url, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        return resp.text
```

### 3.7 Naver News Source (`sources/naver_news_source.py`)

```python
# app/tools/news_pipeline/sources/naver_news_source.py
"""네이버 뉴스 검색 API 기반 수집"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime
from email.utils import parsedate_to_datetime

import httpx

from app.core.config import settings
from app.tools.news_pipeline.config import NewsPipelineConfig
from app.tools.news_pipeline.exceptions import SourceFetchError
from app.tools.news_pipeline.models import NewsSourceType, RawArticle
from app.tools.news_pipeline.sources import BaseNewsSource

logger = logging.getLogger(__name__)

NAVER_SEARCH_URL = "https://openapi.naver.com/v1/search/news.json"
_HTML_TAG_RE = re.compile(r"<[^>]+>")


class NaverNewsSource(BaseNewsSource):
    """네이버 뉴스 검색 API 데이터 소스

    기존 app/tools/trend/sources/naver_source.py 패턴을 참고하되,
    뉴스 파이프라인 전용으로 독립 구현:
    - 키워드 목록 순회하며 검색
    - 날짜 필터링 (target_date)
    - 원문 URL 추출 (originallink 우선)
    """

    def __init__(self, config: NewsPipelineConfig) -> None:
        self._config = config

    @property
    def source_type(self) -> NewsSourceType:
        return NewsSourceType.NAVER

    @property
    def is_available(self) -> bool:
        return bool(
            self._config.naver_enabled
            and settings.NAVER_CLIENT_ID
            and settings.NAVER_CLIENT_SECRET
        )

    async def fetch(self, target_date: date) -> list[RawArticle]:
        """네이버 뉴스 키워드 검색으로 수집"""
        if not self.is_available:
            return []

        all_articles: list[RawArticle] = []
        seen_urls: set[str] = set()

        headers = {
            "X-Naver-Client-Id": settings.NAVER_CLIENT_ID,
            "X-Naver-Client-Secret": settings.NAVER_CLIENT_SECRET,
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            for keyword in self._config.naver_keywords:
                try:
                    articles = await self._search_keyword(
                        client, headers, keyword, target_date, seen_urls,
                    )
                    all_articles.extend(articles)
                except Exception as exc:
                    logger.warning("네이버 키워드 [%s] 검색 실패: %s", keyword, exc)
                    continue

                if len(all_articles) >= self._config.max_articles_per_run:
                    break

        logger.info("네이버 뉴스 수집 완료: %d건 (대상: %s)", len(all_articles), target_date)
        return all_articles

    async def _search_keyword(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        keyword: str,
        target_date: date,
        seen_urls: set[str],
    ) -> list[RawArticle]:
        """단일 키워드 검색"""
        params: dict[str, str | int] = {
            "query": keyword,
            "display": 100,     # 최대 100건
            "sort": "date",     # 최신순
        }

        resp = await client.get(NAVER_SEARCH_URL, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()

        articles: list[RawArticle] = []
        for item in data.get("items", []):
            url = item.get("originallink") or item.get("link", "")
            if not url or url in seen_urls:
                continue

            published_at = self._parse_date(item.get("pubDate"))
            if published_at and published_at.date() != target_date:
                continue

            seen_urls.add(url)
            articles.append(RawArticle(
                url=url,
                title=self._strip_html(item.get("title", "")),
                raw_html=item.get("description", ""),  # 네이버 API는 snippet만 제공
                source=NewsSourceType.NAVER,
                publisher=self._extract_publisher(url),
                published_at=published_at,
                section=None,
                tags=[keyword],
                raw_metadata={"naver_link": item.get("link", "")},
            ))

        return articles

    @staticmethod
    def _strip_html(text: str) -> str:
        return _HTML_TAG_RE.sub("", text).strip()

    @staticmethod
    def _parse_date(date_str: str | None) -> datetime | None:
        if not date_str:
            return None
        try:
            return parsedate_to_datetime(date_str)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_publisher(url: str) -> str:
        """URL에서 매체명 추출 (도메인 기반)"""
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        # 주요 법률 매체 매핑
        publisher_map: dict[str, str] = {
            "www.lawtimes.co.kr": "법률신문",
            "www.lec.co.kr": "법률저널",
            "www.legaltimes.co.kr": "리걸타임즈",
            "news.law.go.kr": "법제처",
        }
        return publisher_map.get(domain, domain)
```

### 3.8 Deduplicator (`deduplicator.py`)

```python
# app/tools/news_pipeline/deduplicator.py
"""4단계 중복 제거 (v0.3.0: Stage 4 SimHash Fuzzy Dedup 추가)"""

from __future__ import annotations

import hashlib
import logging
from datetime import date

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.news_article import NewsArticle
from app.tools.news_pipeline.models import RawArticle

logger = logging.getLogger(__name__)


class Deduplicator:
    """4단계 중복 제거

    Stage 1: URL 완전 일치 (DB 조회)
    Stage 2: 제목 + 발행일 조합 (DB 조회)
    Stage 3: 본문 해시 비교 (정제 후, content_hash)
    Stage 4: SimHash Fuzzy Dedup (v0.3.0, 유사도 95%+ 기사 탐지)
    """

    def __init__(self, simhash_threshold: int = 3) -> None:
        self._simhash_threshold = simhash_threshold

    async def deduplicate(
        self,
        articles: list[RawArticle],
        db: AsyncSession,
    ) -> list[RawArticle]:
        """중복 제거된 기사 목록 반환"""
        if not articles:
            return []

        # Stage 1: URL 기반 (DB에 이미 있는 URL 제외)
        urls = [a.url for a in articles]
        existing_urls = await self._get_existing_urls(db, urls)
        after_url = [a for a in articles if a.url not in existing_urls]
        logger.info("중복 제거 Stage 1 (URL): %d → %d", len(articles), len(after_url))

        # Stage 2: 제목+발행일 기반 (같은 배치 내 중복)
        after_title = self._deduplicate_by_title_date(after_url)
        logger.info("중복 제거 Stage 2 (제목+일자): %d → %d", len(after_url), len(after_title))

        # Stage 4: SimHash Fuzzy Dedup (v0.3.0)
        after_fuzzy = self._deduplicate_by_simhash(after_title)
        logger.info("중복 제거 Stage 4 (SimHash): %d → %d", len(after_title), len(after_fuzzy))

        return after_fuzzy

    async def check_content_hash(
        self, content_hash: str, db: AsyncSession
    ) -> bool:
        """Stage 3: 본문 해시 중복 여부 확인 (정제 후 호출)"""
        result = await db.execute(
            select(NewsArticle.id).where(
                NewsArticle.content_hash == content_hash
            ).limit(1)
        )
        return result.scalar_one_or_none() is not None

    def _deduplicate_by_simhash(
        self, articles: list[RawArticle]
    ) -> list[RawArticle]:
        """v0.3.0 Stage 4: SimHash 기반 유사 기사 제거 (배치 내)

        해밍 거리가 threshold 이하면 유사 기사로 판정하여 먼저 나온 기사만 유지.
        """
        from app.tools.news_pipeline.fuzzy_dedup import compute_simhash, hamming_distance

        unique: list[RawArticle] = []
        seen_hashes: list[int] = []

        for article in articles:
            text = f"{article.title} {article.raw_html[:2000]}"
            article_hash = compute_simhash(text)

            is_duplicate = False
            for existing_hash in seen_hashes:
                if hamming_distance(article_hash, existing_hash) <= self._simhash_threshold:
                    is_duplicate = True
                    logger.debug("SimHash 유사 기사 탐지: %s", article.url)
                    break

            if not is_duplicate:
                unique.append(article)
                seen_hashes.append(article_hash)

        return unique

    async def _get_existing_urls(
        self, db: AsyncSession, urls: list[str]
    ) -> set[str]:
        """DB에 이미 존재하는 URL 집합 반환"""
        if not urls:
            return set()
        result = await db.execute(
            select(NewsArticle.url).where(NewsArticle.url.in_(urls))
        )
        return {row[0] for row in result.all()}

    @staticmethod
    def _deduplicate_by_title_date(articles: list[RawArticle]) -> list[RawArticle]:
        """제목+발행일 기반 배치 내 중복 제거"""
        seen: set[str] = set()
        unique: list[RawArticle] = []
        for article in articles:
            pub_date = article.published_at.date().isoformat() if article.published_at else "unknown"
            key = f"{article.title.strip()}|{pub_date}"
            if key not in seen:
                seen.add(key)
                unique.append(article)
        return unique
```

### 3.9 Cleaner (`cleaner.py`)

```python
# app/tools/news_pipeline/cleaner.py
"""기사 본문 정제/정규화"""

from __future__ import annotations

import hashlib
import logging
import re

from bs4 import BeautifulSoup

from app.tools.news_pipeline.config import NewsPipelineConfig
from app.tools.news_pipeline.models import CleanedArticle, RawArticle

logger = logging.getLogger(__name__)

# 제거 대상 패턴
_AD_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"[\[【]\s*광고\s*[\]】]", re.IGNORECASE),
    re.compile(r"기자\s*[가-힣]+@[a-zA-Z0-9.]+"),        # 기자 이메일
    re.compile(r"ⓒ\s*.+$", re.MULTILINE),               # 저작권 표시
    re.compile(r"무단\s*전재.+금지", re.IGNORECASE),
    re.compile(r"<저작권자.+>", re.IGNORECASE),
]

# 연속 공백/줄바꿈 정규화
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_MULTI_SPACE = re.compile(r" {2,}")


class Cleaner:
    """기사 본문 정제기

    1. HTML 태그 제거 (본문 영역만 추출)
    2. 광고/네비/저작권/기자 정보 제거
    3. 공백/줄바꿈 정규화
    4. 길이 검증 (min_article_length 이상)
    5. content_hash (SHA256) 생성
    """

    def __init__(self, config: NewsPipelineConfig) -> None:
        self._min_length = config.min_article_length

    def clean(self, article: RawArticle) -> CleanedArticle | None:
        """원시 기사 → 정제된 기사. 길이 미달 시 None 반환."""
        # 1. HTML → 텍스트
        text = self._extract_text(article.raw_html)

        # 2. 광고/메타 제거
        text = self._remove_patterns(text)

        # 3. 공백 정규화
        text = _MULTI_NEWLINE.sub("\n\n", text)
        text = _MULTI_SPACE.sub(" ", text)
        text = text.strip()

        # 4. 길이 검증
        if len(text) < self._min_length:
            logger.debug("본문 길이 미달 (%d < %d): %s", len(text), self._min_length, article.url)
            return None

        # 5. 해시 생성
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        return CleanedArticle(
            url=article.url,
            title=article.title,
            cleaned_text=text,
            content_hash=content_hash,
            source=article.source,
            publisher=article.publisher,
            published_at=article.published_at,
            author=article.author,
            section=article.section,
            tags=article.tags,
            char_count=len(text),
        )

    @staticmethod
    def _extract_text(html: str) -> str:
        """HTML에서 텍스트 추출"""
        soup = BeautifulSoup(html, "lxml")
        # script, style, nav, header, footer 제거
        for tag in soup.find_all(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()
        return soup.get_text(separator="\n")

    @staticmethod
    def _remove_patterns(text: str) -> str:
        """광고/메타/저작권 패턴 제거"""
        for pattern in _AD_PATTERNS:
            text = pattern.sub("", text)
        return text
```

### 3.10 PII Filter (`pii_filter.py`)

```python
# app/tools/news_pipeline/pii_filter.py
"""개인정보(PII) 마스킹 필터"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# PII 패턴 (한국 법률 뉴스 특화)
_PHONE_PATTERN = re.compile(
    r"0\d{1,2}[-.\s]?\d{3,4}[-.\s]?\d{4}"  # 전화번호
)
_EMAIL_PATTERN = re.compile(
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
)
_ADDRESS_PATTERN = re.compile(
    r"(?:서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|전북|전남|경북|경남|제주)"
    r"(?:특별시|광역시|특별자치시|도|특별자치도)?\s+"
    r"[가-힣]+(?:시|군|구)\s+"
    r"[가-힣]+(?:읍|면|동|로|길)\s*"
    r"[\d\-가-힣]*"
)
_RESIDENT_ID_PATTERN = re.compile(
    r"\d{6}[-\s]?[1-4]\d{6}"  # 주민등록번호
)

# 마스킹 대체 문자열
MASK_MAP: dict[str, str] = {
    "phone": "[PHONE]",
    "email": "[EMAIL]",
    "address": "[ADDRESS]",
    "resident_id": "[RESIDENT_ID]",
}


class PIIFilter:
    """PII 마스킹 필터

    정규식 기반으로 전화번호, 이메일, 주소, 주민번호를 마스킹.
    법률 뉴스 특성상 판사/검사/변호사 실명은 공인으로 분류하여 유지.
    피고인/피해자 등 사건 관계자의 경우, 기사 원문에서 이미 익명 처리된 경우가 많음.
    """

    def mask(self, text: str) -> tuple[str, int]:
        """텍스트 내 PII를 마스킹하고 (마스킹된 텍스트, 마스킹 건수) 반환"""
        count = 0

        # 주민번호 (가장 먼저 — 가장 민감)
        text, n = _RESIDENT_ID_PATTERN.subn(MASK_MAP["resident_id"], text)
        count += n

        # 전화번호
        text, n = _PHONE_PATTERN.subn(MASK_MAP["phone"], text)
        count += n

        # 이메일
        text, n = _EMAIL_PATTERN.subn(MASK_MAP["email"], text)
        count += n

        # 주소
        text, n = _ADDRESS_PATTERN.subn(MASK_MAP["address"], text)
        count += n

        if count > 0:
            logger.info("PII 마스킹 %d건 적용", count)

        return text, count
```

### 3.11 Summarizer (`summarizer.py`)

```python
# app/tools/news_pipeline/summarizer.py
"""LLM 기반 구조화 요약 생성"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.tools.llm import get_chat_model
from app.tools.news_pipeline.config import NewsPipelineConfig
from app.tools.news_pipeline.exceptions import SummaryGenerationError
from app.tools.news_pipeline.models import ArticleSummary, CleanedArticle

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """당신은 법률 전문 기자입니다. 아래 기사를 법률 AI 보조 코퍼스용으로 요약해주세요.

## 출력 형식 (JSON만 출력, 다른 텍스트 없이)
{
  "one_liner": "핵심 한 문장 요약",
  "issues": ["쟁점1", "쟁점2"],
  "mentions": {
    "laws": ["관련 법령명"],
    "cases": ["관련 판례"],
    "institutions": ["관련 기관"]
  },
  "implications": ["시사점1"]
}

## 규칙
- 기사에 없는 판례/법령 조문 번호를 만들어내지 마세요
- "확정적 법률 결론"을 단정하지 마세요 (기사 요약임)
- 한국어로 작성하세요
- issues는 3~5개, implications는 1~3개"""


class Summarizer:
    """LLM 기반 구조화 요약 생성기

    Primary: Upstage Solar-Pro2
    Fallback: OpenAI → 건너뛰기
    """

    def __init__(self, config: NewsPipelineConfig) -> None:
        self._config = config

    async def summarize(self, article: CleanedArticle) -> ArticleSummary:
        """기사를 구조화 요약으로 변환"""
        user_prompt = self._build_user_prompt(article)

        # Primary LLM 시도
        try:
            return await self._call_llm(
                provider=self._config.summary_provider,
                model=self._config.summary_model,
                user_prompt=user_prompt,
            )
        except Exception as primary_exc:
            logger.warning(
                "Primary LLM 실패 (%s/%s): %s",
                self._config.summary_provider,
                self._config.summary_model,
                primary_exc,
            )

        # Fallback LLM 시도
        try:
            return await self._call_llm(
                provider=self._config.fallback_provider,
                model=None,  # 프로바이더 기본 모델
                user_prompt=user_prompt,
            )
        except Exception as fallback_exc:
            raise SummaryGenerationError(
                article.url,
                f"Primary + Fallback 모두 실패: {primary_exc} / {fallback_exc}",
            ) from fallback_exc

    async def _call_llm(
        self,
        provider: str,
        model: str | None,
        user_prompt: str,
    ) -> ArticleSummary:
        """LLM 호출 및 JSON 파싱"""
        llm = get_chat_model(
            provider=provider,
            model=model,
            temperature=self._config.summary_temperature,
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        response = await llm.ainvoke(messages)
        raw_text = response.content
        if isinstance(raw_text, list):
            raw_text = raw_text[0] if raw_text else ""

        return self._parse_response(str(raw_text))

    @staticmethod
    def _build_user_prompt(article: CleanedArticle) -> str:
        """사용자 프롬프트 생성"""
        pub_date = article.published_at.isoformat() if article.published_at else "불명"
        return (
            f"## 기사 원문\n"
            f"제목: {article.title}\n"
            f"매체: {article.publisher}\n"
            f"발행일: {pub_date}\n\n"
            f"{article.cleaned_text[:8000]}"  # 토큰 제한 방지
        )

    @staticmethod
    def _parse_response(text: str) -> ArticleSummary:
        """LLM 응답 JSON 파싱 + Pydantic 품질 게이트 검증 (v0.2.0)"""
        # JSON 블록 추출 (```json ... ``` 또는 순수 JSON)
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        data: dict[str, Any] = json.loads(text)

        # v0.2.0: Pydantic strict validation (품질 게이트)
        validated = SummarySchema.model_validate(data)
        mentions = validated.mentions

        return ArticleSummary(
            one_liner=validated.one_liner,
            issues=validated.issues,
            laws=mentions.laws if mentions else [],
            cases=mentions.cases if mentions else [],
            institutions=mentions.institutions if mentions else [],
            implications=validated.implications,
        )


# v0.2.0: 요약 JSON 품질 게이트 (Pydantic strict validation)
from pydantic import BaseModel, Field


class MentionsSchema(BaseModel):
    """요약 내 참조 스키마"""
    laws: list[str] = Field(default_factory=list, max_length=20)
    cases: list[str] = Field(default_factory=list, max_length=20)
    institutions: list[str] = Field(default_factory=list, max_length=20)


class SummarySchema(BaseModel):
    """LLM 요약 출력 검증 스키마"""
    one_liner: str = Field(min_length=5, max_length=500)
    issues: list[str] = Field(min_length=1, max_length=10)
    mentions: MentionsSchema = Field(default_factory=MentionsSchema)
    implications: list[str] = Field(min_length=1, max_length=5)
```

### 3.12 Reference Validator (`reference_validator.py`)

```python
# app/tools/news_pipeline/reference_validator.py
"""요약 내 법령/판례 Cross-Reference 검증"""

from __future__ import annotations

import logging

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.law_document import LawDocument
from app.models.precedent_document import PrecedentDocument
from app.tools.news_pipeline.models import ArticleSummary

logger = logging.getLogger(__name__)


class ReferenceValidator:
    """요약 내 법령/판례 번호가 DB에 존재하는지 검증

    존재하지 않는 법령/판례는 요약에서 제거하여 환각(hallucination) 방지.
    단, 기사에서 언급된 '법령명'은 정확한 조문 번호가 아닐 수 있으므로,
    법령명 기반의 유사 매칭도 허용.

    v0.2.0: 검증된 참조의 DB ID도 함께 반환하여 프론트엔드 직접 링크 지원.
    """

    async def validate_and_filter(
        self,
        summary: ArticleSummary,
        db: AsyncSession,
    ) -> ArticleSummary:
        """검증 후 유효한 참조만 남긴 요약 반환 (DB ID 포함)"""
        validated_laws, law_ids = await self._validate_laws(summary.laws, db)
        validated_cases, case_ids = await self._validate_cases(summary.cases, db)

        removed_laws = set(summary.laws) - set(validated_laws)
        removed_cases = set(summary.cases) - set(validated_cases)

        if removed_laws:
            logger.info("환각 법령 제거: %s", removed_laws)
        if removed_cases:
            logger.info("환각 판례 제거: %s", removed_cases)

        return ArticleSummary(
            one_liner=summary.one_liner,
            issues=summary.issues,
            laws=validated_laws,
            cases=validated_cases,
            institutions=summary.institutions,
            implications=summary.implications,
            law_ids=law_ids,
            case_ids=case_ids,
        )

    async def _validate_laws(
        self, laws: list[str], db: AsyncSession
    ) -> tuple[list[str], list[int]]:
        """법령명이 DB에 존재하는지 확인 (부분 매칭 허용). ID도 반환."""
        valid: list[str] = []
        valid_ids: list[int] = []
        for law_name in laws:
            base_name = law_name.split(" 제")[0].strip()  # "민법 제750조" → "민법"
            result = await db.execute(
                select(LawDocument.id).where(
                    LawDocument.law_name.contains(base_name)
                ).limit(1)
            )
            row = result.scalar_one_or_none()
            if row is not None:
                valid.append(law_name)
                valid_ids.append(row)
        return valid, valid_ids

    async def _validate_cases(
        self, cases: list[str], db: AsyncSession
    ) -> tuple[list[str], list[int]]:
        """판례 번호가 DB에 존재하는지 확인. ID도 반환."""
        valid: list[str] = []
        valid_ids: list[int] = []
        for case_ref in cases:
            result = await db.execute(
                select(PrecedentDocument.id).where(
                    PrecedentDocument.case_number.contains(case_ref)
                ).limit(1)
            )
            row = result.scalar_one_or_none()
            if row is not None:
                valid.append(case_ref)
                valid_ids.append(row)
        return valid, valid_ids
```

### 3.13 Chunker (`chunker.py`)

```python
# app/tools/news_pipeline/chunker.py
"""RAG용 청킹 + 임베딩 + LanceDB 저장"""

from __future__ import annotations

import logging
from typing import Any

from app.services.rag.embedding import create_query_embedding
from app.tools.news_pipeline.config import NewsPipelineConfig
from app.tools.news_pipeline.models import ProcessedArticle

logger = logging.getLogger(__name__)


@dataclass
class NewsChunk:
    """LanceDB 저장용 청크"""
    chunk_id: str
    doc_id: str
    chunk_text: str
    chunk_type: str           # "summary" | "body"
    published_at: str
    source: str
    publisher: str
    title: str
    url: str
    is_secondary: bool = True  # 항상 True
    data_type: str = "news_article"


class Chunker:
    """기사를 RAG용 청크로 분할 + 임베딩 생성

    청킹 전략:
    1. 요약 → 별도 chunk (항상 포함, chunk_type="summary")
    2. 본문 → 고정 크기 분절 + overlap (chunk_type="body")
    3. 모든 청크에 is_secondary=true 메타데이터 필수
    """

    def __init__(self, config: NewsPipelineConfig) -> None:
        self._chunk_size = config.chunk_size
        self._chunk_overlap = config.chunk_overlap

    def create_chunks(self, article: ProcessedArticle) -> list[NewsChunk]:
        """기사를 청크 목록으로 변환"""
        chunks: list[NewsChunk] = []
        pub_at = article.published_at.isoformat() if article.published_at else ""

        # 1. 요약 청크
        summary_text = self._build_summary_text(article)
        chunks.append(NewsChunk(
            chunk_id=f"{article.doc_id}_summary",
            doc_id=article.doc_id,
            chunk_text=summary_text,
            chunk_type="summary",
            published_at=pub_at,
            source=article.source.value,
            publisher=article.publisher,
            title=article.title,
            url=article.url,
        ))

        # 2. 본문 청크
        body_chunks = self._split_text(article.cleaned_text)
        for idx, chunk_text in enumerate(body_chunks):
            chunks.append(NewsChunk(
                chunk_id=f"{article.doc_id}_body_{idx}",
                doc_id=article.doc_id,
                chunk_text=chunk_text,
                chunk_type="body",
                published_at=pub_at,
                source=article.source.value,
                publisher=article.publisher,
                title=article.title,
                url=article.url,
            ))

        return chunks

    async def embed_and_store(
        self, chunks: list[NewsChunk], lancedb_table_name: str
    ) -> int:
        """청크를 임베딩하고 LanceDB에 저장. 저장 건수 반환.

        v0.3.0: FTS 인덱스 자동 생성 → 하이브리드 검색 지원
        """
        import lancedb
        from app.core.config import settings

        if not chunks:
            return 0

        # 임베딩 생성
        records: list[dict[str, Any]] = []
        for chunk in chunks:
            vector = create_query_embedding(chunk.chunk_text)
            records.append({
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "chunk_text": chunk.chunk_text,
                "chunk_type": chunk.chunk_type,
                "published_at": chunk.published_at,
                "source": chunk.source,
                "publisher": chunk.publisher,
                "title": chunk.title,
                "url": chunk.url,
                "is_secondary": chunk.is_secondary,
                "data_type": chunk.data_type,
                "vector": vector,
            })

        # LanceDB 저장
        db = lancedb.connect(settings.LANCEDB_URI)
        try:
            table = db.open_table(lancedb_table_name)
            table.add(records)
        except Exception:
            # 테이블 미존재 시 생성
            table = db.create_table(lancedb_table_name, data=records)

        # v0.3.0: FTS 인덱스 (하이브리드 검색용, 최초 1회만 생성)
        self._ensure_fts_index(table)

        logger.info("LanceDB 저장 완료: %d 청크 → %s", len(records), lancedb_table_name)
        return len(records)

    @staticmethod
    def _ensure_fts_index(table: Any) -> None:
        """FTS 인덱스 존재 확인 및 생성 (v0.3.0)"""
        try:
            table.create_fts_index("chunk_text", replace=False)
        except Exception:
            pass  # 이미 존재하면 무시

    @staticmethod
    async def hybrid_search(
        query: str,
        lancedb_table_name: str,
        *,
        limit: int = 20,
        rerank_top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """v0.3.0: 하이브리드 검색 (Vector + FTS) + 리랭커

        Consultant 피드백: LanceDB 저장만으로는 검색 품질 부족.
        Vector + FTS 결합 후 리랭커로 정밀 정렬.

        Args:
            query: 검색 쿼리
            lancedb_table_name: LanceDB 테이블명
            limit: 1차 검색 결과 수
            rerank_top_k: 리랭킹 후 반환할 최종 결과 수

        Returns:
            정렬된 검색 결과 목록
        """
        import lancedb
        from app.core.config import settings

        db = lancedb.connect(settings.LANCEDB_URI)
        table = db.open_table(lancedb_table_name)

        # 1. 벡터 검색
        query_vector = create_query_embedding(query)
        vector_results = (
            table.search(query_vector)
            .limit(limit)
            .to_list()
        )

        # 2. FTS 검색
        fts_results = (
            table.search(query, query_type="fts")
            .limit(limit)
            .to_list()
        )

        # 3. RRF (Reciprocal Rank Fusion) 결합
        merged = _reciprocal_rank_fusion(vector_results, fts_results, k=60)

        # 4. 리랭커 적용 (기존 인프라 재활용)
        reranked = await _apply_reranker(query, merged[:rerank_top_k * 2])

        return reranked[:rerank_top_k]


async def _apply_reranker(
    query: str, candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """기존 리랭커 인프라로 후보 재정렬

    app/services/rag/ 하위 리랭커를 재사용.
    리랭커 미설정 시 원래 순서 유지.
    """
    try:
        from app.services.rag.reranker import rerank_documents
        texts = [c.get("chunk_text", "") for c in candidates]
        scores = await rerank_documents(query, texts)
        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = score
        candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
    except ImportError:
        logger.debug("리랭커 미설치, RRF 순서 유지")
    return candidates


def _reciprocal_rank_fusion(
    results_a: list[dict],
    results_b: list[dict],
    k: int = 60,
) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion — 두 검색 결과를 통합 정렬"""
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for rank, item in enumerate(results_a):
        doc_id = item.get("chunk_id", str(rank))
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
        items[doc_id] = item

    for rank, item in enumerate(results_b):
        doc_id = item.get("chunk_id", str(rank))
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
        items[doc_id] = item

    sorted_ids = sorted(scores, key=scores.get, reverse=True)
    return [items[doc_id] for doc_id in sorted_ids if doc_id in items]

    @staticmethod
    def _build_summary_text(article: ProcessedArticle) -> str:
        """요약 정보를 검색용 텍스트로 조합"""
        s = article.summary
        parts = [
            f"[요약] {s.one_liner}",
            f"[쟁점] {', '.join(s.issues)}" if s.issues else "",
            f"[법령] {', '.join(s.laws)}" if s.laws else "",
            f"[판례] {', '.join(s.cases)}" if s.cases else "",
            f"[기관] {', '.join(s.institutions)}" if s.institutions else "",
            f"[시사점] {', '.join(s.implications)}" if s.implications else "",
        ]
        return "\n".join(p for p in parts if p)

    def _split_text(self, text: str) -> list[str]:
        """본문을 고정 크기 청크로 분할 (overlap 적용)"""
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + self._chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk.strip())
            start += self._chunk_size - self._chunk_overlap
        return chunks
```

### 3.14 Pipeline Reporter (`reporter.py`)

```python
# app/tools/news_pipeline/reporter.py
"""파이프라인 운영 리포트 생성"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from app.tools.news_pipeline.models import PipelineResult

logger = logging.getLogger(__name__)

REPORT_DIR = Path("data/news_pipeline/reports")


class PipelineReporter:
    """일일 운영 리포트 생성기

    v0.2.0: Slack Webhook 알림 + KPI 메트릭 추가
    """

    def __init__(self, webhook_url: str | None = None) -> None:
        self._webhook_url = webhook_url

    def generate(self, result: PipelineResult) -> Path:
        """리포트 파일 생성 후 경로 반환"""
        REPORT_DIR.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = REPORT_DIR / f"report_{timestamp}.json"

        duration = (
            (result.finished_at - result.started_at).total_seconds()
            if result.finished_at else None
        )

        # v0.2.0: KPI 메트릭 계산
        kpi = self._compute_kpi(result, duration)

        report_data = {
            "run_id": result.run_id,
            "started_at": result.started_at.isoformat(),
            "finished_at": result.finished_at.isoformat() if result.finished_at else None,
            "duration_seconds": duration,
            "kpi": kpi,
            "stats": {
                "total_collected": result.total_collected,
                "total_deduplicated": result.total_deduplicated,
                "total_cleaned": result.total_cleaned,
                "total_summarized": result.total_summarized,
                "total_stored": result.total_stored,
                "total_chunked": result.total_chunked,
            },
            "source_stats": {
                name: {
                    "collected": stat.collected,
                    "deduplicated": stat.deduplicated,
                    "failed": stat.failed,
                    "errors": stat.errors,
                }
                for name, stat in result.source_stats.items()
            },
            "errors": [
                {
                    "stage": e.stage,
                    "article_url": e.article_url,
                    "error_type": e.error_type,
                    "error_message": e.error_message,
                    "timestamp": e.timestamp.isoformat(),
                }
                for e in result.errors
            ],
        }

        report_path.write_text(
            json.dumps(report_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # 콘솔 요약
        logger.info(
            "━━━ 파이프라인 리포트 ━━━\n"
            "수집: %d건 | 중복제거: -%d건 | 정제: %d건 | 요약: %d건 | 저장: %d건 | 청킹: %d건\n"
            "에러: %d건 | 소요: %s\n"
            "KPI → 수집성공률: %.1f%% | 요약성공률: %.1f%% | 중복률: %.1f%%",
            result.total_collected,
            result.total_deduplicated,
            result.total_cleaned,
            result.total_summarized,
            result.total_stored,
            result.total_chunked,
            len(result.errors),
            f"{duration:.1f}초" if duration else "진행중",
            kpi["collect_success_rate"],
            kpi["summary_success_rate"],
            kpi["duplicate_rate"],
        )

        # v0.2.0: 에러 발생 시 Slack 알림
        if result.errors and self._webhook_url:
            self._send_alert(result, kpi)

        return report_path

    @staticmethod
    def _compute_kpi(result: PipelineResult, duration: float | None) -> dict[str, float]:
        """KPI 메트릭 산출 (Consultant 피드백 반영)"""
        total = result.total_collected or 1  # ZeroDivision 방지
        return {
            "collect_success_rate": (total - len([
                e for e in result.errors if e.stage == "collect"
            ])) / total * 100,
            "summary_success_rate": result.total_summarized / max(result.total_cleaned, 1) * 100,
            "duplicate_rate": result.total_deduplicated / total * 100,
            "avg_processing_seconds": duration / max(result.total_stored, 1) if duration else 0,
        }

    def _send_alert(self, result: PipelineResult, kpi: dict[str, float]) -> None:
        """Slack Webhook으로 에러 알림 전송 (Red Team 피드백 반영)"""
        import httpx

        error_summary = "\n".join(
            f"- [{e.stage}] {e.error_type}: {e.error_message[:100]}"
            for e in result.errors[:5]  # 최대 5건
        )
        text = (
            f"⚠️ *뉴스 파이프라인 에러 알림*\n"
            f"Run: `{result.run_id}`\n"
            f"에러: {len(result.errors)}건\n"
            f"수집성공률: {kpi['collect_success_rate']:.1f}%\n"
            f"```{error_summary}```"
        )

        try:
            httpx.post(self._webhook_url, json={"text": text}, timeout=10.0)
        except Exception as exc:
            logger.warning("Slack 알림 전송 실패: %s", exc)
```

### 3.15 Fuzzy Deduplicator (`fuzzy_dedup.py`) — v0.3.0

```python
# app/tools/news_pipeline/fuzzy_dedup.py
"""SimHash 기반 Fuzzy 중복 탐지 (v0.3.0)

Red Team 피드백: 동일 사건을 다른 매체가 약간 다른 표현으로 보도하는 '재탕 기사' 탐지.
SimHash 해밍 거리 기반으로 95%+ 유사도 기사를 필터링.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter

# SimHash 비트 수
SIMHASH_BITS = 64


def compute_simhash(text: str, *, ngram_size: int = 3) -> int:
    """텍스트의 SimHash 값을 계산

    Args:
        text: 입력 텍스트 (한국어 포함)
        ngram_size: 문자 n-gram 크기 (기본 3)

    Returns:
        64비트 SimHash 정수값
    """
    # 전처리: 공백/특수문자 정규화
    text = re.sub(r"\s+", " ", text.strip().lower())

    # 문자 n-gram 생성 (한국어 형태소 분석 대신 n-gram 사용 — 외부 의존성 최소화)
    tokens = [text[i:i + ngram_size] for i in range(len(text) - ngram_size + 1)]
    token_counts = Counter(tokens)

    # 가중치 벡터 초기화
    vector = [0] * SIMHASH_BITS

    for token, weight in token_counts.items():
        # 각 토큰의 해시값
        token_hash = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        for i in range(SIMHASH_BITS):
            bitmask = 1 << i
            if token_hash & bitmask:
                vector[i] += weight
            else:
                vector[i] -= weight

    # 최종 SimHash 생성
    fingerprint = 0
    for i in range(SIMHASH_BITS):
        if vector[i] > 0:
            fingerprint |= (1 << i)

    return fingerprint


def hamming_distance(hash_a: int, hash_b: int) -> int:
    """두 SimHash 값 간의 해밍 거리 계산

    Returns:
        0~64 범위의 해밍 거리 (작을수록 유사)
    """
    xor = hash_a ^ hash_b
    return bin(xor).count("1")


def is_near_duplicate(
    hash_a: int,
    hash_b: int,
    threshold: int = 3,
) -> bool:
    """두 해시가 유사 문서인지 판정

    threshold=3 → 해밍 거리 3 이하 → 약 95%+ 유사도
    """
    return hamming_distance(hash_a, hash_b) <= threshold
```

### 3.16 Telemetry (`telemetry.py`) — v0.3.0

```python
# app/tools/news_pipeline/telemetry.py
"""OpenTelemetry 기반 파이프라인 관측성 (v0.3.0)

Consultant 피드백: 단계별 통계만으로는 운영 가시성 부족.
run/article 단위 Trace ID 연결로 병목 구간 즉시 파악 가능.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from app.core.config import settings

logger = logging.getLogger(__name__)

# 전역 트레이서 (lazy init)
_tracer = None


def _init_tracer() -> Any:
    """OpenTelemetry TracerProvider 초기화 (최초 1회)"""
    global _tracer

    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource

    resource = Resource.create({
        "service.name": "news-pipeline",
        "service.version": "0.3.0",
    })

    provider = TracerProvider(resource=resource)

    otel_endpoint = getattr(settings, "NEWS_PIPELINE_OTEL_ENDPOINT", "")
    if otel_endpoint:
        # OTLP gRPC exporter
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        exporter = OTLPSpanExporter(endpoint=otel_endpoint)
    else:
        # 콘솔 출력 (개발/로컬용)
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter
        exporter = ConsoleSpanExporter()

    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer("news_pipeline", "0.3.0")

    logger.info("OpenTelemetry 초기화 완료 (endpoint=%s)", otel_endpoint or "console")
    return _tracer


def get_tracer() -> Any:
    """트레이서 인스턴스 반환 (lazy init)"""
    global _tracer
    if _tracer is None:
        return _init_tracer()
    return _tracer


@asynccontextmanager
async def pipeline_span(
    name: str,
    *,
    run_id: str = "",
    article_url: str = "",
    attributes: dict[str, str] | None = None,
) -> AsyncIterator[Any]:
    """파이프라인 단계별 span 컨텍스트 매니저

    사용 예:
        async with pipeline_span("collect", run_id=run_id):
            articles = await source.fetch(target_date)
    """
    tracer = get_tracer()
    span_attrs: dict[str, str] = {
        "pipeline.run_id": run_id,
    }
    if article_url:
        span_attrs["pipeline.article_url"] = article_url
    if attributes:
        span_attrs.update(attributes)

    with tracer.start_as_current_span(name, attributes=span_attrs) as span:
        try:
            yield span
        except Exception as exc:
            span.set_status(trace.StatusCode.ERROR, str(exc))
            span.record_exception(exc)
            raise


def record_metric(span: Any, key: str, value: int | float) -> None:
    """span에 커스텀 메트릭 기록"""
    if span is not None:
        span.set_attribute(f"pipeline.metric.{key}", value)
```

### 3.17 Lineage Emitter (`lineage.py`) — v0.3.0

```python
# app/tools/news_pipeline/lineage.py
"""OpenLineage 기반 데이터 계보 추적 (v0.3.0)

Consultant 피드백: 기사→요약→청크→벡터 인덱스의 lineage 명시 필요.
OpenLineage 이벤트 발행으로 데이터 흐름 추적 및 영향도 분석 지원.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import uuid4

from app.core.config import settings

logger = logging.getLogger(__name__)


class LineageEmitter:
    """OpenLineage 이벤트 발행기

    파이프라인 각 단계(수집→정제→요약→저장→청킹)를
    OpenLineage RunEvent로 기록하여 데이터 계보(lineage) 추적.

    lineage_url이 빈 문자열이면 로그 출력만 수행 (비활성화).
    """

    def __init__(self) -> None:
        self._lineage_url = getattr(settings, "NEWS_PIPELINE_LINEAGE_URL", "")
        self._client = None

    @property
    def is_enabled(self) -> bool:
        return bool(self._lineage_url)

    def _get_client(self) -> Any:
        """OpenLineage 클라이언트 (lazy init)"""
        if self._client is None and self.is_enabled:
            from openlineage.client import OpenLineageClient
            from openlineage.client.transport.http import HttpConfig, HttpTransport
            transport = HttpTransport(HttpConfig(url=self._lineage_url))
            self._client = OpenLineageClient(transport=transport)
        return self._client

    def emit_run_start(
        self,
        run_id: str,
        job_name: str,
        inputs: list[dict[str, str]] | None = None,
    ) -> None:
        """파이프라인 단계 시작 이벤트 발행

        Args:
            run_id: 파이프라인 실행 ID
            job_name: 단계명 (예: "news_pipeline.collect", "news_pipeline.summarize")
            inputs: 입력 데이터셋 [{namespace, name}]
        """
        if not self.is_enabled:
            logger.debug("Lineage(disabled): START %s", job_name)
            return

        from openlineage.client.run import (
            InputDataset,
            Job,
            Run,
            RunEvent,
            RunState,
        )

        input_datasets = [
            InputDataset(namespace=ds["namespace"], name=ds["name"])
            for ds in (inputs or [])
        ]

        event = RunEvent(
            eventType=RunState.START,
            eventTime=datetime.now(timezone.utc).isoformat(),
            run=Run(runId=run_id),
            job=Job(namespace="news_pipeline", name=job_name),
            inputs=input_datasets,
            outputs=[],
        )

        try:
            self._get_client().emit(event)
        except Exception as exc:
            logger.warning("Lineage START 이벤트 발행 실패: %s", exc)

    def emit_run_complete(
        self,
        run_id: str,
        job_name: str,
        outputs: list[dict[str, str]] | None = None,
        record_count: int = 0,
    ) -> None:
        """파이프라인 단계 완료 이벤트 발행

        Args:
            run_id: 파이프라인 실행 ID
            job_name: 단계명
            outputs: 출력 데이터셋 [{namespace, name}]
            record_count: 처리된 레코드 수
        """
        if not self.is_enabled:
            logger.debug("Lineage(disabled): COMPLETE %s (records=%d)", job_name, record_count)
            return

        from openlineage.client.run import (
            Job,
            OutputDataset,
            Run,
            RunEvent,
            RunState,
        )

        output_datasets = [
            OutputDataset(namespace=ds["namespace"], name=ds["name"])
            for ds in (outputs or [])
        ]

        event = RunEvent(
            eventType=RunState.COMPLETE,
            eventTime=datetime.now(timezone.utc).isoformat(),
            run=Run(runId=run_id),
            job=Job(namespace="news_pipeline", name=job_name),
            inputs=[],
            outputs=output_datasets,
        )

        try:
            self._get_client().emit(event)
        except Exception as exc:
            logger.warning("Lineage COMPLETE 이벤트 발행 실패: %s", exc)

    def emit_run_fail(self, run_id: str, job_name: str, error: str) -> None:
        """파이프라인 단계 실패 이벤트 발행"""
        if not self.is_enabled:
            logger.debug("Lineage(disabled): FAIL %s — %s", job_name, error)
            return

        from openlineage.client.run import (
            Job,
            Run,
            RunEvent,
            RunState,
        )

        event = RunEvent(
            eventType=RunState.FAIL,
            eventTime=datetime.now(timezone.utc).isoformat(),
            run=Run(runId=run_id),
            job=Job(namespace="news_pipeline", name=job_name),
            inputs=[],
            outputs=[],
        )

        try:
            self._get_client().emit(event)
        except Exception as exc:
            logger.warning("Lineage FAIL 이벤트 발행 실패: %s", exc)
```

> **의존성**: `openlineage-python` (`pyproject.toml`에 추가)
> **비활성화**: `NEWS_PIPELINE_LINEAGE_URL`이 빈 문자열이면 로그 출력만 수행

### 3.18 TaskIQ Broker (`task_broker.py`) — v0.3.0

```python
# app/tools/news_pipeline/task_broker.py
"""TaskIQ 비동기 작업 큐 (v0.3.0)

Red Team 피드백: 소스 10개+ 확장 시 순차 수집은 병목.
TaskIQ 기반으로 소스별 수집을 병렬 태스크로 분리.
Redis URL이 비어 있으면 InMemoryBroker 사용 (개발/테스트).
"""

from __future__ import annotations

import logging
from datetime import date

from app.core.config import settings

logger = logging.getLogger(__name__)

# --- Broker 싱글턴 (lazy init) ---

_broker = None


def get_broker():
    """TaskIQ 브로커 인스턴스 (lazy init)

    Redis URL이 설정되어 있으면 Redis 브로커, 아니면 InMemoryBroker.
    """
    global _broker
    if _broker is not None:
        return _broker

    redis_url = getattr(settings, "NEWS_PIPELINE_REDIS_URL", "")

    if redis_url:
        from taskiq_redis import RedisAsyncResultBackend, ListQueueBroker
        _broker = ListQueueBroker(url=redis_url).with_result_backend(
            RedisAsyncResultBackend(redis_url=redis_url)
        )
        logger.info("TaskIQ Redis 브로커 초기화: %s", redis_url)
    else:
        from taskiq import InMemoryBroker
        _broker = InMemoryBroker()
        logger.info("TaskIQ InMemory 브로커 초기화 (개발 모드)")

    return _broker


# --- 태스크 정의 ---

async def fetch_source_task(
    source_type: str,
    target_date_iso: str,
) -> dict:
    """개별 소스 수집 태스크

    TaskIQ 워커에서 비동기 실행됨.
    결과: {"source": str, "articles": list[dict], "error": str|None}
    """
    from app.tools.news_pipeline.config import NewsPipelineConfig
    from app.tools.news_pipeline.sources.lawtimes_source import LawtimesSource
    from app.tools.news_pipeline.sources.naver_news_source import NaverNewsSource

    config = NewsPipelineConfig.from_settings()
    target_date = date.fromisoformat(target_date_iso)

    source_map = {
        "lawtimes": lambda: LawtimesSource(config),
        "naver": lambda: NaverNewsSource(config),
    }

    source_factory = source_map.get(source_type)
    if not source_factory:
        return {"source": source_type, "articles": [], "error": f"Unknown source: {source_type}"}

    source = source_factory()
    if not source.is_available:
        return {"source": source_type, "articles": [], "error": "Source disabled"}

    try:
        articles = await source.fetch(target_date)
        # RawArticle → dict 직렬화 (TaskIQ 결과 전달용)
        return {
            "source": source_type,
            "articles": [
                {
                    "url": a.url,
                    "title": a.title,
                    "raw_html": a.raw_html,
                    "source": a.source.value,
                    "publisher": a.publisher,
                    "published_at": a.published_at.isoformat() if a.published_at else None,
                    "author": a.author,
                    "section": a.section,
                    "tags": a.tags,
                }
                for a in articles
            ],
            "error": None,
        }
    except Exception as exc:
        logger.error("소스 [%s] 수집 태스크 실패: %s", source_type, exc)
        return {"source": source_type, "articles": [], "error": str(exc)}


# 브로커에 태스크 등록 (모듈 로드 시)
def register_tasks() -> None:
    """브로커에 태스크 데코레이터 적용 (앱 시작 시 호출)"""
    broker = get_broker()
    global fetch_source_task
    fetch_source_task = broker.task(fetch_source_task)
    logger.info("TaskIQ 태스크 등록 완료: fetch_source_task")
```

> **의존성**: `taskiq`, `taskiq-redis` (Redis 사용 시), `taskiq-fastapi` (라이프사이클 통합)
> **비활성화**: `NEWS_PIPELINE_REDIS_URL`이 빈 문자열이면 `InMemoryBroker` → 실질적 순차 실행

### 3.19 Dead-Letter Queue (`dead_letter.py`) — v0.3.0

```python
# app/tools/news_pipeline/dead_letter.py
"""Dead-Letter Queue: 실패 기사 재처리 서비스 (v0.3.0)

Consultant 피드백: 에러 로그+알림만으로는 실패 기사 추적/재처리 어려움.
PostgreSQL 테이블 기반 DLQ로 실패 기사를 저장하고 자동 재시도.
"""

from __future__ import annotations

import logging
from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# 최대 재시도 횟수
MAX_RETRY_COUNT = 3


class DeadLetterService:
    """Dead-Letter Queue 관리 서비스

    실패한 기사를 news_article_dlq 테이블에 저장하고,
    재시도 가능한 기사를 조회하여 파이프라인에 재투입.
    """

    async def enqueue(
        self,
        db: AsyncSession,
        *,
        article_url: str,
        stage: str,
        error_type: str,
        error_message: str,
        raw_payload: str | None = None,
    ) -> None:
        """실패 기사를 DLQ에 등록

        이미 등록된 URL이면 retry_count 증가.
        """
        from app.models.news_article_dlq import NewsArticleDLQ
        from sqlalchemy.dialects.postgresql import insert

        stmt = insert(NewsArticleDLQ).values(
            article_url=article_url,
            stage=stage,
            error_type=error_type,
            error_message=error_message[:2000],
            raw_payload=raw_payload,
            retry_count=0,
        ).on_conflict_do_update(
            index_elements=["article_url"],
            set_={
                "retry_count": NewsArticleDLQ.retry_count + 1,
                "error_type": error_type,
                "error_message": error_message[:2000],
                "stage": stage,
                "updated_at": datetime.utcnow(),
            },
        )
        await db.execute(stmt)
        await db.flush()
        logger.info("DLQ 등록: [%s] %s — %s", stage, article_url, error_type)

    async def get_retryable(
        self,
        db: AsyncSession,
        *,
        limit: int = 50,
    ) -> list[dict]:
        """재시도 가능한 기사 목록 조회

        조건: retry_count < MAX_RETRY_COUNT, is_resolved=False
        """
        from app.models.news_article_dlq import NewsArticleDLQ

        result = await db.execute(
            select(NewsArticleDLQ)
            .where(
                NewsArticleDLQ.retry_count < MAX_RETRY_COUNT,
                NewsArticleDLQ.is_resolved == False,  # noqa: E712
            )
            .order_by(NewsArticleDLQ.created_at)
            .limit(limit)
        )
        rows = result.scalars().all()

        return [
            {
                "id": row.id,
                "article_url": row.article_url,
                "stage": row.stage,
                "retry_count": row.retry_count,
                "raw_payload": row.raw_payload,
            }
            for row in rows
        ]

    async def mark_resolved(
        self,
        db: AsyncSession,
        article_url: str,
    ) -> None:
        """재처리 성공 시 해결 완료 표시"""
        from app.models.news_article_dlq import NewsArticleDLQ

        await db.execute(
            update(NewsArticleDLQ)
            .where(NewsArticleDLQ.article_url == article_url)
            .values(is_resolved=True, updated_at=datetime.utcnow())
        )
        await db.flush()
        logger.info("DLQ 해결 완료: %s", article_url)
```

---

## 4. Database Schema (DDL)

### 4.1 ORM Model (`app/models/news_article.py`)

```python
# app/models/news_article.py
"""뉴스 기사 ORM 모델"""

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Index,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import ARRAY

from app.core.database import Base


class NewsArticle(Base):
    """뉴스 기사 테이블

    법률 뉴스 파이프라인에서 수집/정제/요약된 기사를 저장.
    LanceDB news_chunks 테이블과 doc_id로 연결.
    """

    __tablename__ = "news_articles"

    # PK: SHA256(url)
    id = Column(
        String(64), primary_key=True,
        comment="문서 ID (SHA256(url))",
    )

    # 소스 정보
    source = Column(
        String(20), nullable=False, index=True,
        comment="소스 유형 (lawtimes | naver)",
    )
    publisher = Column(
        String(200), nullable=False,
        comment="매체명",
    )

    # 기사 기본 정보
    title = Column(Text, nullable=False, comment="기사 제목")
    author = Column(String(100), nullable=True, comment="기자명")
    published_at = Column(
        DateTime(timezone=True), nullable=True, index=True,
        comment="기사 발행일시 (KST)",
    )
    collected_at = Column(
        DateTime(timezone=True), nullable=False,
        default=datetime.utcnow,
        comment="수집 일시",
    )
    url = Column(
        Text, nullable=False, unique=True,
        comment="기사 원문 URL",
    )
    section = Column(String(50), nullable=True, comment="기사 섹션")
    tags = Column(
        ARRAY(String), nullable=True,
        comment="기사 태그/키워드",
    )

    # 정제된 본문
    cleaned_text = Column(Text, nullable=False, comment="정제된 본문")

    # 구조화 요약
    summary_one_liner = Column(Text, nullable=False, comment="한줄 요지")
    summary_issues = Column(ARRAY(String), nullable=True, comment="주요 쟁점")
    summary_laws = Column(ARRAY(String), nullable=True, comment="언급 법령")
    summary_cases = Column(ARRAY(String), nullable=True, comment="언급 판례")
    summary_institutions = Column(ARRAY(String), nullable=True, comment="언급 기관")
    summary_implications = Column(ARRAY(String), nullable=True, comment="시사점")

    # 메타데이터
    content_hash = Column(
        String(64), nullable=False, index=True,
        comment="정제 본문 SHA256 해시",
    )
    disclaimer = Column(
        Text, nullable=False,
        default="본 문서는 기사 요약이며 법령/판례 원문이 아닙니다",
        comment="면책 고지",
    )
    schema_version = Column(
        String(10), nullable=False, default="1.0",
        comment="스키마 버전",
    )
    is_indexed = Column(
        Boolean, nullable=False, default=False,
        comment="LanceDB 임베딩 완료 여부",
    )

    # 타임스탬프
    created_at = Column(DateTime, default=datetime.utcnow, comment="레코드 생성일시")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="수정일시")

    # 인덱스
    __table_args__ = (
        Index("idx_news_source_published", "source", "published_at"),
        Index("idx_news_tags", "tags", postgresql_using="gin"),
    )

    def __repr__(self) -> str:
        return f"<NewsArticle(id={self.id}, source={self.source}, title={self.title[:30]})>"
```

### 4.2 Dead-Letter Queue ORM Model (`app/models/news_article_dlq.py`) — v0.3.0

```python
# app/models/news_article_dlq.py
"""뉴스 기사 Dead-Letter Queue ORM 모델 (v0.3.0)"""

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Integer,
    String,
    Text,
)

from app.core.database import Base


class NewsArticleDLQ(Base):
    """뉴스 기사 Dead-Letter Queue 테이블

    파이프라인에서 처리 실패한 기사를 저장하고 자동 재시도 관리.
    retry_count < MAX_RETRY_COUNT 인 기사만 재시도 대상.
    """

    __tablename__ = "news_article_dlq"

    id = Column(Integer, primary_key=True, autoincrement=True)
    article_url = Column(
        Text, nullable=False, unique=True,
        comment="실패한 기사 URL",
    )
    stage = Column(
        String(30), nullable=False,
        comment="실패 단계 (collect|clean|summarize|store|chunk)",
    )
    error_type = Column(String(200), nullable=False, comment="예외 클래스명")
    error_message = Column(Text, nullable=False, comment="에러 메시지 (최대 2000자)")
    raw_payload = Column(Text, nullable=True, comment="실패 시점 원시 데이터 (JSON)")

    retry_count = Column(
        Integer, nullable=False, default=0,
        comment="재시도 횟수 (3회 초과 시 재시도 중단)",
    )
    is_resolved = Column(
        Boolean, nullable=False, default=False,
        comment="재처리 성공 여부",
    )

    created_at = Column(DateTime, default=datetime.utcnow, comment="최초 실패 일시")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="최종 업데이트 일시")

    def __repr__(self) -> str:
        return f"<NewsArticleDLQ(id={self.id}, url={self.article_url[:50]}, retries={self.retry_count})>"
```

### 4.3 Alembic Migration

```python
# alembic/versions/NNN_add_news_articles_table.py
"""뉴스 기사 테이블 생성

Revision ID: {auto-generated}
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

def upgrade() -> None:
    op.create_table(
        "news_articles",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("source", sa.String(20), nullable=False, index=True),
        sa.Column("publisher", sa.String(200), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("author", sa.String(100), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True, index=True),
        sa.Column("collected_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("url", sa.Text(), nullable=False, unique=True),
        sa.Column("section", sa.String(50), nullable=True),
        sa.Column("tags", postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column("cleaned_text", sa.Text(), nullable=False),
        sa.Column("summary_one_liner", sa.Text(), nullable=False),
        sa.Column("summary_issues", postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column("summary_laws", postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column("summary_cases", postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column("summary_institutions", postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column("summary_implications", postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column("content_hash", sa.String(64), nullable=False, index=True),
        sa.Column("disclaimer", sa.Text(), nullable=False),
        sa.Column("schema_version", sa.String(10), nullable=False, server_default="1.0"),
        sa.Column("is_indexed", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
    )

    # 복합 인덱스
    op.create_index("idx_news_source_published", "news_articles", ["source", "published_at"])
    op.create_index("idx_news_tags", "news_articles", ["tags"], postgresql_using="gin")


    # v0.3.0: Dead-Letter Queue 테이블
    op.create_table(
        "news_article_dlq",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("article_url", sa.Text(), nullable=False, unique=True),
        sa.Column("stage", sa.String(30), nullable=False),
        sa.Column("error_type", sa.String(200), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=False),
        sa.Column("raw_payload", sa.Text(), nullable=True),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("is_resolved", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("news_article_dlq")
    op.drop_index("idx_news_tags", table_name="news_articles")
    op.drop_index("idx_news_source_published", table_name="news_articles")
    op.drop_table("news_articles")
```

---

## 5. Pipeline Orchestrator

### 5.1 Service (`news_pipeline_service.py`)

```python
# app/services/service_function/news_pipeline_service.py
"""뉴스 파이프라인 오케스트레이터"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import date, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import async_session_factory
from app.models.news_article import NewsArticle
from app.tools.news_pipeline.chunker import Chunker
from app.tools.news_pipeline.cleaner import Cleaner
from app.tools.news_pipeline.config import NewsPipelineConfig
from app.tools.news_pipeline.deduplicator import Deduplicator
from app.tools.news_pipeline.models import (
    PipelineError,
    PipelineResult,
    ProcessedArticle,
    SourceStat,
)
from app.tools.news_pipeline.pii_filter import PIIFilter
from app.tools.news_pipeline.reference_validator import ReferenceValidator
from app.tools.news_pipeline.reporter import PipelineReporter
from app.tools.news_pipeline.sources import BaseNewsSource
from app.tools.news_pipeline.sources.lawtimes_source import LawtimesSource
from app.tools.news_pipeline.sources.naver_news_source import NaverNewsSource
from app.tools.news_pipeline.summarizer import Summarizer

logger = logging.getLogger(__name__)

# 배치 저장 크기
DB_BATCH_SIZE = 50


async def run_pipeline(
    target_date: date,
    config: NewsPipelineConfig | None = None,
    sources: list[str] | None = None,
) -> PipelineResult:
    """메인 파이프라인 실행

    Args:
        target_date: 수집 대상 날짜
        config: 파이프라인 설정 (None이면 환경 변수에서 로드)
        sources: 실행할 소스 목록 (None이면 활성화된 전체)

    Returns:
        PipelineResult: 실행 결과 (리포트용)
    """
    if config is None:
        config = NewsPipelineConfig.from_settings()

    result = PipelineResult(
        run_id=str(uuid.uuid4()),
        started_at=datetime.now(),
    )

    # 컴포넌트 초기화
    available_sources = _build_sources(config, sources)
    dedup = Deduplicator()
    cleaner = Cleaner(config)
    pii_filter = PIIFilter()
    summarizer = Summarizer(config)
    ref_validator = ReferenceValidator()
    chunker = Chunker(config)
    reporter = PipelineReporter()

    async with async_session_factory() as db:
        # === Phase 1: 수집 + 중복제거 ===
        all_raw = []
        for source in available_sources:
            try:
                raw_articles = await source.fetch(target_date)
                src_name = source.source_type.value
                result.source_stats[src_name] = SourceStat(collected=len(raw_articles))
                all_raw.extend(raw_articles)
            except Exception as exc:
                src_name = source.source_type.value
                result.source_stats[src_name] = SourceStat(failed=1, errors=[str(exc)])
                result.errors.append(PipelineError(
                    stage="collect", article_url=None,
                    error_type=type(exc).__name__, error_message=str(exc),
                ))
                logger.error("소스 [%s] 수집 실패: %s", src_name, exc)

        result.total_collected = len(all_raw)

        unique_articles = await dedup.deduplicate(all_raw, db)
        result.total_deduplicated = result.total_collected - len(unique_articles)

        # === Phase 2: 정제 + PII + 요약 ===
        processed: list[ProcessedArticle] = []
        for raw in unique_articles:
            try:
                # 정제
                cleaned = cleaner.clean(raw)
                if cleaned is None:
                    continue
                result.total_cleaned += 1

                # Stage 3 중복 체크 (본문 해시)
                if await dedup.check_content_hash(cleaned.content_hash, db):
                    continue

                # PII 마스킹
                cleaned.cleaned_text, _ = pii_filter.mask(cleaned.cleaned_text)

                # 요약 생성
                summary = await summarizer.summarize(cleaned)
                result.total_summarized += 1

                # Cross-Reference 검증
                summary = await ref_validator.validate_and_filter(summary, db)

                # ProcessedArticle 조합
                doc_id = hashlib.sha256(cleaned.url.encode()).hexdigest()
                processed.append(ProcessedArticle(
                    doc_id=doc_id,
                    url=cleaned.url,
                    title=cleaned.title,
                    cleaned_text=cleaned.cleaned_text,
                    content_hash=cleaned.content_hash,
                    source=cleaned.source,
                    publisher=cleaned.publisher,
                    published_at=cleaned.published_at,
                    collected_at=datetime.now(),
                    author=cleaned.author,
                    section=cleaned.section,
                    tags=cleaned.tags,
                    summary=summary,
                ))

            except Exception as exc:
                result.errors.append(PipelineError(
                    stage="process", article_url=raw.url,
                    error_type=type(exc).__name__, error_message=str(exc),
                ))
                logger.warning("기사 처리 실패 [%s]: %s", raw.url, exc)

        # === Phase 3: DB 저장 + 청킹 ===
        for i in range(0, len(processed), DB_BATCH_SIZE):
            batch = processed[i:i + DB_BATCH_SIZE]
            try:
                await _store_batch(batch, db)
                result.total_stored += len(batch)
            except Exception as exc:
                result.errors.append(PipelineError(
                    stage="store", article_url=None,
                    error_type=type(exc).__name__, error_message=str(exc),
                ))

        # 청킹 + LanceDB
        for article in processed:
            try:
                chunks = chunker.create_chunks(article)
                stored = await chunker.embed_and_store(chunks, config.lancedb_table)
                result.total_chunked += stored

                # is_indexed 플래그 업데이트
                await _mark_indexed(article.doc_id, db)
            except Exception as exc:
                result.errors.append(PipelineError(
                    stage="chunk", article_url=article.url,
                    error_type=type(exc).__name__, error_message=str(exc),
                ))

        await db.commit()

    # === Phase 4: 리포트 ===
    result.finished_at = datetime.now()
    reporter.generate(result)

    return result


def _build_sources(
    config: NewsPipelineConfig, filter_sources: list[str] | None
) -> list[BaseNewsSource]:
    """활성화된 소스 인스턴스 목록 생성"""
    all_sources: list[BaseNewsSource] = [
        LawtimesSource(config),
        NaverNewsSource(config),
    ]

    available = [s for s in all_sources if s.is_available]

    if filter_sources:
        available = [
            s for s in available
            if s.source_type.value in filter_sources
        ]

    return available


async def _store_batch(
    articles: list[ProcessedArticle], db: AsyncSession
) -> None:
    """배치 단위 DB 저장 (ON CONFLICT 멱등)"""
    from sqlalchemy.dialects.postgresql import insert

    for article in articles:
        stmt = insert(NewsArticle).values(
            id=article.doc_id,
            source=article.source.value,
            publisher=article.publisher,
            title=article.title,
            author=article.author,
            published_at=article.published_at,
            collected_at=article.collected_at,
            url=article.url,
            section=article.section,
            tags=article.tags,
            cleaned_text=article.cleaned_text,
            summary_one_liner=article.summary.one_liner,
            summary_issues=article.summary.issues,
            summary_laws=article.summary.laws,
            summary_cases=article.summary.cases,
            summary_institutions=article.summary.institutions,
            summary_implications=article.summary.implications,
            content_hash=article.content_hash,
            disclaimer=article.disclaimer,
            schema_version=article.schema_version,
        ).on_conflict_do_update(
            index_elements=["id"],
            set_={
                "content_hash": article.content_hash,
                "cleaned_text": article.cleaned_text,
                "summary_one_liner": article.summary.one_liner,
                "updated_at": datetime.utcnow(),
            },
        )
        await db.execute(stmt)

    await db.flush()


async def _mark_indexed(doc_id: str, db: AsyncSession) -> None:
    """LanceDB 임베딩 완료 플래그 설정"""
    from sqlalchemy import update
    await db.execute(
        update(NewsArticle).where(NewsArticle.id == doc_id).values(is_indexed=True)
    )
```

---

## 6. CLI Interface

### 6.1 CLI (`scripts/news_pipeline/cli.py`)

```python
# scripts/news_pipeline/cli.py
"""뉴스 파이프라인 CLI"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import date, timedelta

from app.tools.news_pipeline.config import NewsPipelineConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="법률 뉴스 수집/요약 파이프라인",
    )
    parser.add_argument(
        "--date", type=str, default="yesterday",
        help="수집 대상 날짜 (YYYY-MM-DD 또는 'today'/'yesterday')",
    )
    parser.add_argument(
        "--date-range", nargs=2, type=str, metavar=("START", "END"),
        help="날짜 범위 (YYYY-MM-DD YYYY-MM-DD)",
    )
    parser.add_argument(
        "--source", type=str, choices=["lawtimes", "naver"],
        help="특정 소스만 실행",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="수집만 실행 (요약/저장 건너뜀)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="상세 로그 출력",
    )
    # v0.2.0: Red Team 피드백 — 인덱싱 복구 CLI
    parser.add_argument(
        "--reindex-pending", action="store_true",
        help="is_indexed=false인 문서를 일괄 재처리 (LanceDB 청킹/임베딩)",
    )
    return parser.parse_args()


def resolve_date(date_str: str) -> date:
    """날짜 문자열 → date 변환"""
    if date_str == "today":
        return date.today()
    if date_str == "yesterday":
        return date.today() - timedelta(days=1)
    return date.fromisoformat(date_str)


async def main() -> None:
    args = parse_args()

    # 로깅 설정
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from app.services.service_function.news_pipeline_service import run_pipeline

    config = NewsPipelineConfig.from_settings()
    sources = [args.source] if args.source else None

    # v0.2.0: 인덱싱 복구 모드
    if args.reindex_pending:
        from app.services.service_function.news_pipeline_service import reindex_pending
        count = await reindex_pending(config)
        logging.info("재인덱싱 완료: %d건", count)
        return

    if args.date_range:
        start = date.fromisoformat(args.date_range[0])
        end = date.fromisoformat(args.date_range[1])
        current = start
        while current <= end:
            logging.info("━━━ %s 수집 시작 ━━━", current)
            await run_pipeline(current, config=config, sources=sources)
            current += timedelta(days=1)
    else:
        target = resolve_date(args.date)
        await run_pipeline(target, config=config, sources=sources)


if __name__ == "__main__":
    asyncio.run(main())
```

### 6.2 `__main__.py`

```python
# scripts/news_pipeline/__main__.py
"""python -m scripts.news_pipeline 실행 지원"""

import asyncio
from scripts.news_pipeline.cli import main

asyncio.run(main())
```

### 6.3 Cron Setup (`cron_setup.sh`)

```bash
#!/bin/bash
# scripts/news_pipeline/cron_setup.sh
# KST 06:00 자동 실행 cron 설정

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
UV_PATH="$(which uv)"
LOG_DIR="${PROJECT_DIR}/data/news_pipeline/reports"

mkdir -p "$LOG_DIR"

CRON_LINE="0 6 * * * cd ${PROJECT_DIR}/backend && ${UV_PATH} run python -m scripts.news_pipeline --date yesterday >> ${LOG_DIR}/cron.log 2>&1"

echo "등록할 cron 라인:"
echo "$CRON_LINE"
echo ""
echo "crontab에 추가하려면:"
echo "(crontab -l 2>/dev/null; echo '$CRON_LINE') | crontab -"
```

---

## 7. API Module (`app/modules/legal_news/`) — v0.3.0

> Red Team 피드백: 프론트엔드 뉴스 조회용 API 모듈 필요.
> 기존 모듈 패턴 (`registry.py` 자동 등록, Pydantic 스키마) 100% 준수.

### 7.1 Router (`router/__init__.py`)

```python
# app/modules/legal_news/router/__init__.py
"""법률 뉴스 API 라우터"""

from __future__ import annotations

from datetime import date
from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.modules.legal_news.schema import (
    NewsArticleResponse,
    NewsListResponse,
    NewsSearchRequest,
    NewsSearchResponse,
)

router = APIRouter()


@router.get(
    "/list",
    response_model=NewsListResponse,
    summary="뉴스 목록 조회",
)
async def get_news_list(
    source: str | None = Query(None, description="소스 필터 (lawtimes|naver)"),
    published_date: date | None = Query(None, description="발행일 필터"),
    page: int = Query(1, ge=1, description="페이지 번호"),
    page_size: int = Query(20, ge=1, le=100, description="페이지 크기"),
    db: AsyncSession = Depends(get_db),
) -> NewsListResponse:
    """법률 뉴스 목록 조회 (페이지네이션, 소스/날짜 필터)"""
    from app.modules.legal_news.service import get_news_list_service
    return await get_news_list_service(
        db, source=source, published_date=published_date,
        page=page, page_size=page_size,
    )


@router.get(
    "/{article_id}",
    response_model=NewsArticleResponse,
    summary="뉴스 상세 조회",
)
async def get_news_detail(
    article_id: str,
    db: AsyncSession = Depends(get_db),
) -> NewsArticleResponse:
    """개별 뉴스 기사 상세 조회 (doc_id로 조회)"""
    from app.modules.legal_news.service import get_news_detail_service
    return await get_news_detail_service(db, article_id=article_id)


@router.post(
    "/search",
    response_model=NewsSearchResponse,
    summary="뉴스 하이브리드 검색",
)
async def search_news(
    request: NewsSearchRequest,
    db: AsyncSession = Depends(get_db),
) -> NewsSearchResponse:
    """v0.3.0: 하이브리드 검색 (Vector + FTS + 리랭커)"""
    from app.modules.legal_news.service import search_news_service
    return await search_news_service(db, request=request)
```

### 7.2 Schema (`schema/__init__.py`)

```python
# app/modules/legal_news/schema/__init__.py
"""법률 뉴스 API 스키마"""

from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field


class NewsArticleSummary(BaseModel):
    """기사 목록용 요약 스키마"""
    model_config = ConfigDict(from_attributes=True, extra="ignore")

    id: str = Field(description="문서 ID (SHA256)")
    title: str
    source: str
    publisher: str
    published_at: datetime | None
    summary_one_liner: str = Field(description="한줄 요약")
    section: str | None = None
    tags: list[str] | None = None


class NewsArticleResponse(BaseModel):
    """기사 상세 스키마"""
    model_config = ConfigDict(from_attributes=True, extra="ignore")

    id: str
    title: str
    source: str
    publisher: str
    published_at: datetime | None
    collected_at: datetime
    url: str
    author: str | None = None
    section: str | None = None
    tags: list[str] | None = None
    cleaned_text: str
    summary_one_liner: str
    summary_issues: list[str] | None = None
    summary_laws: list[str] | None = None
    summary_cases: list[str] | None = None
    summary_institutions: list[str] | None = None
    summary_implications: list[str] | None = None
    disclaimer: str
    schema_version: str


class NewsListResponse(BaseModel):
    """기사 목록 응답"""
    items: list[NewsArticleSummary]
    total: int
    page: int
    page_size: int
    has_next: bool


class NewsSearchRequest(BaseModel):
    """검색 요청"""
    query: str = Field(min_length=2, max_length=500, description="검색 쿼리")
    limit: int = Field(default=10, ge=1, le=50, description="결과 수")
    source: str | None = Field(default=None, description="소스 필터")


class NewsSearchResult(BaseModel):
    """검색 결과 항목"""
    chunk_id: str
    doc_id: str
    title: str
    chunk_text: str
    chunk_type: str
    source: str
    publisher: str
    url: str
    published_at: str | None = None
    rerank_score: float | None = None


class NewsSearchResponse(BaseModel):
    """검색 응답"""
    results: list[NewsSearchResult]
    query: str
    total: int
```

### 7.3 Service (`service.py`)

```python
# app/modules/legal_news/service.py
"""법률 뉴스 API 비즈니스 로직"""

from __future__ import annotations

from datetime import date

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.news_article import NewsArticle
from app.modules.legal_news.schema import (
    NewsArticleResponse,
    NewsArticleSummary,
    NewsListResponse,
    NewsSearchRequest,
    NewsSearchResponse,
    NewsSearchResult,
)


async def get_news_list_service(
    db: AsyncSession,
    *,
    source: str | None,
    published_date: date | None,
    page: int,
    page_size: int,
) -> NewsListResponse:
    """뉴스 목록 조회 서비스"""
    query = select(NewsArticle).order_by(NewsArticle.published_at.desc())

    if source:
        query = query.where(NewsArticle.source == source)
    if published_date:
        query = query.where(
            func.date(NewsArticle.published_at) == published_date
        )

    # 전체 건수
    count_query = select(func.count()).select_from(query.subquery())
    total = (await db.execute(count_query)).scalar_one()

    # 페이지네이션
    offset = (page - 1) * page_size
    rows = (await db.execute(query.offset(offset).limit(page_size))).scalars().all()

    items = [NewsArticleSummary.model_validate(row) for row in rows]

    return NewsListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        has_next=(offset + page_size) < total,
    )


async def get_news_detail_service(
    db: AsyncSession,
    *,
    article_id: str,
) -> NewsArticleResponse:
    """뉴스 상세 조회 서비스"""
    result = await db.execute(
        select(NewsArticle).where(NewsArticle.id == article_id)
    )
    article = result.scalar_one_or_none()
    if not article:
        raise HTTPException(status_code=404, detail="기사를 찾을 수 없습니다")
    return NewsArticleResponse.model_validate(article)


async def search_news_service(
    db: AsyncSession,
    *,
    request: NewsSearchRequest,
) -> NewsSearchResponse:
    """뉴스 하이브리드 검색 서비스 (v0.3.0)"""
    from app.tools.news_pipeline.chunker import Chunker
    from app.tools.news_pipeline.config import NewsPipelineConfig

    config = NewsPipelineConfig.from_settings()
    results = await Chunker.hybrid_search(
        query=request.query,
        lancedb_table_name=config.lancedb_table,
        limit=request.limit * 2,
        rerank_top_k=request.limit,
    )

    # 소스 필터 적용
    if request.source:
        results = [r for r in results if r.get("source") == request.source]

    search_results = [
        NewsSearchResult(
            chunk_id=r.get("chunk_id", ""),
            doc_id=r.get("doc_id", ""),
            title=r.get("title", ""),
            chunk_text=r.get("chunk_text", ""),
            chunk_type=r.get("chunk_type", ""),
            source=r.get("source", ""),
            publisher=r.get("publisher", ""),
            url=r.get("url", ""),
            published_at=r.get("published_at"),
            rerank_score=r.get("rerank_score"),
        )
        for r in results
    ]

    return NewsSearchResponse(
        results=search_results,
        query=request.query,
        total=len(search_results),
    )
```

> **모듈 등록**: `registry.py`가 자동 스캔하므로 `router/__init__.py`에 `router = APIRouter()`만 정의하면 `/api/legal-news/*` 경로로 자동 등록됨.
> **프론트엔드 연동 시**: `frontend/src/lib/modules.ts`에 `legal-news` 모듈 추가 + `api.ts` endpoints 추가 필요.

---

## 8. Error Handling Strategy

### 8.1 에러 전파 정책

| 단계 | 에러 발생 시 | 전파 |
|------|-------------|------|
| **소스 수집** | 해당 소스 건너뜀, 다른 소스 계속 | PipelineError 기록 |
| **개별 기사 정제** | 해당 기사 건너뜀 | PipelineError 기록 |
| **PII 마스킹** | 마스킹 실패 시 원문 그대로 사용 | 경고 로그 |
| **요약 생성** | Primary → Fallback → 해당 기사 건너뜀 | PipelineError 기록 |
| **Cross-Reference** | 검증 실패 시 요약 그대로 사용 | 경고 로그 |
| **DB 저장** | 배치 실패 시 해당 배치 건너뜀 | PipelineError 기록 |
| **청킹/임베딩** | 해당 기사 건너뜀 (is_indexed=false) | PipelineError 기록 |

### 8.2 재시도 정책

```
소스 수집: 최대 3회, exponential backoff (2, 4, 8초)
LLM 호출: Primary 1회 → Fallback 1회 → 건너뜀
DB 저장: 배치 실패 시 개별 저장 재시도
```

---

## 9. Environment Variables

`app/core/config.py` Settings에 추가할 필드:

```python
# News Pipeline
NEWS_PIPELINE_ENABLED: bool = True
NEWS_PIPELINE_SUMMARY_PROVIDER: str = "upstage"
NEWS_PIPELINE_SUMMARY_MODEL: str = "solar-pro2"
NEWS_PIPELINE_MIN_ARTICLE_LENGTH: int = 500
NEWS_PIPELINE_CHUNK_SIZE: int = 1500
NEWS_PIPELINE_CHUNK_OVERLAP: int = 200
NEWS_PIPELINE_MAX_ARTICLES_PER_RUN: int = 200
NEWS_PIPELINE_RATE_LIMIT_CRAWL: float = 2.0
NEWS_PIPELINE_LAWTIMES_ENABLED: bool = True
NEWS_PIPELINE_NAVER_ENABLED: bool = True
NEWS_PIPELINE_LANCEDB_TABLE: str = "news_chunks"
NEWS_PIPELINE_RETENTION_DAYS: int = 90
NEWS_PIPELINE_LLM_FALLBACK_PROVIDER: str = "openai"

# v0.2.0: 운영 알림
NEWS_PIPELINE_SLACK_WEBHOOK_URL: str = ""  # Slack Webhook URL (빈 문자열이면 알림 비활성화)

# v0.3.0: OpenTelemetry
NEWS_PIPELINE_OTEL_ENDPOINT: str = ""  # OTLP gRPC endpoint (빈 문자열이면 콘솔 출력)

# v0.3.0: OpenLineage
NEWS_PIPELINE_LINEAGE_URL: str = ""  # OpenLineage 서버 URL (빈 문자열이면 비활성화)

# v0.3.0: TaskIQ / Redis
NEWS_PIPELINE_REDIS_URL: str = ""  # Redis URL (빈 문자열이면 InMemoryBroker)

# v0.3.0: Fuzzy Dedup
NEWS_PIPELINE_SIMHASH_THRESHOLD: int = 3  # SimHash 해밍 거리 임계값
```

---

## 10. Test Strategy (v0.2.0)

> Consultant 피드백 반영: 4계층 테스트 피라미드

| 레이어 | 비율 | 대상 | 도구 |
|--------|------|------|------|
| **Unit** | 60% | Cleaner, PIIFilter, Deduplicator, Chunker, SummarySchema, SSRFGuard | pytest |
| **Contract** | 20% | 소스별 입력 형식 검증, DB upsert 계약, SummarySchema validation | pytest + pydantic |
| **Integration** | 15% | httpx mock, DB/LanceDB 테스트 컨테이너, upsert·재처리·중복 시나리오 | pytest + httpx mock |
| **E2E** | 5% | 4개 골든 시나리오: 정상, 소스 장애, LLM 타임아웃, 부분 실패 후 재실행 | pytest + mock |

### 운영 검증 (KPI)

| 지표 | 목표 | 측정 |
|------|------|------|
| 수집 성공률 | >= 95% | `collect_success_rate` |
| 요약 성공률 | >= 90% | `summary_success_rate` |
| 중복률 | < 30% | `duplicate_rate` |
| 평균 처리시간 | < 5초/건 | `avg_processing_seconds` |

---

## 11. Implementation Checklist

Plan v0.3의 구현 순서 + v0.3.0 추가 컴포넌트 매핑:

### Phase 1: 기반 구조
- [ ] `app/models/news_article.py` — ORM 모델
- [ ] `app/models/news_article_dlq.py` — Dead-Letter Queue ORM 모델 (v0.3.0)
- [ ] `alembic/versions/NNN_add_news_articles_table.py` — 마이그레이션 (news_articles + news_article_dlq)
- [ ] `app/core/config.py` — 환경 변수 20개 추가 (v0.3.0: +OTel, Lineage, Redis, SimHash)
- [ ] `app/tools/news_pipeline/__init__.py` — 패키지
- [ ] `app/tools/news_pipeline/models.py` — 내부 데이터 모델
- [ ] `app/tools/news_pipeline/config.py` — 파이프라인 설정
- [ ] `app/tools/news_pipeline/exceptions.py` — 예외 클래스
- [ ] `app/tools/news_pipeline/ssrf_guard.py` — SSRF 방어

### Phase 2: 수집 레이어
- [ ] `app/tools/news_pipeline/sources/__init__.py` — BaseNewsSource ABC
- [ ] `app/tools/news_pipeline/sources/lawtimes_source.py` — 법률신문 크롤러
- [ ] `app/tools/news_pipeline/sources/naver_news_source.py` — 네이버 뉴스
- [ ] `app/tools/news_pipeline/deduplicator.py` — 4단계 중복 제거 (v0.3.0: Stage 4 SimHash)
- [ ] `app/tools/news_pipeline/fuzzy_dedup.py` — SimHash 모듈 (v0.3.0)

### Phase 3: 정제 + 요약 레이어
- [ ] `app/tools/news_pipeline/cleaner.py` — 본문 정제
- [ ] `app/tools/news_pipeline/pii_filter.py` — PII 마스킹
- [ ] `app/tools/news_pipeline/summarizer.py` — Solar-Pro2 요약
- [ ] `app/tools/news_pipeline/reference_validator.py` — Cross-Reference 검증

### Phase 4: 저장 + 청킹
- [ ] DB 저장 로직 (news_pipeline_service.py 내 `_store_batch`)
- [ ] `app/tools/news_pipeline/chunker.py` — 청킹 + LanceDB 저장 + 하이브리드 검색 (v0.3.0)
- [ ] `app/tools/news_pipeline/reporter.py` — 운영 리포트

### Phase 5: 오케스트레이션 + CLI
- [ ] `app/services/service_function/news_pipeline_service.py` — 오케스트레이터
- [ ] `scripts/news_pipeline/__init__.py`
- [ ] `scripts/news_pipeline/__main__.py`
- [ ] `scripts/news_pipeline/cli.py` — CLI (+ `--retry-dlq` v0.3.0)
- [ ] `scripts/news_pipeline/cron_setup.sh` — cron 설정

### Phase 6: v0.3.0 관측성 + 비동기 레이어
- [ ] `app/tools/news_pipeline/telemetry.py` — OpenTelemetry 계측
- [ ] `app/tools/news_pipeline/lineage.py` — OpenLineage 계보 추적
- [ ] `app/tools/news_pipeline/task_broker.py` — TaskIQ 비동기 큐
- [ ] `app/tools/news_pipeline/dead_letter.py` — Dead-Letter Queue 서비스

### Phase 7: v0.3.0 API 모듈
- [ ] `app/modules/legal_news/__init__.py` — 모듈 패키지
- [ ] `app/modules/legal_news/router/__init__.py` — 뉴스 API 라우터
- [ ] `app/modules/legal_news/schema/__init__.py` — Pydantic 스키마
- [ ] `app/modules/legal_news/service.py` — 비즈니스 로직

### Phase 8: 검증
- [ ] `pyproject.toml` 의존성 추가: `feedparser`, `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp`, `openlineage-python`, `taskiq`, `taskiq-redis`
- [ ] 정적 검증 통과 (`ruff check` + `mypy`)
- [ ] Unit 테스트: Cleaner, PIIFilter, Deduplicator, SSRFGuard, SummarySchema, SimHash
- [ ] Contract 테스트: API 스키마 검증, DLQ upsert
- [ ] 로컬 1회 실행 테스트
- [ ] `app/models/__init__.py`에 NewsArticle, NewsArticleDLQ import 추가
- [ ] `frontend/src/lib/modules.ts`에 `legal-news` 모듈 등록

---

## 12. Design Decisions Log

| 결정 | 이유 |
|------|------|
| `app/tools/` 하위 배치 (모듈 아님) | API 엔드포인트 불필요, CLI 전용 배치 파이프라인 |
| SHA256(url)을 PK로 사용 | URL 기반 멱등성, autoincrement 대비 중복 제거 간편 |
| `ARRAY(String)` for 요약 필드 | 구조화 데이터를 단일 테이블에 저장, 별도 JOIN 불필요 |
| `dataclass` 사용 (Pydantic 아닌) | 내부 파이프라인 모델은 직렬화/검증 불필요, 가벼운 구조 |
| `news_chunks` 별도 LanceDB 테이블 | 기존 `legal_chunks` 오염 방지, 독립 관리 |
| `is_secondary=True` 메타데이터 | RAG 검색 시 1차 근거(법령/판례)와 명확한 구분 |
| **v0.2.0** SSRF Guard 추가 | Red Team: httpx에서 내부망 IP 요청 차단 필수 |
| **v0.2.0** Selector config 분리 | Red Team: HTML 크롤링 구조 변경 시 코드 수정 최소화 |
| **v0.2.0** ReferenceValidator ID 반환 | Red Team: 법령/판례 직접 링크를 위한 DB ID 연계 |
| **v0.2.0** SummarySchema (Pydantic) | Consultant: 요약 JSON 품질 게이트 (구조 검증) |
| **v0.2.0** Slack Webhook 알림 | Red Team: 운영 에러 즉시 알림 수단 필수 |
| **v0.2.0** `--reindex-pending` CLI | Red Team: is_indexed=false 문서 일괄 재처리 |
| **v0.2.0** KPI 메트릭 | Consultant: 수집/요약 성공률, 중복률, 처리시간 |
| **v0.3.0** SimHash Fuzzy Dedup | Red Team: 동일 사건 '재탕 기사' 유사도 95%+ 탐지 |
| **v0.3.0** OpenTelemetry 계측 | Consultant: run/article 단위 Trace 연결로 병목 파악 |
| **v0.3.0** OpenLineage 계보 | Consultant: 기사→요약→청크→벡터 데이터 흐름 추적 |
| **v0.3.0** 하이브리드 검색+리랭커 | Consultant: Vector+FTS RRF + 리랭커로 검색 품질 고도화 |
| **v0.3.0** Dead-Letter Queue | Consultant: 실패 기사 자동 재시도, 최대 3회 |
| **v0.3.0** TaskIQ 비동기 큐 | Red Team: 소스별 병렬 수집 → 확장성 확보 |
| **v0.3.0** 뉴스 소비 API 모듈 | Red Team: 프론트엔드 뉴스 조회/검색 엔드포인트 |

### v0.2.0 보류 항목 → v0.3.0 전량 반영 완료

> v0.2.0에서 보류했던 7건의 제안(뉴스 소비 API 모듈, Fuzzy Dedup, OpenTelemetry, OpenLineage, 하이브리드 검색+리랭킹, Dead-letter queue, TaskIQ 비동기 큐)이 **v0.3.0에서 전부 설계에 반영**되었습니다.
> 모든 컴포넌트는 Feature Flag / 빈 문자열 기반 비활성화를 지원하여 점진적 도입이 가능합니다.

---

## Revision History

| 버전 | 날짜 | 변경 내용 |
|------|------|----------|
| v0.1.0 | 2026-02-26 | 초안 작성 (13개 컴포넌트 코드 수준 설계) |
| v0.2.0 | 2026-02-26 | 3중 검증 피드백 반영 — 채택 7건: SSRF Guard, Selector config, ReferenceValidator ID 반환, SummarySchema Pydantic, Slack Webhook, --reindex-pending CLI, KPI 메트릭/테스트 전략. 보류 7건 로드맵 기록 |
| v0.3.0 | 2026-02-26 | 보류 7건 전량 반영: (1) 뉴스 소비 API 모듈 `app/modules/legal_news`, (2) SimHash Fuzzy Dedup (Stage 4), (3) OpenTelemetry 관측성, (4) OpenLineage 계보 추적, (5) 하이브리드 검색+리랭커 (RRF), (6) Dead-Letter Queue + 자동 재시도, (7) TaskIQ 비동기 큐. 섹션 구조 재정리 (12개 섹션) |
