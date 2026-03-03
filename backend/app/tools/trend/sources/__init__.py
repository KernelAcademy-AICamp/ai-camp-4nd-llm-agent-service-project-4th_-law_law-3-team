"""트렌드 데이터 소스 어댑터 (Strategy 패턴)

v2.2: safe_fetch_with_status() + sanitize_error_message() 보안 필터링
"""

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.modules.content_marketing.schema import TrendSource
from app.tools.trend.models import RawTrendItem

logger = logging.getLogger(__name__)


# ── 보안 필터링 (FR-08, FR-09) ──

# API 키 패턴 (마스킹 대상)
_SECRET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(AIza[0-9A-Za-z_-]{35})"),           # Google API Key
    re.compile(r"(sk-[a-zA-Z0-9]{20,})"),              # OpenAI-style key
    re.compile(r"(Bearer\s+[a-zA-Z0-9._~+/=-]{20,})"),  # Bearer token
    re.compile(r"(api[_-]?key[=:]\s*\S+)", re.IGNORECASE),
    re.compile(r"(token[=:]\s*\S+)", re.IGNORECASE),
]

# 내부 기술 스택 정보 (추상화 대상)
_TECH_STACK_KEYWORDS = [
    "httpx", "aiohttp", "requests", "urllib3", "asyncio",
    "Traceback", 'File "', "line ", "ModuleNotFoundError",
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
            return _classify_user_facing_message(raw_message)

    return sanitized[:200]


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


def _classify_error(exc: Exception) -> str:
    """예외를 표준 에러 유형으로 분류한다."""
    exc_str = str(exc).lower()
    exc_type = type(exc).__name__.lower()

    if "timeout" in exc_str or "timeout" in exc_type:
        return "timeout"
    if "401" in exc_str or "403" in exc_str or "unauthorized" in exc_str:
        return "auth"
    if "429" in exc_str or "quota" in exc_str or "rate" in exc_str:
        return "rate_limit"
    if "connect" in exc_str or "network" in exc_str or "dns" in exc_str:
        return "network"
    if "json" in exc_str or "parse" in exc_str or "decode" in exc_str:
        return "parse"
    return "unknown"


@dataclass
class SourceFetchResult:
    """소스 수집 결과 + 에러 정보"""

    items: list[RawTrendItem]
    source_name: str
    is_success: bool
    error_type: str | None = None  # timeout, auth, rate_limit, network, parse, unknown
    error_message: str | None = None  # sanitize_error_message() 적용 후 값
    fetched_at: datetime | None = None


@dataclass
class SourceConfig:
    """소스별 설정"""

    time_range: str = "24h"
    max_results: int = 20
    language: str = "ko"
    search_query: str = ""  # collector가 구성한 최종 검색 쿼리 (비어있으면 소스 기본값)
    community_context: str = ""  # Stage 1 커뮤니티 토픽 맥락 (Perplexity 심층 분석용)
    include_domains: list[str] = field(default_factory=list)  # Tavily include_domains
    category: str = "all"  # 법률 카테고리 (all, criminal, civil, labor, family, ...)


class BaseTrendSource(ABC):
    """트렌드 데이터 소스 추상 클래스"""

    @property
    @abstractmethod
    def name(self) -> TrendSource:
        """소스 식별자"""
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """API 키 설정 여부로 사용 가능 판단"""
        ...

    @abstractmethod
    async def fetch(
        self,
        query: str | None,
        config: SourceConfig,
    ) -> list[RawTrendItem]:
        """트렌드 데이터 수집

        Args:
            query: 키워드 필터 (None이면 전체 트렌드)
            config: 소스 설정

        Returns:
            수집된 원시 트렌드 항목 목록

        Raises:
            TrendSourceError: 소스 API 호출 실패 시
        """
        ...

    async def safe_fetch(
        self,
        query: str | None,
        config: SourceConfig,
    ) -> list[RawTrendItem]:
        """실패 시 빈 목록 반환 (graceful degradation)

        기존 인터페이스 유지 (EC-1 합의). 신규 코드에서는 safe_fetch_with_status() 사용 권장.
        """
        try:
            return await self.fetch(query, config)
        except Exception:
            logger.warning("소스 %s 수집 실패, 건너뜀", self.name.value)
            return []

    async def safe_fetch_with_status(
        self,
        query: str | None,
        config: SourceConfig,
    ) -> SourceFetchResult:
        """에러 정보를 포함한 수집 결과 반환

        safe_fetch()와 달리 실패 시에도 에러 유형/메시지를 SourceFetchResult로 반환한다.
        에러 메시지는 sanitize_error_message()로 민감 정보가 제거된 상태.
        """
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
                items=[],
                source_name=self.name.value,
                is_success=False,
                error_type="timeout",
                error_message="응답 시간 초과",
            )
        except Exception as exc:
            error_type = _classify_error(exc)
            logger.warning(
                "소스 %s 수집 실패 (%s): %s",
                self.name.value,
                error_type,
                str(exc)[:200],
            )
            safe_message = sanitize_error_message(str(exc)[:500])
            return SourceFetchResult(
                items=[],
                source_name=self.name.value,
                is_success=False,
                error_type=error_type,
                error_message=safe_message,
            )
