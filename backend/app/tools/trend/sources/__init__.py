"""트렌드 데이터 소스 어댑터 (Strategy 패턴)"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from app.modules.content_marketing.schema import TrendSource
from app.tools.trend.models import RawTrendItem

logger = logging.getLogger(__name__)


@dataclass
class SourceConfig:
    """소스별 설정"""

    time_range: str = "24h"
    max_results: int = 20
    language: str = "ko"
    search_query: str = ""  # collector가 구성한 최종 검색 쿼리 (비어있으면 소스 기본값)
    community_context: str = ""  # Stage 1 커뮤니티 토픽 맥락 (Perplexity 심층 분석용)
    include_domains: list[str] = field(default_factory=list)  # Tavily include_domains


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
        """실패 시 빈 목록 반환 (graceful degradation)"""
        try:
            return await self.fetch(query, config)
        except Exception:
            logger.warning("소스 %s 수집 실패, 건너뜀", self.name.value)
            return []
