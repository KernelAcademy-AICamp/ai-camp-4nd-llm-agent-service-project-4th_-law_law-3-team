"""뉴스 소스 추상 인터페이스"""

from __future__ import annotations

import abc
from datetime import date

from app.tools.news_pipeline.models import NewsSourceType, RawArticle


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
