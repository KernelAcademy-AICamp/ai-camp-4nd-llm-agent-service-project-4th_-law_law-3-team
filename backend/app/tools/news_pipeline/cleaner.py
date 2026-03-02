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
    re.compile(r"기자\s*[가-힣]+@[a-zA-Z0-9.]+"),
    re.compile(r"ⓒ\s*.+$", re.MULTILINE),
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
        for tag in soup.find_all(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()
        return soup.get_text(separator="\n")

    @staticmethod
    def _remove_patterns(text: str) -> str:
        """광고/메타/저작권 패턴 제거"""
        for pattern in _AD_PATTERNS:
            text = pattern.sub("", text)
        return text
