"""트렌드 수집 URL 유효성 검증"""

import re
from urllib.parse import urlparse

# 포털/뉴스 홈페이지 및 목록 페이지 패턴 (기사가 아닌 URL)
_BLOCKED_PATTERNS: list[re.Pattern[str]] = [
    # 포털 홈페이지
    re.compile(r"^https?://(www\.)?(news\.)?naver\.com/?$"),
    re.compile(r"^https?://(www\.)?news\.daum\.net/?$"),
    re.compile(r"^https?://(www\.)?daum\.net/?$"),
    re.compile(r"^https?://(www\.)?google\.com/?$"),
    re.compile(r"^https?://(www\.)?youtube\.com/?$"),
    # Google 뉴스 홈
    re.compile(r"^https?://news\.google\.com/(home|rss)"),
    # Perplexity 가짜 URL
    re.compile(r"^https?://perplexity\.ai/search"),
]

# 뉴스 목록/인덱스 경로 패턴 (기사가 아닌 목록 페이지)
_LIST_PATH_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^/?(news/?|latest/?|recent\.php|hot/?|index)$", re.IGNORECASE),
    re.compile(r"^/?news/(recent|latest|hot|ranking|index)(\.\w+)?$", re.IGNORECASE),
    re.compile(r"^/?news/keywordList\.do", re.IGNORECASE),
    re.compile(r"^/?\w+/index$", re.IGNORECASE),
]


def is_article_url(url: str) -> bool:
    """기사 URL인지 검증. 홈페이지/목록 페이지/가짜 URL이면 False."""
    if not url or not url.startswith("http"):
        return False

    # 홈페이지/차단 패턴 매칭
    for pattern in _BLOCKED_PATTERNS:
        if pattern.match(url):
            return False

    parsed = urlparse(url)
    path = parsed.path.strip("/")

    # 루트 URL (예: https://www.nodongilbo.com/)
    if not path:
        return False

    # 뉴스 목록/인덱스 페이지 (예: /news, /news/recent.php)
    for pattern in _LIST_PATH_PATTERNS:
        if pattern.match(f"/{path}"):
            return False

    return True
