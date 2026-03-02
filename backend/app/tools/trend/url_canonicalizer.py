"""URL 정규화 (Canonicalization)

수집된 뉴스 기사 URL을 표준 형태로 통일하여 중복 탐지 정확도를 높인다.

처리:
1. UTM 파라미터 제거 (utm_source, utm_medium, utm_campaign, ...)
2. 추적/광고 파라미터 제거 (fbclid, gclid, ref, ...)
3. 모바일 도메인 통일 (m.naver.com → naver.com)
4. 프래그먼트(#) 제거
5. 후행 슬래시 통일
6. 소문자 호스트 정규화
"""

import logging
import re
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

logger = logging.getLogger(__name__)

# 제거 대상 쿼리 파라미터 (추적/광고/분석용)
_STRIP_PARAMS: frozenset[str] = frozenset({
    # UTM
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "utm_id", "utm_source_platform", "utm_creative_format",
    # 소셜/광고
    "fbclid", "gclid", "gclsrc", "dclid", "msclkid", "twclid",
    "mc_cid", "mc_eid", "yclid",
    # 추적
    "ref", "ref_src", "referer", "referrer", "source",
    "share", "share_id", "from",
    # 기타
    "_ga", "_gl", "si", "feature",
})

# 모바일 → 데스크톱 도메인 매핑
_MOBILE_DOMAIN_MAP: dict[str, str] = {
    "m.naver.com": "naver.com",
    "m.news.naver.com": "news.naver.com",
    "m.blog.naver.com": "blog.naver.com",
    "m.cafe.naver.com": "cafe.naver.com",
    "m.kin.naver.com": "kin.naver.com",
    "m.daum.net": "daum.net",
    "m.news.daum.net": "news.daum.net",
    "m.entertain.naver.com": "entertain.naver.com",
    "m.sports.naver.com": "sports.naver.com",
    "mobile.twitter.com": "twitter.com",
    "m.youtube.com": "www.youtube.com",
    "m.khan.co.kr": "www.khan.co.kr",
    "m.hani.co.kr": "www.hani.co.kr",
    "m.mk.co.kr": "www.mk.co.kr",
    "m.sedaily.com": "www.sedaily.com",
    "m.lawtimes.co.kr": "www.lawtimes.co.kr",
}

# Naver 뉴스 AMP/리다이렉트 URL 정규화 패턴
_NAVER_NEWS_PATTERN = re.compile(
    r"^https?://(?:m\.)?news\.naver\.com/article/(\d+)/(\d+)",
)


def canonicalize_url(url: str) -> str:
    """URL을 정규화된 표준 형태로 변환

    Args:
        url: 원본 URL

    Returns:
        정규화된 URL (변경 없으면 원본 반환)
    """
    if not url or not url.startswith(("http://", "https://")):
        return url

    try:
        parsed = urlparse(url)
    except Exception:
        return url

    # 1. 호스트 소문자 + 모바일 도메인 통일
    host = (parsed.hostname or "").lower()
    host = _MOBILE_DOMAIN_MAP.get(host, host)

    # 2. 쿼리 파라미터 필터링
    if parsed.query:
        params = parse_qs(parsed.query, keep_blank_values=False)
        filtered = {
            k: v for k, v in params.items()
            if k.lower() not in _STRIP_PARAMS
        }
        query = urlencode(filtered, doseq=True)
    else:
        query = ""

    # 3. 프래그먼트 제거
    fragment = ""

    # 4. 경로 후행 슬래시 통일 (루트 제외)
    path = parsed.path
    if path and path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    # 5. 스킴 통일 (http → https)
    scheme = "https"

    # 6. 포트 제거 (기본 포트)
    netloc = host
    if parsed.port and parsed.port not in (80, 443):
        netloc = f"{host}:{parsed.port}"

    canonical = urlunparse((scheme, netloc, path, "", query, fragment))
    return canonical


def canonicalize_urls(urls: list[str]) -> list[str]:
    """URL 목록을 정규화

    Args:
        urls: 원본 URL 리스트

    Returns:
        정규화된 URL 리스트
    """
    return [canonicalize_url(u) for u in urls]
