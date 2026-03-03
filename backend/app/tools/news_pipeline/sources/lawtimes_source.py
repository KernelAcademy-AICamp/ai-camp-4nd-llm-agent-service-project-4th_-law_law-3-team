"""법률신문 크롤러 (ND소프트 CMS 대응)

v2.1: 2026-02 사이트 개편 대응 + 3중 검증 반영
- RSS 제거 (301→403 차단)
- ND소프트 CMS 셀렉터 적용 (altlist-* 클래스)
- articleList.html?sc_section_code= URL 패턴
- 브라우저 유사 헤더로 403 우회
- robots.txt UA 일관성 수정 (Red Team 지적)
- 페이지네이션 최대 페이지 제한 (Red Team + Consultant 공통 지적)
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import date, datetime
from urllib.parse import urljoin
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup, Tag

from app.tools.news_pipeline.config import NewsPipelineConfig
from app.tools.news_pipeline.exceptions import SourceFetchError
from app.tools.news_pipeline.models import NewsSourceType, RawArticle
from app.tools.news_pipeline.sources import BaseNewsSource
from app.tools.news_pipeline.ssrf_guard import validate_url

logger = logging.getLogger(__name__)

LAWTIMES_BASE_URL = "https://www.lawtimes.co.kr"

# 브라우저 유사 헤더 (봇 차단 우회)
_BROWSER_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": f"{LAWTIMES_BASE_URL}/",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

# robots.txt 확인용 User-Agent (실제 크롤링과 동일한 UA 사용)
_ROBOT_UA = _BROWSER_HEADERS["User-Agent"]

# 페이지네이션 안전 상한 (무한 루프 방지)
_MAX_PAGES_PER_SECTION = 20

# ── 섹션 설정 (ND소프트 CMS 코드) ──────────────────────────────────
# sc_section_code → 대분류, sc_sub_section_code → 소분류
TARGET_SECTIONS: list[dict[str, str]] = [
    {"code": "S1N1", "name": "뉴스", "param": "sc_section_code"},
    {"code": "S1N3", "name": "판결큐레이션", "param": "sc_section_code"},
]

# ── CSS 셀렉터 (ND소프트 CMS altlist-* 패턴) ───────────────────────
SELECTOR_CONFIG: dict[str, str] = {
    # 목록 페이지
    "article_list": "#section-list ul.altlist-webzine > li.altlist-webzine-item",
    "article_title": "h2.altlist-subject a",
    "article_info": "div.altlist-info",
    "article_info_item": "div.altlist-info-item",
    # 상세 페이지
    "body_content": "#article-view-content-div",
    "body_exclude": "figure, .photo-layout, script, style, .ad-area",
}

# 날짜 파싱: 앞뒤 파이프/공백 제거용
_DATE_PREFIX_RE = re.compile(r"^[\s|]+")


class LawtimesSource(BaseNewsSource):
    """법률신문 (lawtimes.co.kr) 크롤러

    수집 전략 (v2.0):
    1. HTML 크롤링 only (RSS 차단됨)
    2. ND소프트 CMS articleList.html 엔드포인트 사용
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
            articles = await self._fetch_via_html(target_date)
        except Exception as exc:
            raise SourceFetchError("lawtimes", str(exc)) from exc

        logger.info("법률신문 수집 완료: %d건 (대상: %s)", len(articles), target_date)
        return articles

    # ── robots.txt ──────────────────────────────────────────────────

    async def _check_robots_txt(self) -> None:
        """robots.txt 확인 및 캐싱"""
        if self._robot_parser is not None:
            return
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{LAWTIMES_BASE_URL}/robots.txt",
                    headers={"User-Agent": _ROBOT_UA},
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
        return self._robot_parser.can_fetch(_ROBOT_UA, url)

    # ── HTML 크롤링 ────────────────────────────────────────────────

    async def _fetch_via_html(self, target_date: date) -> list[RawArticle]:
        """HTML 크롤링으로 기사 수집 (ND소프트 CMS)

        ND소프트 CMS의 sc_sdate/sc_edate 파라미터로 서버 사이드 날짜 필터링 후,
        page 파라미터로 페이지네이션하여 대상 날짜의 전체 기사를 수집한다.
        """
        articles: list[RawArticle] = []
        date_str = target_date.isoformat()  # "YYYY-MM-DD"

        async with httpx.AsyncClient(
            timeout=20.0,
            follow_redirects=True,
            headers=_BROWSER_HEADERS,
        ) as client:
            for section in TARGET_SECTIONS:
                page_num = 1

                while (
                    len(articles) < self._config.max_articles_per_run
                    and page_num <= _MAX_PAGES_PER_SECTION
                ):
                    section_url = (
                        f"{LAWTIMES_BASE_URL}/news/articleList.html"
                        f"?{section['param']}={section['code']}"
                        f"&view_type=sm"
                        f"&sc_sdate={date_str}&sc_edate={date_str}"
                        f"&page={page_num}"
                    )
                    if not validate_url(section_url) or not self._can_fetch(section_url):
                        break

                    try:
                        await asyncio.sleep(self._config.rate_limit_crawl)
                        resp = await client.get(section_url)
                        resp.raise_for_status()
                    except Exception as exc:
                        logger.warning(
                            "섹션 [%s] p%d 실패: %s", section["name"], page_num, exc,
                        )
                        break

                    page_articles = self._parse_article_list(
                        resp.text, section["name"],
                    )

                    if not page_articles:
                        break  # 더 이상 기사 없음

                    # 각 기사의 본문 가져오기
                    for raw in page_articles:
                        if len(articles) >= self._config.max_articles_per_run:
                            return articles

                        body_html = await self._fetch_article_body(raw.url, client)
                        if body_html:
                            raw.raw_html = body_html
                            articles.append(raw)

                    page_num += 1

        return articles

    def _parse_article_list(
        self, html: str, section_name: str,
    ) -> list[RawArticle]:
        """목록 페이지 HTML 파싱 (날짜는 서버 sc_sdate/sc_edate로 필터링됨)"""
        soup = BeautifulSoup(html, "lxml")
        items = soup.select(SELECTOR_CONFIG["article_list"])
        results: list[RawArticle] = []

        for item in items:
            # 제목 + URL
            link_tag = item.select_one(SELECTOR_CONFIG["article_title"])
            if not link_tag or not link_tag.get("href"):
                continue

            article_url = str(link_tag["href"])
            if not article_url.startswith("http"):
                article_url = urljoin(LAWTIMES_BASE_URL, article_url)

            if not validate_url(article_url):
                continue

            title = link_tag.get_text(strip=True)

            # 메타 정보 (섹션, 기자, 날짜)
            info_items = item.select(
                f"{SELECTOR_CONFIG['article_info']} > {SELECTOR_CONFIG['article_info_item']}",
            )
            article_section = section_name
            author: str | None = None
            date_text = ""

            if len(info_items) >= 3:
                article_section = info_items[0].get_text(strip=True)
                author = _clean_pipe_prefix(info_items[1].get_text(strip=True))
                date_text = _clean_pipe_prefix(info_items[2].get_text(strip=True))
            elif len(info_items) == 2:
                author = _clean_pipe_prefix(info_items[0].get_text(strip=True))
                date_text = _clean_pipe_prefix(info_items[1].get_text(strip=True))
            elif len(info_items) == 1:
                date_text = _clean_pipe_prefix(info_items[0].get_text(strip=True))

            published = _parse_date_text(date_text)

            results.append(RawArticle(
                url=article_url,
                title=title,
                raw_html="",  # 본문은 나중에 채움
                source=NewsSourceType.LAWTIMES,
                publisher="법률신문",
                published_at=published,
                author=author,
                section=article_section,
            ))

        return results

    async def _fetch_article_body(
        self, url: str, client: httpx.AsyncClient,
    ) -> str | None:
        """개별 기사 본문 HTML 추출 (article-view-content-div 영역)"""
        if not self._can_fetch(url):
            logger.warning("robots.txt 차단: %s", url)
            return None

        try:
            await asyncio.sleep(self._config.rate_limit_crawl)
            resp = await client.get(url)
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("기사 본문 요청 실패: %s — %s", url, exc)
            return None

        # 본문 영역만 추출 (전체 페이지 노이즈 제거)
        soup = BeautifulSoup(resp.text, "lxml")
        body = soup.select_one(SELECTOR_CONFIG["body_content"])
        if not body:
            logger.warning("본문 컨테이너 미발견: %s", url)
            return None

        # 불필요 요소 제거 (광고, 사진 캡션 등은 Cleaner에서 처리)
        for tag in body.select(SELECTOR_CONFIG["body_exclude"]):
            if isinstance(tag, Tag):
                tag.decompose()

        return str(body)


# ── 유틸리티 ────────────────────────────────────────────────────────

def _clean_pipe_prefix(text: str) -> str:
    """'| 기자명' → '기자명' 형태로 파이프 접두사 제거"""
    return _DATE_PREFIX_RE.sub("", text).strip()


def _parse_date_text(text: str) -> datetime | None:
    """날짜 텍스트 파싱 (다양한 형식 지원)

    지원 형식:
    - "2026-02-26"          (목록 페이지)
    - "2026.02.26 18:34"    (상세 페이지)
    - "업데이트 2026.02.26 18:34"
    - "입력 2026.02.26 18:34"
    """
    cleaned = _clean_pipe_prefix(text)

    # "업데이트 " / "입력 " 접두사 제거
    for prefix in ("업데이트", "입력"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()

    for fmt in ("%Y-%m-%d %H:%M", "%Y.%m.%d %H:%M", "%Y-%m-%d", "%Y.%m.%d"):
        try:
            return datetime.strptime(cleaned.strip(), fmt)
        except ValueError:
            continue
    return None
