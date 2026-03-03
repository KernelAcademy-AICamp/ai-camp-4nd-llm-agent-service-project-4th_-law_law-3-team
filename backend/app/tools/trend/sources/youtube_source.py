"""YouTube Data API v3 데이터 소스

v2: 2단계 API — search.list → videos.list(statistics)
  Stage 1: search.list로 비디오 검색
  Stage 2: videos.list(part=statistics)로 조회수/댓글수/좋아요수 보강
"""

import logging
from datetime import datetime

import httpx

from app.core.config import settings
from app.modules.content_marketing.schema import TrendSource
from app.tools.trend.models import RawTrendItem
from app.tools.trend.sources import BaseTrendSource, SourceConfig

logger = logging.getLogger(__name__)

YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"

# time_range → publishedAfter 매핑용 (시간 단위)
_TIME_RANGE_HOURS: dict[str, int] = {
    "48h": 48,
    "7d": 168,
    "30d": 720,
}


def _parse_youtube_date(date_str: str | None) -> datetime | None:
    """YouTube ISO 8601 날짜 파싱 (timezone-aware)"""
    if not date_str:
        return None
    from datetime import timezone

    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ"):
        try:
            return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


class YouTubeSource(BaseTrendSource):
    """YouTube Data API v3 데이터 소스"""

    @property
    def name(self) -> TrendSource:
        return TrendSource.YOUTUBE

    @property
    def is_available(self) -> bool:
        return bool(settings.YOUTUBE_API_KEY)

    async def fetch(
        self,
        query: str | None,
        config: SourceConfig,
    ) -> list[RawTrendItem]:
        search_query = config.search_query or query or "법률 뉴스"

        # publishedAfter 계산
        hours = _TIME_RANGE_HOURS.get(config.time_range, 24)
        from datetime import timedelta, timezone

        published_after = (
            datetime.now(tz=timezone.utc) - timedelta(hours=hours)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

        params: dict[str, str | int] = {
            "part": "snippet",
            "q": search_query,
            "type": "video",
            "order": "date",
            "relevanceLanguage": "ko",
            "regionCode": "KR",
            "maxResults": min(config.max_results, 25),
            "publishedAfter": published_after,
            "key": settings.YOUTUBE_API_KEY,
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(YOUTUBE_SEARCH_URL, params=params)
            response.raise_for_status()
            data = response.json()

        # Stage 1: search 결과에서 비디오 ID 추출
        video_ids: list[str] = []
        search_items: list[dict[str, object]] = []
        for item in data.get("items", []):
            snippet = item.get("snippet", {})
            video_id = item.get("id", {}).get("videoId", "")
            if not video_id:
                continue
            video_ids.append(video_id)
            search_items.append(item)

        # Stage 2: videos.list로 statistics 보강
        stats_map: dict[str, dict[str, int]] = {}
        if video_ids:
            stats_map = await self._fetch_video_statistics(
                video_ids, client=None,
            )

        items: list[RawTrendItem] = []
        for item in search_items:
            snippet = item.get("snippet", {})
            video_id = item.get("id", {}).get("videoId", "")
            if not video_id:
                continue

            stats = stats_map.get(video_id, {})
            is_shorts = self._detect_shorts(video_id, snippet)

            items.append(
                RawTrendItem(
                    title=snippet.get("title", ""),
                    url=f"https://www.youtube.com/watch?v={video_id}",
                    snippet=snippet.get("description", "")[:300],
                    source=TrendSource.YOUTUBE,
                    published_at=_parse_youtube_date(snippet.get("publishedAt")),
                    raw_data=item,
                    view_count=stats.get("view_count"),
                    comment_count=stats.get("comment_count"),
                    like_count=stats.get("like_count"),
                    is_shorts=is_shorts,
                )
            )

        logger.info("YouTube 수집 완료: %d건 (statistics %d건)", len(items), len(stats_map))
        return items

    async def _fetch_video_statistics(
        self,
        video_ids: list[str],
        client: httpx.AsyncClient | None = None,
    ) -> dict[str, dict[str, int]]:
        """videos.list(statistics)로 조회수/댓글수/좋아요수 조회

        Args:
            video_ids: 조회할 비디오 ID 리스트 (최대 50개씩 분할)
            client: 기존 httpx 클라이언트 (None이면 새로 생성)

        Returns:
            {video_id: {"view_count": N, "comment_count": N, "like_count": N}}
        """
        result: dict[str, dict[str, int]] = {}

        # YouTube API는 videos.list에서 최대 50개씩 조회 가능
        for i in range(0, len(video_ids), 50):
            batch = video_ids[i:i + 50]
            params: dict[str, str | int] = {
                "part": "statistics",
                "id": ",".join(batch),
                "key": settings.YOUTUBE_API_KEY,
            }

            try:
                async with httpx.AsyncClient(timeout=10.0) as http_client:
                    response = await http_client.get(YOUTUBE_VIDEOS_URL, params=params)
                    response.raise_for_status()
                    data = response.json()

                for item in data.get("items", []):
                    vid = item.get("id", "")
                    stats = item.get("statistics", {})
                    result[vid] = {
                        "view_count": _safe_int(stats.get("viewCount")),
                        "comment_count": _safe_int(stats.get("commentCount")),
                        "like_count": _safe_int(stats.get("likeCount")),
                    }
            except Exception:
                logger.warning("YouTube statistics 조회 실패 (batch %d)", i, exc_info=True)

        return result

    @staticmethod
    def _detect_shorts(video_id: str, snippet: dict[str, object]) -> bool:
        """Shorts 영상 감지 (제목/설명 기반 휴리스틱)"""
        title = str(snippet.get("title", "")).lower()
        description = str(snippet.get("description", "")).lower()
        return "#shorts" in title or "#shorts" in description


def _safe_int(value: object) -> int | None:
    """문자열/숫자를 int로 안전 변환"""
    if value is None:
        return None
    try:
        return int(str(value))
    except (ValueError, TypeError):
        return None
