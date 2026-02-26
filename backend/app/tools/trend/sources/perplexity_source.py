"""Perplexity Sonar API 데이터 소스"""

import logging
import re
from datetime import datetime, timezone

import httpx

from app.core.config import settings
from app.modules.content_marketing.schema import TrendSource
from app.tools.trend.models import RawTrendItem
from app.tools.trend.sources import BaseTrendSource, SourceConfig

logger = logging.getLogger(__name__)

PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"

# URL 추출용 정규식
_URL_RE = re.compile(r"https?://[^\s\)\]\>\"']+")


class PerplexitySource(BaseTrendSource):
    """Perplexity Sonar API 데이터 소스

    Perplexity Sonar는 검색 강화 LLM이므로,
    법률 트렌드 검색 쿼리를 보내고 응답에서 이슈를 추출합니다.
    """

    @property
    def name(self) -> TrendSource:
        return TrendSource.PERPLEXITY

    @property
    def is_available(self) -> bool:
        return bool(settings.PERPLEXITY_API_KEY)

    async def fetch(
        self,
        query: str | None,
        config: SourceConfig,
    ) -> list[RawTrendItem]:
        search_query = config.search_query or query or "최신 법률 뉴스"

        time_label = {
            "48h": "최근 48시간",
            "7d": "최근 일주일",
            "30d": "최근 한 달",
        }.get(config.time_range, "최근 48시간")

        max_issues = min(config.max_results, 10)

        if config.community_context:
            # 심층 분석 모드: 커뮤니티 토픽의 핵심 쟁점과 최신 뉴스 분석
            prompt = (
                f"{time_label} 아래 커뮤니티 인기 토픽의 핵심 쟁점과 최신 뉴스를 심층 분석해주세요.\n\n"
                f"커뮤니티 인기 토픽:\n{config.community_context}\n\n"
                f"키워드: {search_query}\n\n"
                "각 이슈마다 다음 형식으로 작성:\n"
                "- 제목: [이슈 제목]\n"
                "- 요약: [법적 쟁점 포함 1-2문장 요약]\n"
                "- 출처: [URL]\n\n"
                f"최대 {max_issues}개 이슈를 알려주세요."
            )
        else:
            # 기존 모드: 일반 뉴스 이슈 검색
            prompt = (
                f"{time_label} 한국의 주요 뉴스 이슈를 검색해서 알려주세요.\n"
                f"키워드: {search_query}\n\n"
                "각 이슈마다 다음 형식으로 작성:\n"
                "- 제목: [이슈 제목]\n"
                "- 요약: [1-2문장 요약]\n"
                "- 출처: [URL]\n\n"
                f"최대 {max_issues}개 이슈를 알려주세요."
            )

        headers = {
            "Authorization": f"Bearer {settings.PERPLEXITY_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "sonar",
            "messages": [
                {
                    "role": "system",
                    "content": "한국 트렌드 분석 전문가입니다. 최신 뉴스 이슈를 검색하여 정리합니다.",
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 2000,
            "temperature": 0.1,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                PERPLEXITY_API_URL,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        citations: list[str] = data.get("citations", [])

        items = self._parse_response(content, citations)
        logger.info("Perplexity 수집 완료: %d건", len(items))
        return items

    def _parse_response(
        self,
        content: str,
        citations: list[str],
    ) -> list[RawTrendItem]:
        """Perplexity 응답에서 이슈 추출"""
        items: list[RawTrendItem] = []
        now = datetime.now(tz=timezone.utc)
        # citations 복사 (pop으로 순서대로 소비)
        remaining_citations = list(citations)

        # "- 제목:" 패턴으로 이슈 블록 분리
        blocks = re.split(r"(?=- 제목:)", content)

        for block in blocks:
            block = block.strip()
            if not block.startswith("- 제목:"):
                continue

            title = ""
            snippet = ""
            url = ""

            for line in block.split("\n"):
                line = line.strip().lstrip("- ")
                if line.startswith("제목:"):
                    title = line[3:].strip()
                elif line.startswith("요약:"):
                    snippet = line[3:].strip()
                elif line.startswith("출처:"):
                    url_match = _URL_RE.search(line)
                    if url_match:
                        url = url_match.group(0)

            if not title:
                continue

            # URL이 없으면 citations에서 순서대로 가져오기
            if not url and remaining_citations:
                url = remaining_citations.pop(0)

            # URL이 여전히 없으면 건너뛰기
            if not url:
                continue

            items.append(
                RawTrendItem(
                    title=title,
                    url=url,
                    snippet=snippet[:300],
                    source=TrendSource.PERPLEXITY,
                    published_at=now,
                    raw_data={"content": block},
                )
            )

        # 구조화 파싱 실패 시 전체 텍스트에서 추출
        if not items and content.strip():
            urls = _URL_RE.findall(content)
            lines = [
                line.strip()
                for line in content.split("\n")
                if line.strip() and len(line.strip()) > 10
            ]
            for i, line in enumerate(lines[:10]):
                cleaned = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
                if len(cleaned) < 5:
                    continue
                url = urls[i] if i < len(urls) else ""
                if not url:
                    continue
                items.append(
                    RawTrendItem(
                        title=cleaned[:100],
                        url=url,
                        snippet=cleaned[:300],
                        source=TrendSource.PERPLEXITY,
                        published_at=now,
                        raw_data={"line": cleaned},
                    )
                )

        return items
