"""TaskIQ 비동기 작업 큐 (v0.3.0)

Red Team 피드백: 소스 10개+ 확장 시 순차 수집은 병목.
TaskIQ 기반으로 소스별 수집을 병렬 태스크로 분리.
Redis URL이 비어 있으면 InMemoryBroker 사용 (개발/테스트).
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)

# --- Broker 싱글턴 (lazy init) ---

_broker: Any = None


def get_broker() -> Any:
    """TaskIQ 브로커 인스턴스 (lazy init)

    Redis URL이 설정되어 있으면 Redis 브로커, 아니면 InMemoryBroker.
    """
    global _broker  # noqa: PLW0603
    if _broker is not None:
        return _broker

    redis_url = getattr(settings, "NEWS_PIPELINE_REDIS_URL", "")

    if redis_url:
        from taskiq_redis import ListQueueBroker, RedisAsyncResultBackend
        _broker = ListQueueBroker(url=redis_url).with_result_backend(
            RedisAsyncResultBackend(redis_url=redis_url),
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
) -> dict[str, Any]:
    """개별 소스 수집 태스크

    TaskIQ 워커에서 비동기 실행됨.
    결과: {"source": str, "articles": list[dict], "error": str|None}
    """
    from app.tools.news_pipeline.config import NewsPipelineConfig
    from app.tools.news_pipeline.sources.lawtimes_source import LawtimesSource
    from app.tools.news_pipeline.sources.naver_news_source import NaverNewsSource

    config = NewsPipelineConfig.from_settings()
    target_date = date.fromisoformat(target_date_iso)

    source_map: dict[str, Any] = {
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
    global fetch_source_task  # noqa: PLW0603
    fetch_source_task = broker.task(fetch_source_task)
    logger.info("TaskIQ 태스크 등록 완료: fetch_source_task")
