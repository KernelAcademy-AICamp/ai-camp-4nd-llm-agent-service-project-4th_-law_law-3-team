"""뉴스 파이프라인 설정"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.config import settings


@dataclass(frozen=True)
class NewsPipelineConfig:
    """파이프라인 설정 (환경 변수에서 로드)"""

    # 활성화
    enabled: bool = True

    # 요약 LLM
    summary_provider: str = "upstage"
    summary_model: str = "solar-pro2"
    summary_temperature: float = 0.3
    fallback_provider: str = "openai"

    # 수집
    min_article_length: int = 500  # 최소 본문 길이 (자)
    max_articles_per_run: int = 200  # 1회 최대 수집 수
    rate_limit_crawl: float = 2.0  # 크롤링 요청 간격 (초)

    # 소스 활성화
    lawtimes_enabled: bool = True
    naver_enabled: bool = True

    # 청킹
    chunk_size: int = 1500  # 청크 크기 (자)
    chunk_overlap: int = 200  # 청크 겹침 (자)

    # LanceDB
    lancedb_table: str = "news_chunks"

    # 보관
    retention_days: int = 90

    # 네이버 키워드
    naver_keywords: list[str] = field(
        default_factory=lambda: [
            "법원 판결",
            "대법원",
            "헌법재판소",
            "검찰 기소",
            "로펌",
            "변호사",
            "민사소송",
            "형사재판",
            "행정소송",
            "노동법",
            "세무 판결",
            "공정거래",
        ]
    )

    @classmethod
    def from_settings(cls) -> NewsPipelineConfig:
        """settings에서 환경 변수를 읽어 설정 생성"""
        return cls(
            enabled=getattr(settings, "NEWS_PIPELINE_ENABLED", True),
            summary_provider=getattr(settings, "NEWS_PIPELINE_SUMMARY_PROVIDER", "upstage"),
            summary_model=getattr(settings, "NEWS_PIPELINE_SUMMARY_MODEL", "solar-pro2"),
            min_article_length=getattr(settings, "NEWS_PIPELINE_MIN_ARTICLE_LENGTH", 500),
            max_articles_per_run=getattr(settings, "NEWS_PIPELINE_MAX_ARTICLES_PER_RUN", 200),
            rate_limit_crawl=getattr(settings, "NEWS_PIPELINE_RATE_LIMIT_CRAWL", 2.0),
            lawtimes_enabled=getattr(settings, "NEWS_PIPELINE_LAWTIMES_ENABLED", True),
            naver_enabled=getattr(settings, "NEWS_PIPELINE_NAVER_ENABLED", True),
            chunk_size=getattr(settings, "NEWS_PIPELINE_CHUNK_SIZE", 1500),
            chunk_overlap=getattr(settings, "NEWS_PIPELINE_CHUNK_OVERLAP", 200),
            lancedb_table=getattr(settings, "NEWS_PIPELINE_LANCEDB_TABLE", "news_chunks"),
            retention_days=getattr(settings, "NEWS_PIPELINE_RETENTION_DAYS", 90),
            fallback_provider=getattr(settings, "NEWS_PIPELINE_LLM_FALLBACK_PROVIDER", "openai"),
        )
