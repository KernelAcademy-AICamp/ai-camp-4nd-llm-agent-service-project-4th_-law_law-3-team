"""뉴스 파이프라인 내부 데이터 모델"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class NewsSourceType(str, Enum):
    """뉴스 소스 유형"""

    LAWTIMES = "lawtimes"
    NAVER = "naver"


@dataclass
class RawArticle:
    """수집 단계 출력: 원시 기사 데이터"""

    url: str
    title: str
    raw_html: str  # 원문 HTML
    source: NewsSourceType
    publisher: str  # 매체명
    published_at: datetime | None
    author: str | None = None
    section: str | None = None
    tags: list[str] = field(default_factory=list)
    raw_metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class CleanedArticle:
    """정제 단계 출력: 정제된 기사"""

    url: str
    title: str
    cleaned_text: str  # HTML 제거, 정규화된 본문
    content_hash: str  # SHA256(cleaned_text)
    source: NewsSourceType
    publisher: str
    published_at: datetime | None
    author: str | None = None
    section: str | None = None
    tags: list[str] = field(default_factory=list)
    char_count: int = 0  # 본문 글자 수


@dataclass
class ArticleSummary:
    """요약 결과"""

    one_liner: str  # 한줄 요지
    issues: list[str]  # 주요 쟁점 (3~5개)
    laws: list[str]  # 언급 법령
    cases: list[str]  # 언급 판례
    institutions: list[str]  # 언급 기관
    implications: list[str]  # 시사점 (1~3개)
    law_ids: list[int] = field(default_factory=list)  # v0.2.0: 검증된 법령 DB ID
    case_ids: list[int] = field(default_factory=list)  # v0.2.0: 검증된 판례 DB ID


@dataclass
class ProcessedArticle:
    """최종 처리 완료 기사 (DB 저장 직전)"""

    doc_id: str  # SHA256(url)
    url: str
    title: str
    cleaned_text: str
    content_hash: str
    source: NewsSourceType
    publisher: str
    published_at: datetime | None
    collected_at: datetime
    author: str | None
    section: str | None
    tags: list[str]
    summary: ArticleSummary
    disclaimer: str = "본 문서는 기사 요약이며 법령/판례 원문이 아닙니다"
    schema_version: str = "1.0"


@dataclass
class PipelineResult:
    """파이프라인 실행 결과 (리포트용)"""

    run_id: str  # UUID
    started_at: datetime
    finished_at: datetime | None = None
    source_stats: dict[str, SourceStat] = field(default_factory=dict)
    total_collected: int = 0
    total_deduplicated: int = 0  # 중복 제거된 수
    total_cleaned: int = 0
    total_summarized: int = 0
    total_stored: int = 0
    total_chunked: int = 0
    errors: list[PipelineError] = field(default_factory=list)


@dataclass
class SourceStat:
    """소스별 통계"""

    collected: int = 0
    deduplicated: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class PipelineError:
    """파이프라인 에러 기록"""

    stage: str  # "collect" | "clean" | "summarize" | "store" | "chunk"
    article_url: str | None
    error_type: str
    error_message: str
    timestamp: datetime = field(default_factory=datetime.now)
