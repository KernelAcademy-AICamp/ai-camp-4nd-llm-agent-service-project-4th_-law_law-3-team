"""법률 뉴스 API 스키마"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class NewsArticleSummary(BaseModel):
    """기사 목록용 요약 스키마"""

    model_config = ConfigDict(from_attributes=True, extra="ignore")

    id: str = Field(description="문서 ID (SHA256)")
    title: str
    source: str
    publisher: str
    published_at: datetime | None
    summary_one_liner: str = Field(description="한줄 요약")
    section: str | None = None
    tags: list[str] | None = None


class NewsArticleResponse(BaseModel):
    """기사 상세 스키마"""

    model_config = ConfigDict(from_attributes=True, extra="ignore")

    id: str
    title: str
    source: str
    publisher: str
    published_at: datetime | None
    collected_at: datetime
    url: str
    author: str | None = None
    section: str | None = None
    tags: list[str] | None = None
    cleaned_text: str
    summary_one_liner: str
    summary_issues: list[str] | None = None
    summary_laws: list[str] | None = None
    summary_cases: list[str] | None = None
    summary_institutions: list[str] | None = None
    summary_implications: list[str] | None = None
    disclaimer: str
    schema_version: str


class NewsListResponse(BaseModel):
    """기사 목록 응답"""

    items: list[NewsArticleSummary]
    total: int
    page: int
    page_size: int
    has_next: bool


class NewsSearchRequest(BaseModel):
    """검색 요청"""

    query: str = Field(min_length=2, max_length=500, description="검색 쿼리")
    limit: int = Field(default=10, ge=1, le=50, description="결과 수")
    source: str | None = Field(default=None, description="소스 필터")


class NewsSearchResult(BaseModel):
    """검색 결과 항목 (BM25 기반)"""

    model_config = ConfigDict(from_attributes=True, extra="ignore")

    id: str = Field(description="기사 ID (SHA256)")
    title: str
    source: str
    publisher: str
    url: str
    published_at: datetime | None = None
    summary_one_liner: str = Field(default="", description="한줄 요약")
    section: str | None = None
    tags: list[str] | None = None
    relevance_score: float | None = Field(default=None, description="BM25 점수")


class NewsSearchResponse(BaseModel):
    """검색 응답"""

    results: list[NewsSearchResult]
    query: str
    total: int


# ── 통계 스키마 ──


class DailyStatItem(BaseModel):
    """일별 수집 건수 항목"""

    date: str = Field(description="날짜 (YYYY-MM-DD)")
    source: str = Field(description="소스 (lawtimes|naver)")
    count: int = Field(description="수집 건수")


class NewsStatsDaily(BaseModel):
    """일별 수집 통계 응답"""

    items: list[DailyStatItem]
    total: int = Field(description="전체 기사 수")
    period_days: int = Field(description="조회 기간 (일)")


class CategoryStatItem(BaseModel):
    """카테고리 분포 항목"""

    category: str = Field(description="카테고리명")
    count: int = Field(description="기사 건수")


class NewsCategoryStats(BaseModel):
    """카테고리 분포 통계 응답"""

    items: list[CategoryStatItem]
    total: int = Field(description="전체 기사 수")
    period_days: int | None = Field(default=None, description="조회 기간 (일, None=전체)")


class MainRagSourceItem(BaseModel):
    """Main RAG 데이터 소스별 건수"""

    table_name: str = Field(description="테이블명 (예: law_documents)")
    label: str = Field(description="한글명 (예: 법령)")
    count: int = Field(description="문서 건수")


class AssistRagSourceItem(BaseModel):
    """Assist RAG 데이터 소스별 건수"""

    source: str = Field(description="소스 (lawtimes|naver)")
    label: str = Field(description="한글명 (로타임즈|네이버뉴스)")
    count: int = Field(description="문서 건수")
    indexed_count: int = Field(description="임베딩 완료 건수")


class RagContributionStats(BaseModel):
    """RAG 기여도 통계 응답 (Main RAG + Assist RAG)"""

    main_rag_total: int = Field(description="Main RAG 전체 문서 수")
    main_rag_sources: list[MainRagSourceItem] = Field(description="Main RAG 소스별 통계")
    assist_rag_total: int = Field(description="Assist RAG 전체 문서 수")
    assist_rag_sources: list[AssistRagSourceItem] = Field(description="Assist RAG 소스별 통계")
    assist_contribution_percent: float = Field(description="Assist RAG 기여 비율 (%)")
