"""법률 뉴스 API 비즈니스 로직"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from fastapi import HTTPException
from sqlalchemy import case, func, literal, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from app.models.news_article import NewsArticle
from app.modules.legal_news.schema import (
    AssistRagSourceItem,
    CategoryStatItem,
    DailyStatItem,
    MainRagSourceItem,
    NewsArticleResponse,
    NewsArticleSummary,
    NewsCategoryStats,
    NewsListResponse,
    NewsSearchRequest,
    NewsSearchResponse,
    NewsSearchResult,
    NewsStatsDaily,
    RagContributionStats,
)


async def get_news_list_service(
    db: AsyncSession,
    *,
    source: str | None,
    published_date: date | None,
    page: int,
    page_size: int,
) -> NewsListResponse:
    """뉴스 목록 조회 서비스"""
    query = select(NewsArticle).order_by(NewsArticle.published_at.desc())

    if source:
        query = query.where(NewsArticle.source == source)
    if published_date:
        query = query.where(
            func.date(NewsArticle.published_at) == published_date
        )

    # 전체 건수
    count_query = select(func.count()).select_from(query.subquery())
    total = (await db.execute(count_query)).scalar_one()

    # 페이지네이션
    offset = (page - 1) * page_size
    rows = (await db.execute(query.offset(offset).limit(page_size))).scalars().all()

    items = [NewsArticleSummary.model_validate(row) for row in rows]

    return NewsListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        has_next=(offset + page_size) < total,
    )


async def get_news_detail_service(
    db: AsyncSession,
    *,
    article_id: str,
) -> NewsArticleResponse:
    """뉴스 상세 조회 서비스"""
    result = await db.execute(
        select(NewsArticle).where(NewsArticle.id == article_id)
    )
    article = result.scalar_one_or_none()
    if not article:
        raise HTTPException(status_code=404, detail="기사를 찾을 수 없습니다")
    return NewsArticleResponse.model_validate(article)


async def search_news_service(
    db: AsyncSession,
    *,
    request: NewsSearchRequest,
) -> NewsSearchResponse:
    """뉴스 하이브리드 검색 서비스 (v0.3.0)"""
    from app.tools.news_pipeline.chunker import Chunker
    from app.tools.news_pipeline.config import NewsPipelineConfig

    config = NewsPipelineConfig.from_settings()
    results = await Chunker.hybrid_search(
        query=request.query,
        lancedb_table_name=config.lancedb_table,
        limit=request.limit * 2,
        rerank_top_k=request.limit,
    )

    # 소스 필터 적용
    if request.source:
        results = [r for r in results if r.get("source") == request.source]

    search_results = [
        NewsSearchResult(
            chunk_id=r.get("chunk_id", ""),
            doc_id=r.get("doc_id", ""),
            title=r.get("title", ""),
            chunk_text=r.get("chunk_text", ""),
            chunk_type=r.get("chunk_type", ""),
            source=r.get("source", ""),
            publisher=r.get("publisher", ""),
            url=r.get("url", ""),
            published_at=r.get("published_at"),
            rerank_score=r.get("rerank_score"),
        )
        for r in results
    ]

    return NewsSearchResponse(
        results=search_results,
        query=request.query,
        total=len(search_results),
    )


# ── 카테고리 분류 규칙 ──

_CATEGORY_RULES: list[tuple[str, set[str], set[str]]] = [
    # (카테고리명, section 매칭, 키워드 매칭)
    ("판결 큐레이션", {"판결큐레이션", "판결"}, set()),
    ("법조계 인사", {"인사", "사람"}, {"인사", "취임", "임명"}),
    ("법령 동향", set(), {"법령", "개정", "시행", "공포"}),
    ("형사·검찰", set(), {"형사", "검찰", "수사", "기소", "구속"}),
    ("소송·재판", set(), {"소송", "재판", "항소", "상고"}),
]


def _classify_category(
    source: str,
    section: str | None,
    tags: list[str] | None,
    title: str,
) -> str:
    """규칙 기반 카테고리 분류"""
    section_lower = (section or "").strip()
    title_lower = title or ""
    tags_str = " ".join(tags) if tags else ""
    combined_text = f"{title_lower} {tags_str}"

    for category_name, section_matches, keyword_matches in _CATEGORY_RULES:
        if section_matches and section_lower in section_matches:
            return category_name
        if keyword_matches and any(kw in combined_text for kw in keyword_matches):
            return category_name

    if source == "lawtimes":
        return "법조계 동향"
    return "기타"


async def get_news_stats_daily(
    db: AsyncSession,
    *,
    days: int,
) -> NewsStatsDaily:
    """일별 수집 통계 조회"""
    kst = timezone(timedelta(hours=9))
    since = (datetime.now(kst) - timedelta(days=days)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    query = (
        select(
            func.date(NewsArticle.published_at).label("pub_date"),
            NewsArticle.source,
            func.count().label("cnt"),
        )
        .where(NewsArticle.published_at >= since)
        .group_by(func.date(NewsArticle.published_at), NewsArticle.source)
        .order_by(func.date(NewsArticle.published_at))
    )

    rows = (await db.execute(query)).all()

    items = [
        DailyStatItem(
            date=str(row.pub_date),
            source=row.source,
            count=row.cnt,
        )
        for row in rows
    ]

    total = sum(item.count for item in items)

    return NewsStatsDaily(items=items, total=total, period_days=days)


def _build_category_expression() -> ColumnElement[str]:
    """SQL CASE WHEN 기반 카테고리 분류 표현식"""
    title_tags = func.concat(
        func.coalesce(NewsArticle.title, literal("")),
        literal(" "),
        func.coalesce(func.array_to_string(NewsArticle.tags, literal(" ")), literal("")),
    )
    return case(
        # section 기반 매칭 (우선순위 높음)
        (NewsArticle.section.in_(["판결큐레이션", "판결"]), literal("판결 큐레이션")),
        (
            or_(
                NewsArticle.section.in_(["인사", "사람"]),
                or_(
                    title_tags.contains("인사"),
                    title_tags.contains("취임"),
                    title_tags.contains("임명"),
                ),
            ),
            literal("법조계 인사"),
        ),
        (
            or_(
                title_tags.contains("법령"),
                title_tags.contains("개정"),
                title_tags.contains("시행"),
                title_tags.contains("공포"),
            ),
            literal("법령 동향"),
        ),
        (
            or_(
                title_tags.contains("형사"),
                title_tags.contains("검찰"),
                title_tags.contains("수사"),
                title_tags.contains("기소"),
                title_tags.contains("구속"),
            ),
            literal("형사·검찰"),
        ),
        (
            or_(
                title_tags.contains("소송"),
                title_tags.contains("재판"),
                title_tags.contains("항소"),
                title_tags.contains("상고"),
            ),
            literal("소송·재판"),
        ),
        (NewsArticle.source == "lawtimes", literal("법조계 동향")),
        else_=literal("기타"),
    )


async def get_news_category_stats(
    db: AsyncSession,
    *,
    days: int | None = None,
) -> NewsCategoryStats:
    """카테고리 분포 통계 조회 (SQL 집계, 기간 필터 지원)"""
    category_expr = _build_category_expression()

    query = (
        select(
            category_expr.label("category"),
            func.count().label("cnt"),
        )
        .group_by(category_expr)
        .order_by(func.count().desc())
    )

    # 기간 필터 (SQLAlchemy ORM, SQL Injection 안전)
    if days is not None:
        kst = timezone(timedelta(hours=9))
        since = (datetime.now(kst) - timedelta(days=days)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        query = query.where(NewsArticle.published_at >= since)

    rows = (await db.execute(query)).all()

    items = [
        CategoryStatItem(category=row.category, count=row.cnt)
        for row in rows
    ]
    total = sum(row.cnt for row in rows)

    return NewsCategoryStats(items=items, total=total, period_days=days)


_MAIN_RAG_TABLES: list[tuple[str, str]] = [
    # (테이블명, 한글 라벨) — 모델 클래스는 아래 _get_model_class()로 동적 매핑
    ("local_ordinance_documents", "자치법규"),
    ("special_admin_appeal_documents", "특별행정심판"),
    ("precedent_documents", "판례"),
    ("dec_labor_documents", "노동위원회 결정례"),
    ("interpretation_ministry_documents", "부처 유권해석"),
    ("administration_documents", "행정심판"),
    ("constitutional_documents", "헌법재판소 결정례"),
    ("admin_rule_documents", "행정규칙"),
    ("legislation_documents", "법령해석"),
    ("dec_fair_trade_documents", "공정거래 결정례"),
    ("law_documents", "법령"),
    ("dec_human_rights_documents", "국가인권위 결정례"),
    ("treaty_documents", "조약"),
    ("dec_privacy_documents", "개인정보보호 결정례"),
    ("dec_media_documents", "방송미디어 결정례"),
    ("dec_industrial_documents", "산업재해 결정례"),
    ("dec_financial_documents", "금융위 결정례"),
    ("dec_securities_documents", "증권선물 결정례"),
    ("dec_civil_rights_documents", "국민권익위 결정례"),
    ("dec_environment_documents", "환경분쟁 결정례"),
    ("dec_employment_documents", "고용보험 결정례"),
]

_ASSIST_RAG_LABELS: dict[str, str] = {
    "lawtimes": "로타임즈",
    "naver": "네이버뉴스",
}


def _get_model_class(table_name: str) -> type:
    """테이블명으로 ORM 모델 클래스를 반환"""
    import app.models as models

    _table_to_model: dict[str, str] = {
        "local_ordinance_documents": "LocalOrdinanceDocument",
        "special_admin_appeal_documents": "SpecialAdminAppealDocument",
        "precedent_documents": "PrecedentDocument",
        "dec_labor_documents": "DecLaborDocument",
        "interpretation_ministry_documents": "InterpretationMinistryDocument",
        "administration_documents": "AdministrationDocument",
        "constitutional_documents": "ConstitutionalDocument",
        "admin_rule_documents": "AdminRuleDocument",
        "legislation_documents": "LegislationDocument",
        "dec_fair_trade_documents": "DecFairTradeDocument",
        "law_documents": "LawDocument",
        "dec_human_rights_documents": "DecHumanRightsDocument",
        "treaty_documents": "TreatyDocument",
        "dec_privacy_documents": "DecPrivacyDocument",
        "dec_media_documents": "DecMediaDocument",
        "dec_industrial_documents": "DecIndustrialDocument",
        "dec_financial_documents": "DecFinancialDocument",
        "dec_securities_documents": "DecSecuritiesDocument",
        "dec_civil_rights_documents": "DecCivilRightsDocument",
        "dec_environment_documents": "DecEnvironmentDocument",
        "dec_employment_documents": "DecEmploymentDocument",
    }
    class_name = _table_to_model[table_name]
    return getattr(models, class_name)


async def get_rag_contribution_stats(
    db: AsyncSession,
) -> RagContributionStats:
    """RAG 기여도 통계 조회 (Main RAG 21개 테이블 + Assist RAG 뉴스)"""
    # Main RAG: 21개 원본 문서 테이블 COUNT
    main_sources: list[MainRagSourceItem] = []
    main_total = 0
    for table_name, label in _MAIN_RAG_TABLES:
        model_cls = _get_model_class(table_name)
        result = await db.execute(select(func.count()).select_from(model_cls))
        count = result.scalar_one()
        main_sources.append(
            MainRagSourceItem(table_name=table_name, label=label, count=count)
        )
        main_total += count

    # 건수 내림차순 정렬
    main_sources.sort(key=lambda x: x.count, reverse=True)

    # Assist RAG: news_articles를 source별 GROUP BY
    news_query = select(
        NewsArticle.source,
        func.count().label("cnt"),
        func.count(case((NewsArticle.is_indexed == True, 1))).label("indexed_cnt"),  # noqa: E712
    ).group_by(NewsArticle.source)
    news_rows = (await db.execute(news_query)).all()

    assist_sources: list[AssistRagSourceItem] = []
    assist_total = 0
    for row in news_rows:
        label = _ASSIST_RAG_LABELS.get(row.source, row.source)
        assist_sources.append(
            AssistRagSourceItem(
                source=row.source,
                label=label,
                count=row.cnt,
                indexed_count=row.indexed_cnt,
            )
        )
        assist_total += row.cnt

    # 건수 내림차순 정렬
    assist_sources.sort(key=lambda x: x.count, reverse=True)

    grand_total = main_total + assist_total
    contribution_pct = (assist_total / grand_total * 100) if grand_total > 0 else 0.0

    return RagContributionStats(
        main_rag_total=main_total,
        main_rag_sources=main_sources,
        assist_rag_total=assist_total,
        assist_rag_sources=assist_sources,
        assist_contribution_percent=round(contribution_pct, 2),
    )
