"""법률 뉴스 API 라우터"""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.modules.legal_news.schema import (
    NewsArticleResponse,
    NewsCategoryStats,
    NewsListResponse,
    NewsSearchRequest,
    NewsSearchResponse,
    NewsStatsDaily,
    RagContributionStats,
)

router = APIRouter()


@router.get(
    "/list",
    response_model=NewsListResponse,
    summary="뉴스 목록 조회",
)
async def get_news_list(
    source: str | None = Query(None, description="소스 필터 (lawtimes|naver)"),
    published_date: date | None = Query(None, description="발행일 필터"),
    page: int = Query(1, ge=1, description="페이지 번호"),
    page_size: int = Query(20, ge=1, le=100, description="페이지 크기"),
    db: AsyncSession = Depends(get_db),
) -> NewsListResponse:
    """법률 뉴스 목록 조회 (페이지네이션, 소스/날짜 필터)"""
    from app.modules.legal_news.service import get_news_list_service
    return await get_news_list_service(
        db, source=source, published_date=published_date,
        page=page, page_size=page_size,
    )


@router.get(
    "/stats/daily",
    response_model=NewsStatsDaily,
    summary="일별 수집 통계",
)
async def get_news_stats_daily(
    days: int = Query(7, ge=1, le=90, description="조회 기간 (일)"),
    db: AsyncSession = Depends(get_db),
) -> NewsStatsDaily:
    """일별 뉴스 수집 건수 통계 (소스별 구분)"""
    from app.modules.legal_news.service import get_news_stats_daily as svc
    return await svc(db, days=days)


@router.get(
    "/stats/category",
    response_model=NewsCategoryStats,
    summary="카테고리 분포 통계",
)
async def get_news_category_stats(
    days: int | None = Query(None, ge=1, le=90, description="조회 기간 (일, 미지정 시 전체)"),
    db: AsyncSession = Depends(get_db),
) -> NewsCategoryStats:
    """뉴스 기사 카테고리 분포 통계 (기간 필터 지원)"""
    from app.modules.legal_news.service import get_news_category_stats as svc
    return await svc(db, days=days)


@router.get(
    "/stats/rag-contribution",
    response_model=RagContributionStats,
    summary="RAG 기여도 통계",
)
async def get_rag_contribution_stats(
    db: AsyncSession = Depends(get_db),
) -> RagContributionStats:
    """법령 DB vs 뉴스 데이터 비교 — RAG 시스템 기여도"""
    from app.modules.legal_news.service import get_rag_contribution_stats as svc
    return await svc(db)


# NOTE: /{article_id} 는 반드시 정적 경로(/list, /stats/*, /search) 뒤에 위치해야 함.
# FastAPI는 경로를 등록 순서대로 매칭하므로, 이 경로가 앞에 있으면
# /stats/daily 등이 article_id="stats"로 잘못 매칭됨.
@router.get(
    "/{article_id}",
    response_model=NewsArticleResponse,
    summary="뉴스 상세 조회",
)
async def get_news_detail(
    article_id: str,
    db: AsyncSession = Depends(get_db),
) -> NewsArticleResponse:
    """개별 뉴스 기사 상세 조회 (doc_id로 조회)"""
    from app.modules.legal_news.service import get_news_detail_service
    return await get_news_detail_service(db, article_id=article_id)


@router.post(
    "/search",
    response_model=NewsSearchResponse,
    summary="뉴스 하이브리드 검색",
)
async def search_news(
    request: NewsSearchRequest,
    db: AsyncSession = Depends(get_db),
) -> NewsSearchResponse:
    """v0.3.0: 하이브리드 검색 (Vector + FTS + 리랭커)"""
    from app.modules.legal_news.service import search_news_service
    return await search_news_service(db, request=request)
