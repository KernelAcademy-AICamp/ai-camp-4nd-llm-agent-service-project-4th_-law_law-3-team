"""뉴스 파이프라인 오케스트레이터"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import date, datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import async_session_factory
from app.models.news_article import NewsArticle
from app.tools.news_pipeline.chunker import Chunker
from app.tools.news_pipeline.cleaner import Cleaner
from app.tools.news_pipeline.config import NewsPipelineConfig
from app.tools.news_pipeline.deduplicator import Deduplicator
from app.tools.news_pipeline.models import (
    PipelineError,
    PipelineResult,
    ProcessedArticle,
    SourceStat,
)
from app.tools.news_pipeline.pii_filter import PIIFilter
from app.tools.news_pipeline.reference_validator import ReferenceValidator
from app.tools.news_pipeline.reporter import PipelineReporter
from app.tools.news_pipeline.sources import BaseNewsSource
from app.tools.news_pipeline.sources.lawtimes_source import LawtimesSource
from app.tools.news_pipeline.sources.naver_news_source import NaverNewsSource
from app.tools.news_pipeline.summarizer import Summarizer

logger = logging.getLogger(__name__)

# 배치 저장 크기
DB_BATCH_SIZE = 50


async def run_pipeline(
    target_date: date,
    config: NewsPipelineConfig | None = None,
    sources: list[str] | None = None,
) -> PipelineResult:
    """메인 파이프라인 실행

    Args:
        target_date: 수집 대상 날짜
        config: 파이프라인 설정 (None이면 환경 변수에서 로드)
        sources: 실행할 소스 목록 (None이면 활성화된 전체)

    Returns:
        PipelineResult: 실행 결과 (리포트용)
    """
    if config is None:
        config = NewsPipelineConfig.from_settings()

    result = PipelineResult(
        run_id=str(uuid.uuid4()),
        started_at=datetime.now(),
    )

    # 컴포넌트 초기화
    available_sources = _build_sources(config, sources)
    dedup = Deduplicator()
    cleaner = Cleaner(config)
    pii_filter = PIIFilter()
    summarizer = Summarizer(config)
    ref_validator = ReferenceValidator()
    chunker = Chunker(config)
    reporter = PipelineReporter()

    async with async_session_factory() as db:
        # === Phase 1: 수집 + 중복제거 ===
        all_raw = []
        for source in available_sources:
            try:
                raw_articles = await source.fetch(target_date)
                src_name = source.source_type.value
                result.source_stats[src_name] = SourceStat(collected=len(raw_articles))
                all_raw.extend(raw_articles)
            except Exception as exc:
                src_name = source.source_type.value
                result.source_stats[src_name] = SourceStat(failed=1, errors=[str(exc)])
                result.errors.append(PipelineError(
                    stage="collect", article_url=None,
                    error_type=type(exc).__name__, error_message=str(exc),
                ))
                logger.error("소스 [%s] 수집 실패: %s", src_name, exc)

        result.total_collected = len(all_raw)

        unique_articles = await dedup.deduplicate(all_raw, db)
        result.total_deduplicated = result.total_collected - len(unique_articles)

        # === Phase 2: 정제 + PII + 요약 ===
        processed: list[ProcessedArticle] = []
        for raw in unique_articles:
            try:
                # 정제
                cleaned = cleaner.clean(raw)
                if cleaned is None:
                    continue
                result.total_cleaned += 1

                # Stage 3 중복 체크 (본문 해시)
                if await dedup.check_content_hash(cleaned.content_hash, db):
                    continue

                # PII 마스킹
                cleaned.cleaned_text, _ = pii_filter.mask(cleaned.cleaned_text)

                # 요약 생성
                summary = await summarizer.summarize(cleaned)
                result.total_summarized += 1

                # Cross-Reference 검증
                summary = await ref_validator.validate_and_filter(summary, db)

                # ProcessedArticle 조합
                doc_id = hashlib.sha256(cleaned.url.encode()).hexdigest()
                processed.append(ProcessedArticle(
                    doc_id=doc_id,
                    url=cleaned.url,
                    title=cleaned.title,
                    cleaned_text=cleaned.cleaned_text,
                    content_hash=cleaned.content_hash,
                    source=cleaned.source,
                    publisher=cleaned.publisher,
                    published_at=cleaned.published_at,
                    collected_at=datetime.now(),
                    author=cleaned.author,
                    section=cleaned.section,
                    tags=cleaned.tags,
                    summary=summary,
                ))

            except Exception as exc:
                result.errors.append(PipelineError(
                    stage="process", article_url=raw.url,
                    error_type=type(exc).__name__, error_message=str(exc),
                ))
                logger.warning("기사 처리 실패 [%s]: %s", raw.url, exc)

        # === Phase 3: DB 저장 + 청킹 ===
        for i in range(0, len(processed), DB_BATCH_SIZE):
            batch = processed[i:i + DB_BATCH_SIZE]
            try:
                await _store_batch(batch, db)
                result.total_stored += len(batch)
            except Exception as exc:
                result.errors.append(PipelineError(
                    stage="store", article_url=None,
                    error_type=type(exc).__name__, error_message=str(exc),
                ))

        # 청킹 + LanceDB
        for article in processed:
            try:
                chunks = chunker.create_chunks(article)
                stored = await chunker.embed_and_store(chunks, config.lancedb_table)
                result.total_chunked += stored

                # is_indexed 플래그 업데이트
                await _mark_indexed(article.doc_id, db)
            except Exception as exc:
                result.errors.append(PipelineError(
                    stage="chunk", article_url=article.url,
                    error_type=type(exc).__name__, error_message=str(exc),
                ))

        await db.commit()

    # === Phase 4: 리포트 ===
    result.finished_at = datetime.now()
    reporter.generate(result)

    return result


async def reindex_pending(
    config: NewsPipelineConfig | None = None,
) -> int:
    """is_indexed=false인 문서를 일괄 재처리 (LanceDB 청킹/임베딩)

    CLI --reindex-pending 옵션에서 호출.

    Returns:
        재인덱싱된 문서 수
    """
    if config is None:
        config = NewsPipelineConfig.from_settings()

    chunker = Chunker(config)
    count = 0

    async with async_session_factory() as db:
        result = await db.execute(
            select(NewsArticle).where(NewsArticle.is_indexed == False)  # noqa: E712
        )
        pending_articles = result.scalars().all()

        for article in pending_articles:
            try:
                from app.tools.news_pipeline.models import ArticleSummary

                doc_id = str(article.id)
                summary = ArticleSummary(
                    one_liner=str(article.summary_one_liner),
                    issues=list(article.summary_issues or []),
                    laws=list(article.summary_laws or []),
                    cases=list(article.summary_cases or []),
                    institutions=list(article.summary_institutions or []),
                    implications=list(article.summary_implications or []),
                )

                processed = ProcessedArticle(
                    doc_id=doc_id,
                    url=str(article.url),
                    title=str(article.title),
                    cleaned_text=str(article.cleaned_text),
                    content_hash=str(article.content_hash),
                    source=str(article.source),  # type: ignore[arg-type]
                    publisher=str(article.publisher),
                    published_at=article.published_at,  # type: ignore[arg-type]
                    collected_at=article.collected_at,  # type: ignore[arg-type]
                    author=str(article.author) if article.author else None,
                    section=str(article.section) if article.section else None,
                    tags=list(article.tags) if article.tags else [],
                    summary=summary,
                )

                chunks = chunker.create_chunks(processed)
                await chunker.embed_and_store(chunks, config.lancedb_table)
                await _mark_indexed(doc_id, db)
                count += 1

            except Exception as exc:
                logger.warning("재인덱싱 실패 [%s]: %s", article.id, exc)

        await db.commit()

    logger.info("재인덱싱 완료: %d건", count)
    return count


def _build_sources(
    config: NewsPipelineConfig, filter_sources: list[str] | None,
) -> list[BaseNewsSource]:
    """활성화된 소스 인스턴스 목록 생성"""
    all_sources: list[BaseNewsSource] = [
        LawtimesSource(config),
        NaverNewsSource(config),
    ]

    available = [s for s in all_sources if s.is_available]

    if filter_sources:
        available = [
            s for s in available
            if s.source_type.value in filter_sources
        ]

    return available


async def _store_batch(
    articles: list[ProcessedArticle], db: AsyncSession,
) -> None:
    """배치 단위 DB 저장 (ON CONFLICT 멱등)"""
    from sqlalchemy.dialects.postgresql import insert

    for article in articles:
        stmt = insert(NewsArticle).values(
            id=article.doc_id,
            source=article.source.value,
            publisher=article.publisher,
            title=article.title,
            author=article.author,
            published_at=article.published_at,
            collected_at=article.collected_at,
            url=article.url,
            section=article.section,
            tags=article.tags,
            cleaned_text=article.cleaned_text,
            summary_one_liner=article.summary.one_liner,
            summary_issues=article.summary.issues,
            summary_laws=article.summary.laws,
            summary_cases=article.summary.cases,
            summary_institutions=article.summary.institutions,
            summary_implications=article.summary.implications,
            content_hash=article.content_hash,
            disclaimer=article.disclaimer,
            schema_version=article.schema_version,
        ).on_conflict_do_update(
            index_elements=["id"],
            set_={
                "content_hash": article.content_hash,
                "cleaned_text": article.cleaned_text,
                "summary_one_liner": article.summary.one_liner,
                "updated_at": datetime.utcnow(),
            },
        )
        await db.execute(stmt)

    await db.flush()


async def _mark_indexed(doc_id: str, db: AsyncSession) -> None:
    """LanceDB 임베딩 완료 플래그 설정"""
    await db.execute(
        update(NewsArticle).where(NewsArticle.id == doc_id).values(is_indexed=True)
    )
