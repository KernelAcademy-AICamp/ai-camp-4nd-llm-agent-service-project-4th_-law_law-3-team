"""news_articles.search_text 컬럼 적재 스크립트

news_articles 테이블의 title + summary_one_liner + tags + cleaned_text를
MeCab 토크나이징하여 search_text 컬럼에 저장한다.

Usage:
    cd backend
    uv run python scripts/build_news_search_text.py
    uv run python scripts/build_news_search_text.py --verify
    uv run python scripts/build_news_search_text.py --batch-size 500
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# 프로젝트 루트 설정
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import func, select  # noqa: E402

from app.core.database import sync_session_factory  # noqa: E402
from app.models.news_article import NewsArticle  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# MeCab 토크나이저 (keyword_search.py와 동일 패턴)
_tokenizer = None


def _get_tokenizer():  # type: ignore[no-untyped-def]
    """MeCab 토크나이저 싱글턴."""
    global _tokenizer  # noqa: PLW0603
    if _tokenizer is None:
        from app.tools.vectorstore.lancedb import _get_thread_tokenizer

        _tokenizer = _get_thread_tokenizer()
    return _tokenizer


def _build_search_text(
    title: str,
    summary_one_liner: str,
    tags: list[str] | None,
    cleaned_text: str,
) -> str:
    """텍스트 필드를 MeCab 토크나이징하여 search_text 생성."""
    tokenizer = _get_tokenizer()
    parts = [title, summary_one_liner]
    if tags:
        parts.append(" ".join(tags))
    parts.append(cleaned_text)

    combined = " ".join(parts)
    tokens: list[str] = list(tokenizer.morphs(combined))
    return " ".join(tokens)


def build_all(batch_size: int = 1000) -> int:
    """전체 news_articles의 search_text를 배치 생성."""
    session_factory = sync_session_factory
    processed = 0

    with session_factory() as session:
        # 전체 건수
        total = session.execute(
            select(func.count()).select_from(NewsArticle)
        ).scalar_one()
        logger.info("전체 %d건 처리 시작 (배치: %d)", total, batch_size)

        # search_text가 NULL인 행만 처리
        null_count = session.execute(
            select(func.count())
            .select_from(NewsArticle)
            .where(NewsArticle.search_text.is_(None))
        ).scalar_one()
        logger.info("search_text NULL: %d건", null_count)

        if null_count == 0:
            logger.info("모든 행에 search_text가 이미 존재합니다.")
            return 0

        offset = 0
        start = time.time()

        while offset < total:
            rows = (
                session.execute(
                    select(NewsArticle)
                    .where(NewsArticle.search_text.is_(None))
                    .order_by(NewsArticle.id)
                    .limit(batch_size)
                )
                .scalars()
                .all()
            )

            if not rows:
                break

            for row in rows:
                row.search_text = _build_search_text(
                    title=row.title or "",
                    summary_one_liner=row.summary_one_liner or "",
                    tags=row.tags,
                    cleaned_text=row.cleaned_text or "",
                )
                processed += 1

            session.commit()
            elapsed = time.time() - start
            rate = processed / elapsed if elapsed > 0 else 0
            logger.info(
                "  진행: %d / %d (%.1f건/초)",
                processed,
                null_count,
                rate,
            )

        elapsed = time.time() - start
        logger.info(
            "완료: %d건 처리, %.1f초 소요",
            processed,
            elapsed,
        )

    return processed


def verify() -> None:
    """search_text 적재 상태 확인."""
    session_factory = sync_session_factory

    with session_factory() as session:
        total = session.execute(
            select(func.count()).select_from(NewsArticle)
        ).scalar_one()

        filled = session.execute(
            select(func.count())
            .select_from(NewsArticle)
            .where(NewsArticle.search_text.isnot(None))
        ).scalar_one()

        null_count = total - filled

        logger.info("=== search_text 적재 상태 ===")
        logger.info("전체: %d건", total)
        logger.info("적재: %d건 (%.1f%%)", filled, filled / total * 100 if total else 0)
        logger.info("미적재: %d건", null_count)

        # 샘플 출력
        if filled > 0:
            sample = session.execute(
                select(
                    NewsArticle.id,
                    NewsArticle.title,
                    func.left(NewsArticle.search_text, 100).label("search_text_preview"),
                )
                .where(NewsArticle.search_text.isnot(None))
                .limit(3)
            ).all()
            logger.info("--- 샘플 ---")
            for row in sample:
                logger.info("  [%s] %s", row.id[:8], row.title[:40])
                logger.info("    search_text: %s...", row.search_text_preview)


def main() -> None:
    """메인 엔트리포인트."""
    parser = argparse.ArgumentParser(
        description="news_articles search_text 적재"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="적재 상태만 확인",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="배치 크기 (기본: 1000)",
    )
    args = parser.parse_args()

    if args.verify:
        verify()
    else:
        build_all(batch_size=args.batch_size)
        verify()


if __name__ == "__main__":
    main()
