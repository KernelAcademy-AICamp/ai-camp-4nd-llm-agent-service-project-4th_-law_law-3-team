"""뉴스 기사 JSONL 덤프 스크립트

news_articles 테이블 → data/news_articles.jsonl

Usage:
    uv run python scripts/dump_news_data.py            # 덤프
    uv run python scripts/dump_news_data.py --stats     # 통계만
    uv run python scripts/dump_news_data.py --output /path/to/out.jsonl  # 경로 지정
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import func, select

from app.models.news_article import NewsArticle
from scripts.common.db import create_sync_session_factory
from scripts.common.logging_config import setup_logging

logger = setup_logging(__name__)

DATA_DIR = PROJECT_ROOT.parent / "data"
NEWS_DIR = DATA_DIR / "news"
DEFAULT_OUTPUT = NEWS_DIR / "news_articles.jsonl"

BATCH_SIZE = 500


def _dt(val: object) -> str | None:
    """datetime 컬럼 값을 ISO 문자열로 변환."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val.isoformat()
    return str(val)


def _serialize_row(row: NewsArticle) -> dict[str, object]:
    """ORM 객체를 JSON 직렬화 가능한 dict로 변환."""
    return {
        "id": row.id,
        "source": row.source,
        "publisher": row.publisher,
        "title": row.title,
        "author": row.author,
        "published_at": _dt(row.published_at),
        "collected_at": _dt(row.collected_at),
        "url": row.url,
        "section": row.section,
        "tags": row.tags or [],
        "cleaned_text": row.cleaned_text,
        "summary_one_liner": row.summary_one_liner,
        "summary_issues": row.summary_issues or [],
        "summary_laws": row.summary_laws or [],
        "summary_cases": row.summary_cases or [],
        "summary_institutions": row.summary_institutions or [],
        "summary_implications": row.summary_implications or [],
        "content_hash": row.content_hash,
        "disclaimer": row.disclaimer,
        "schema_version": row.schema_version,
        "is_indexed": row.is_indexed,
        "created_at": _dt(row.created_at),
        "updated_at": _dt(row.updated_at),
    }


def dump_news(output_path: Path) -> int:
    """news_articles 테이블을 JSONL 파일로 덤프.

    Returns:
        덤프된 레코드 수
    """
    session_factory = create_sync_session_factory()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with session_factory() as db:
        # 전체 건수 확인
        count = db.execute(
            select(func.count()).select_from(NewsArticle)
        ).scalar_one()
        logger.info("news_articles 테이블: %d건", count)

        if count == 0:
            logger.warning("덤프할 데이터가 없습니다")
            return 0

        # 배치 단위 조회 → JSONL 쓰기
        with open(output_path, "w", encoding="utf-8") as f:
            offset = 0
            while offset < count:
                rows = (
                    db.execute(
                        select(NewsArticle)
                        .order_by(NewsArticle.published_at.desc())
                        .offset(offset)
                        .limit(BATCH_SIZE)
                    )
                    .scalars()
                    .all()
                )
                for row in rows:
                    line = json.dumps(
                        _serialize_row(row), ensure_ascii=False
                    )
                    f.write(line + "\n")
                    total += 1

                offset += BATCH_SIZE
                logger.info("  진행: %d / %d", min(offset, count), count)

    logger.info("덤프 완료: %d건 → %s", total, output_path)
    return total


def show_stats() -> None:
    """news_articles 테이블 통계 출력."""
    session_factory = create_sync_session_factory()
    with session_factory() as db:
        total = db.execute(
            select(func.count()).select_from(NewsArticle)
        ).scalar_one()

        sources = db.execute(
            select(NewsArticle.source, func.count())
            .group_by(NewsArticle.source)
            .order_by(func.count().desc())
        ).all()

        date_range = db.execute(
            select(
                func.min(NewsArticle.published_at),
                func.max(NewsArticle.published_at),
            )
        ).one()

    print(f"\n{'='*50}")
    print("news_articles 통계")
    print(f"{'='*50}")
    print(f"총 건수: {total:,}건")
    print(f"기간: {date_range[0]} ~ {date_range[1]}")
    print("\n소스별:")
    for source, cnt in sources:
        print(f"  {source}: {cnt:,}건")
    print(f"{'='*50}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="news_articles 테이블을 JSONL 파일로 덤프"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"출력 파일 경로 (기본: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="통계만 출력 (덤프 안 함)",
    )
    args = parser.parse_args()

    if args.stats:
        show_stats()
        return

    dumped = dump_news(args.output)
    if dumped > 0:
        file_size_mb = args.output.stat().st_size / 1024 / 1024
        print(f"\n덤프 완료: {dumped:,}건, {file_size_mb:.1f}MB → {args.output}")


if __name__ == "__main__":
    main()
