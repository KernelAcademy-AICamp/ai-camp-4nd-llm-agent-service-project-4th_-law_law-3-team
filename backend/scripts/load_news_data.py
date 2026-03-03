"""뉴스 기사 JSONL → PostgreSQL 로드 스크립트

data/news_articles.jsonl → news_articles 테이블

Usage:
    uv run python scripts/load_news_data.py            # 로드
    uv run python scripts/load_news_data.py --reset     # 삭제 후 재로드
    uv run python scripts/load_news_data.py --verify    # 검증만
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.models.news_article import NewsArticle
from scripts.common.db import create_sync_session_factory
from scripts.common.logging_config import setup_logging

logger = setup_logging(__name__)

DATA_DIR = PROJECT_ROOT.parent / "data"
NEWS_DIR = DATA_DIR / "news"
DEFAULT_INPUT = NEWS_DIR / "news_articles.jsonl"

BATCH_SIZE = 1000


def _parse_datetime(val: str | None) -> datetime | None:
    """ISO 형식 문자열을 datetime으로 변환."""
    if val is None:
        return None
    return datetime.fromisoformat(val)


def _str_or_none(val: object) -> str | None:
    """object를 str | None으로 변환."""
    if val is None:
        return None
    return str(val)


def _record_to_dict(record: dict[str, object]) -> dict[str, object]:
    """JSONL 레코드를 ORM insert용 dict로 변환."""
    return {
        "id": record["id"],
        "source": record["source"],
        "publisher": record["publisher"],
        "title": record["title"],
        "author": record.get("author"),
        "published_at": _parse_datetime(_str_or_none(record.get("published_at"))),
        "collected_at": _parse_datetime(_str_or_none(record.get("collected_at"))),
        "url": record["url"],
        "section": record.get("section"),
        "tags": record.get("tags") or [],
        "cleaned_text": record["cleaned_text"],
        "summary_one_liner": record["summary_one_liner"],
        "summary_issues": record.get("summary_issues") or [],
        "summary_laws": record.get("summary_laws") or [],
        "summary_cases": record.get("summary_cases") or [],
        "summary_institutions": record.get("summary_institutions") or [],
        "summary_implications": record.get("summary_implications") or [],
        "content_hash": record["content_hash"],
        "disclaimer": record.get(
            "disclaimer",
            "본 문서는 기사 요약이며 법령/판례 원문이 아닙니다",
        ),
        "schema_version": record.get("schema_version", "1.0"),
        "is_indexed": record.get("is_indexed", False),
        "created_at": _parse_datetime(_str_or_none(record.get("created_at"))),
        "updated_at": _parse_datetime(_str_or_none(record.get("updated_at"))),
    }


def load_news(input_path: Path, *, reset: bool = False) -> int:
    """JSONL 파일을 news_articles 테이블로 로드.

    ON CONFLICT (id) DO UPDATE로 멱등성 보장.

    Returns:
        로드된 레코드 수
    """
    if not input_path.exists():
        logger.error("입력 파일이 없습니다: %s", input_path)
        return 0

    session_factory = create_sync_session_factory()

    with session_factory() as db:
        if reset:
            db.execute(text("TRUNCATE TABLE news_articles"))
            db.commit()
            logger.info("기존 데이터 삭제 완료 (TRUNCATE)")

        # JSONL 읽기 → 배치 insert
        batch: list[dict[str, object]] = []
        total = 0

        with open(input_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                batch.append(_record_to_dict(record))

                if len(batch) >= BATCH_SIZE:
                    total += _upsert_batch(db, batch)
                    batch.clear()

            # 나머지 배치
            if batch:
                total += _upsert_batch(db, batch)

        logger.info("로드 완료: %d건", total)
        return total


def _upsert_batch(db: Session, batch: list[dict[str, object]]) -> int:
    """배치 upsert (ON CONFLICT DO UPDATE)."""
    stmt = insert(NewsArticle).values(batch)
    update_cols = {
        col.name: col
        for col in stmt.excluded
        if col.name not in ("id", "created_at")
    }
    stmt = stmt.on_conflict_do_update(
        index_elements=["id"],
        set_=update_cols,
    )
    db.execute(stmt)
    db.commit()
    logger.info("  배치 upsert: %d건", len(batch))
    return len(batch)


def verify(input_path: Path) -> None:
    """DB와 JSONL 파일의 데이터 정합성 검증."""
    session_factory = create_sync_session_factory()

    # DB 건수
    with session_factory() as db:
        db_count = db.execute(
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

    # JSONL 건수
    file_count = 0
    if input_path.exists():
        with open(input_path, encoding="utf-8") as f:
            file_count = sum(1 for line in f if line.strip())

    print(f"\n{'='*50}")
    print("news_articles 검증 결과")
    print(f"{'='*50}")
    print(f"DB 건수:   {db_count:,}건")
    print(f"JSONL 건수: {file_count:,}건")
    match_icon = "✅" if db_count == file_count else "❌"
    print(f"일치 여부:  {match_icon} {'일치' if db_count == file_count else '불일치'}")
    print(f"\n기간: {date_range[0]} ~ {date_range[1]}")
    print("\n소스별:")
    for source, cnt in sources:
        print(f"  {source}: {cnt:,}건")
    print(f"{'='*50}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="news_articles JSONL → PostgreSQL 로드"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"입력 파일 경로 (기본: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="기존 데이터 삭제 후 재로드",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="검증만 수행 (로드 안 함)",
    )
    args = parser.parse_args()

    if args.verify:
        verify(args.input)
        return

    loaded = load_news(args.input, reset=args.reset)
    if loaded > 0:
        print(f"\n로드 완료: {loaded:,}건")
        verify(args.input)


if __name__ == "__main__":
    main()
