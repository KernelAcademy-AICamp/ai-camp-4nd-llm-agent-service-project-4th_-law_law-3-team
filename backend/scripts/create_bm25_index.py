"""
BM25 인덱스 생성 스크립트

fts_index.search_text 컬럼에 pg_textsearch BM25 인덱스를 생성합니다.
병렬 빌드 설정 → CONCURRENTLY 시도 → 실패 시 일반 CREATE INDEX fallback.

사전 조건:
    1. PostgreSQL 17 + pg_textsearch 확장 설치
    2. fts_index 테이블에 search_text 데이터 적재 완료
       (uv run python -m scripts.ingest.cli --type all --step fts --reset)

사용법:
    uv run python scripts/create_bm25_index.py
    uv run python scripts/create_bm25_index.py --drop   # 기존 인덱스 삭제 후 재생성
    uv run python scripts/create_bm25_index.py --check   # 인덱스 존재 여부만 확인
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

_backend_root = Path(__file__).parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))

from sqlalchemy import text

from app.core.database import sync_session_factory

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

INDEX_NAME = "idx_fts_bm25"
TABLE_NAME = "fts_index"
COLUMN_NAME = "search_text"
TEXT_CONFIG = "simple"


def _index_exists(session: object) -> bool:
    """BM25 인덱스 존재 여부 확인."""
    result = session.execute(  # type: ignore[union-attr]
        text("SELECT 1 FROM pg_indexes WHERE indexname = :name LIMIT 1"),
        {"name": INDEX_NAME},
    )
    return result.scalar_one_or_none() is not None


def _check_search_text_count(session: object) -> int:
    """search_text가 채워진 행 수 확인."""
    result = session.execute(  # type: ignore[union-attr]
        text(
            f"SELECT COUNT(*) FROM {TABLE_NAME} "  # noqa: S608
            f"WHERE {COLUMN_NAME} IS NOT NULL AND {COLUMN_NAME} != ''"
        ),
    )
    return result.scalar_one()  # type: ignore[return-value]


def check_index() -> None:
    """인덱스 상태 확인."""
    with sync_session_factory() as session:
        exists = _index_exists(session)
        count = _check_search_text_count(session)

        if exists:
            logger.info("✅ BM25 인덱스 '%s' 존재", INDEX_NAME)
        else:
            logger.info("❌ BM25 인덱스 '%s' 없음", INDEX_NAME)

        logger.info("search_text 적재 건수: %d", count)


def drop_index() -> None:
    """기존 BM25 인덱스 삭제."""
    with sync_session_factory() as session:
        if not _index_exists(session):
            logger.info("인덱스 '%s' 없음 — 삭제 불필요", INDEX_NAME)
            return

        session.execute(text(f"DROP INDEX IF EXISTS {INDEX_NAME}"))  # noqa: S608
        session.commit()
        logger.info("인덱스 '%s' 삭제 완료", INDEX_NAME)


def create_index() -> None:
    """BM25 인덱스 생성 (병렬 빌드 설정 포함)."""
    with sync_session_factory() as session:
        if _index_exists(session):
            logger.info("인덱스 '%s' 이미 존재 — 건너뜀", INDEX_NAME)
            return

        count = _check_search_text_count(session)
        if count == 0:
            logger.error(
                "search_text 데이터가 없습니다. 먼저 FTS 재빌드를 실행하세요:\n"
                "  uv run python -m scripts.ingest.cli --type all --step fts --reset"
            )
            sys.exit(1)

        logger.info("search_text 적재 건수: %d", count)

        create_sql = (
            f"CREATE INDEX {INDEX_NAME} ON {TABLE_NAME} "
            f"USING bm25({COLUMN_NAME}) WITH (text_config='{TEXT_CONFIG}')"
        )

        # 병렬 빌드 시도 → shm 부족 시 비병렬 fallback
        session.execute(text("SET max_parallel_maintenance_workers = 4"))
        session.execute(text("SET maintenance_work_mem = '256MB'"))
        logger.info("병렬 빌드 설정: workers=4, work_mem=256MB")

        start = time.time()
        try:
            session.execute(text(create_sql))
            session.commit()
            logger.info(
                "BM25 인덱스 생성 완료 (병렬): %.1f초", time.time() - start
            )
        except Exception as e:
            session.rollback()
            logger.warning("병렬 빌드 실패, 비병렬 모드로 재시도: %s", e)

            session.execute(text("SET max_parallel_maintenance_workers = 0"))
            session.execute(text("SET maintenance_work_mem = '64MB'"))

            start = time.time()
            session.execute(text(create_sql))
            session.commit()
            logger.info(
                "BM25 인덱스 생성 완료 (비병렬): %.1f초", time.time() - start
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="BM25 인덱스 생성")
    parser.add_argument("--drop", action="store_true", help="기존 인덱스 삭제 후 재생성")
    parser.add_argument("--check", action="store_true", help="인덱스 상태만 확인")
    args = parser.parse_args()

    if args.check:
        check_index()
        return

    if args.drop:
        drop_index()

    create_index()
    check_index()


if __name__ == "__main__":
    main()
