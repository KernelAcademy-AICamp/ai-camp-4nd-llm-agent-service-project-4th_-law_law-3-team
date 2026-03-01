"""
법령 조문 데이터 PostgreSQL 로드 스크립트

data/ingest_source/law_v3.json → law_articles 테이블
법령 JSON의 조문 배열을 조문 단위로 분리하여 적재.

Usage:
    uv run python scripts/load_law_articles_data.py           # 로드
    uv run python scripts/load_law_articles_data.py --reset    # 삭제 후 재로드
    uv run python scripts/load_law_articles_data.py --verify   # 검증만
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import func, text  # noqa: E402
from sqlalchemy.dialects.postgresql import insert  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.models.law_article import LawArticle  # noqa: E402
from scripts.common.db import create_sync_session_factory  # noqa: E402
from scripts.common.json_loader import load_json_file  # noqa: E402
from scripts.common.logging_config import setup_logging  # noqa: E402
from scripts.ingest.types.law import (  # noqa: E402
    _extract_article_body,
)

logger = setup_logging(__name__)

# 데이터 소스 경로 (ingest 파이프라인과 동일)
DATA_DIR = PROJECT_ROOT.parent / "data" / "ingest_source"
LAW_FILE = DATA_DIR / "law_v3.json"

BATCH_SIZE = 1000


def _build_article_content(article: dict[str, Any]) -> str:
    """단일 조문 → 조문내용 + 항·호 텍스트 결합.

    조문내용(본문)과 항·호를 합쳐서 하나의 텍스트로 만든다.
    """
    parts: list[str] = []

    # 조문내용 (조문 본문)
    content_text = article.get("조문내용", "")
    if content_text:
        parts.append(str(content_text))

    # 항·호 (기존 파싱 함수 재사용)
    parts.extend(_extract_article_body(article))

    return "\n".join(parts)


def extract_articles(law_item: dict[str, Any]) -> list[dict[str, str]]:
    """법령 JSON 아이템에서 조문 레코드 리스트 추출.

    Returns:
        [{"law_id": "...", "article_number": "...",
          "article_title": "...", "article_content": "..."}, ...]
    """
    law_id = str(law_item.get("법령ID", "") or law_item.get("law_id", ""))
    if not law_id:
        return []

    articles_data = law_item.get("조문")
    if not articles_data or not isinstance(articles_data, list):
        return []

    records: list[dict[str, str]] = []
    for article in articles_data:
        if not isinstance(article, dict):
            continue

        # LanceDB와 동일한 키 형식: JSON 조문번호 원본값 그대로
        article_number = str(article.get("조문번호", "")).strip()
        if not article_number:
            continue

        article_title = str(article.get("조문제목", "") or "").strip()
        article_content = _build_article_content(article)

        if not article_content.strip():
            continue

        records.append({
            "law_id": law_id,
            "article_number": article_number,
            "article_title": article_title or None,
            "article_content": article_content,
        })

    return records


def load_to_db(
    session_factory: sessionmaker,
    law_items: list[dict[str, Any]],
    reset: bool = False,
) -> int:
    """조문 데이터를 DB에 로드."""
    with session_factory() as db:
        if reset:
            count = db.query(func.count(LawArticle.id)).scalar() or 0
            db.execute(
                text("TRUNCATE TABLE law_articles RESTART IDENTITY CASCADE")
            )
            db.commit()
            logger.info(f"기존 데이터 {count}건 삭제 완료")

        # 모든 법령에서 조문 레코드 추출
        all_records: list[dict[str, Any]] = []
        for item in law_items:
            all_records.extend(extract_articles(item))

        logger.info(
            f"법령 {len(law_items)}건에서 조문 {len(all_records)}건 추출"
        )

        if not all_records:
            logger.warning("추출된 조문이 없습니다")
            return 0

        total_loaded = 0
        start_time = time.time()

        for i in range(0, len(all_records), BATCH_SIZE):
            batch = all_records[i : i + BATCH_SIZE]

            # ON CONFLICT (law_id, article_number) DO UPDATE 멱등성 보장
            stmt = insert(LawArticle).values(batch)
            stmt = stmt.on_conflict_do_update(
                constraint="uq_law_articles_law_article",
                set_={
                    "article_title": stmt.excluded.article_title,
                    "article_content": stmt.excluded.article_content,
                },
            )
            db.execute(stmt)
            db.commit()

            total_loaded += len(batch)
            elapsed = time.time() - start_time
            logger.info(
                f"  진행: {total_loaded}/{len(all_records)} "
                f"({total_loaded / len(all_records) * 100:.1f}%) "
                f"[{elapsed:.1f}s]"
            )

        return total_loaded


def verify_data(session_factory: sessionmaker) -> None:
    """로드된 데이터 검증."""
    with session_factory() as db:
        total = db.query(func.count(LawArticle.id)).scalar() or 0
        law_count = (
            db.query(func.count(func.distinct(LawArticle.law_id))).scalar()
            or 0
        )

        logger.info("=" * 50)
        logger.info("law_articles 테이블 검증 결과")
        logger.info("=" * 50)
        logger.info(f"총 조문 수: {total:,}건")
        logger.info(f"법령 수 (고유 law_id): {law_count:,}건")

        if total == 0:
            logger.warning("데이터가 없습니다. 로드를 먼저 실행하세요.")
            return

        # 법령별 조문 수 상위 5개
        top_laws = (
            db.query(
                LawArticle.law_id,
                func.count(LawArticle.id).label("cnt"),
            )
            .group_by(LawArticle.law_id)
            .order_by(func.count(LawArticle.id).desc())
            .limit(5)
            .all()
        )

        logger.info("\n법령별 조문 수 상위 5개:")
        for law_id, cnt in top_laws:
            logger.info(f"  law_id={law_id}: {cnt}건")

        # article_number 형식 샘플
        samples = (
            db.query(
                LawArticle.law_id,
                LawArticle.article_number,
                LawArticle.article_title,
            )
            .limit(5)
            .all()
        )

        logger.info("\narticle_number 형식 샘플:")
        for law_id, article_number, title in samples:
            logger.info(
                f"  law_id={law_id}, "
                f"article_number='{article_number}', "
                f"title='{title or ''}'"
            )

        logger.info("=" * 50)


def main() -> None:
    """메인 실행."""
    parser = argparse.ArgumentParser(
        description="법령 조문 데이터 PostgreSQL 로드"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="기존 데이터 삭제 후 재로드",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="검증만 실행 (로드 안 함)",
    )
    args = parser.parse_args()

    session_factory = create_sync_session_factory()

    if args.verify:
        verify_data(session_factory)
        return

    if not LAW_FILE.exists():
        logger.error(f"파일이 없습니다: {LAW_FILE}")
        sys.exit(1)

    logger.info(f"법령 JSON 로드 중: {LAW_FILE}")
    data = load_json_file(str(LAW_FILE))

    # law_v3.json은 리스트 형태
    if isinstance(data, list):
        law_items = data
    elif isinstance(data, dict):
        law_items = data.get("법령", data.get("items", []))
    else:
        logger.error("지원하지 않는 JSON 구조입니다")
        sys.exit(1)

    logger.info(f"법령 {len(law_items)}건 로드 완료")

    start = time.time()
    count = load_to_db(session_factory, law_items, reset=args.reset)
    elapsed = time.time() - start

    logger.info(f"적재 완료: {count:,}건 [{elapsed:.1f}s]")

    # 적재 후 자동 검증
    verify_data(session_factory)


if __name__ == "__main__":
    main()
