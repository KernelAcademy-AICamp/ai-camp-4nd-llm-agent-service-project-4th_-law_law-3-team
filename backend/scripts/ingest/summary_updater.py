"""
ai_summary 컬럼 일괄 업데이트

JSON 소스에서 요약 필드만 읽어 PostgreSQL ORM 테이블의 ai_summary 컬럼을 갱신합니다.
FTS tsvector는 ai_summary를 사용하지 않으므로 재빌드하지 않습니다.
MeCab 초기화도 불필요합니다.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

_backend_root = Path(__file__).parent.parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))

from sqlalchemy import Table, text

from app.core.database import sync_session_factory
from scripts.ingest.config import IngestConfig, get_config, list_configs
from scripts.ingest.db_writer import _load_json

logger = logging.getLogger(__name__)

BATCH_SIZE = 1000


def _update_summaries_for_type(
    config: IngestConfig,
    batch_size: int = BATCH_SIZE,
) -> dict[str, Any]:
    """단일 타입의 ai_summary 컬럼을 JSON 소스에서 일괄 업데이트

    Args:
        config: 인제스트 설정
        batch_size: 배치 크기

    Returns:
        통계 dict: total, updated, skipped, errors
    """
    items = _load_json(config.source_path)

    stats: dict[str, Any] = {
        "type": config.name,
        "label": config.data_type_label,
        "total": len(items),
        "updated": 0,
        "skipped": 0,
        "errors": 0,
    }

    if not items:
        return stats

    start_time = time.time()

    orm_table: Table = config.orm_class.__table__  # type: ignore[assignment]
    table_name = str(orm_table.name)
    id_col_name = config.orm_id_attr

    update_sql = text(
        f"UPDATE {table_name} "  # noqa: S608
        f"SET ai_summary = :summary, updated_at = NOW() "
        f"WHERE {id_col_name} = :doc_id"
    )

    with sync_session_factory() as session:
        batch: list[dict[str, str]] = []

        for item in items:
            doc_id = str(item.get(config.id_field, ""))
            summary = item.get(config.summary_field) or ""

            if not doc_id:
                stats["errors"] += 1
                continue

            if not summary:
                stats["skipped"] += 1
                continue

            batch.append({"doc_id": doc_id, "summary": summary})

            if len(batch) >= batch_size:
                session.execute(update_sql, batch)
                session.commit()
                stats["updated"] += len(batch)
                batch = []

        if batch:
            session.execute(update_sql, batch)
            session.commit()
            stats["updated"] += len(batch)

    elapsed = time.time() - start_time
    stats["elapsed_sec"] = round(elapsed, 1)

    logger.info(
        "[%s] ai_summary 업데이트 완료: %d/%d건, %.1f초",
        config.data_type_label,
        stats["updated"],
        stats["total"],
        elapsed,
    )

    return stats


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # types/ 임포트 → 전체 config 자동 등록
    import scripts.ingest.types  # noqa: F401

    all_types = list_configs()

    parser = argparse.ArgumentParser(
        description="ai_summary 컬럼 일괄 업데이트 (FTS 재빌드 없음)",
    )
    parser.add_argument(
        "--type",
        choices=["all"] + all_types,
        default="all",
        help="업데이트 대상 타입 (기본: all)",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="제외할 타입 (예: --exclude admin_rule treaty)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"배치 크기 (기본: {BATCH_SIZE})",
    )
    args = parser.parse_args()

    # 대상 타입 결정
    if args.type == "all":
        target_types = all_types
    else:
        target_types = [args.type]

    # 제외 필터
    exclude_set = set(args.exclude)
    if exclude_set:
        target_types = [t for t in target_types if t not in exclude_set]
        logger.info("제외된 타입: %s", ", ".join(sorted(exclude_set)))

    logger.info("대상 타입: %d개 — %s", len(target_types), ", ".join(target_types))

    # 실행
    results: list[dict[str, Any]] = []
    total_start = time.time()

    for type_name in target_types:
        config = get_config(type_name)
        result = _update_summaries_for_type(config, batch_size=args.batch_size)
        results.append(result)

    total_elapsed = time.time() - total_start

    # 요약 보고
    print("\n" + "=" * 60)
    print("ai_summary 업데이트 결과")
    print("=" * 60)
    print(f"{'타입':<30} {'전체':>8} {'업데이트':>8} {'건너뜀':>8} {'시간':>8}")
    print("-" * 60)

    total_updated = 0
    for r in results:
        print(
            f"{r['label']:<30} {r['total']:>8,} {r['updated']:>8,} "
            f"{r['skipped']:>8,} {r.get('elapsed_sec', 0):>7.1f}s"
        )
        total_updated += r["updated"]

    print("-" * 60)
    print(f"{'합계':<30} {'':>8} {total_updated:>8,} {'':>8} {total_elapsed:>7.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
