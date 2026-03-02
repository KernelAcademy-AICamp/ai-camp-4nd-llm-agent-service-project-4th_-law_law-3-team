"""뉴스 파이프라인 CLI"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import date, timedelta

from app.tools.news_pipeline.config import NewsPipelineConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="법률 뉴스 수집/요약 파이프라인",
    )
    parser.add_argument(
        "--date", type=str, default="yesterday",
        help="수집 대상 날짜 (YYYY-MM-DD 또는 'today'/'yesterday')",
    )
    parser.add_argument(
        "--date-range", nargs=2, type=str, metavar=("START", "END"),
        help="날짜 범위 (YYYY-MM-DD YYYY-MM-DD)",
    )
    parser.add_argument(
        "--source", type=str, choices=["lawtimes", "naver"],
        help="특정 소스만 실행",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="수집만 실행 (요약/저장 건너뜀)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="상세 로그 출력",
    )
    # v0.2.0: Red Team 피드백 — 인덱싱 복구 CLI
    parser.add_argument(
        "--reindex-pending", action="store_true",
        help="is_indexed=false인 문서를 일괄 재처리 (LanceDB 청킹/임베딩)",
    )
    return parser.parse_args()


def resolve_date(date_str: str) -> date:
    """날짜 문자열 → date 변환"""
    if date_str == "today":
        return date.today()
    if date_str == "yesterday":
        return date.today() - timedelta(days=1)
    return date.fromisoformat(date_str)


async def main() -> None:
    args = parse_args()

    # 로깅 설정
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from app.services.service_function.news_pipeline_service import run_pipeline

    config = NewsPipelineConfig.from_settings()
    sources = [args.source] if args.source else None

    # v0.2.0: 인덱싱 복구 모드
    if args.reindex_pending:
        from app.services.service_function.news_pipeline_service import reindex_pending
        count = await reindex_pending(config)
        logging.info("재인덱싱 완료: %d건", count)
        return

    if args.date_range:
        start = date.fromisoformat(args.date_range[0])
        end = date.fromisoformat(args.date_range[1])
        current = start
        while current <= end:
            logging.info("━━━ %s 수집 시작 ━━━", current)
            await run_pipeline(current, config=config, sources=sources)
            current += timedelta(days=1)
    else:
        target = resolve_date(args.date)
        await run_pipeline(target, config=config, sources=sources)


if __name__ == "__main__":
    asyncio.run(main())
