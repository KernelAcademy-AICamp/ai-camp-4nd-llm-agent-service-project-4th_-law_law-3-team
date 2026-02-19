"""배치 처리 유틸리티.

대량 데이터를 일정 크기 단위로 분할 처리하는 공통 패턴을 제공합니다.

Usage:
    from scripts.common.batch import batch_iterate, process_in_batches
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def batch_iterate(
    items: Iterable[T],
    batch_size: int = 1000,
) -> Iterable[list[T]]:
    """아이템을 batch_size 단위로 분할하여 yield.

    Args:
        items: 분할할 아이템 (list 또는 iterable)
        batch_size: 배치 크기 (기본 1000)

    Yields:
        batch_size 크기의 리스트
    """
    # list인 경우 슬라이싱 (효율적)
    if isinstance(items, (list, tuple)):
        for i in range(0, len(items), batch_size):
            yield items[i : i + batch_size]
        return

    # 일반 iterable인 경우
    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def process_in_batches(
    items: list[T],
    process_fn: Callable[[list[T]], int],
    batch_size: int = 1000,
    label: str = "",
    log: logging.Logger | None = None,
) -> dict[str, int]:
    """배치 단위로 처리하면서 진행률을 로깅.

    Args:
        items: 처리할 아이템 리스트
        process_fn: 배치를 받아 처리한 건수를 반환하는 함수
        batch_size: 배치 크기 (기본 1000)
        label: 로그에 표시할 라벨
        log: 사용할 로거 (None이면 모듈 로거 사용)

    Returns:
        {"total": 전체 건수, "processed": 처리 건수, "batches": 배치 수}
    """
    _log = log or logger
    total = len(items)
    processed = 0
    batch_count = 0
    start_time = time.time()

    for batch in batch_iterate(items, batch_size):
        count = process_fn(batch)
        processed += count
        batch_count += 1
        elapsed = time.time() - start_time
        pct = processed / total * 100 if total > 0 else 0
        prefix = f"  [{label}] " if label else "  "
        _log.info(
            "%s진행: %d/%d (%.1f%%) [%.1fs]",
            prefix, processed, total, pct, elapsed,
        )

    return {"total": total, "processed": processed, "batches": batch_count}
