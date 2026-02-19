"""로깅 설정 유틸리티.

스크립트 공통 로깅 포맷을 통합합니다.

Usage:
    from scripts.common.logging_config import setup_logging
    logger = setup_logging(__name__)
"""

from __future__ import annotations

import logging


def setup_logging(
    name: str,
    level: int = logging.INFO,
    fmt: str = "%(asctime)s [%(levelname)s] %(message)s",
    datefmt: str = "%H:%M:%S",
) -> logging.Logger:
    """로깅 설정 후 Logger 반환.

    Args:
        name: 로거 이름 (보통 __name__)
        level: 로그 레벨 (기본: INFO)
        fmt: 로그 포맷
        datefmt: 날짜/시간 포맷

    Returns:
        설정된 Logger 인스턴스
    """
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)
    return logging.getLogger(name)
