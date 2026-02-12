"""
인제스트 타입 정의

각 데이터 타입별 IngestConfig 설정을 등록합니다.
이 패키지를 import하면 자동으로 모든 타입이 등록됩니다.
"""

from scripts.ingest.types import law, precedent  # noqa: F401

__all__ = ["law", "precedent"]
