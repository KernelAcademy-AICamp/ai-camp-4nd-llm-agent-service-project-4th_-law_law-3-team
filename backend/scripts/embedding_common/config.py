"""
하드웨어 프로필 및 배치 크기 설정

디바이스에 따른 최적 설정을 자동 결정합니다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from scripts.embedding_common.device import DeviceInfo


class HardwareProfile(str, Enum):
    """하드웨어 프로필"""

    DESKTOP = "desktop"  # 5060Ti 등 데스크탑 GPU (충분한 쿨링)
    LAPTOP = "laptop"  # 3060 Laptop 등 (발열 보호 필요)
    MAC = "mac"  # Apple Silicon MPS
    CPU = "cpu"  # CPU only


@dataclass
class OptimalConfig:
    """환경별 최적 설정"""

    batch_size: int
    num_workers: int
    prefetch_factor: int = 2
    gc_interval: int = 10
    temp_monitoring: bool = False
    temp_threshold: int = 85  # 섭씨


def _get_default_config() -> dict[str, str | int]:
    """환경변수에서 기본 설정 로드"""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    return {
        "LANCEDB_URI": os.getenv("LANCEDB_URI", "./lancedb_data"),
        "LANCEDB_TABLE_NAME": os.getenv("LANCEDB_TABLE_NAME", "legal_chunks"),
        "EMBEDDING_MODEL": os.getenv("LOCAL_EMBEDDING_MODEL", "nlpai-lab/KURE-v1"),
        "VECTOR_DIM": 1024,
        "BATCH_SIZE": 100,
        # 판례 청킹
        "PRECEDENT_CHUNK_SIZE": 1250,
        "PRECEDENT_CHUNK_OVERLAP": 125,
        "PRECEDENT_MIN_CHUNK_SIZE": 100,
        # 법령 청킹 (토큰 기반)
        "LAW_MAX_TOKENS": 800,
        "LAW_MIN_TOKENS": 100,
        # 텍스트 처리 제한
        "MAX_TEXT_LENGTH": 4000,
        "DEFAULT_EMPTY_TEXT": "(내용 없음)",
        "MAX_QUERY_LIMIT": 1_000_000,
    }


DEFAULT_CONFIG = _get_default_config()


def get_hardware_profile(
    device_info: Optional["DeviceInfo"] = None,
) -> HardwareProfile:
    """디바이스 정보에서 하드웨어 프로필 결정"""
    if device_info is None:
        from scripts.embedding_common.device import get_device_info

        device_info = get_device_info()

    if device_info.device == "mps":
        return HardwareProfile.MAC
    elif device_info.device == "cpu":
        return HardwareProfile.CPU
    elif device_info.is_laptop:
        return HardwareProfile.LAPTOP
    else:
        return HardwareProfile.DESKTOP


def get_optimal_config(
    device_info: Optional["DeviceInfo"] = None,
    profile: Optional[HardwareProfile] = None,
) -> OptimalConfig:
    """디바이스에 따른 최적 설정 반환"""
    if device_info is None:
        from scripts.embedding_common.device import get_device_info

        device_info = get_device_info()

    if profile is None:
        profile = get_hardware_profile(device_info)

    vram = device_info.vram_gb

    if profile == HardwareProfile.DESKTOP:
        # 데스크탑: 쿨링 충분, 최대 성능
        if vram >= 20:
            return OptimalConfig(
                batch_size=128, num_workers=4, gc_interval=25,
                temp_monitoring=False,
            )
        elif vram >= 14:
            return OptimalConfig(
                batch_size=100, num_workers=4, gc_interval=20,
                temp_monitoring=False,
            )
        elif vram >= 8:
            return OptimalConfig(
                batch_size=70, num_workers=2, gc_interval=15,
                temp_monitoring=False,
            )
        else:
            return OptimalConfig(
                batch_size=50, num_workers=2, gc_interval=10,
                temp_monitoring=False,
            )

    elif profile == HardwareProfile.LAPTOP:
        # 노트북: 발열 보호 필수
        if vram >= 8:
            return OptimalConfig(
                batch_size=50, num_workers=2, gc_interval=10,
                temp_monitoring=True, temp_threshold=85,
            )
        else:
            return OptimalConfig(
                batch_size=30, num_workers=2, gc_interval=5,
                temp_monitoring=True, temp_threshold=85,
            )

    elif profile == HardwareProfile.MAC:
        # Mac: MPS 백엔드, 온도 모니터링 불필요
        if vram >= 12:
            return OptimalConfig(
                batch_size=50, num_workers=0, gc_interval=10,
                temp_monitoring=False,
            )
        else:
            return OptimalConfig(
                batch_size=30, num_workers=0, gc_interval=5,
                temp_monitoring=False,
            )

    else:
        # CPU
        return OptimalConfig(
            batch_size=20, num_workers=2, gc_interval=5,
            temp_monitoring=False,
        )
