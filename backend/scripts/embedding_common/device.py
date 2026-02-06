"""
GPU/CPU/MPS 디바이스 감지 및 정보 조회

임베딩 스크립트에서 사용 가능한 최적 디바이스를 자동 감지합니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from scripts.embedding_common.config import OptimalConfig

import torch


@dataclass
class DeviceInfo:
    """디바이스 정보"""

    device: str  # "cuda", "mps", "cpu"
    name: str
    vram_gb: float
    is_laptop: bool = False
    compute_capability: Optional[tuple[int, int]] = None

    def __str__(self) -> str:
        return f"{self.name} ({self.device}, {self.vram_gb:.1f}GB)"


def get_device() -> str:
    """사용 가능한 최적 디바이스 반환"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_optimal_cuda_device() -> int:
    """VRAM이 가장 큰 CUDA 디바이스 ID 반환 (멀티 GPU 환경)"""
    if not torch.cuda.is_available():
        return 0
    device_count = torch.cuda.device_count()
    if device_count == 1:
        return 0
    return max(
        range(device_count),
        key=lambda i: torch.cuda.get_device_properties(i).total_memory,
    )


def get_device_info() -> DeviceInfo:
    """디바이스 상세 정보 조회"""
    device = get_device()

    if device == "cuda":
        device_id = get_optimal_cuda_device()
        props = torch.cuda.get_device_properties(device_id)
        name = props.name
        vram_gb = props.total_memory / (1024**3)
        compute_capability = (props.major, props.minor)

        is_laptop = any(
            kw in name.lower()
            for kw in ["laptop", "mobile", "max-q", "notebook"]
        )

        return DeviceInfo(
            device=device,
            name=name,
            vram_gb=vram_gb,
            is_laptop=is_laptop,
            compute_capability=compute_capability,
        )

    elif device == "mps":
        try:
            import psutil

            total_ram = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            total_ram = 16.0

        usable_memory = total_ram * 0.75

        return DeviceInfo(
            device=device,
            name="Apple Silicon (MPS)",
            vram_gb=usable_memory,
            is_laptop=True,
        )

    else:
        try:
            import psutil

            total_ram = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            total_ram = 8.0

        return DeviceInfo(
            device=device,
            name="CPU",
            vram_gb=total_ram,
        )


def print_device_info() -> tuple[DeviceInfo, "OptimalConfig"]:
    """디바이스 정보 및 최적 설정 출력"""
    from scripts.embedding_common.config import get_optimal_config

    device_info = get_device_info()
    config = get_optimal_config(device_info)

    print("=" * 60)
    print("Device Information")
    print("=" * 60)
    print(f"  Device: {device_info.device.upper()}")
    print(f"  Name: {device_info.name}")
    print(f"  Memory: {device_info.vram_gb:.1f} GB")
    if device_info.is_laptop:
        print("  Type: Laptop/Mobile")
    if device_info.compute_capability:
        cc = device_info.compute_capability
        print(f"  Compute Capability: {cc[0]}.{cc[1]}")

    print("\nRecommended Settings:")
    print(f"  batch_size: {config.batch_size}")
    print(f"  num_workers: {config.num_workers}")
    print(f"  gc_interval: {config.gc_interval} batches")
    print("=" * 60)

    return device_info, config
