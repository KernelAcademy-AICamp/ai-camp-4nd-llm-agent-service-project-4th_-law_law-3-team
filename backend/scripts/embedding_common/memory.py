"""
GPU/시스템 메모리 사용량 모니터링

CUDA, MPS, CPU별 메모리 관리 유틸리티.
"""

from dataclasses import dataclass

import torch


@dataclass
class MemoryInfo:
    """메모리 사용량 정보 (단위: GB)"""

    total: float = 0.0
    used: float = 0.0
    free: float = 0.0
    device_type: str = "unknown"

    @property
    def usage_percent(self) -> float:
        if self.total <= 0:
            return 0.0
        return (self.used / self.total) * 100


def get_gpu_memory() -> MemoryInfo:
    """GPU 메모리 사용량 조회 (CUDA)"""
    if not torch.cuda.is_available():
        return MemoryInfo(device_type="no_cuda")

    device_id = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)
    reserved = torch.cuda.memory_reserved(device_id) / (1024**3)
    allocated = torch.cuda.memory_allocated(device_id) / (1024**3)

    return MemoryInfo(
        total=total,
        used=allocated,
        free=total - reserved,
        device_type="cuda",
    )


def get_system_memory() -> MemoryInfo:
    """시스템 RAM 사용량 조회"""
    try:
        import psutil

        mem = psutil.virtual_memory()
        return MemoryInfo(
            total=mem.total / (1024**3),
            used=mem.used / (1024**3),
            free=mem.available / (1024**3),
            device_type="system",
        )
    except ImportError:
        return MemoryInfo(device_type="system_unknown")


def print_memory_status() -> None:
    """현재 메모리 상태 출력"""
    sys_mem = get_system_memory()
    print(f"[MEM] System: {sys_mem.used:.1f}/{sys_mem.total:.1f} GB "
          f"({sys_mem.usage_percent:.0f}%)")

    if torch.cuda.is_available():
        gpu_mem = get_gpu_memory()
        print(f"[MEM] GPU: {gpu_mem.used:.1f}/{gpu_mem.total:.1f} GB "
              f"({gpu_mem.usage_percent:.0f}%)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS는 정확한 VRAM 조회가 불가하므로 시스템 메모리 기반
        print("[MEM] MPS: 시스템 메모리 공유 (별도 VRAM 없음)")


def check_memory_pressure(threshold_percent: float = 90.0) -> bool:
    """
    메모리 압력 체크

    Args:
        threshold_percent: 경고 임계값 (%)

    Returns:
        True면 메모리 부족 경고
    """
    if torch.cuda.is_available():
        gpu_mem = get_gpu_memory()
        return gpu_mem.usage_percent >= threshold_percent

    sys_mem = get_system_memory()
    return sys_mem.usage_percent >= threshold_percent
