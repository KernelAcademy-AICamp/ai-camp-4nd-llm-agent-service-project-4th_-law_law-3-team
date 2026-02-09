"""
임베딩 스크립트 공통 모듈

3개 임베딩 스크립트(runpod, colab, local)에서 공유되는 코드를 모듈화.
- device: GPU/CPU/MPS 감지
- config: 하드웨어 프로필 설정
- store: LanceDB 테이블 생성/연결
- chunking: 텍스트 청킹 (법령/판례)
- model: 임베딩 모델 로딩
- cache: 임베딩 캐시 (MD5 기반)
- schema: schema_v2 re-export + 검증
- temperature: GPU 온도 모니터링
- memory: GPU/시스템 메모리 모니터링
"""

from scripts.embedding_common.config import (
    DEFAULT_CONFIG,
    HardwareProfile,
    OptimalConfig,
    get_hardware_profile,
    get_optimal_config,
)
from scripts.embedding_common.device import (
    DeviceInfo,
    get_device,
    get_device_info,
    get_optimal_cuda_device,
    print_device_info,
)

__all__ = [
    # device
    "DeviceInfo",
    "get_device",
    "get_device_info",
    "get_optimal_cuda_device",
    "print_device_info",
    # config
    "DEFAULT_CONFIG",
    "HardwareProfile",
    "OptimalConfig",
    "get_hardware_profile",
    "get_optimal_config",
]
