"""
Device Manager - 디바이스 자동 선택 및 관리
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional


@dataclass
class DeviceInfo:
    """디바이스 정보"""
    device: torch.device
    dtype: torch.dtype
    device_name: str
    total_memory: Optional[float] = None  # GB


class DeviceManager:
    """
    디바이스 관리 및 자동 최적화
    
    사용 예시:
        dm = DeviceManager()
        model = model.to(dm.device)
        
        # 또는
        dm = DeviceManager(force_cpu=True)
    """
    
    def __init__(self, force_cpu: bool = False, verbose: bool = True):
        """
        Args:
            force_cpu: CPU 강제 사용
            verbose: 디바이스 정보 출력
        """
        self.device = self._detect_optimal_device(force_cpu)
        self.dtype = self._get_optimal_dtype()
        self.info = self._get_device_info()
        
        if verbose:
            self._print_device_info()
    
    def _detect_optimal_device(self, force_cpu: bool) -> torch.device:
        """사용 가능한 최적 디바이스 탐지"""
        if force_cpu:
            return torch.device("cpu")
        
        if torch.cuda.is_available():
            # VRAM이 가장 큰 GPU 선택
            device_id = max(
                range(torch.cuda.device_count()),
                key=lambda i: torch.cuda.get_device_properties(i).total_memory
            )
            return torch.device(f"cuda:{device_id}")
        
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        
        return torch.device("cpu")
    
    def _get_optimal_dtype(self) -> torch.dtype:
        """디바이스에 맞는 최적 dtype 반환"""
        if self.device.type == "cuda":
            # bf16 지원 확인 (Ampere 이상)
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        
        elif self.device.type == "mps":
            # MPS는 fp16 지원이 제한적
            return torch.float32
        
        return torch.float32
    
    def _get_device_info(self) -> DeviceInfo:
        """디바이스 상세 정보 수집"""
        device_name = "CPU"
        total_memory = None
        
        if self.device.type == "cuda":
            props = torch.cuda.get_device_properties(self.device)
            device_name = props.name
            total_memory = props.total_memory / 1e9
        
        elif self.device.type == "mps":
            device_name = "Apple Silicon (MPS)"
        
        return DeviceInfo(
            device=self.device,
            dtype=self.dtype,
            device_name=device_name,
            total_memory=total_memory
        )
    
    def _print_device_info(self):
        """디바이스 정보 출력"""
        print(f"{'='*50}")
        print(f"Device: {self.info.device_name}")
        print(f"Type: {self.device.type}")
        print(f"Dtype: {self.dtype}")
        if self.info.total_memory:
            print(f"VRAM: {self.info.total_memory:.1f} GB")
        print(f"{'='*50}")
    
    def to_device(self, model: nn.Module) -> nn.Module:
        """모델을 최적 디바이스로 이동"""
        return model.to(self.device)
    
    def get_autocast_context(self, enabled: bool = True):
        """AMP autocast 컨텍스트 반환"""
        return torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
            enabled=enabled and self.device.type in ("cuda", "cpu")
        )
    
    def get_grad_scaler(self) -> Optional[torch.amp.GradScaler]:
        """GradScaler 반환 (fp16일 때만)"""
        if self.dtype == torch.float16 and self.device.type == "cuda":
            return torch.amp.GradScaler('cuda')
        return None
    
    @property
    def use_amp(self) -> bool:
        """AMP 사용 여부"""
        return self.device.type == "cuda"


# 함수형 인터페이스
def get_optimal_device(force_cpu: bool = False) -> torch.device:
    """사용 가능한 최적 디바이스 반환"""
    if force_cpu:
        return torch.device("cpu")
    
    if torch.cuda.is_available():
        device_id = max(
            range(torch.cuda.device_count()),
            key=lambda i: torch.cuda.get_device_properties(i).total_memory
        )
        return torch.device(f"cuda:{device_id}")
    
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    
    return torch.device("cpu")


def get_optimal_dtype(device: torch.device) -> torch.dtype:
    """디바이스에 맞는 최적 dtype 반환"""
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def get_device_info(device: torch.device) -> dict:
    """디바이스 정보 반환"""
    info = {
        "device": str(device),
        "type": device.type,
    }
    
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info.update({
            "name": props.name,
            "total_memory_gb": props.total_memory / 1e9,
            "compute_capability": f"{props.major}.{props.minor}",
        })
    
    return info


if __name__ == "__main__":
    # 테스트
    dm = DeviceManager()
    print(f"\nDevice: {dm.device}")
    print(f"Dtype: {dm.dtype}")
    print(f"Use AMP: {dm.use_amp}")
