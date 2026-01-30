"""
Profiling & Debugging - 프로파일링 및 디버깅 유틸리티
"""

import gc
import functools
import time
from contextlib import contextmanager
from typing import Optional, Dict, Any, Callable

import torch
import torch.nn as nn


# =============================================================================
# Memory Monitoring
# =============================================================================

def get_memory_stats(device: torch.device) -> Dict[str, float]:
    """
    현재 메모리 사용량 반환 (GB)
    
    Args:
        device: 디바이스
    
    Returns:
        메모리 통계 딕셔너리
    """
    if device.type == "cuda":
        return {
            'allocated': torch.cuda.memory_allocated(device) / 1e9,
            'reserved': torch.cuda.memory_reserved(device) / 1e9,
            'max_allocated': torch.cuda.max_memory_allocated(device) / 1e9,
            'free': (
                torch.cuda.get_device_properties(device).total_memory -
                torch.cuda.memory_allocated(device)
            ) / 1e9,
        }
    elif device.type == "mps":
        return {
            'allocated': torch.mps.current_allocated_memory() / 1e9,
        }
    return {}


def print_memory_stats(device: torch.device, prefix: str = ""):
    """메모리 상태 출력"""
    stats = get_memory_stats(device)
    
    if not stats:
        print(f"{prefix}Memory stats not available for {device}")
        return
    
    parts = []
    if 'allocated' in stats:
        parts.append(f"Allocated: {stats['allocated']:.2f}GB")
    if 'reserved' in stats:
        parts.append(f"Reserved: {stats['reserved']:.2f}GB")
    if 'max_allocated' in stats:
        parts.append(f"Max: {stats['max_allocated']:.2f}GB")
    if 'free' in stats:
        parts.append(f"Free: {stats['free']:.2f}GB")
    
    print(f"{prefix}Memory - {', '.join(parts)}")


def clear_memory(device: Optional[torch.device] = None):
    """
    메모리 정리 유틸리티
    
    Args:
        device: 정리할 디바이스 (None이면 모든 CUDA)
    """
    gc.collect()
    
    if torch.cuda.is_available():
        if device and device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        else:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    if torch.backends.mps.is_available():
        # MPS는 명시적 캐시 클리어 없음
        pass


def reset_peak_memory(device: torch.device):
    """피크 메모리 카운터 리셋"""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


@contextmanager
def track_memory(device: torch.device, description: str = ""):
    """
    메모리 사용량 추적 컨텍스트 매니저
    
    Example:
        with track_memory(device, "Forward pass"):
            output = model(input)
    """
    if device.type != "cuda":
        yield
        return
    
    torch.cuda.synchronize(device)
    clear_memory(device)
    
    start_allocated = torch.cuda.memory_allocated(device)
    start_reserved = torch.cuda.memory_reserved(device)
    
    yield
    
    torch.cuda.synchronize(device)
    
    end_allocated = torch.cuda.memory_allocated(device)
    end_reserved = torch.cuda.memory_reserved(device)
    
    delta_allocated = (end_allocated - start_allocated) / 1e6
    delta_reserved = (end_reserved - start_reserved) / 1e6
    
    print(f"[{description}] Memory delta - Allocated: {delta_allocated:+.2f}MB, "
          f"Reserved: {delta_reserved:+.2f}MB")


# =============================================================================
# Timing Utilities
# =============================================================================

class Timer:
    """
    시간 측정 유틸리티
    
    Example:
        timer = Timer()
        
        timer.start("forward")
        output = model(input)
        timer.stop("forward")
        
        timer.report()
    """
    
    def __init__(self, cuda_sync: bool = True):
        self.cuda_sync = cuda_sync
        self.times: Dict[str, list] = {}
        self._start_times: Dict[str, float] = {}
    
    def start(self, name: str):
        """타이머 시작"""
        if self.cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        self._start_times[name] = time.perf_counter()
    
    def stop(self, name: str) -> float:
        """타이머 중지, 경과 시간 반환"""
        if self.cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - self._start_times[name]
        
        if name not in self.times:
            self.times[name] = []
        self.times[name].append(elapsed)
        
        return elapsed
    
    def report(self):
        """결과 출력"""
        print("\n" + "="*60)
        print("Timing Report")
        print("="*60)
        
        for name, times in self.times.items():
            avg = sum(times) / len(times)
            total = sum(times)
            print(f"{name}: avg={avg*1000:.2f}ms, total={total:.2f}s, count={len(times)}")
        
        print("="*60)


def timed(func: Callable) -> Callable:
    """
    함수 실행 시간 측정 데코레이터
    
    Example:
        @timed
        def train_step():
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        result = func(*args, **kwargs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start
        print(f"{func.__name__}: {elapsed*1000:.2f}ms")
        
        return result
    
    return wrapper


# =============================================================================
# Model Analysis
# =============================================================================

def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    모델 파라미터 수 계산
    
    Args:
        model: 모델
        trainable_only: 학습 가능한 파라미터만 카운트
    
    Returns:
        파라미터 수
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def print_model_summary(model: nn.Module, input_shape: Optional[tuple] = None):
    """
    모델 요약 출력
    
    Args:
        model: 모델
        input_shape: 입력 shape (배치 제외)
    """
    total_params = count_parameters(model)
    trainable_params = count_parameters(model, trainable_only=True)
    
    print("\n" + "="*60)
    print("Model Summary")
    print("="*60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"Model size (fp32): {total_params * 4 / 1e6:.2f} MB")
    print(f"Model size (fp16): {total_params * 2 / 1e6:.2f} MB")
    print("="*60)


def find_tensor_leaks(threshold_mb: float = 100):
    """
    큰 텐서 찾기 (메모리 누수 디버깅)
    
    Args:
        threshold_mb: 출력할 최소 크기 (MB)
    """
    import gc
    
    gc.collect()
    
    print(f"\nTensors > {threshold_mb}MB:")
    print("-" * 60)
    
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                size_mb = obj.element_size() * obj.nelement() / 1e6
                if size_mb > threshold_mb:
                    print(f"Shape: {obj.shape}, dtype: {obj.dtype}, "
                          f"size: {size_mb:.2f}MB, device: {obj.device}")
        except Exception:
            pass


# =============================================================================
# PyTorch Profiler
# =============================================================================

def profile_training_step(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    output_path: Optional[str] = None
):
    """
    학습 스텝 프로파일링
    
    Args:
        model: 모델
        inputs: 입력 텐서
        targets: 타겟 텐서
        criterion: 손실 함수
        optimizer: 옵티마이저
        device: 디바이스
        output_path: 결과 저장 경로 (Chrome trace)
    
    Returns:
        프로파일러 객체
    """
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        # Forward
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward
        loss.backward()
        optimizer.step()
    
    # 결과 출력
    print("\n" + "="*80)
    print("Profiler Results (sorted by CUDA time)")
    print("="*80)
    
    if device.type == "cuda":
        print(prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=15
        ))
    else:
        print(prof.key_averages().table(
            sort_by="cpu_time_total",
            row_limit=15
        ))
    
    # Chrome trace 저장
    if output_path:
        prof.export_chrome_trace(output_path)
        print(f"\nChrome trace saved: {output_path}")
        print("Open chrome://tracing in Chrome to view")
    
    return prof


@contextmanager
def profiler_context(
    device: torch.device,
    wait: int = 1,
    warmup: int = 1,
    active: int = 3,
    repeat: int = 1
):
    """
    프로파일러 컨텍스트 매니저
    
    Example:
        with profiler_context(device) as prof:
            for step, batch in enumerate(dataloader):
                train_step(batch)
                prof.step()
    """
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    
    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=wait,
            warmup=warmup,
            active=active,
            repeat=repeat
        ),
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        yield prof
    
    print(prof.key_averages().table(
        sort_by="cuda_time_total" if device.type == "cuda" else "cpu_time_total",
        row_limit=15
    ))


# =============================================================================
# Gradient Debugging
# =============================================================================

def check_gradients(model: nn.Module) -> Dict[str, Any]:
    """
    그래디언트 상태 체크
    
    Returns:
        그래디언트 통계
    """
    stats = {
        "has_nan": False,
        "has_inf": False,
        "max_grad": 0.0,
        "min_grad": float('inf'),
        "layers_with_issues": []
    }
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            
            if torch.isnan(grad).any():
                stats["has_nan"] = True
                stats["layers_with_issues"].append((name, "NaN"))
            
            if torch.isinf(grad).any():
                stats["has_inf"] = True
                stats["layers_with_issues"].append((name, "Inf"))
            
            grad_max = grad.abs().max().item()
            grad_min = grad.abs().min().item()
            
            stats["max_grad"] = max(stats["max_grad"], grad_max)
            stats["min_grad"] = min(stats["min_grad"], grad_min)
    
    return stats


def print_gradient_stats(model: nn.Module):
    """그래디언트 통계 출력"""
    stats = check_gradients(model)
    
    print("\n" + "="*60)
    print("Gradient Statistics")
    print("="*60)
    print(f"Has NaN: {stats['has_nan']}")
    print(f"Has Inf: {stats['has_inf']}")
    print(f"Max gradient: {stats['max_grad']:.6f}")
    print(f"Min gradient: {stats['min_grad']:.6f}")
    
    if stats["layers_with_issues"]:
        print("\nLayers with issues:")
        for name, issue in stats["layers_with_issues"]:
            print(f"  - {name}: {issue}")
    
    print("="*60)


if __name__ == "__main__":
    # 테스트
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print_memory_stats(device)
