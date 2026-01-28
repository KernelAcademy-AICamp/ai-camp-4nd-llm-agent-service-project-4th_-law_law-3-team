"""
Error Handlers - 에러 핸들링 유틸리티
"""

import gc
import functools
import traceback
from contextlib import contextmanager
from typing import Optional, Callable, Any, TypeVar

import torch
import torch.nn as nn

T = TypeVar('T')


# =============================================================================
# CUDA Error Handling
# =============================================================================

class CUDAError(Exception):
    """CUDA 관련 에러"""
    pass


class OOMError(CUDAError):
    """Out of Memory 에러"""
    pass


def handle_cuda_error(func: Callable[..., T]) -> Callable[..., T]:
    """
    CUDA 에러 핸들링 데코레이터
    
    Example:
        @handle_cuda_error
        def train_step(model, batch):
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except torch.cuda.OutOfMemoryError as e:
            clear_cuda_memory()
            raise OOMError(
                f"CUDA OOM in {func.__name__}: {e}\n"
                "Suggestions:\n"
                "  1. Reduce batch size\n"
                "  2. Enable gradient checkpointing\n"
                "  3. Use mixed precision training\n"
                "  4. Reduce model size"
            ) from e
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "cuda" in error_msg or "gpu" in error_msg:
                clear_cuda_memory()
                raise CUDAError(f"CUDA error in {func.__name__}: {e}") from e
            raise
    
    return wrapper


def clear_cuda_memory():
    """CUDA 메모리 정리"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# =============================================================================
# OOM Recovery
# =============================================================================

def train_with_oom_recovery(
    train_fn: Callable[[int], Any],
    initial_batch_size: int,
    min_batch_size: int = 1,
    max_retries: int = 5
) -> tuple[Any, int]:
    """
    OOM 발생 시 배치 사이즈 자동 축소하며 재시도
    
    Args:
        train_fn: 배치 사이즈를 받는 학습 함수
        initial_batch_size: 초기 배치 사이즈
        min_batch_size: 최소 배치 사이즈
        max_retries: 최대 재시도 횟수
    
    Returns:
        (학습 결과, 성공한 배치 사이즈)
    
    Example:
        def train_epoch(batch_size):
            dataloader = create_dataloader(batch_size)
            for batch in dataloader:
                train_step(batch)
            return metrics
        
        result, batch_size = train_with_oom_recovery(train_epoch, 32, min_batch_size=4)
    """
    batch_size = initial_batch_size
    retries = 0
    
    while batch_size >= min_batch_size and retries < max_retries:
        try:
            result = train_fn(batch_size)
            return result, batch_size
        except (torch.cuda.OutOfMemoryError, OOMError) as e:
            clear_cuda_memory()
            
            old_batch_size = batch_size
            batch_size = max(batch_size // 2, min_batch_size)
            retries += 1
            
            print(f"OOM detected, reducing batch size: {old_batch_size} -> {batch_size}")
            
            if batch_size < min_batch_size:
                break
    
    raise OOMError(
        f"Cannot train even with minimum batch size {min_batch_size}.\n"
        "Suggestions:\n"
        "  1. Enable gradient checkpointing\n"
        "  2. Use a smaller model\n"
        "  3. Use more aggressive mixed precision"
    )


@contextmanager
def oom_guard(
    cleanup_fn: Optional[Callable] = None,
    reraise: bool = True
):
    """
    OOM 발생 시 cleanup 후 처리하는 컨텍스트 매니저
    
    Example:
        with oom_guard(cleanup_fn=lambda: model.cpu()):
            outputs = model(large_batch)
    """
    try:
        yield
    except torch.cuda.OutOfMemoryError as e:
        clear_cuda_memory()
        
        if cleanup_fn:
            cleanup_fn()
        
        if reraise:
            raise OOMError(f"OOM: {e}") from e


# =============================================================================
# Data Loading Error Handling
# =============================================================================

class DataLoadingError(Exception):
    """데이터 로딩 에러"""
    pass


def handle_data_loading_error(func: Callable[..., T]) -> Callable[..., T]:
    """
    데이터 로딩 에러 핸들링 데코레이터
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            raise DataLoadingError(f"File not found: {e}") from e
        except PermissionError as e:
            raise DataLoadingError(f"Permission denied: {e}") from e
        except UnicodeDecodeError as e:
            raise DataLoadingError(
                f"Encoding error: {e}\n"
                "Try specifying encoding='utf-8' or encoding='cp949'"
            ) from e
        except Exception as e:
            raise DataLoadingError(f"Data loading error: {e}") from e
    
    return wrapper


def safe_collate_fn(batch: list) -> Any:
    """
    안전한 collate 함수 - None 값 필터링
    """
    batch = [item for item in batch if item is not None]
    
    if not batch:
        return None
    
    return torch.utils.data.dataloader.default_collate(batch)


class RobustDataLoader:
    """
    에러 발생 시 스킵하는 DataLoader 래퍼
    
    Example:
        robust_loader = RobustDataLoader(dataloader, max_errors=10)
        for batch in robust_loader:
            train_step(batch)
    """
    
    def __init__(
        self,
        dataloader: torch.utils.data.DataLoader,
        max_errors: int = 100,
        log_errors: bool = True
    ):
        self.dataloader = dataloader
        self.max_errors = max_errors
        self.log_errors = log_errors
        self.error_count = 0
    
    def __iter__(self):
        for batch in self.dataloader:
            try:
                yield batch
            except Exception as e:
                self.error_count += 1
                
                if self.log_errors:
                    print(f"DataLoader error ({self.error_count}): {e}")
                
                if self.error_count >= self.max_errors:
                    raise DataLoadingError(
                        f"Too many data loading errors: {self.error_count}"
                    )
                
                continue
    
    def __len__(self):
        return len(self.dataloader)


# =============================================================================
# Safe Training Context
# =============================================================================

class SafeTrainingContext:
    """
    안전한 학습 컨텍스트 매니저
    
    메모리 관리, 에러 로깅, 체크포인트 자동 저장 등
    
    Example:
        with SafeTrainingContext(device, checkpoint_dir="checkpoints") as ctx:
            for epoch in range(100):
                train_epoch(model, dataloader)
                ctx.checkpoint(model, optimizer, epoch)
    """
    
    def __init__(
        self,
        device: torch.device,
        checkpoint_dir: Optional[str] = None,
        auto_clear_memory: bool = True
    ):
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.auto_clear_memory = auto_clear_memory
        self.error_log = []
    
    def __enter__(self):
        if self.auto_clear_memory:
            clear_cuda_memory()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.error_log.append({
                'type': exc_type.__name__,
                'message': str(exc_val),
                'traceback': traceback.format_exc()
            })
            
            if self.auto_clear_memory:
                clear_cuda_memory()
            
            print(f"Error occurred: {exc_type.__name__}: {exc_val}")
        
        return False  # 예외 전파
    
    def checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        **kwargs
    ):
        """체크포인트 저장"""
        if self.checkpoint_dir is None:
            return
        
        from pathlib import Path
        checkpoint_dir = Path(self.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            **kwargs
        }, path)
        
        print(f"Checkpoint saved: {path}")
    
    def log_error(self, error: Exception, context: str = ""):
        """에러 로깅"""
        self.error_log.append({
            'type': type(error).__name__,
            'message': str(error),
            'context': context,
            'traceback': traceback.format_exc()
        })


# =============================================================================
# Retry Decorator
# =============================================================================

def retry_on_error(
    max_retries: int = 3,
    delay: float = 1.0,
    exceptions: tuple = (Exception,),
    cleanup_fn: Optional[Callable] = None
):
    """
    에러 발생 시 재시도 데코레이터
    
    Example:
        @retry_on_error(max_retries=3, exceptions=(ConnectionError,))
        def download_data():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import time
            
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                    
                    if cleanup_fn:
                        cleanup_fn()
                    
                    if attempt < max_retries - 1:
                        time.sleep(delay * (attempt + 1))  # Exponential backoff
            
            raise last_exception
        
        return wrapper
    
    return decorator


# =============================================================================
# Validation Utilities
# =============================================================================

def validate_tensor(
    tensor: torch.Tensor,
    name: str = "tensor",
    check_nan: bool = True,
    check_inf: bool = True,
    check_range: Optional[tuple] = None
) -> bool:
    """
    텐서 유효성 검사
    
    Args:
        tensor: 검사할 텐서
        name: 텐서 이름 (에러 메시지용)
        check_nan: NaN 체크
        check_inf: Inf 체크
        check_range: 값 범위 (min, max)
    
    Returns:
        유효성 여부
    
    Raises:
        ValueError: 유효하지 않은 경우
    """
    if check_nan and torch.isnan(tensor).any():
        raise ValueError(f"{name} contains NaN values")
    
    if check_inf and torch.isinf(tensor).any():
        raise ValueError(f"{name} contains Inf values")
    
    if check_range is not None:
        min_val, max_val = check_range
        if tensor.min() < min_val or tensor.max() > max_val:
            raise ValueError(
                f"{name} values out of range [{min_val}, {max_val}]: "
                f"got [{tensor.min():.4f}, {tensor.max():.4f}]"
            )
    
    return True


def validate_model_output(
    output: torch.Tensor,
    expected_shape: Optional[tuple] = None,
    check_nan: bool = True
) -> bool:
    """모델 출력 유효성 검사"""
    validate_tensor(output, "model output", check_nan=check_nan, check_inf=True)
    
    if expected_shape is not None:
        if output.shape != expected_shape:
            raise ValueError(
                f"Output shape mismatch: expected {expected_shape}, got {output.shape}"
            )
    
    return True


if __name__ == "__main__":
    print("Error handlers 모듈 로드 완료")
    
    # 테스트
    @handle_cuda_error
    def test_function():
        return torch.randn(100, 100)
    
    result = test_function()
    print(f"Test passed: {result.shape}")
