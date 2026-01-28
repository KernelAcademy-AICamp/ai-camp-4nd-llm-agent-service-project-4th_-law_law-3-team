# PyTorch Optimization Skill

## Overview
PyTorch 코드 작성 시 자동으로 최적의 디바이스를 선택하고, 메모리 효율적인 데이터 로딩 및 학습을 구현하는 가이드라인.

## When to Use
- PyTorch 모델 학습/추론 코드 작성 시
- 임베딩 모델 구현 시
- 대규모 데이터셋 처리 시
- 메모리 최적화가 필요할 때
- 학습 파이프라인 구축 시

## Templates Location
- `templates/device_manager.py` - 디바이스 관리
- `templates/data_optimizer.py` - 데이터 로딩 최적화
- `templates/training_utils.py` - 학습 유틸리티
- `templates/profiling.py` - 프로파일링 & 디버깅
- `templates/inference.py` - 추론 최적화
- `templates/error_handlers.py` - 에러 핸들링

---

## 1. Device Selection (자동 자원 탐지)

### 규칙
- 항상 사용 가능한 최적 디바이스를 자동 탐지
- 우선순위: CUDA (VRAM 최대) > MPS > CPU
- 디바이스 선택 로직은 코드 최상단에 배치
- dtype도 디바이스에 맞게 자동 선택

### 함수형 패턴
```python
import torch

def get_optimal_device(force_cpu: bool = False) -> torch.device:
    """사용 가능한 최적 디바이스 반환"""
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

def get_optimal_dtype(device: torch.device) -> torch.dtype:
    """디바이스에 맞는 최적 dtype 반환"""
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    elif device.type == "mps":
        return torch.float32  # MPS는 fp16 지원 제한적
    return torch.float32
```

### 클래스 기반 패턴
```python
class DeviceManager:
    """디바이스 관리 및 자동 최적화"""
    
    def __init__(self, force_cpu: bool = False):
        self.device = self._detect_optimal_device(force_cpu)
        self.dtype = self._get_optimal_dtype()
        self._print_device_info()
    
    def _detect_optimal_device(self, force_cpu: bool) -> torch.device:
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
    
    def _get_optimal_dtype(self) -> torch.dtype:
        if self.device.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    
    def _print_device_info(self):
        print(f"Device: {self.device}, dtype: {self.dtype}")
        if self.device.type == "cuda":
            props = torch.cuda.get_device_properties(self.device)
            print(f"GPU: {props.name}, VRAM: {props.total_memory / 1e9:.1f}GB")
```

---

## 2. Memory-Optimized Data Loading

### 규칙
- 대규모 데이터는 항상 Lazy Loading 사용
- DataLoader는 `pin_memory=True` (CUDA), `num_workers` 자동 설정
- CSV/Parquet은 streaming 또는 chunked 로딩
- 이미지는 경로만 저장, 접근 시 로드

### 텍스트 데이터 (Streaming)
```python
import pandas as pd
from torch.utils.data import IterableDataset

class StreamingTextDataset(IterableDataset):
    """메모리 효율적 텍스트 스트리밍"""
    
    def __init__(self, file_path: str, text_column: str = 'text', chunk_size: int = 10000):
        self.file_path = file_path
        self.text_column = text_column
        self.chunk_size = chunk_size
    
    def __iter__(self):
        for chunk in pd.read_csv(self.file_path, chunksize=self.chunk_size):
            for text in chunk[self.text_column]:
                yield text
```

### 이미지 데이터 (Lazy Loading)
```python
from PIL import Image
from torch.utils.data import Dataset

class LazyImageDataset(Dataset):
    """이미지를 필요할 때만 로드"""
    
    def __init__(self, image_paths: list[str], transform=None):
        self.paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image
```

### Parquet 스트리밍
```python
import pyarrow.parquet as pq

class StreamingParquetDataset(IterableDataset):
    """Parquet 파일 스트리밍"""
    
    def __init__(self, file_path: str, columns: list[str] = None, batch_size: int = 1000):
        self.file_path = file_path
        self.columns = columns
        self.batch_size = batch_size
    
    def __iter__(self):
        parquet_file = pq.ParquetFile(self.file_path)
        for batch in parquet_file.iter_batches(batch_size=self.batch_size, columns=self.columns):
            df = batch.to_pandas()
            for _, row in df.iterrows():
                yield dict(row)
```

### DataLoader 최적화 팩토리
```python
import os
from torch.utils.data import DataLoader

def create_optimized_dataloader(
    dataset,
    batch_size: int,
    device: torch.device,
    shuffle: bool = True,
    drop_last: bool = False
) -> DataLoader:
    """디바이스에 최적화된 DataLoader 생성"""
    
    # MPS는 num_workers > 0에서 문제 발생 가능
    num_workers = 0 if device.type == "mps" else min(4, os.cpu_count() or 1)
    pin_memory = device.type == "cuda"
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=drop_last,
    )
```

---

## 3. Mixed Precision Training

### 규칙
- CUDA: `torch.amp` 사용 (bf16 우선, 미지원 시 fp16)
- MPS: fp32 유지 (amp 미지원)
- GradScaler는 fp16일 때만 사용 (bf16은 불필요)

### 패턴
```python
class MixedPrecisionTrainer:
    def __init__(self, model, optimizer, criterion, device_manager: DeviceManager):
        self.model = model.to(device_manager.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device_manager.device
        self.dtype = device_manager.dtype
        
        self.use_amp = self.device.type == "cuda"
        # fp16만 GradScaler 필요, bf16은 불필요
        self.scaler = torch.amp.GradScaler('cuda') if self.dtype == torch.float16 else None
    
    def train_step(self, inputs, targets):
        self.optimizer.zero_grad(set_to_none=True)  # 메모리 효율
        
        with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.use_amp):
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
        
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        
        return loss.item()
```

---

## 4. Gradient Checkpointing

### 규칙
- 메모리 부족 시 활성화
- 연산 시간 ~20% 증가 vs 메모리 ~60% 절약
- Transformer 모델에서 특히 효과적

### 패턴
```python
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

class CheckpointedModel(nn.Module):
    def __init__(self, layers: nn.ModuleList, use_checkpointing: bool = False):
        super().__init__()
        self.layers = layers
        self.use_checkpointing = use_checkpointing
        self.checkpoint_segments = 4
    
    def forward(self, x):
        if self.use_checkpointing and self.training:
            return checkpoint_sequential(
                self.layers,
                self.checkpoint_segments,
                x,
                use_reentrant=False
            )
        
        for layer in self.layers:
            x = layer(x)
        return x

# HuggingFace 모델용
def enable_gradient_checkpointing(model):
    """HuggingFace 모델 gradient checkpointing 활성화"""
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    return model
```

---

## 5. torch.compile() 사용

### 규칙
- PyTorch 2.0+ 환경에서 사용
- 첫 실행 시 컴파일 오버헤드 있음 (이후 빨라짐)
- 동적 shape이 많으면 `dynamic=True` 설정

### 패턴
```python
def compile_model_if_available(
    model: nn.Module,
    mode: str = "reduce-overhead",
    dynamic: bool = False
) -> nn.Module:
    """가능하면 torch.compile 적용"""
    if not hasattr(torch, 'compile'):
        return model
    
    try:
        compiled = torch.compile(model, mode=mode, dynamic=dynamic)
        print(f"torch.compile 적용 완료 (mode={mode})")
        return compiled
    except Exception as e:
        print(f"torch.compile 실패, 원본 모델 사용: {e}")
        return model

# Mode 선택 가이드:
# - "default": 균형 잡힌 최적화
# - "reduce-overhead": 작은 배치에서 오버헤드 최소화
# - "max-autotune": 최대 성능 (컴파일 시간 김)
```

---

## 6. OOM 대응 패턴

### 규칙
- OOM 발생 시 캐시 클리어 후 배치 사이즈 축소
- 최소 배치 사이즈까지 재시도
- 실패 시 명확한 에러 메시지

### 패턴
```python
import gc

def train_with_oom_recovery(
    train_fn,
    initial_batch_size: int,
    min_batch_size: int = 1,
    max_retries: int = 5
):
    """OOM 발생 시 자동 복구"""
    batch_size = initial_batch_size
    retries = 0
    
    while batch_size >= min_batch_size and retries < max_retries:
        try:
            return train_fn(batch_size)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            
            batch_size //= 2
            retries += 1
            print(f"OOM 발생, 배치 사이즈 축소: {batch_size}")
    
    raise RuntimeError(
        f"최소 배치 사이즈 {min_batch_size}에서도 OOM 발생. "
        "Gradient Checkpointing 활성화 또는 모델 크기 축소 필요."
    )

def clear_memory():
    """메모리 정리 유틸리티"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
```

---

## 7. 모델 저장/로드 최적화

### 규칙
- 저장: `state_dict`만 저장
- 로드: `map_location`으로 디바이스 지정
- 대용량: `_use_new_zipfile_serialization=True`

### 패턴
```python
def save_checkpoint(
    model: nn.Module,
    optimizer,
    epoch: int,
    path: str,
    **kwargs
):
    """체크포인트 저장"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        **kwargs
    }
    torch.save(checkpoint, path, _use_new_zipfile_serialization=True)
    print(f"Checkpoint saved: {path}")

def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer=None,
    device: torch.device = None
) -> dict:
    """체크포인트 로드"""
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False  # optimizer 포함 시 False
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint
```

---

## 8. Reproducibility (재현성)

### 규칙
- 모든 랜덤 시드 고정
- CUDA 결정론적 모드 활성화
- DataLoader worker 시드 고정

### 패턴
```python
import random
import numpy as np

def set_seed(seed: int = 42, deterministic: bool = True):
    """모든 랜덤 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # PyTorch 2.0+
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True, warn_only=True)

def seed_worker(worker_id):
    """DataLoader worker 시드 고정"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# DataLoader에 적용
# DataLoader(..., worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(42))
```

---

## 9. Learning Rate Scheduler 패턴

### 규칙
- Warmup은 대부분의 학습에 도움됨
- Cosine Annealing이 일반적으로 좋은 성능

### 패턴
```python
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, OneCycleLR

def get_linear_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear Warmup + Linear Decay"""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.0, (total_steps - step) / (total_steps - warmup_steps))
    return LambdaLR(optimizer, lr_lambda)

def get_cosine_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear Warmup + Cosine Decay"""
    import math
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)
```

---

## 10. Early Stopping

### 패턴
```python
class EarlyStopping:
    """조기 종료 유틸리티"""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        improved = (
            score < self.best_score - self.min_delta if self.mode == 'min'
            else score > self.best_score + self.min_delta
        )
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop
```

---

## 11. Gradient Clipping

### 패턴
```python
def train_step_with_clipping(model, optimizer, loss, max_norm: float = 1.0):
    """Gradient Clipping 포함 학습 스텝"""
    loss.backward()
    
    # Gradient Clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```

---

## 12. Model EMA (Exponential Moving Average)

### 패턴
```python
class ModelEMA:
    """모델 가중치 EMA"""
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.decay = decay
        
        for param in self.ema_model.parameters():
            param.requires_grad_(False)
    
    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(self.decay).add_(param.data, alpha=1 - self.decay)
    
    def forward(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)
```

---

## 13. Profiling & Debugging

### 메모리 모니터링
```python
def get_memory_stats(device: torch.device) -> dict:
    """현재 메모리 사용량 반환"""
    if device.type == "cuda":
        return {
            'allocated': torch.cuda.memory_allocated(device) / 1e9,
            'reserved': torch.cuda.memory_reserved(device) / 1e9,
            'max_allocated': torch.cuda.max_memory_allocated(device) / 1e9,
        }
    return {}

def print_memory_stats(device: torch.device, prefix: str = ""):
    """메모리 상태 출력"""
    stats = get_memory_stats(device)
    if stats:
        print(f"{prefix}Memory - Allocated: {stats['allocated']:.2f}GB, "
              f"Reserved: {stats['reserved']:.2f}GB, Max: {stats['max_allocated']:.2f}GB")
```

### Profiler 사용
```python
def profile_training_step(model, inputs, targets, criterion, optimizer):
    """학습 스텝 프로파일링"""
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    return prof
```

---

## 14. Inference 최적화

### 규칙
- 추론 시 `torch.inference_mode()` 사용 (no_grad보다 빠름)
- 배치 처리로 throughput 향상
- 필요시 양자화 적용

### 패턴
```python
@torch.inference_mode()
def batch_inference(model, dataloader, device):
    """배치 추론"""
    model.eval()
    results = []
    
    for batch in dataloader:
        batch = batch.to(device)
        outputs = model(batch)
        results.append(outputs.cpu())
    
    return torch.cat(results, dim=0)

# Dynamic Quantization (CPU 추론 최적화)
def quantize_model_dynamic(model: nn.Module) -> nn.Module:
    """동적 양자화 적용"""
    return torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.LSTM, nn.GRU},
        dtype=torch.qint8
    )
```

---

## 15. Distributed 준비 (미래 대비)

### 패턴
```python
def setup_distributed():
    """분산 학습 초기화 (미래 대비)"""
    import torch.distributed as dist
    
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    
    return local_rank

def cleanup_distributed():
    """분산 학습 정리"""
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()
```

---

## 16. Hyperparameter 관리

### 패턴
```python
from dataclasses import dataclass, asdict
import yaml

@dataclass
class TrainingConfig:
    # Model
    model_name: str = "bert-base-uncased"
    hidden_size: int = 768
    
    # Training
    batch_size: int = 32
    learning_rate: float = 2e-5
    num_epochs: int = 10
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # Optimization
    use_amp: bool = True
    gradient_checkpointing: bool = False
    max_grad_norm: float = 1.0
    
    # Seed
    seed: int = 42
    
    def save(self, path: str):
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f)
    
    @classmethod
    def load(cls, path: str):
        with open(path, 'r') as f:
            return cls(**yaml.safe_load(f))
```

---

## 17. Unit Test 패턴

### 패턴
```python
import unittest

class TestModel(unittest.TestCase):
    def setUp(self):
        self.device = get_optimal_device()
        self.model = MyModel().to(self.device)
    
    def test_forward_shape(self):
        """출력 shape 검증"""
        batch_size, seq_len = 4, 128
        x = torch.randn(batch_size, seq_len).to(self.device)
        output = self.model(x)
        self.assertEqual(output.shape[0], batch_size)
    
    def test_backward(self):
        """역전파 검증"""
        x = torch.randn(4, 128, requires_grad=True).to(self.device)
        output = self.model(x)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
    
    def test_deterministic(self):
        """재현성 검증"""
        set_seed(42)
        x = torch.randn(4, 128).to(self.device)
        out1 = self.model(x)
        
        set_seed(42)
        x = torch.randn(4, 128).to(self.device)
        out2 = self.model(x)
        
        self.assertTrue(torch.allclose(out1, out2))
```

---

## Quick Reference

| 상황 | 해결책 |
|------|--------|
| 디바이스 선택 | `get_optimal_device()` 또는 `DeviceManager` |
| 대규모 텍스트 | `StreamingTextDataset` |
| 이미지 로딩 | `LazyImageDataset` |
| 학습 속도 | Mixed Precision + `torch.compile()` |
| 메모리 부족 | Gradient Checkpointing + 배치 축소 |
| OOM 에러 | `train_with_oom_recovery()` |
| 재현성 | `set_seed()` + deterministic mode |
| 추론 최적화 | `@torch.inference_mode()` + quantization |
| 디버깅 | `print_memory_stats()` + profiler |

---

## Error Handling

자세한 에러 핸들링은 `templates/error_handlers.py` 참조.

```python
from templates.error_handlers import (
    handle_cuda_error,
    handle_data_loading_error,
    SafeTrainingContext
)

# 사용 예시
with SafeTrainingContext(device) as ctx:
    train_epoch(model, dataloader)
```
