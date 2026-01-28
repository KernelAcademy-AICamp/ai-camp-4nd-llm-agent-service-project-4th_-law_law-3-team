"""
Training Utilities - 학습 관련 유틸리티
"""

import copy
import math
import random
from typing import Optional, Callable, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler
from torch.utils.checkpoint import checkpoint_sequential


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int = 42, deterministic: bool = True):
    """
    모든 랜덤 시드 고정
    
    Args:
        seed: 시드 값
        deterministic: 결정론적 알고리즘 사용
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass


# =============================================================================
# Mixed Precision Training
# =============================================================================

class MixedPrecisionTrainer:
    """
    Mixed Precision 학습 트레이너
    
    Args:
        model: PyTorch 모델
        optimizer: Optimizer
        criterion: Loss 함수
        device: 디바이스
        dtype: 데이터 타입 (float16, bfloat16)
        max_grad_norm: Gradient clipping 값
    
    Example:
        trainer = MixedPrecisionTrainer(model, optimizer, criterion, device)
        loss = trainer.train_step(inputs, targets)
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        max_grad_norm: Optional[float] = 1.0
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.dtype = dtype
        self.max_grad_norm = max_grad_norm
        
        self.use_amp = device.type == "cuda"
        # bf16은 GradScaler 불필요
        self.scaler = (
            torch.amp.GradScaler('cuda')
            if self.use_amp and dtype == torch.float16
            else None
        )
    
    def train_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        """단일 학습 스텝"""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
            enabled=self.use_amp
        ):
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
        
        if self.scaler:
            self.scaler.scale(loss).backward()
            
            if self.max_grad_norm:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            
            if self.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
            
            self.optimizer.step()
        
        return loss.item()
    
    @torch.inference_mode()
    def eval_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """평가 스텝"""
        self.model.eval()
        
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
            enabled=self.use_amp
        ):
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
        
        return {"loss": loss.item()}


# =============================================================================
# Gradient Checkpointing
# =============================================================================

class CheckpointedModel(nn.Module):
    """
    Gradient Checkpointing 적용 모델 래퍼
    
    메모리 ~60% 절약, 연산 ~20% 증가
    
    Args:
        layers: nn.ModuleList
        use_checkpointing: checkpointing 활성화
        segments: 세그먼트 수
    """
    
    def __init__(
        self,
        layers: nn.ModuleList,
        use_checkpointing: bool = True,
        segments: int = 4
    ):
        super().__init__()
        self.layers = layers
        self.use_checkpointing = use_checkpointing
        self.segments = segments
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpointing and self.training:
            return checkpoint_sequential(
                self.layers,
                self.segments,
                x,
                use_reentrant=False
            )
        
        for layer in self.layers:
            x = layer(x)
        return x


def enable_gradient_checkpointing(model: nn.Module) -> nn.Module:
    """
    HuggingFace 모델 gradient checkpointing 활성화
    """
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    return model


# =============================================================================
# Learning Rate Schedulers
# =============================================================================

def get_linear_warmup_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int
) -> LambdaLR:
    """
    Linear Warmup + Linear Decay 스케줄러
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.0, (total_steps - step) / (total_steps - warmup_steps))
    
    return LambdaLR(optimizer, lr_lambda)


def get_cosine_warmup_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0
) -> LambdaLR:
    """
    Linear Warmup + Cosine Decay 스케줄러
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        
        return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay
    
    return LambdaLR(optimizer, lr_lambda)


def get_constant_warmup_scheduler(
    optimizer: Optimizer,
    warmup_steps: int
) -> LambdaLR:
    """
    Linear Warmup + Constant LR 스케줄러
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0
    
    return LambdaLR(optimizer, lr_lambda)


# =============================================================================
# Early Stopping
# =============================================================================

class EarlyStopping:
    """
    조기 종료 유틸리티
    
    Args:
        patience: 개선 없이 기다릴 에폭 수
        min_delta: 개선으로 인정할 최소 변화량
        mode: 'min' (loss) 또는 'max' (accuracy)
    
    Example:
        early_stop = EarlyStopping(patience=5, mode='min')
        
        for epoch in range(100):
            val_loss = validate()
            if early_stop(val_loss):
                print("Early stopping!")
                break
    """
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
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
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop
    
    def reset(self):
        """상태 초기화"""
        self.counter = 0
        self.best_score = None
        self.should_stop = False


# =============================================================================
# Model EMA
# =============================================================================

class ModelEMA:
    """
    Exponential Moving Average 모델
    
    학습된 가중치의 이동 평균으로 더 안정적인 모델 생성
    
    Args:
        model: 원본 모델
        decay: EMA 감쇠율 (0.999 권장)
        device: 디바이스
    
    Example:
        ema = ModelEMA(model, decay=0.999)
        
        for batch in dataloader:
            train_step(model, batch)
            ema.update(model)
        
        # 평가 시 EMA 모델 사용
        eval_output = ema.ema_model(eval_input)
    """
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        device: Optional[torch.device] = None
    ):
        self.decay = decay
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        
        if device:
            self.ema_model.to(device)
        
        for param in self.ema_model.parameters():
            param.requires_grad_(False)
    
    @torch.no_grad()
    def update(self, model: nn.Module):
        """EMA 가중치 업데이트"""
        for ema_param, param in zip(
            self.ema_model.parameters(),
            model.parameters()
        ):
            ema_param.data.mul_(self.decay).add_(
                param.data,
                alpha=1 - self.decay
            )
    
    def forward(self, *args, **kwargs):
        """EMA 모델 forward"""
        return self.ema_model(*args, **kwargs)
    
    def state_dict(self) -> dict:
        """EMA 모델 state dict"""
        return self.ema_model.state_dict()
    
    def load_state_dict(self, state_dict: dict):
        """EMA 모델 state dict 로드"""
        self.ema_model.load_state_dict(state_dict)


# =============================================================================
# Checkpoint Management
# =============================================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    path: str,
    scheduler: Optional[_LRScheduler] = None,
    ema: Optional[ModelEMA] = None,
    **kwargs
):
    """
    체크포인트 저장
    
    Args:
        model: 모델
        optimizer: Optimizer
        epoch: 현재 에폭
        path: 저장 경로
        scheduler: LR 스케줄러 (선택)
        ema: EMA 모델 (선택)
        **kwargs: 추가 저장할 데이터
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        **kwargs
    }
    
    if scheduler:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if ema:
        checkpoint['ema_state_dict'] = ema.state_dict()
    
    torch.save(checkpoint, path, _use_new_zipfile_serialization=True)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    ema: Optional[ModelEMA] = None,
    device: Optional[torch.device] = None
) -> dict:
    """
    체크포인트 로드
    
    Args:
        path: 체크포인트 경로
        model: 모델
        optimizer: Optimizer (선택)
        scheduler: LR 스케줄러 (선택)
        ema: EMA 모델 (선택)
        device: 로드할 디바이스
    
    Returns:
        체크포인트 딕셔너리
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    if ema and 'ema_state_dict' in checkpoint:
        ema.load_state_dict(checkpoint['ema_state_dict'])
    
    print(f"Checkpoint loaded: {path} (epoch {checkpoint.get('epoch', 'unknown')})")
    return checkpoint


# =============================================================================
# torch.compile Wrapper
# =============================================================================

def compile_model_if_available(
    model: nn.Module,
    mode: str = "reduce-overhead",
    dynamic: bool = False
) -> nn.Module:
    """
    가능하면 torch.compile 적용
    
    Args:
        model: 모델
        mode: 컴파일 모드
            - "default": 균형 잡힌 최적화
            - "reduce-overhead": 작은 배치 최적화
            - "max-autotune": 최대 성능 (컴파일 오래 걸림)
        dynamic: 동적 shape 지원
    
    Returns:
        컴파일된 모델 (또는 원본)
    """
    if not hasattr(torch, 'compile'):
        print("torch.compile 미지원 (PyTorch 2.0+ 필요)")
        return model
    
    try:
        compiled = torch.compile(model, mode=mode, dynamic=dynamic)
        print(f"torch.compile 적용 완료 (mode={mode}, dynamic={dynamic})")
        return compiled
    except Exception as e:
        print(f"torch.compile 실패, 원본 모델 사용: {e}")
        return model


if __name__ == "__main__":
    # 테스트
    set_seed(42)
    print("Training utilities 로드 완료")
