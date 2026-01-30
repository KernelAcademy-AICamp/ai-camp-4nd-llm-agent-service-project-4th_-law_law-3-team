"""
Inference Optimization - 추론 최적화 유틸리티
"""

from typing import Optional, List, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# =============================================================================
# Inference Utilities
# =============================================================================

@torch.inference_mode()
def batch_inference(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> torch.Tensor:
    """
    배치 추론
    
    Args:
        model: 모델
        dataloader: 데이터로더
        device: 디바이스
        dtype: 데이터 타입 (AMP용)
        progress_callback: 진행 콜백 (current, total)
    
    Returns:
        모든 출력 결과 텐서
    """
    model.eval()
    results = []
    total = len(dataloader)
    
    use_amp = device.type == "cuda" and dtype is not None
    
    for i, batch in enumerate(dataloader):
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        
        batch = batch.to(device)
        
        with torch.amp.autocast(
            device_type=device.type,
            dtype=dtype,
            enabled=use_amp
        ):
            outputs = model(batch)
        
        results.append(outputs.cpu())
        
        if progress_callback:
            progress_callback(i + 1, total)
    
    return torch.cat(results, dim=0)


@torch.inference_mode()
def single_inference(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """단일 입력 추론"""
    model.eval()
    
    input_tensor = input_tensor.to(device)
    
    if input_tensor.dim() == 1:
        input_tensor = input_tensor.unsqueeze(0)
    
    use_amp = device.type == "cuda" and dtype is not None
    
    with torch.amp.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=use_amp
    ):
        output = model(input_tensor)
    
    return output.cpu()


# =============================================================================
# Quantization
# =============================================================================

def quantize_dynamic(
    model: nn.Module,
    dtype: torch.dtype = torch.qint8
) -> nn.Module:
    """
    동적 양자화 적용 (CPU 추론용)
    
    Args:
        model: 원본 모델
        dtype: 양자화 타입 (qint8, float16)
    
    Returns:
        양자화된 모델
    
    Note:
        - CPU 추론에서만 효과 있음
        - Linear, LSTM, GRU 레이어 양자화
    """
    return torch.quantization.quantize_dynamic(
        model.cpu(),
        {nn.Linear, nn.LSTM, nn.GRU},
        dtype=dtype
    )


def quantize_static(
    model: nn.Module,
    calibration_dataloader: DataLoader,
    device: torch.device = torch.device('cpu')
) -> nn.Module:
    """
    정적 양자화 적용
    
    Args:
        model: 원본 모델 (fuse 완료 상태)
        calibration_dataloader: 캘리브레이션 데이터
        device: 디바이스 (CPU만 지원)
    
    Returns:
        양자화된 모델
    """
    model = model.cpu()
    model.eval()
    
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    
    with torch.no_grad():
        for batch in calibration_dataloader:
            if isinstance(batch, (tuple, list)):
                batch = batch[0]
            model(batch.cpu())
    
    torch.quantization.convert(model, inplace=True)
    
    return model


# =============================================================================
# ONNX Export
# =============================================================================

def export_to_onnx(
    model: nn.Module,
    dummy_input: torch.Tensor,
    output_path: str,
    input_names: List[str] = None,
    output_names: List[str] = None,
    dynamic_axes: Optional[dict] = None,
    opset_version: int = 14
) -> str:
    """
    ONNX 형식으로 모델 내보내기
    """
    if input_names is None:
        input_names = ["input"]
    if output_names is None:
        output_names = ["output"]
    
    model = model.cpu()
    model.eval()
    
    if dynamic_axes is None:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    
    torch.onnx.export(
        model,
        dummy_input.cpu(),
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )
    
    print(f"ONNX model exported: {output_path}")
    return output_path


def verify_onnx(onnx_path: str, dummy_input: torch.Tensor, model: nn.Module) -> bool:
    """ONNX 모델 검증"""
    try:
        import onnx
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        print("onnx, onnxruntime이 필요합니다")
        return False
    
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)[0]
    
    model.eval()
    with torch.no_grad():
        torch_outputs = model(dummy_input).numpy()
    
    is_close = np.allclose(ort_outputs, torch_outputs, rtol=1e-3, atol=1e-5)
    
    if is_close:
        print("ONNX verification passed!")
    else:
        print("ONNX verification failed - outputs differ")
    
    return is_close


# =============================================================================
# TorchScript
# =============================================================================

def export_to_torchscript(
    model: nn.Module,
    dummy_input: torch.Tensor,
    output_path: str,
    method: str = "trace"
) -> str:
    """TorchScript로 모델 내보내기"""
    model.eval()
    
    if method == "trace":
        scripted = torch.jit.trace(model, dummy_input)
    elif method == "script":
        scripted = torch.jit.script(model)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    scripted.save(output_path)
    print(f"TorchScript model saved: {output_path}")
    
    return output_path


def load_torchscript(path: str, device: torch.device) -> torch.jit.ScriptModule:
    """TorchScript 모델 로드"""
    model = torch.jit.load(path, map_location=device)
    model.eval()
    return model


# =============================================================================
# Optimized Inference Wrapper
# =============================================================================

class OptimizedInferenceModel:
    """
    최적화된 추론 모델 래퍼
    
    Example:
        wrapper = OptimizedInferenceModel(
            model,
            device=device,
            dtype=torch.float16,
            use_compile=True
        )
        
        outputs = wrapper.predict(inputs)
        embeddings = wrapper.predict_batch(dataloader)
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
        use_compile: bool = True
    ):
        self.device = device
        self.dtype = dtype
        self.use_amp = device.type == "cuda" and dtype is not None
        
        model = model.to(device)
        model.eval()
        
        if use_compile and hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode="reduce-overhead")
                print("Model compiled with torch.compile")
            except Exception as e:
                print(f"torch.compile failed: {e}")
        
        self.model = model
    
    @torch.inference_mode()
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """단일 배치 예측"""
        inputs = inputs.to(self.device)
        
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
            enabled=self.use_amp
        ):
            outputs = self.model(inputs)
        
        return outputs.cpu()
    
    @torch.inference_mode()
    def predict_batch(
        self,
        dataloader: DataLoader,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> torch.Tensor:
        """배치 예측"""
        results = []
        total = len(dataloader)
        
        for i, batch in enumerate(dataloader):
            if isinstance(batch, (tuple, list)):
                batch = batch[0]
            
            outputs = self.predict(batch)
            results.append(outputs)
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        return torch.cat(results, dim=0)


if __name__ == "__main__":
    print("Inference optimization 모듈 로드 완료")
