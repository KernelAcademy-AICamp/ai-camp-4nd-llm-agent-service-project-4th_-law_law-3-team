"""
임베딩 모델 로딩 유틸리티

KURE-v1 등 sentence-transformers 모델을 로드합니다.
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Optional

import torch

from scripts.embedding_common.config import DEFAULT_CONFIG
from scripts.embedding_common.device import get_device, get_optimal_cuda_device

if TYPE_CHECKING:
    from pathlib import Path

    from sentence_transformers import SentenceTransformer

_cached_model: Optional[object] = None
_cached_device: Optional[str] = None
_cached_model_name: Optional[str] = None
_cached_backend: Optional[str] = None

# ONNX 모델 경로 (DATA_DIR 기준)
_ONNX_MODEL_DIRS: dict[str, str] = {
    "onnx": "models/kure-v1-onnx",
    "onnx-int8": "models/kure-v1-onnx-int8",
}


def _auto_convert_onnx(backend: str, data_dir: Path) -> None:
    """ONNX 모델이 없을 때 자동 변환 (FP32 export + INT8 양자화)"""
    fp32_path = data_dir / _ONNX_MODEL_DIRS["onnx"]

    # FP32 ONNX가 없으면 먼저 변환
    if not fp32_path.exists():
        print("[INFO] ONNX FP32 모델이 없습니다. 자동 변환 중...")
        try:
            from scripts.benchmark_embedding_quantize import export_onnx

            if not export_onnx():
                raise RuntimeError("ONNX FP32 변환 실패")
        except ImportError:
            raise FileNotFoundError(
                f"ONNX 모델을 찾을 수 없고 자동 변환도 실패했습니다: {fp32_path}\n"
                "optimum 설치 확인: uv sync --dev"
            )

    # INT8 요청인데 INT8 디렉토리가 없으면 양자화
    if backend == "onnx-int8":
        int8_path = data_dir / _ONNX_MODEL_DIRS["onnx-int8"]
        if not int8_path.exists():
            print("[INFO] ONNX INT8 모델이 없습니다. 자동 양자화 중...")
            try:
                from scripts.benchmark_embedding_quantize import quantize_int8

                if not quantize_int8():
                    raise RuntimeError("INT8 양자화 실패")
            except ImportError:
                raise FileNotFoundError(
                    f"INT8 모델을 찾을 수 없고 자동 양자화도 실패했습니다: {int8_path}\n"
                    "optimum 설치 확인: uv sync --dev"
                )


def get_embedding_model(
    device: Optional[str] = None,
    model_name: Optional[str] = None,
    backend: Optional[str] = None,
) -> SentenceTransformer:
    """
    임베딩 모델 로드 (캐싱)

    Args:
        device: 디바이스 ("cuda", "mps", "cpu", None=자동)
        model_name: 모델명 (기본: KURE-v1)
        backend: ONNX 백엔드 ("onnx", "onnx-int8", None=PyTorch)

    Returns:
        SentenceTransformer 모델
    """
    global _cached_model, _cached_device, _cached_model_name, _cached_backend
    from sentence_transformers import SentenceTransformer

    model_name = model_name or str(DEFAULT_CONFIG["EMBEDDING_MODEL"])

    if device is None:
        device = get_device()
        if device == "cuda":
            device_id = get_optimal_cuda_device()
            device = f"cuda:{device_id}"

    # 캐시된 모델이 같은 디바이스 + 같은 모델 + 같은 백엔드이면 반환
    if (
        _cached_model is not None
        and _cached_device == device
        and _cached_model_name == model_name
        and _cached_backend == backend
    ):
        return _cached_model  # type: ignore[return-value]

    if backend in _ONNX_MODEL_DIRS:
        # ONNX 백엔드: 로컬 변환된 모델 경로 사용
        from pathlib import Path

        data_dir = Path(__file__).parent.parent.parent / "data"
        onnx_path = data_dir / _ONNX_MODEL_DIRS[backend]

        if not onnx_path.exists():
            # ONNX 모델 자동 변환
            _auto_convert_onnx(backend, data_dir)

        # INT8: model_quantized.onnx → model.onnx 자동 리네임
        quantized = onnx_path / "model_quantized.onnx"
        model_onnx = onnx_path / "model.onnx"
        if quantized.exists() and not model_onnx.exists():
            quantized.rename(model_onnx)
            print("[INFO] Renamed model_quantized.onnx → model.onnx")

        # INT8: config/tokenizer 파일 누락 시 FP32 ONNX에서 복사
        if not (onnx_path / "config.json").exists():
            fp32_path = data_dir / _ONNX_MODEL_DIRS["onnx"]
            if fp32_path.exists():
                import shutil

                for f in fp32_path.iterdir():
                    if f.is_file() and f.suffix in (".json", ".txt"):
                        dst = onnx_path / f.name
                        if not dst.exists():
                            shutil.copy2(f, dst)
                print(f"[INFO] Copied config/tokenizer from {fp32_path}")

        print(f"[INFO] Loading ONNX model: {onnx_path} (backend={backend})")
        model = SentenceTransformer(
            str(onnx_path),
            backend="onnx",
            trust_remote_code=True,
        )
    else:
        # 기존 PyTorch 모델
        print(f"[INFO] Loading embedding model: {model_name} on {device}")
        model = SentenceTransformer(
            model_name, device=device, trust_remote_code=True
        )

    model.eval()

    _cached_model = model
    _cached_device = device
    _cached_model_name = model_name
    _cached_backend = backend

    return model


def create_embeddings(
    texts: list[str],
    model: Optional[SentenceTransformer] = None,
    batch_size: int = 32,
    normalize: bool = True,
) -> list[list[float]]:
    """
    텍스트 목록을 임베딩 벡터로 변환

    Args:
        texts: 텍스트 목록
        model: 임베딩 모델 (None이면 자동 로드)
        batch_size: 배치 크기
        normalize: L2 정규화 적용

    Returns:
        임베딩 벡터 목록
    """
    if model is None:
        model = get_embedding_model()

    with torch.no_grad():
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=normalize,
        )

    return embeddings.tolist()


def clear_model_cache() -> None:
    """모델 캐시 및 GPU 메모리 해제"""
    global _cached_model, _cached_device, _cached_model_name
    _cached_model = None
    _cached_device = None
    _cached_model_name = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def clear_memory() -> None:
    """GC + GPU 캐시 정리"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """랜덤 시드 고정 (재현성)"""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
