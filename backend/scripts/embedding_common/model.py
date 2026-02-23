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
    from sentence_transformers import SentenceTransformer

_cached_model: Optional[object] = None
_cached_device: Optional[str] = None
_cached_model_name: Optional[str] = None

# ONNX 배치 임베딩 1회 확인 플래그
_onnx_embedding_checked: bool = False
_onnx_embedding_available: bool = False


def get_embedding_model(
    device: Optional[str] = None,
    model_name: Optional[str] = None,
) -> SentenceTransformer:
    """
    임베딩 모델 로드 (캐싱)

    Args:
        device: 디바이스 ("cuda", "mps", "cpu", None=자동)
        model_name: 모델명 (기본: KURE-v1)

    Returns:
        SentenceTransformer 모델
    """
    global _cached_model, _cached_device, _cached_model_name
    from sentence_transformers import SentenceTransformer

    model_name = model_name or str(DEFAULT_CONFIG["EMBEDDING_MODEL"])

    if device is None:
        device = get_device()
        if device == "cuda":
            device_id = get_optimal_cuda_device()
            device = f"cuda:{device_id}"

    # 캐시된 모델이 같은 디바이스 + 같은 모델이면 반환
    if (
        _cached_model is not None
        and _cached_device == device
        and _cached_model_name == model_name
    ):
        return _cached_model  # type: ignore[return-value]

    print(f"[INFO] Loading embedding model: {model_name} on {device}")
    model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
    model.eval()

    _cached_model = model
    _cached_device = device
    _cached_model_name = model_name

    return model


def _should_use_onnx_embedding() -> bool:
    """ONNX 배치 임베딩 사용 가능 여부를 확인한다.

    settings.USE_ONNX_EMBEDDING 확인 + 세션 로드를 1회만 시도.
    """
    global _onnx_embedding_checked, _onnx_embedding_available

    if _onnx_embedding_checked:
        return _onnx_embedding_available

    _onnx_embedding_checked = True

    try:
        from app.core.config import settings

        if not settings.USE_ONNX_EMBEDDING:
            _onnx_embedding_available = False
            return False

        from app.services.rag.onnx_session import (
            is_embedding_onnx_loaded,
            load_embedding_session,
        )

        if not is_embedding_onnx_loaded():
            loaded = load_embedding_session()
            if not loaded:
                print("[WARN] ONNX 임베딩 세션 로드 실패 → PyTorch fallback")
                _onnx_embedding_available = False
                return False

        _onnx_embedding_available = True
        print("[INFO] ONNX 배치 임베딩 활성화")
        return True
    except Exception as e:
        print(f"[WARN] ONNX 확인 중 오류 → PyTorch fallback: {e}")
        _onnx_embedding_available = False
        return False


def _create_embeddings_onnx(
    texts: list[str],
    batch_size: int = 32,
    normalize: bool = True,
) -> list[list[float]]:
    """ONNX 세션으로 배치 임베딩을 생성한다."""
    from app.services.rag.onnx_session import encode_embedding_onnx_batch

    return encode_embedding_onnx_batch(texts, batch_size=batch_size, normalize=normalize)


def create_embeddings(
    texts: list[str],
    model: Optional[SentenceTransformer] = None,
    batch_size: int = 32,
    normalize: bool = True,
) -> list[list[float]]:
    """
    텍스트 목록을 임베딩 벡터로 변환

    ONNX 모드 활성화 시 ONNX 배치 임베딩을 우선 사용하고,
    실패 시 PyTorch로 자동 fallback한다.

    Args:
        texts: 텍스트 목록
        model: 임베딩 모델 (None이면 자동 로드, ONNX 모드에서는 무시)
        batch_size: 배치 크기
        normalize: L2 정규화 적용

    Returns:
        임베딩 벡터 목록
    """
    # ONNX 경로: model=None (인제스트에서 ONNX 모드일 때) 또는 ONNX 활성화 상태
    if _should_use_onnx_embedding():
        try:
            return _create_embeddings_onnx(texts, batch_size, normalize)
        except Exception as e:
            print(f"[WARN] ONNX 배치 임베딩 실패 → PyTorch fallback: {e}")

    # PyTorch 경로
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
