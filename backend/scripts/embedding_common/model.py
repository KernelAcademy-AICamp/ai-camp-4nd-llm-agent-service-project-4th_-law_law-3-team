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
    global _cached_model, _cached_device
    from sentence_transformers import SentenceTransformer

    model_name = model_name or str(DEFAULT_CONFIG["EMBEDDING_MODEL"])

    if device is None:
        device = get_device()
        if device == "cuda":
            device_id = get_optimal_cuda_device()
            device = f"cuda:{device_id}"

    # 캐시된 모델이 같은 디바이스면 반환
    if _cached_model is not None and _cached_device == device:
        return _cached_model  # type: ignore[return-value]

    print(f"[INFO] Loading embedding model: {model_name} on {device}")
    model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
    model.eval()

    _cached_model = model
    _cached_device = device

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
    global _cached_model, _cached_device
    _cached_model = None
    _cached_device = None
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
