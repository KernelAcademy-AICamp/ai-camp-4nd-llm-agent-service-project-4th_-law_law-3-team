"""
Data Optimizer - 메모리 효율적 데이터 로딩
"""

import os
from pathlib import Path
from typing import Iterator, Optional, Callable, Union, List, Dict, Any

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader

# Optional imports
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# =============================================================================
# Text Datasets
# =============================================================================

class StreamingTextDataset(IterableDataset):
    """
    메모리 효율적 텍스트 스트리밍 데이터셋
    
    CSV/TSV 파일을 청크 단위로 읽어 메모리 사용량 최소화
    
    Args:
        file_path: 데이터 파일 경로
        text_column: 텍스트 컬럼명
        chunk_size: 청크 크기
        encoding: 파일 인코딩
    
    Example:
        dataset = StreamingTextDataset("data.csv", text_column="content")
        for text in dataset:
            print(text)
    """
    
    def __init__(
        self,
        file_path: str,
        text_column: str = 'text',
        chunk_size: int = 10000,
        encoding: str = 'utf-8'
    ):
        if not HAS_PANDAS:
            raise ImportError("pandas가 필요합니다: pip install pandas")
        
        self.file_path = file_path
        self.text_column = text_column
        self.chunk_size = chunk_size
        self.encoding = encoding
    
    def __iter__(self) -> Iterator[str]:
        for chunk in pd.read_csv(
            self.file_path,
            chunksize=self.chunk_size,
            encoding=self.encoding
        ):
            for text in chunk[self.text_column]:
                if pd.notna(text):
                    yield str(text)


class StreamingJSONDataset(IterableDataset):
    """
    JSON Lines 파일 스트리밍 데이터셋
    
    Args:
        file_path: JSONL 파일 경로
        text_field: 텍스트 필드명
        chunk_size: 청크 크기
    """
    
    def __init__(
        self,
        file_path: str,
        text_field: str = 'text',
        chunk_size: int = 10000
    ):
        if not HAS_PANDAS:
            raise ImportError("pandas가 필요합니다: pip install pandas")
        
        self.file_path = file_path
        self.text_field = text_field
        self.chunk_size = chunk_size
    
    def __iter__(self) -> Iterator[str]:
        for chunk in pd.read_json(
            self.file_path,
            lines=True,
            chunksize=self.chunk_size
        ):
            for text in chunk[self.text_field]:
                if pd.notna(text):
                    yield str(text)


class StreamingParquetDataset(IterableDataset):
    """
    Parquet 파일 스트리밍 데이터셋
    
    Args:
        file_path: Parquet 파일 경로
        columns: 로드할 컬럼 (None이면 전체)
        batch_size: 배치 크기
    """
    
    def __init__(
        self,
        file_path: str,
        columns: Optional[List[str]] = None,
        batch_size: int = 1000
    ):
        if not HAS_PYARROW:
            raise ImportError("pyarrow가 필요합니다: pip install pyarrow")
        
        self.file_path = file_path
        self.columns = columns
        self.batch_size = batch_size
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        parquet_file = pq.ParquetFile(self.file_path)
        
        for batch in parquet_file.iter_batches(
            batch_size=self.batch_size,
            columns=self.columns
        ):
            df = batch.to_pandas()
            for _, row in df.iterrows():
                yield dict(row)


# =============================================================================
# Image Datasets
# =============================================================================

class LazyImageDataset(Dataset):
    """
    Lazy Loading 이미지 데이터셋
    
    이미지 경로만 저장하고, 접근 시점에 로드
    
    Args:
        image_paths: 이미지 경로 리스트
        transform: 이미지 변환 함수
        labels: 레이블 리스트 (선택)
    
    Example:
        paths = ["img1.jpg", "img2.jpg"]
        dataset = LazyImageDataset(paths, transform=transforms.ToTensor())
    """
    
    def __init__(
        self,
        image_paths: List[str],
        transform: Optional[Callable] = None,
        labels: Optional[List[Any]] = None
    ):
        if not HAS_PIL:
            raise ImportError("Pillow가 필요합니다: pip install Pillow")
        
        self.paths = image_paths
        self.transform = transform
        self.labels = labels
    
    def __len__(self) -> int:
        return len(self.paths)
    
    def __getitem__(self, idx: int):
        image = Image.open(self.paths[idx]).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        if self.labels is not None:
            return image, self.labels[idx]
        return image


class LazyImageFolderDataset(Dataset):
    """
    폴더 구조 기반 Lazy Loading 이미지 데이터셋
    
    폴더명을 레이블로 사용
    
    Args:
        root_dir: 루트 디렉토리 (하위 폴더가 클래스)
        transform: 이미지 변환
        extensions: 이미지 확장자
    """
    
    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        extensions: tuple = ('.jpg', '.jpeg', '.png', '.webp')
    ):
        if not HAS_PIL:
            raise ImportError("Pillow가 필요합니다: pip install Pillow")
        
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in extensions:
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


# =============================================================================
# DataLoader Factory
# =============================================================================

def create_optimized_dataloader(
    dataset: Dataset,
    batch_size: int,
    device: torch.device,
    shuffle: bool = True,
    drop_last: bool = False,
    collate_fn: Optional[Callable] = None
) -> DataLoader:
    """
    디바이스에 최적화된 DataLoader 생성
    
    Args:
        dataset: 데이터셋
        batch_size: 배치 크기
        device: 타겟 디바이스
        shuffle: 셔플 여부
        drop_last: 마지막 불완전 배치 드롭
        collate_fn: 커스텀 collate 함수
    
    Returns:
        최적화된 DataLoader
    """
    # MPS는 num_workers > 0에서 문제 발생 가능
    if device.type == "mps":
        num_workers = 0
    else:
        num_workers = min(4, os.cpu_count() or 1)
    
    pin_memory = device.type == "cuda"
    
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
    }
    
    if collate_fn:
        kwargs["collate_fn"] = collate_fn
    
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    
    return DataLoader(dataset, **kwargs)


def seed_worker(worker_id: int):
    """
    DataLoader worker 시드 고정
    
    Usage:
        DataLoader(..., worker_init_fn=seed_worker)
    """
    import random
    import numpy as np
    
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_reproducible_dataloader(
    dataset: Dataset,
    batch_size: int,
    device: torch.device,
    seed: int = 42,
    shuffle: bool = True,
    **kwargs
) -> DataLoader:
    """재현 가능한 DataLoader 생성"""
    
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    return create_optimized_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        device=device,
        shuffle=shuffle,
        **kwargs
    )


# =============================================================================
# Memory Utilities
# =============================================================================

def estimate_dataset_memory(dataset: Dataset, sample_size: int = 100) -> float:
    """
    데이터셋 메모리 사용량 추정 (GB)
    
    Args:
        dataset: 데이터셋
        sample_size: 샘플 크기
    
    Returns:
        추정 메모리 사용량 (GB)
    """
    import sys
    
    sample_size = min(sample_size, len(dataset))
    total_size = 0
    
    for i in range(sample_size):
        item = dataset[i]
        if isinstance(item, tuple):
            for x in item:
                if isinstance(x, torch.Tensor):
                    total_size += x.element_size() * x.nelement()
                else:
                    total_size += sys.getsizeof(x)
        elif isinstance(item, torch.Tensor):
            total_size += item.element_size() * item.nelement()
        else:
            total_size += sys.getsizeof(item)
    
    avg_size = total_size / sample_size
    estimated_total = avg_size * len(dataset)
    
    return estimated_total / 1e9


if __name__ == "__main__":
    # 테스트
    print("Data Optimizer 모듈 로드 완료")
    print(f"Pandas: {HAS_PANDAS}")
    print(f"PyArrow: {HAS_PYARROW}")
    print(f"PIL: {HAS_PIL}")
