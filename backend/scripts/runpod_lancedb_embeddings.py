#!/usr/bin/env python3
"""
LanceDB 임베딩 생성 스크립트 (RunPod 전용)

RunPod GPU 환경에서 실행하는 독립형 스크립트입니다.
법령/판례 데이터를 읽어 KURE-v1 임베딩을 생성하고 LanceDB에 저장합니다.

=============================================================================
RunPod 사용법
=============================================================================

1. RunPod에서 GPU Pod 생성
   - GPU: RTX 3090 (24GB) 권장
   - Template: RunPod Pytorch 2.1
   - Container Disk: 20GB

2. Jupyter Lab 접속 후 새 노트북 생성

3. 셀 1: 패키지 설치
   !pip install lancedb sentence-transformers pyarrow ijson psutil tqdm gdown -q

4. 셀 2: Google Drive에서 데이터 다운로드 (권장 - 빠름!)

   # 방법 A: Google Drive 사용 (서버간 전송이라 빠름)
   # 1) 파일을 Google Drive에 업로드
   # 2) 공유 -> "링크가 있는 모든 사용자"로 설정
   # 3) 링크에서 FILE_ID 추출
   #    예: https://drive.google.com/file/d/ABC123xyz/view
   #        FILE_ID = ABC123xyz

   !gdown --id YOUR_LAW_FILE_ID -O law_cleaned.json
   !gdown --id YOUR_PRECEDENT_FILE_ID -O precedents_cleaned.json

   # 방법 B: Jupyter 파일 업로드 (느림)
   # 왼쪽 파일 브라우저에서 업로드 아이콘 클릭

5. 셀 3: 스크립트 실행
   # 이 파일 내용을 새 .py 파일로 저장하거나, 아래처럼 직접 실행
   exec(open('runpod_lancedb_embeddings.py').read())

   # 디바이스 확인
   print_device_info()

6. 셀 4: 임베딩 생성
   # 법령 임베딩
   run_law_embedding('law_cleaned.json', reset=True)

   # 판례 임베딩
   run_precedent_embedding('precedents_cleaned.json', reset=True)

7. 셀 5: 결과 다운로드
   !zip -r lancedb_data.zip ./lancedb_data
   # 왼쪽 파일 브라우저에서 zip 파일 우클릭 -> Download

   # 또는 Google Drive로 업로드
   # from google.colab import drive  # Colab만 해당
   # !cp -r ./lancedb_data /content/drive/MyDrive/

=============================================================================
"""

import gc
import json
import os
import re
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any, Iterator, Union, TypeVar, Generic
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# 청크 설정 타입 변수
ChunkConfigT = TypeVar("ChunkConfigT", bound="BaseChunkConfig")

import pyarrow as pa
from tqdm import tqdm
import torch
from torch.utils.data import IterableDataset, DataLoader

try:
    import ijson
    IJSON_AVAILABLE = True
except ImportError:
    IJSON_AVAILABLE = False
    print("[WARN] ijson not installed. Using full JSON load (higher memory usage)")
    print("       Install with: pip install ijson")


# ============================================================================
# 설정
# ============================================================================

CONFIG = {
    # LanceDB 저장 경로 (현재 작업 디렉토리 기준)
    "LANCEDB_URI": "./lancedb_data",
    "LANCEDB_TABLE_NAME": "legal_chunks",

    # 임베딩 모델
    "EMBEDDING_MODEL": "nlpai-lab/KURE-v1",
    "VECTOR_DIM": 1024,

    # 배치 크기 (자동 설정됨)
    "BATCH_SIZE": 100,

    # 판례 청킹 설정
    "PRECEDENT_CHUNK_SIZE": 1250,
    "PRECEDENT_CHUNK_OVERLAP": 125,
    "PRECEDENT_MIN_CHUNK_SIZE": 100,

    # 법령 청킹 설정 (토큰 기반)
    "LAW_MAX_TOKENS": 800,
    "LAW_MIN_TOKENS": 100,

    # 텍스트 처리 제한
    "MAX_TEXT_LENGTH": 4000,
    "DEFAULT_EMPTY_TEXT": "(내용 없음)",

    # 쿼리 제한
    "MAX_QUERY_LIMIT": 1_000_000,
}


# ============================================================================
# 디바이스 감지 및 최적화
# ============================================================================

@dataclass
class DeviceInfo:
    """디바이스 정보"""
    device: str  # "cuda", "mps", "cpu"
    name: str
    vram_gb: float
    is_laptop: bool = False
    compute_capability: Optional[tuple] = None

    def __str__(self) -> str:
        return f"{self.name} ({self.device}, {self.vram_gb:.1f}GB)"


@dataclass
class OptimalConfig:
    """환경별 최적 설정"""
    batch_size: int
    num_workers: int
    prefetch_factor: int = 2
    gc_interval: int = 10


def get_device() -> str:
    """사용 가능한 최적 디바이스 반환"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_optimal_cuda_device() -> int:
    """VRAM이 가장 큰 CUDA 디바이스 ID 반환 (멀티 GPU 환경)"""
    if not torch.cuda.is_available():
        return 0
    device_count = torch.cuda.device_count()
    if device_count == 1:
        return 0
    return max(
        range(device_count),
        key=lambda i: torch.cuda.get_device_properties(i).total_memory
    )


def get_device_info() -> DeviceInfo:
    """디바이스 상세 정보 조회"""
    device = get_device()

    if device == "cuda":
        device_id = get_optimal_cuda_device()
        props = torch.cuda.get_device_properties(device_id)
        name = props.name
        vram_gb = props.total_memory / (1024**3)
        compute_capability = (props.major, props.minor)

        is_laptop = any(
            kw in name.lower()
            for kw in ["laptop", "mobile", "max-q", "notebook"]
        )

        return DeviceInfo(
            device=device,
            name=name,
            vram_gb=vram_gb,
            is_laptop=is_laptop,
            compute_capability=compute_capability,
        )

    elif device == "mps":
        try:
            import psutil
            total_ram = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            total_ram = 16.0

        usable_memory = total_ram * 0.75

        return DeviceInfo(
            device=device,
            name="Apple Silicon (MPS)",
            vram_gb=usable_memory,
            is_laptop=True,
        )

    else:
        try:
            import psutil
            total_ram = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            total_ram = 8.0

        return DeviceInfo(
            device=device,
            name="CPU",
            vram_gb=total_ram,
        )


def get_optimal_config(device_info: Optional[DeviceInfo] = None) -> OptimalConfig:
    """디바이스에 따른 최적 설정 반환"""
    if device_info is None:
        device_info = get_device_info()

    device = device_info.device
    vram = device_info.vram_gb

    if device == "cuda":
        if vram >= 20:  # RTX 3090/4090 (24GB)
            return OptimalConfig(
                batch_size=128,
                num_workers=4,
                gc_interval=25,
            )
        elif vram >= 14:  # RTX 4080, 3080 Ti
            return OptimalConfig(
                batch_size=100,
                num_workers=4,
                gc_interval=20,
            )
        elif vram >= 8:  # RTX 3070, 4060
            return OptimalConfig(
                batch_size=70,
                num_workers=2,
                gc_interval=15,
            )
        else:
            return OptimalConfig(
                batch_size=50,
                num_workers=2,
                gc_interval=10,
            )

    elif device == "mps":
        if vram >= 12:
            return OptimalConfig(
                batch_size=50,
                num_workers=0,
                gc_interval=10,
            )
        else:
            return OptimalConfig(
                batch_size=30,
                num_workers=0,
                gc_interval=5,
            )

    else:
        return OptimalConfig(
            batch_size=20,
            num_workers=2,
            gc_interval=5,
        )


def print_device_info():
    """디바이스 정보 및 최적 설정 출력"""
    device_info = get_device_info()
    config = get_optimal_config(device_info)

    print("=" * 60)
    print("Device Information")
    print("=" * 60)
    print(f"  Device: {device_info.device.upper()}")
    print(f"  Name: {device_info.name}")
    print(f"  Memory: {device_info.vram_gb:.1f} GB")
    if device_info.is_laptop:
        print(f"  Type: Laptop/Mobile")
    if device_info.compute_capability:
        cc = device_info.compute_capability
        print(f"  Compute Capability: {cc[0]}.{cc[1]}")

    print("\nRecommended Settings:")
    print(f"  batch_size: {config.batch_size}")
    print(f"  num_workers: {config.num_workers}")
    print(f"  gc_interval: {config.gc_interval} batches")
    print("=" * 60)

    return device_info, config


# ============================================================================
# LanceDB 스키마 정의 (v2)
# ============================================================================
#
# ⚠️ 중요: 이 스키마는 백엔드의 schema_v2.py와 반드시 동기화되어야 합니다!
# 위치: backend/app/common/vectorstore/schema_v2.py
#
# 스키마 변경 시 체크리스트:
# 1. 백엔드 schema_v2.py 수정
# 2. 이 파일의 LEGAL_CHUNKS_SCHEMA 수정
# 3. runpod_split_embeddings.py의 스키마도 수정
# ============================================================================

VECTOR_DIM = CONFIG["VECTOR_DIM"]

LEGAL_CHUNKS_SCHEMA = pa.schema([
    # 공통 필드
    pa.field("id", pa.utf8()),
    pa.field("source_id", pa.utf8()),
    pa.field("data_type", pa.utf8()),
    pa.field("title", pa.utf8()),
    pa.field("content", pa.utf8()),
    pa.field("vector", pa.list_(pa.float32(), VECTOR_DIM)),
    pa.field("date", pa.utf8()),
    pa.field("source_name", pa.utf8()),
    pa.field("chunk_index", pa.int32()),
    pa.field("total_chunks", pa.int32()),

    # 법령 전용
    pa.field("promulgation_date", pa.utf8()),
    pa.field("promulgation_no", pa.utf8()),
    pa.field("law_type", pa.utf8()),
    pa.field("article_no", pa.utf8()),

    # 판례 전용
    pa.field("case_number", pa.utf8()),
    pa.field("case_type", pa.utf8()),
    pa.field("judgment_type", pa.utf8()),
    pa.field("judgment_status", pa.utf8()),
    pa.field("reference_provisions", pa.utf8()),
    pa.field("reference_cases", pa.utf8()),
])


def create_law_chunk(
    source_id: str,
    chunk_index: int,
    title: str,
    content: str,
    vector: List[float],
    enforcement_date: str,
    department: str,
    total_chunks: int = 1,
    promulgation_date: Optional[str] = None,
    promulgation_no: Optional[str] = None,
    law_type: Optional[str] = None,
    article_no: Optional[str] = None,
) -> Dict[str, Any]:
    """법령 청크 레코드 생성"""
    return {
        "id": f"law_{source_id}_{chunk_index}",  # 접두사로 ID 충돌 방지
        "source_id": source_id,
        "data_type": "법령",
        "title": title,
        "content": content,
        "vector": vector,
        "date": enforcement_date,
        "source_name": department,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "promulgation_date": promulgation_date,
        "promulgation_no": promulgation_no,
        "law_type": law_type,
        "article_no": article_no,
        "case_number": None,
        "case_type": None,
        "judgment_type": None,
        "judgment_status": None,
        "reference_provisions": None,
        "reference_cases": None,
    }


def create_precedent_chunk(
    source_id: str,
    chunk_index: int,
    title: str,
    content: str,
    vector: List[float],
    decision_date: str,
    court_name: str,
    total_chunks: int = 1,
    case_number: Optional[str] = None,
    case_type: Optional[str] = None,
    judgment_type: Optional[str] = None,
    judgment_status: Optional[str] = None,
    reference_provisions: Optional[str] = None,
    reference_cases: Optional[str] = None,
) -> Dict[str, Any]:
    """판례 청크 레코드 생성"""
    return {
        "id": f"prec_{source_id}_{chunk_index}",  # 접두사로 ID 충돌 방지
        "source_id": source_id,
        "data_type": "판례",
        "title": title,
        "content": content,
        "vector": vector,
        "date": decision_date,
        "source_name": court_name,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "promulgation_date": None,
        "promulgation_no": None,
        "law_type": None,
        "article_no": None,
        "case_number": case_number,
        "case_type": case_type,
        "judgment_type": judgment_type,
        "judgment_status": judgment_status,
        "reference_provisions": reference_provisions,
        "reference_cases": reference_cases,
    }


# ============================================================================
# LanceDB Store
# ============================================================================

class LanceDBStore:
    """LanceDB 벡터 저장소"""

    def __init__(self, db_path: Optional[str] = None, table_name: Optional[str] = None):
        import lancedb

        db_path = db_path or CONFIG["LANCEDB_URI"]
        self.table_name = table_name or CONFIG["LANCEDB_TABLE_NAME"]

        Path(db_path).mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(db_path)
        self._table = None

        if self.table_name in self.db.table_names():
            self._table = self.db.open_table(self.table_name)

    def _ensure_table(self):
        if self._table is None:
            self._table = self.db.create_table(
                self.table_name,
                schema=LEGAL_CHUNKS_SCHEMA,
            )
        return self._table

    def add_law_documents(
        self,
        source_ids: List[str],
        chunk_indices: List[int],
        embeddings: List[List[float]],
        titles: List[str],
        contents: List[str],
        enforcement_dates: List[str],
        departments: List[str],
        total_chunks_list: List[int],
        promulgation_dates: List[str],
        promulgation_nos: List[str],
        law_types: List[str],
        article_nos: List[str],
    ) -> None:
        """법령 문서 배치 추가"""
        if not source_ids:
            return

        table = self._ensure_table()
        data = []

        for i in range(len(source_ids)):
            chunk = create_law_chunk(
                source_id=source_ids[i],
                chunk_index=chunk_indices[i],
                title=titles[i],
                content=contents[i],
                vector=embeddings[i],
                enforcement_date=enforcement_dates[i],
                department=departments[i],
                total_chunks=total_chunks_list[i],
                promulgation_date=promulgation_dates[i] if promulgation_dates else None,
                promulgation_no=promulgation_nos[i] if promulgation_nos else None,
                law_type=law_types[i] if law_types else None,
                article_no=article_nos[i] if article_nos else None,
            )
            data.append(chunk)

        if data:
            table.add(data)

    def add_precedent_documents(
        self,
        source_ids: List[str],
        chunk_indices: List[int],
        embeddings: List[List[float]],
        titles: List[str],
        contents: List[str],
        decision_dates: List[str],
        court_names: List[str],
        total_chunks_list: List[int],
        case_numbers: Optional[List[str]] = None,
        case_types: Optional[List[str]] = None,
        reference_provisions_list: Optional[List[str]] = None,
        reference_cases_list: Optional[List[str]] = None,
    ) -> None:
        """판례 문서 배치 추가"""
        if not source_ids:
            return

        table = self._ensure_table()
        data = []

        for i in range(len(source_ids)):
            chunk = create_precedent_chunk(
                source_id=source_ids[i],
                chunk_index=chunk_indices[i],
                title=titles[i],
                content=contents[i],
                vector=embeddings[i],
                decision_date=decision_dates[i],
                court_name=court_names[i],
                total_chunks=total_chunks_list[i],
                case_number=case_numbers[i] if case_numbers else None,
                case_type=case_types[i] if case_types else None,
                reference_provisions=reference_provisions_list[i] if reference_provisions_list else None,
                reference_cases=reference_cases_list[i] if reference_cases_list else None,
            )
            data.append(chunk)

        if data:
            table.add(data)

    def count(self) -> int:
        if self._table is None:
            return 0
        return len(self._table)

    def count_by_type(self, data_type: str) -> int:
        if self._table is None:
            return 0
        try:
            max_limit = CONFIG["MAX_QUERY_LIMIT"]
            result = self._table.search().where(f"data_type = '{data_type}'").limit(max_limit).to_arrow()
            return result.num_rows
        except (ValueError, KeyError, AttributeError) as e:
            print(f"[WARN] count_by_type failed: {e}")
            return 0

    def get_existing_source_ids(self, data_type: str) -> set:
        """이미 저장된 source_id 조회"""
        if self._table is None:
            return set()
        try:
            max_limit = CONFIG["MAX_QUERY_LIMIT"]
            result = self._table.search().where(f"data_type = '{data_type}'").select(["source_id"]).limit(max_limit).to_arrow()
            source_ids = result.column("source_id").to_pylist()
            return set(source_ids)
        except (ValueError, KeyError, AttributeError) as e:
            print(f"[WARN] get_existing_source_ids failed: {e}")
            return set()

    def delete_by_type(self, data_type: str) -> int:
        """특정 유형 데이터 삭제"""
        if self._table is None:
            return 0
        count = self.count_by_type(data_type)
        if count > 0:
            self._table.delete(f"data_type = '{data_type}'")
        return count

    def reset(self):
        """테이블 초기화"""
        if self.table_name in self.db.table_names():
            self.db.drop_table(self.table_name)
        self._table = None

    def compact(self):
        """테이블 압축 (메모리 최적화)"""
        if self._table is not None:
            try:
                self._table.compact_files()
                print("[INFO] Table compacted")
            except Exception as e:
                print(f"[WARN] Compact failed (non-critical): {e}")


# ============================================================================
# 임베딩 모델 (KURE-v1)
# ============================================================================

_local_model = None
_current_device = None


def get_embedding_model(device: str = None):
    """KURE-v1 모델 로드"""
    global _local_model, _current_device

    if device is None:
        device = get_device()

    if _local_model is None or _current_device != device:
        from sentence_transformers import SentenceTransformer

        model_name = CONFIG["EMBEDDING_MODEL"]
        print(f"[INFO] Loading model: {model_name}")
        print(f"[INFO] Device: {device.upper()}")

        _local_model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            device=device,
        )
        _current_device = device

        dim = _local_model.get_sentence_embedding_dimension()
        print(f"[INFO] Model loaded. Dimension: {dim}")

        if dim != VECTOR_DIM:
            raise ValueError(f"Model dimension ({dim}) != Schema dimension ({VECTOR_DIM})")

    return _local_model


def create_embeddings(texts: List[str], device: Optional[str] = None) -> List[List[float]]:
    """임베딩 생성"""
    model = get_embedding_model(device)
    max_len = CONFIG["MAX_TEXT_LENGTH"]
    default_text = CONFIG["DEFAULT_EMPTY_TEXT"]
    processed = [t.strip()[:max_len] if t else default_text for t in texts]
    embeddings = model.encode(processed, show_progress_bar=False)
    result = [emb.tolist() for emb in embeddings]

    # 메모리 즉시 해제
    del embeddings
    del processed

    return result


def clear_model_cache():
    """모델 캐시 정리"""
    global _local_model, _current_device

    if _local_model is not None:
        del _local_model
        _local_model = None
        _current_device = None
        clear_memory()
        print("[INFO] Model cache cleared")


def print_memory_status() -> None:
    """현재 메모리 사용량 출력"""
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"[Memory] RAM: {mem.used / 1024**3:.1f}GB / {mem.total / 1024**3:.1f}GB ({mem.percent}%)")

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3
            print(f"[Memory] GPU: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved, {max_allocated:.1f}GB max")
    except ImportError:
        print("[Memory] psutil not installed")


def clear_memory() -> None:
    """메모리 정리 유틸리티 (PyTorch Optimization 패턴)"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """
    모든 랜덤 시드 고정 (재현성 확보)

    Args:
        seed: 시드 값
        deterministic: 결정론적 모드 활성화 (성능 저하 있음)
    """
    import random
    import numpy as np

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
            torch.use_deterministic_algorithms(True, warn_only=True)


# ============================================================================
# 임베딩 캐싱 (Embedding Pipeline 패턴)
# ============================================================================

class EmbeddingCache:
    """
    해시 기반 임베딩 캐시

    동일 텍스트 재임베딩 방지를 위한 디스크 캐시.
    텍스트 해시를 키로 사용하여 임베딩을 저장/조회.
    """

    def __init__(self, cache_dir: str = "./embedding_cache", model_name: Optional[str] = None):
        """
        Args:
            cache_dir: 캐시 저장 디렉토리
            model_name: 모델명 (캐시 키에 포함)
        """
        import hashlib

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name or CONFIG["EMBEDDING_MODEL"]
        self._hashlib = hashlib
        self._memory_cache: Dict[str, List[float]] = {}
        self._hits = 0
        self._misses = 0

    def _get_hash(self, text: str) -> str:
        """텍스트 해시 생성"""
        key = f"{self.model_name}:{text}"
        return self._hashlib.md5(key.encode()).hexdigest()

    def _get_cache_path(self, text_hash: str) -> Path:
        """캐시 파일 경로"""
        # 해시 앞 2글자로 서브디렉토리 생성 (파일 분산)
        subdir = self.cache_dir / text_hash[:2]
        subdir.mkdir(exist_ok=True)
        return subdir / f"{text_hash}.json"

    def get(self, text: str) -> Optional[List[float]]:
        """캐시에서 임베딩 조회"""
        text_hash = self._get_hash(text)

        # 메모리 캐시 확인
        if text_hash in self._memory_cache:
            self._hits += 1
            return self._memory_cache[text_hash]

        # 디스크 캐시 확인
        cache_path = self._get_cache_path(text_hash)
        if cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    embedding = json.load(f)
                self._memory_cache[text_hash] = embedding
                self._hits += 1
                return embedding
            except (json.JSONDecodeError, IOError):
                pass

        self._misses += 1
        return None

    def set(self, text: str, embedding: List[float]) -> None:
        """임베딩 캐시에 저장"""
        text_hash = self._get_hash(text)

        # 메모리 캐시 저장
        self._memory_cache[text_hash] = embedding

        # 디스크 캐시 저장
        cache_path = self._get_cache_path(text_hash)
        try:
            with open(cache_path, "w") as f:
                json.dump(embedding, f)
        except IOError as e:
            print(f"[WARN] Cache write failed: {e}")

    def get_or_compute(
        self,
        text: str,
        compute_fn: callable,
        device: Optional[str] = None
    ) -> List[float]:
        """캐시 조회 후 없으면 계산"""
        cached = self.get(text)
        if cached is not None:
            return cached

        embedding = compute_fn([text], device)[0]
        self.set(text, embedding)
        return embedding

    def get_stats(self) -> Dict[str, int]:
        """캐시 통계"""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total": total,
            "hit_rate": f"{hit_rate:.1f}%",
            "memory_cache_size": len(self._memory_cache),
        }

    def clear_memory_cache(self) -> None:
        """메모리 캐시만 정리"""
        self._memory_cache.clear()

    def clear_all(self) -> None:
        """모든 캐시 삭제"""
        import shutil
        self._memory_cache.clear()
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0
        print("[INFO] All cache cleared")


# ============================================================================
# 임베딩 품질 검증 (Embedding Pipeline 패턴)
# ============================================================================

class EmbeddingQualityChecker:
    """
    임베딩 품질 검증 유틸리티

    유사/비유사 문서 쌍의 코사인 유사도를 측정하여
    임베딩 품질을 평가.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or get_device()

    def compute_similarity(self, text1: str, text2: str) -> float:
        """두 텍스트의 코사인 유사도 계산"""
        embeddings = create_embeddings([text1, text2], self.device)
        emb1 = torch.tensor(embeddings[0])
        emb2 = torch.tensor(embeddings[1])

        # 코사인 유사도
        similarity = torch.nn.functional.cosine_similarity(
            emb1.unsqueeze(0), emb2.unsqueeze(0)
        ).item()

        return similarity

    def evaluate(
        self,
        similar_pairs: List[tuple],
        dissimilar_pairs: List[tuple],
    ) -> Dict[str, float]:
        """
        유사/비유사 쌍으로 임베딩 품질 평가

        Args:
            similar_pairs: 유사해야 하는 텍스트 쌍 리스트 [(text1, text2), ...]
            dissimilar_pairs: 비유사해야 하는 텍스트 쌍 리스트

        Returns:
            평가 결과 딕셔너리
        """
        similar_scores = []
        dissimilar_scores = []

        print("[INFO] Evaluating similar pairs...")
        for text1, text2 in tqdm(similar_pairs, desc="Similar"):
            score = self.compute_similarity(text1, text2)
            similar_scores.append(score)

        print("[INFO] Evaluating dissimilar pairs...")
        for text1, text2 in tqdm(dissimilar_pairs, desc="Dissimilar"):
            score = self.compute_similarity(text1, text2)
            dissimilar_scores.append(score)

        similar_avg = sum(similar_scores) / len(similar_scores) if similar_scores else 0
        dissimilar_avg = sum(dissimilar_scores) / len(dissimilar_scores) if dissimilar_scores else 0
        separation = similar_avg - dissimilar_avg

        report = {
            "similar_avg": similar_avg,
            "similar_min": min(similar_scores) if similar_scores else 0,
            "similar_max": max(similar_scores) if similar_scores else 0,
            "dissimilar_avg": dissimilar_avg,
            "dissimilar_min": min(dissimilar_scores) if dissimilar_scores else 0,
            "dissimilar_max": max(dissimilar_scores) if dissimilar_scores else 0,
            "separation": separation,
            "quality": "good" if separation > 0.2 else "fair" if separation > 0.1 else "poor",
        }

        print("\n" + "=" * 50)
        print("Embedding Quality Report")
        print("=" * 50)
        print(f"  Similar pairs avg:    {similar_avg:.4f}")
        print(f"  Dissimilar pairs avg: {dissimilar_avg:.4f}")
        print(f"  Separation:           {separation:.4f}")
        print(f"  Quality:              {report['quality'].upper()}")
        print("=" * 50)

        return report

    def quick_test(self) -> Dict[str, float]:
        """법률 도메인 기본 테스트"""
        similar_pairs = [
            ("손해배상 청구권", "손해배상 청구"),
            ("민법 제750조 불법행위", "민법상 불법행위 책임"),
            ("임대차 계약 해지", "임대차계약의 해지"),
            ("형사소송법 제309조", "형사소송법 309조"),
        ]

        dissimilar_pairs = [
            ("민법 제750조 불법행위", "형법 제250조 살인죄"),
            ("손해배상 청구권", "회사 설립 절차"),
            ("임대차 계약", "특허권 침해"),
            ("형사소송법", "민사집행법"),
        ]

        return self.evaluate(similar_pairs, dissimilar_pairs)


# ============================================================================
# 통합 임베딩 프로세서 (스트리밍 방식)
# ============================================================================

@dataclass
class EmbeddingStats:
    """임베딩 처리 통계"""
    total_docs: int = 0
    processed_docs: int = 0
    total_chunks: int = 0
    skipped: int = 0
    errors: int = 0
    device: str = ""

    def to_dict(self) -> dict:
        return {
            "total_docs": self.total_docs,
            "processed_docs": self.processed_docs,
            "total_chunks": self.total_chunks,
            "skipped": self.skipped,
            "errors": self.errors,
            "device": self.device,
        }


class StreamingEmbeddingProcessor(ABC, Generic[ChunkConfigT]):
    """
    스트리밍 방식 임베딩 프로세서 (베이스 클래스)

    특징:
    - 개수 세기 스킵으로 즉시 시작
    - tqdm에서 처리 속도(it/s)만 표시 (진행률 % 미표시)
    - 스트리밍으로 메모리 효율적 처리

    Type Parameters:
        ChunkConfigT: 청킹 설정 타입 (PrecedentChunkConfig 또는 LawChunkConfig)
    """

    def __init__(self, data_type: str):
        self.data_type = data_type
        self.device_info = get_device_info()
        self.optimal_config = get_optimal_config(self.device_info)
        self.store = LanceDBStore()
        self.stats = EmbeddingStats(device=str(self.device_info))

    @abstractmethod
    def get_chunk_config(self) -> ChunkConfigT:
        """청킹 설정 반환"""
        pass

    @abstractmethod
    def extract_source_id(self, item: Dict[str, Any], idx: int) -> str:
        """아이템에서 소스 ID 추출"""
        pass

    @abstractmethod
    def extract_text_for_embedding(self, item: Dict[str, Any]) -> str:
        """임베딩할 텍스트 추출"""
        pass

    @abstractmethod
    def chunk_text(self, text: str, config: ChunkConfigT) -> List[tuple]:
        """텍스트를 청크로 분할"""
        pass

    @abstractmethod
    def extract_metadata(self, item: Dict[str, Any]) -> Dict[str, str]:
        """메타데이터 추출"""
        pass

    @abstractmethod
    def create_batch_data(self) -> Dict[str, List]:
        """빈 배치 데이터 구조 생성"""
        pass

    @abstractmethod
    def add_to_batch(
        self,
        batch_data: Dict[str, List],
        source_id: str,
        chunk_idx: int,
        chunk_content: str,
        total_chunks: int,
        metadata: Dict[str, str],
    ) -> None:
        """배치에 데이터 추가"""
        pass

    @abstractmethod
    def save_batch(self, batch_data: Dict[str, List], embeddings: List[List[float]]) -> int:
        """배치 저장, 저장된 개수 반환"""
        pass

    def load_streaming(self, source_path: str) -> tuple:
        """
        스트리밍 로드 (개수 세기 스킵)

        Returns:
            (file_handle, iterator) 또는 (None, list)
        """
        if not IJSON_AVAILABLE:
            print("[INFO] ijson not available, using full load")
            return None, load_json_full(source_path)

        json_format = detect_json_format(source_path)
        print(f"[INFO] JSON format: {json_format}, using streaming (low memory)")
        print("[INFO] Skipping count (immediate start for large files)")
        print("[INFO] Progress bar will show speed (it/s) instead of percentage")

        result = load_json_streaming(source_path)
        if result:
            return result  # (file_handle, iterator)

        print("[WARN] Streaming failed, falling back to full load")
        return None, load_json_full(source_path)

    def run(
        self,
        source_path: str,
        reset: bool = False,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        임베딩 실행

        Args:
            source_path: JSON 파일 경로
            reset: 기존 데이터 삭제 후 시작
            batch_size: 배치 크기 (None=자동)

        Returns:
            처리 통계 딕셔너리
        """
        print("=" * 60)
        print(f"{self.data_type} 임베딩 시작")
        print("=" * 60)

        batch_size = batch_size or self.optimal_config.batch_size

        print(f"  Device: {self.device_info}")
        print(f"  Batch size: {batch_size}")
        print(f"  GC interval: every batch")
        print(f"  Source: {source_path}")
        print(f"  Reset: {reset}")

        chunk_config = self.get_chunk_config()

        print(f"\n[INFO] Loading {self.data_type} data from: {source_path}")

        # 리셋 처리
        if reset:
            deleted = self.store.delete_by_type(self.data_type)
            print(f"[INFO] Deleted {deleted} existing {self.data_type} chunks")

        # 기존 소스 ID 조회 (증분 처리용)
        existing_source_ids = (
            self.store.get_existing_source_ids(self.data_type) if not reset else set()
        )
        if existing_source_ids:
            print(f"[INFO] Found {len(existing_source_ids)} {self.data_type} already embedded")
            self.stats.skipped = len(existing_source_ids)

        # 스트리밍 로드
        file_handle, items = self.load_streaming(source_path)

        # 모델 로드
        print("[INFO] Loading embedding model (this may take a while on first run)...")
        get_embedding_model(self.device_info.device)
        print("[INFO] Model loaded successfully!")

        # 배치 처리
        batch_data = self.create_batch_data()
        start_time = datetime.now()
        batch_count = 0
        first_item_logged = False

        # tqdm: total=None으로 개수 미표시, 속도(it/s)만 표시
        for idx, item in enumerate(tqdm(items, desc=f"{self.data_type} 처리", total=None)):
            if not first_item_logged:
                print(f"\n>>> 첫 번째 {self.data_type} 데이터 로드 성공! 임베딩 진행 중...")
                first_item_logged = True

            source_id = self.extract_source_id(item, idx)

            # 이미 처리된 항목 스킵
            if source_id in existing_source_ids:
                continue

            # 텍스트 추출
            text = self.extract_text_for_embedding(item)
            if not text:
                self.stats.skipped += 1
                continue

            # 청킹
            chunks = self.chunk_text(text, chunk_config)
            if not chunks:
                self.stats.skipped += 1
                continue

            total_chunks = len(chunks)
            self.stats.processed_docs += 1

            # 메타데이터 추출
            metadata = self.extract_metadata(item)

            # 배치에 추가
            for chunk_idx, chunk_content in chunks:
                self.add_to_batch(
                    batch_data, source_id, chunk_idx, chunk_content, total_chunks, metadata
                )

            # 배치 처리
            if len(batch_data["contents"]) >= batch_size:
                try:
                    embeddings = create_embeddings(
                        batch_data["contents"], self.device_info.device
                    )
                    saved = self.save_batch(batch_data, embeddings)
                    self.stats.total_chunks += saved

                    del embeddings
                except (RuntimeError, ValueError, MemoryError) as e:
                    print(f"[ERROR] Batch failed: {type(e).__name__}: {e}")
                    self.stats.errors += 1

                # 배치 초기화
                batch_data = self.create_batch_data()
                batch_count += 1

                # 메모리 정리
                clear_memory()

                # 주기적 압축 및 메모리 상태
                if batch_count % 50 == 0:
                    self.store.compact()
                    print_memory_status()

        # 남은 배치 처리
        if batch_data["contents"]:
            try:
                embeddings = create_embeddings(
                    batch_data["contents"], self.device_info.device
                )
                saved = self.save_batch(batch_data, embeddings)
                self.stats.total_chunks += saved
                del embeddings
            except (RuntimeError, ValueError, MemoryError) as e:
                print(f"[ERROR] Final batch failed: {type(e).__name__}: {e}")
                self.stats.errors += 1

        # 파일 핸들 정리
        if file_handle:
            file_handle.close()

        # 최종 정리
        clear_memory()
        self.store.compact()

        # 결과 출력
        elapsed = (datetime.now() - start_time).total_seconds()
        print("\n" + "=" * 60)
        print(f"{self.data_type} 임베딩 완료")
        print("=" * 60)
        print(f"  처리 문서: {self.stats.processed_docs:,}")
        print(f"  생성 청크: {self.stats.total_chunks:,}")
        print(f"  스킵: {self.stats.skipped:,}")
        print(f"  에러: {self.stats.errors}")
        print(f"  소요 시간: {elapsed:.1f}초")
        if self.stats.total_chunks > 0:
            print(f"  처리 속도: {self.stats.total_chunks / elapsed:.1f} chunks/sec")

        return self.stats.to_dict()

    def run_part(
        self,
        source_path: str,
        reset: bool = False,
        batch_size: int = 64,
    ) -> Dict[str, Any]:
        """
        분할된 JSON 파일 하나를 처리 (전체 로드 방식)

        Args:
            source_path: 분할된 JSON 파일 경로
            reset: 기존 데이터 삭제 (첫 파일에만 True)
            batch_size: 배치 크기 (기본: 64, 메모리 절약)

        Returns:
            처리 통계 딕셔너리
        """
        print("=" * 60)
        print(f"{self.data_type} 임베딩 (분할 파일): {source_path}")
        print("=" * 60)

        print(f"  Device: {self.device_info}")
        print(f"  Batch size: {batch_size}")
        print(f"  Reset: {reset}")

        # 데이터 로드
        with open(source_path, "r", encoding="utf-8") as f:
            items = json.load(f)

        print(f"[INFO] Items: {len(items):,}")
        print_memory_status()

        chunk_config = self.get_chunk_config()

        if reset:
            deleted = self.store.delete_by_type(self.data_type)
            print(f"[INFO] Deleted {deleted} existing {self.data_type} chunks")

        # 통계 초기화
        self.stats = EmbeddingStats(device=str(self.device_info))
        self.stats.total_docs = len(items)

        # 모델 로드
        get_embedding_model(self.device_info.device)

        # 배치 데이터
        batch_data = self.create_batch_data()
        start_time = datetime.now()

        for item in tqdm(items, desc=f"{self.data_type} 처리"):
            source_id = self.extract_source_id(item, 0)

            # 텍스트 추출
            text = self.extract_text_for_embedding(item)
            if not text:
                self.stats.skipped += 1
                continue

            # 청킹
            chunks = self.chunk_text(text, chunk_config)
            if not chunks:
                self.stats.skipped += 1
                continue

            total_chunks = len(chunks)
            self.stats.processed_docs += 1

            # 메타데이터 추출
            metadata = self.extract_metadata(item)

            # 배치에 추가
            for chunk_idx, chunk_content in chunks:
                self.add_to_batch(
                    batch_data, source_id, chunk_idx, chunk_content, total_chunks, metadata
                )

            # 배치 저장
            if len(batch_data["contents"]) >= batch_size:
                try:
                    embeddings = create_embeddings(
                        batch_data["contents"], self.device_info.device
                    )
                    saved = self.save_batch(batch_data, embeddings)
                    self.stats.total_chunks += saved
                    del embeddings
                except (RuntimeError, ValueError, MemoryError) as e:
                    self.stats.errors += len(batch_data["contents"])
                    print(f"  [ERROR] Batch error: {type(e).__name__}: {e}")

                # 배치 데이터 클리어
                batch_data = self.create_batch_data()
                clear_memory()

        # 남은 배치 처리
        if batch_data["contents"]:
            try:
                embeddings = create_embeddings(
                    batch_data["contents"], self.device_info.device
                )
                saved = self.save_batch(batch_data, embeddings)
                self.stats.total_chunks += saved
                del embeddings
            except (RuntimeError, ValueError, MemoryError) as e:
                self.stats.errors += len(batch_data["contents"])
                print(f"  [ERROR] Final batch error: {type(e).__name__}: {e}")

        # 정리
        del items
        self.store.compact()
        clear_memory()

        elapsed = datetime.now() - start_time
        print(f"\n완료! 소요시간: {elapsed}")
        print(f"  Processed: {self.stats.processed_docs:,}")
        print(f"  Chunks: {self.stats.total_chunks:,}")
        print(f"  Skipped: {self.stats.skipped:,}")
        print_memory_status()

        return self.stats.to_dict()


class LawEmbeddingProcessor(StreamingEmbeddingProcessor[LawChunkConfig]):
    """법령 임베딩 프로세서"""

    def __init__(self):
        super().__init__("법령")

    def get_chunk_config(self) -> LawChunkConfig:
        return LawChunkConfig()

    def extract_source_id(self, item: Dict[str, Any], idx: int) -> str:
        return item.get("law_id", "")

    def extract_text_for_embedding(self, item: Dict[str, Any]) -> str:
        return item.get("content", "")

    def chunk_text(self, text: str, config: LawChunkConfig) -> List[tuple]:
        return chunk_law_content(text, config)

    def extract_metadata(self, item: Dict[str, Any]) -> Dict[str, str]:
        return {
            "title": item.get("law_name", ""),
            "enforcement_date": item.get("enforcement_date", ""),
            "department": item.get("department", ""),
            "promulgation_date": item.get("promulgation_date", ""),
            "promulgation_no": item.get("promulgation_no", ""),
            "law_type": item.get("law_type", ""),
        }

    def create_batch_data(self) -> Dict[str, List]:
        return {
            "source_ids": [],
            "chunk_indices": [],
            "contents": [],
            "titles": [],
            "enforcement_dates": [],
            "departments": [],
            "total_chunks_list": [],
            "promulgation_dates": [],
            "promulgation_nos": [],
            "law_types": [],
            "article_nos": [],
        }

    def add_to_batch(
        self,
        batch_data: Dict[str, List],
        source_id: str,
        chunk_idx: int,
        chunk_content: str,
        total_chunks: int,
        metadata: Dict[str, str],
    ) -> None:
        batch_data["source_ids"].append(source_id)
        batch_data["chunk_indices"].append(chunk_idx)
        batch_data["contents"].append(chunk_content)
        batch_data["titles"].append(metadata["title"])
        batch_data["enforcement_dates"].append(metadata["enforcement_date"])
        batch_data["departments"].append(metadata["department"])
        batch_data["total_chunks_list"].append(total_chunks)
        batch_data["promulgation_dates"].append(metadata["promulgation_date"])
        batch_data["promulgation_nos"].append(metadata["promulgation_no"])
        batch_data["law_types"].append(metadata["law_type"])
        batch_data["article_nos"].append("")  # article_no는 chunk_law_content에서 처리

    def save_batch(self, batch_data: Dict[str, List], embeddings: List[List[float]]) -> int:
        self.store.add_law_documents(
            source_ids=batch_data["source_ids"],
            chunk_indices=batch_data["chunk_indices"],
            embeddings=embeddings,
            titles=batch_data["titles"],
            contents=batch_data["contents"],
            enforcement_dates=batch_data["enforcement_dates"],
            departments=batch_data["departments"],
            total_chunks_list=batch_data["total_chunks_list"],
            promulgation_dates=batch_data["promulgation_dates"],
            promulgation_nos=batch_data["promulgation_nos"],
            law_types=batch_data["law_types"],
            article_nos=batch_data["article_nos"],
        )
        return len(batch_data["source_ids"])


class PrecedentEmbeddingProcessor(StreamingEmbeddingProcessor[PrecedentChunkConfig]):
    """판례 임베딩 프로세서"""

    def __init__(self):
        super().__init__("판례")

    def get_chunk_config(self) -> PrecedentChunkConfig:
        return PrecedentChunkConfig()

    def extract_source_id(self, item: Dict[str, Any], idx: int) -> str:
        return str(item.get("판례정보일련번호", item.get("id", idx)))

    def extract_text_for_embedding(self, item: Dict[str, Any]) -> str:
        parts = []
        case_name = item.get("사건명", item.get("case_name", ""))
        if case_name:
            parts.append(f"[{case_name}]")

        summary = item.get("판시사항", item.get("summary", ""))
        if summary:
            parts.append(summary)

        judgment_summary = item.get("판결요지", item.get("judgment_summary", ""))
        if judgment_summary:
            parts.append(judgment_summary)

        return "\n".join(parts)

    def chunk_text(self, text: str, config: PrecedentChunkConfig) -> List[tuple]:
        if not text or len(text) < config.min_chunk_size:
            return [(0, text)] if text else []
        return chunk_precedent_text(text, config)

    def extract_metadata(self, item: Dict[str, Any]) -> Dict[str, str]:
        return {
            "case_name": item.get("사건명", item.get("case_name", "")),
            "case_number": item.get("사건번호", item.get("case_number", "")),
            "decision_date": item.get("선고일자", item.get("decision_date", "")),
            "court_name": item.get("법원명", item.get("court_name", "")),
            "case_type": item.get("사건종류명", item.get("case_type", "")),
            "reference_provisions": item.get("참조조문", item.get("reference_provisions", "")),
            "reference_cases": item.get("참조판례", item.get("reference_cases", "")),
        }

    def create_batch_data(self) -> Dict[str, List]:
        return {
            "source_ids": [],
            "chunk_indices": [],
            "contents": [],
            "titles": [],
            "decision_dates": [],
            "court_names": [],
            "total_chunks_list": [],
            "case_numbers": [],
            "case_types": [],
            "reference_provisions_list": [],
            "reference_cases_list": [],
        }

    def add_to_batch(
        self,
        batch_data: Dict[str, List],
        source_id: str,
        chunk_idx: int,
        chunk_content: str,
        total_chunks: int,
        metadata: Dict[str, str],
    ) -> None:
        batch_data["source_ids"].append(source_id)
        batch_data["chunk_indices"].append(chunk_idx)
        batch_data["contents"].append(chunk_content)
        batch_data["titles"].append(metadata["case_name"])
        batch_data["decision_dates"].append(metadata["decision_date"])
        batch_data["court_names"].append(metadata["court_name"])
        batch_data["total_chunks_list"].append(total_chunks)
        batch_data["case_numbers"].append(metadata["case_number"])
        batch_data["case_types"].append(metadata["case_type"])
        batch_data["reference_provisions_list"].append(metadata["reference_provisions"])
        batch_data["reference_cases_list"].append(metadata["reference_cases"])

    def save_batch(self, batch_data: Dict[str, List], embeddings: List[List[float]]) -> int:
        self.store.add_precedent_documents(
            source_ids=batch_data["source_ids"],
            chunk_indices=batch_data["chunk_indices"],
            embeddings=embeddings,
            titles=batch_data["titles"],
            contents=batch_data["contents"],
            decision_dates=batch_data["decision_dates"],
            court_names=batch_data["court_names"],
            total_chunks_list=batch_data["total_chunks_list"],
            case_numbers=batch_data["case_numbers"],
            case_types=batch_data["case_types"],
            reference_provisions_list=batch_data["reference_provisions_list"],
            reference_cases_list=batch_data["reference_cases_list"],
        )
        return len(batch_data["source_ids"])


# ============================================================================
# 데이터 분할
# ============================================================================

def split_precedents(
    source_path: str,
    chunk_size: int = 5000,
    output_dir: str = ".",
) -> List[str]:
    """
    판례 JSON을 작은 파일들로 분할 (스트리밍 방식 - 메모리 효율적)

    Args:
        source_path: 원본 JSON 파일 경로
        chunk_size: 파일당 항목 수 (기본: 5000)
        output_dir: 출력 디렉토리

    Returns:
        생성된 파일명 리스트
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ijson 사용 가능 여부 확인
    if not IJSON_AVAILABLE:
        print("[WARN] ijson not available. Using full load (higher memory usage).")
        print("       Install with: pip install ijson")
        return _split_precedents_full_load(source_path, chunk_size, output_path)

    print(f"[INFO] Splitting {source_path} using streaming (memory-safe)...")

    # JSON 형식 감지
    json_format = detect_json_format(source_path)
    print(f"[INFO] JSON format: {json_format}")

    part_files = []
    current_chunk = []
    part_num = 1
    total_count = 0

    with open(source_path, "rb") as f:
        # JSON 구조에 따라 파서 설정
        if json_format == "array":
            parser = ijson.items(f, "item")
        else:
            # {"items": [...]} 또는 {"precedents": [...]} 구조
            parser = ijson.items(f, "items.item")

        for item in tqdm(parser, desc="Splitting"):
            current_chunk.append(item)
            total_count += 1

            if len(current_chunk) >= chunk_size:
                filename = f"precedents_part_{part_num:03d}.json"
                filepath = output_path / filename

                with open(filepath, "w", encoding="utf-8") as out:
                    json.dump(current_chunk, out, ensure_ascii=False)

                part_files.append(str(filepath))
                print(f"  - {filename}: {len(current_chunk):,} items")

                current_chunk = []
                part_num += 1
                clear_memory()

    # 남은 데이터 저장
    if current_chunk:
        filename = f"precedents_part_{part_num:03d}.json"
        filepath = output_path / filename

        with open(filepath, "w", encoding="utf-8") as out:
            json.dump(current_chunk, out, ensure_ascii=False)

        part_files.append(str(filepath))
        print(f"  - {filename}: {len(current_chunk):,} items")

    clear_memory()
    print(f"\n[INFO] Split complete! {len(part_files)} files, {total_count:,} items total.")
    return part_files


def _split_precedents_full_load(
    source_path: str,
    chunk_size: int,
    output_path: Path,
) -> List[str]:
    """ijson 없을 때 폴백: 전체 로드 방식 (메모리 주의)"""
    print(f"[INFO] Loading {source_path} (full load)...")

    with open(source_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        items = data.get("items", data.get("precedents", []))
    else:
        items = data

    total = len(items)
    num_parts = (total + chunk_size - 1) // chunk_size

    print(f"[INFO] Total items: {total:,}")
    print(f"[INFO] Splitting into {num_parts} parts ({chunk_size} items each)")

    part_files = []
    for i in range(num_parts):
        start = i * chunk_size
        end = min(start + chunk_size, total)
        part_data = items[start:end]

        filename = f"precedents_part_{i+1:03d}.json"
        filepath = output_path / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(part_data, f, ensure_ascii=False)

        part_files.append(str(filepath))
        print(f"  - {filename}: {len(part_data):,} items")

    del data
    del items
    clear_memory()

    print(f"\n[INFO] Split complete! {num_parts} files created.")
    return part_files


def split_laws(
    source_path: str,
    chunk_size: int = 2000,
    output_dir: str = ".",
) -> List[str]:
    """
    법령 JSON을 작은 파일들로 분할 (스트리밍 방식 - 메모리 효율적)

    Args:
        source_path: 원본 JSON 파일 경로
        chunk_size: 파일당 항목 수 (기본: 2000)
        output_dir: 출력 디렉토리

    Returns:
        생성된 파일명 리스트
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ijson 사용 가능 여부 확인
    if not IJSON_AVAILABLE:
        print("[WARN] ijson not available. Using full load (higher memory usage).")
        return _split_laws_full_load(source_path, chunk_size, output_path)

    print(f"[INFO] Splitting {source_path} using streaming (memory-safe)...")

    # JSON 형식 감지
    json_format = detect_json_format(source_path)
    print(f"[INFO] JSON format: {json_format}")

    part_files = []
    current_chunk = []
    part_num = 1
    total_count = 0

    with open(source_path, "rb") as f:
        # JSON 구조에 따라 파서 설정
        if json_format == "array":
            parser = ijson.items(f, "item")
        else:
            parser = ijson.items(f, "items.item")

        for item in tqdm(parser, desc="Splitting"):
            current_chunk.append(item)
            total_count += 1

            if len(current_chunk) >= chunk_size:
                filename = f"laws_part_{part_num:03d}.json"
                filepath = output_path / filename

                with open(filepath, "w", encoding="utf-8") as out:
                    json.dump(current_chunk, out, ensure_ascii=False)

                part_files.append(str(filepath))
                print(f"  - {filename}: {len(current_chunk):,} items")

                current_chunk = []
                part_num += 1
                clear_memory()

    # 남은 데이터 저장
    if current_chunk:
        filename = f"laws_part_{part_num:03d}.json"
        filepath = output_path / filename

        with open(filepath, "w", encoding="utf-8") as out:
            json.dump(current_chunk, out, ensure_ascii=False)

        part_files.append(str(filepath))
        print(f"  - {filename}: {len(current_chunk):,} items")

    clear_memory()
    print(f"\n[INFO] Split complete! {len(part_files)} files, {total_count:,} items total.")
    return part_files


def _split_laws_full_load(
    source_path: str,
    chunk_size: int,
    output_path: Path,
) -> List[str]:
    """ijson 없을 때 폴백: 전체 로드 방식 (메모리 주의)"""
    print(f"[INFO] Loading {source_path} (full load)...")

    with open(source_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        items = data.get("items", [])
    else:
        items = data

    total = len(items)
    num_parts = (total + chunk_size - 1) // chunk_size

    print(f"[INFO] Total items: {total:,}")
    print(f"[INFO] Splitting into {num_parts} parts ({chunk_size} items each)")

    part_files = []
    for i in range(num_parts):
        start = i * chunk_size
        end = min(start + chunk_size, total)
        part_data = items[start:end]

        filename = f"laws_part_{i+1:03d}.json"
        filepath = output_path / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(part_data, f, ensure_ascii=False)

        part_files.append(str(filepath))
        print(f"  - {filename}: {len(part_data):,} items")

    del data
    del items
    clear_memory()

    print(f"\n[INFO] Split complete! {num_parts} files created.")
    return part_files


# ============================================================================
# 데이터 로딩 헬퍼
# ============================================================================

def detect_json_format(file_path: str) -> str:
    """JSON 파일 형식 감지 (array 또는 object)"""
    with open(file_path, "rb") as f:
        first_char = f.read(1).decode("utf-8").strip()
        while first_char in ("\ufeff", " ", "\n", "\r", "\t", ""):
            first_char = f.read(1).decode("utf-8")
    return "array" if first_char == "[" else "object"


def load_json_streaming(file_path: str, items_path: Optional[str] = None) -> Optional[tuple]:
    """JSON 스트리밍 로드 (ijson 사용)"""
    if not IJSON_AVAILABLE:
        return None

    json_format = detect_json_format(file_path)
    f = open(file_path, "rb")

    if json_format == "array":
        return f, ijson.items(f, "item")
    else:
        path = items_path or "items.item"
        return f, ijson.items(f, path)


def load_json_full(file_path: str) -> List[Dict]:
    """JSON 전체 로드"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    return data.get("items", data.get("precedents", []))


# ============================================================================
# 청킹 설정 베이스 클래스
# ============================================================================

@dataclass
class BaseChunkConfig:
    """청킹 설정 베이스 클래스"""
    pass


# ============================================================================
# 판례 청킹
# ============================================================================

@dataclass
class PrecedentChunkConfig(BaseChunkConfig):
    """판례 청킹 설정"""
    chunk_size: int = CONFIG["PRECEDENT_CHUNK_SIZE"]
    chunk_overlap: int = CONFIG["PRECEDENT_CHUNK_OVERLAP"]
    min_chunk_size: int = CONFIG["PRECEDENT_MIN_CHUNK_SIZE"]


def chunk_precedent_text(text: str, config: PrecedentChunkConfig) -> List[tuple]:
    """판례 텍스트를 청크로 분할"""
    if not text or len(text) < config.min_chunk_size:
        return [(0, text)] if text else []

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = min(start + config.chunk_size, len(text))

        if end < len(text):
            for sep in ['. ', '.\n', '\n\n', '\n', ' ']:
                sep_pos = text.rfind(sep, start + config.min_chunk_size, end)
                if sep_pos > start:
                    end = sep_pos + len(sep)
                    break

        chunk_content = text[start:end].strip()
        if chunk_content and len(chunk_content) >= config.min_chunk_size:
            chunks.append((chunk_index, chunk_content))
            chunk_index += 1

        # 다음 시작 위치 계산
        new_start = end - config.chunk_overlap

        # 무한 루프 방지: start가 진행하지 않으면 종료
        if new_start <= start:
            break

        start = new_start

        # 남은 텍스트가 min_chunk_size보다 작으면 종료
        if start >= len(text) - config.min_chunk_size:
            break

    return chunks


# ============================================================================
# 법령 청킹
# ============================================================================

@dataclass
class LawChunkConfig(BaseChunkConfig):
    """법령 청킹 설정"""
    max_tokens: int = CONFIG["LAW_MAX_TOKENS"]
    min_tokens: int = CONFIG["LAW_MIN_TOKENS"]


PARAGRAPH_PATTERN = re.compile(r"([①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳])")


def estimate_tokens(text: str) -> int:
    """토큰 수 추정"""
    if not text:
        return 0
    korean_chars = len(re.findall(r"[가-힣]", text))
    other_chars = len(text) - korean_chars
    return int(korean_chars / 1.5 + other_chars / 4)


def split_by_paragraphs(article_text: str) -> List[str]:
    """조문을 항 단위로 분리"""
    parts = PARAGRAPH_PATTERN.split(article_text)
    if len(parts) <= 1:
        return [article_text]

    result = []
    if parts[0].strip():
        result.append(parts[0].strip())

    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            paragraph = parts[i] + parts[i + 1]
            if paragraph.strip():
                result.append(paragraph.strip())
        elif parts[i].strip():
            result.append(parts[i].strip())

    return result


def chunk_law_content(content: str, config: LawChunkConfig) -> List[tuple]:
    """법령 내용을 청크로 분할"""
    if not content:
        return []

    articles = content.split("\n\n")
    chunks = []
    chunk_index = 0
    article_no_pattern = re.compile(r"^(제\d+조(?:의\d+)?)")

    for article in articles:
        article = article.strip()
        if not article:
            continue

        match = article_no_pattern.match(article)
        article_no = match.group(1) if match else None
        tokens = estimate_tokens(article)

        if tokens <= config.max_tokens:
            if tokens >= config.min_tokens:
                chunks.append((chunk_index, article, article_no))
                chunk_index += 1
            elif chunks:
                prev_idx, prev_text, prev_article_no = chunks[-1]
                chunks[-1] = (prev_idx, prev_text + "\n\n" + article, prev_article_no)
            else:
                chunks.append((chunk_index, article, article_no))
                chunk_index += 1
        else:
            paragraphs = split_by_paragraphs(article)
            current_chunk = ""
            current_article_no = article_no

            for para in paragraphs:
                if not current_chunk:
                    current_chunk = para
                elif estimate_tokens(current_chunk + "\n" + para) <= config.max_tokens:
                    current_chunk += "\n" + para
                else:
                    if estimate_tokens(current_chunk) >= config.min_tokens:
                        chunks.append((chunk_index, current_chunk, current_article_no))
                        chunk_index += 1
                    current_chunk = para

            if current_chunk:
                if estimate_tokens(current_chunk) >= config.min_tokens:
                    chunks.append((chunk_index, current_chunk, current_article_no))
                    chunk_index += 1
                elif chunks:
                    prev_idx, prev_text, prev_article_no = chunks[-1]
                    chunks[-1] = (prev_idx, prev_text + "\n" + current_chunk, prev_article_no)

    return chunks


# ============================================================================
# 법령 임베딩 생성
# ============================================================================

def run_law_embedding(
    source_path: str,
    reset: bool = False,
    batch_size: Optional[int] = None,
    auto_config: bool = True,
) -> Dict[str, Any]:
    """
    법령 임베딩 실행 (StreamingEmbeddingProcessor 사용)

    Args:
        source_path: JSON 파일 경로
        reset: 기존 데이터 삭제 후 시작
        batch_size: 배치 크기 (None=자동)
        auto_config: 디바이스에 따라 자동 설정 (하위 호환용, 무시됨)

    Returns:
        처리 통계 딕셔너리
    """
    processor = LawEmbeddingProcessor()
    stats = processor.run(source_path, reset=reset, batch_size=batch_size)

    # 통계 출력
    store = LanceDBStore()
    show_stats(store)
    del store
    clear_memory()

    return stats


# ============================================================================
# 판례 임베딩 생성
# ============================================================================

def run_precedent_embedding(
    source_path: str,
    reset: bool = False,
    batch_size: Optional[int] = None,
    auto_config: bool = True,
) -> Dict[str, Any]:
    """
    판례 임베딩 실행 (StreamingEmbeddingProcessor 사용)

    Args:
        source_path: JSON 파일 경로
        reset: 기존 데이터 삭제 후 시작
        batch_size: 배치 크기 (None=자동)
        auto_config: 디바이스에 따라 자동 설정 (하위 호환용, 무시됨)

    Returns:
        처리 통계 딕셔너리
    """
    processor = PrecedentEmbeddingProcessor()
    stats = processor.run(source_path, reset=reset, batch_size=batch_size)

    # 통계 출력
    store = LanceDBStore()
    show_stats(store)
    del store
    clear_memory()

    return stats


# ============================================================================
# 분할 파일 처리
# ============================================================================

def run_precedent_embedding_part(
    source_path: str,
    reset: bool = False,
    batch_size: int = 64,
) -> Dict[str, Any]:
    """
    분할된 판례 파일 하나를 처리

    Args:
        source_path: 분할된 JSON 파일 경로
        reset: 기존 데이터 삭제 (첫 파일에만 True)
        batch_size: 배치 크기 (기본: 64, 메모리 절약)

    Returns:
        처리 통계 딕셔너리
    """
    processor = PrecedentEmbeddingProcessor()
    return processor.run_part(source_path, reset=reset, batch_size=batch_size)


def run_all_precedent_parts(
    pattern: str = "precedents_part_*.json",
    batch_size: int = 64,
) -> Dict[str, Any]:
    """
    모든 분할된 판례 파일 처리

    Args:
        pattern: 파일 glob 패턴
        batch_size: 배치 크기

    Returns:
        전체 처리 통계
    """
    import glob

    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[ERROR] No files matching: {pattern}")
        return {}

    print(f"[INFO] Found {len(files)} files to process")
    print(f"[INFO] Files: {files}")

    total_stats = {
        "processed_docs": 0,
        "total_chunks": 0,
        "skipped": 0,
        "errors": 0,
    }

    for i, filepath in enumerate(files):
        reset = (i == 0)  # 첫 파일에서만 reset

        print(f"\n[{i+1}/{len(files)}] Processing: {filepath}")
        stats = run_precedent_embedding_part(filepath, reset=reset, batch_size=batch_size)

        total_stats["processed_docs"] += stats.get("processed_docs", 0)
        total_stats["total_chunks"] += stats.get("total_chunks", 0)
        total_stats["skipped"] += stats.get("skipped", 0)
        total_stats["errors"] += stats.get("errors", 0)

        # 파일 간 메모리 정리
        clear_memory()

        print(f"\n[Progress] {i+1}/{len(files)} files done")
        print("-" * 60)

    print("\n" + "=" * 60)
    print("ALL DONE!")
    print("=" * 60)
    print(f"Total processed: {total_stats['processed_docs']:,}")
    print(f"Total chunks: {total_stats['total_chunks']:,}")
    print(f"Total skipped: {total_stats['skipped']:,}")
    print(f"Total errors: {total_stats['errors']:,}")

    show_stats()

    return total_stats


def run_law_embedding_part(
    source_path: str,
    reset: bool = False,
    batch_size: int = 64,
) -> Dict[str, Any]:
    """
    분할된 법령 파일 하나를 처리

    Args:
        source_path: 분할된 JSON 파일 경로
        reset: 기존 데이터 삭제 (첫 파일에만 True)
        batch_size: 배치 크기 (기본: 64)

    Returns:
        처리 통계 딕셔너리
    """
    processor = LawEmbeddingProcessor()
    return processor.run_part(source_path, reset=reset, batch_size=batch_size)


def run_all_law_parts(
    pattern: str = "laws_part_*.json",
    batch_size: int = 64,
) -> Dict[str, Any]:
    """
    모든 분할된 법령 파일 처리

    Args:
        pattern: 파일 glob 패턴
        batch_size: 배치 크기

    Returns:
        전체 처리 통계
    """
    import glob

    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[ERROR] No files matching: {pattern}")
        return {}

    print(f"[INFO] Found {len(files)} files to process")
    print(f"[INFO] Files: {files}")

    total_stats = {
        "processed_docs": 0,
        "total_chunks": 0,
        "skipped": 0,
        "errors": 0,
    }

    for i, filepath in enumerate(files):
        reset = (i == 0)  # 첫 파일에서만 reset

        print(f"\n[{i+1}/{len(files)}] Processing: {filepath}")
        stats = run_law_embedding_part(filepath, reset=reset, batch_size=batch_size)

        total_stats["processed_docs"] += stats.get("processed_docs", 0)
        total_stats["total_chunks"] += stats.get("total_chunks", 0)
        total_stats["skipped"] += stats.get("skipped", 0)
        total_stats["errors"] += stats.get("errors", 0)

        # 파일 간 메모리 정리
        clear_memory()

        print(f"\n[Progress] {i+1}/{len(files)} files done")
        print("-" * 60)

    print("\n" + "=" * 60)
    print("ALL DONE!")
    print("=" * 60)
    print(f"Total processed: {total_stats['processed_docs']:,}")
    print(f"Total chunks: {total_stats['total_chunks']:,}")
    print(f"Total skipped: {total_stats['skipped']:,}")
    print(f"Total errors: {total_stats['errors']:,}")

    show_stats()

    return total_stats


# ============================================================================
# 통계 출력
# ============================================================================

def show_stats(store: Optional[LanceDBStore] = None) -> None:
    """LanceDB 통계 출력"""
    if store is None:
        store = LanceDBStore()

    print("\n" + "=" * 60)
    print("LanceDB Statistics")
    print("=" * 60)

    total = store.count()
    print(f"Total chunks: {total:,}")

    if total > 0:
        law_count = store.count_by_type("법령")
        precedent_count = store.count_by_type("판례")

        print(f"\nBy data_type:")
        print(f"  - 법령: {law_count:,}")
        print(f"  - 판례: {precedent_count:,}")


# ============================================================================
# 메인 (CLI)
# ============================================================================

def main():
    """CLI 메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="LanceDB 임베딩 생성 (RunPod)")
    parser.add_argument("--type", choices=["law", "precedent", "all"], default="all")
    parser.add_argument("--law-source", type=str, default="law_cleaned.json")
    parser.add_argument("--precedent-source", type=str, default="precedents_cleaned.json")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--stats", action="store_true")

    args = parser.parse_args()

    print("=" * 60)
    print("LanceDB Embedding Creator (RunPod Edition)")
    print("=" * 60)
    print(f"Model: {CONFIG['EMBEDDING_MODEL']}")
    print(f"Vector dimension: {VECTOR_DIM}")

    if args.stats:
        show_stats()
        return

    if args.type in ("law", "all"):
        run_law_embedding(args.law_source, args.reset, args.batch_size)

    if args.type in ("precedent", "all"):
        run_precedent_embedding(args.precedent_source, args.reset, args.batch_size)


# ============================================================================
# Jupyter/Notebook 환경 감지
# ============================================================================

def is_notebook():
    """Jupyter/Colab/RunPod Notebook 환경인지 확인"""
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            return True
    except ImportError:
        pass
    return False


if __name__ == "__main__":
    if not is_notebook():
        main()
    else:
        print("=" * 60)
        print("LanceDB Embedding Creator (RunPod Edition)")
        print("=" * 60)

        device_info, optimal = print_device_info()

        print("\n[기본 함수]")
        print("  - print_device_info()           # 디바이스 정보")
        print("  - print_memory_status()         # 메모리 사용량 확인")
        print("  - show_stats()                  # 통계 확인")
        print("  - clear_model_cache()           # 모델 메모리 정리")
        print("  - clear_memory()                # GPU/CPU 메모리 정리")
        print("  - set_seed(42)                  # 랜덤 시드 고정")
        print("")
        print("[품질 검증]")
        print("  - checker = EmbeddingQualityChecker()")
        print("  - checker.quick_test()          # 법률 도메인 기본 테스트")
        print("")
        print("[임베딩 캐싱]")
        print("  - cache = EmbeddingCache('./cache')")
        print("  - cache.get_stats()             # 캐시 통계")
        print("")
        print("[분할 처리 (권장)]")
        print("  - split_precedents(path, chunk_size=5000)  # 판례 분할")
        print("  - split_laws(path, chunk_size=2000)        # 법령 분할")
        print("  - run_precedent_embedding_part(path)       # 판례 분할 파일 처리")
        print("  - run_all_precedent_parts(pattern)         # 모든 판례 분할 파일")
        print("  - run_law_embedding_part(path)             # 법령 분할 파일 처리")
        print("  - run_all_law_parts(pattern)               # 모든 법령 분할 파일")
        print("")
        print("[일반 처리 (소규모 데이터)]")
        print("  - run_law_embedding(path)       # 법령 임베딩")
        print("  - run_precedent_embedding(path) # 판례 임베딩")
        print("")
        print("=" * 60)
        print("예시:")
        print("=" * 60)
        print("  # 품질 테스트")
        print("  checker = EmbeddingQualityChecker()")
        print("  checker.quick_test()")
        print("")
        print("  # 분할 처리")
        print("  split_precedents('precedents_cleaned.json', chunk_size=5000)")
        print("  run_all_precedent_parts('precedents_part_*.json', batch_size=64)")
