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
from typing import List, Optional, Dict, Any, Iterator
from dataclasses import dataclass

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


def get_device_info() -> DeviceInfo:
    """디바이스 상세 정보 조회"""
    device = get_device()

    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
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


def get_optimal_config(device_info: DeviceInfo = None) -> OptimalConfig:
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
        "id": f"{source_id}_{chunk_index}",
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
        "id": f"{source_id}_{chunk_index}",
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

    def __init__(self, db_path: str = None, table_name: str = None):
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
        case_numbers: List[str] = None,
        case_types: List[str] = None,
        reference_provisions_list: List[str] = None,
        reference_cases_list: List[str] = None,
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
            # pandas 없이 PyArrow로 직접 카운트
            result = self._table.search().where(f"data_type = '{data_type}'").limit(1000000).to_arrow()
            return result.num_rows
        except Exception:
            return 0

    def get_existing_source_ids(self, data_type: str) -> set:
        """이미 저장된 source_id 조회"""
        if self._table is None:
            return set()
        try:
            # pandas 없이 PyArrow로 직접 조회
            result = self._table.search().where(f"data_type = '{data_type}'").select(["source_id"]).limit(1000000).to_arrow()
            source_ids = result.column("source_id").to_pylist()
            return set(source_ids)
        except Exception:
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


def create_embeddings(texts: List[str], device: str = None) -> List[List[float]]:
    """임베딩 생성"""
    model = get_embedding_model(device)
    processed = [t.strip()[:4000] if t else "(내용 없음)" for t in texts]
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

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()
        print("[INFO] Model cache cleared")


def print_memory_status():
    """현재 메모리 사용량 출력"""
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"[Memory] RAM: {mem.used / 1024**3:.1f}GB / {mem.total / 1024**3:.1f}GB ({mem.percent}%)")

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"[Memory] GPU: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
    except ImportError:
        print("[Memory] psutil not installed")


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
                gc.collect()

    # 남은 데이터 저장
    if current_chunk:
        filename = f"precedents_part_{part_num:03d}.json"
        filepath = output_path / filename

        with open(filepath, "w", encoding="utf-8") as out:
            json.dump(current_chunk, out, ensure_ascii=False)

        part_files.append(str(filepath))
        print(f"  - {filename}: {len(current_chunk):,} items")

    gc.collect()
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
    gc.collect()

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
                gc.collect()

    # 남은 데이터 저장
    if current_chunk:
        filename = f"laws_part_{part_num:03d}.json"
        filepath = output_path / filename

        with open(filepath, "w", encoding="utf-8") as out:
            json.dump(current_chunk, out, ensure_ascii=False)

        part_files.append(str(filepath))
        print(f"  - {filename}: {len(current_chunk):,} items")

    gc.collect()
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
    gc.collect()

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


def load_json_streaming(file_path: str, items_path: str = None):
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
# 판례 청킹
# ============================================================================

@dataclass
class PrecedentChunkConfig:
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

        start = end - config.chunk_overlap
        if start >= len(text) - config.min_chunk_size:
            break

    return chunks


# ============================================================================
# 법령 청킹
# ============================================================================

@dataclass
class LawChunkConfig:
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
    batch_size: int = None,
    auto_config: bool = True,
) -> dict:
    """
    법령 임베딩 실행

    Args:
        source_path: JSON 파일 경로
        reset: 기존 데이터 삭제 후 시작
        batch_size: 배치 크기 (None=자동)
        auto_config: 디바이스에 따라 자동 설정

    Returns:
        처리 통계 딕셔너리
    """
    print("=" * 60)
    print("법령 임베딩 시작")
    print("=" * 60)

    device_info = get_device_info()
    optimal = get_optimal_config(device_info)

    if auto_config:
        batch_size = batch_size or optimal.batch_size
    else:
        batch_size = batch_size or CONFIG["BATCH_SIZE"]

    # 메모리 관리: 매 배치마다 gc 실행
    gc_interval = 1

    print(f"  Device: {device_info}")
    print(f"  Batch size: {batch_size}")
    print(f"  GC interval: every batch")
    print(f"  Source: {source_path}")
    print(f"  Reset: {reset}")

    chunk_config = LawChunkConfig()
    store = LanceDBStore()

    stats = {
        "total_docs": 0,
        "processed_docs": 0,
        "total_chunks": 0,
        "skipped": 0,
        "errors": 0,
        "device": str(device_info),
    }

    # 데이터 로드
    print(f"\n[INFO] Loading law data from: {source_path}")

    if IJSON_AVAILABLE:
        json_format = detect_json_format(source_path)
        print(f"[INFO] JSON format: {json_format}, using streaming")
    else:
        print("[INFO] Using full JSON load")

    # 전체 개수 파악
    items = load_json_full(source_path)
    stats["total_docs"] = len(items)
    print(f"[INFO] Total laws: {len(items):,}")

    if not items:
        return stats

    if reset:
        deleted = store.delete_by_type("법령")
        print(f"[INFO] Deleted {deleted} existing law chunks")

    existing_source_ids = store.get_existing_source_ids("법령") if not reset else set()
    if existing_source_ids:
        print(f"[INFO] Found {len(existing_source_ids)} laws already embedded")
        stats["skipped"] = len(existing_source_ids)

    # 모델 로드
    get_embedding_model(device_info.device)

    # 배치 처리
    batch_data = {
        "source_ids": [], "chunk_indices": [], "embeddings": [],
        "titles": [], "contents": [], "enforcement_dates": [],
        "departments": [], "total_chunks_list": [], "promulgation_dates": [],
        "promulgation_nos": [], "law_types": [], "article_nos": [],
    }

    start_time = datetime.now()
    batch_count = 0

    for idx, item in enumerate(tqdm(items, desc="법령 처리")):
        source_id = item.get("law_id", "")

        if source_id in existing_source_ids:
            continue

        content = item.get("content", "")
        if not content:
            stats["skipped"] += 1
            continue

        chunks = chunk_law_content(content, chunk_config)
        if not chunks:
            stats["skipped"] += 1
            continue

        total_chunks = len(chunks)
        stats["processed_docs"] += 1

        for chunk_idx, chunk_content, article_no in chunks:
            if article_no:
                prefixed_content = f"[법령] {article_no} {chunk_content}"
            else:
                prefixed_content = f"[법령] {chunk_content}"

            batch_data["source_ids"].append(source_id)
            batch_data["chunk_indices"].append(chunk_idx)
            batch_data["titles"].append(item.get("law_name", ""))
            batch_data["contents"].append(prefixed_content)
            batch_data["enforcement_dates"].append(item.get("enforcement_date", ""))
            batch_data["departments"].append(item.get("ministry", ""))
            batch_data["total_chunks_list"].append(total_chunks)
            batch_data["promulgation_dates"].append(item.get("promulgation_date", ""))
            batch_data["promulgation_nos"].append(item.get("promulgation_no", ""))
            batch_data["law_types"].append(item.get("law_type", ""))
            batch_data["article_nos"].append(article_no or "")

        # 배치 저장
        if len(batch_data["source_ids"]) >= batch_size:
            batch_count += 1
            try:
                embeddings = create_embeddings(
                    batch_data["contents"], device_info.device
                )
                batch_data["embeddings"] = embeddings
                store.add_law_documents(**batch_data)
                stats["total_chunks"] += len(batch_data["source_ids"])

                # 임베딩 메모리 즉시 해제
                del embeddings
            except Exception as e:
                stats["errors"] += len(batch_data["source_ids"])
                print(f"  [ERROR] Batch error: {e}")

            # 배치 데이터 명시적 해제
            for key in batch_data:
                batch_data[key].clear()
                batch_data[key] = []

            # 매 배치마다 메모리 정리
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 남은 배치 처리
    if batch_data["source_ids"]:
        try:
            embeddings = create_embeddings(
                batch_data["contents"], device_info.device
            )
            batch_data["embeddings"] = embeddings
            store.add_law_documents(**batch_data)
            stats["total_chunks"] += len(batch_data["source_ids"])
            del embeddings
        except Exception as e:
            stats["errors"] += len(batch_data["source_ids"])
            print(f"  [ERROR] Final batch error: {e}")

    # 최종 메모리 정리
    del batch_data
    del items  # 법령 데이터 해제
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # LanceDB 테이블 압축
    store.compact()

    elapsed = datetime.now() - start_time
    print(f"\n완료! 소요시간: {elapsed}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    show_stats(store)

    # store 객체도 해제
    del store
    gc.collect()

    return stats


# ============================================================================
# 판례 임베딩 생성
# ============================================================================

def run_precedent_embedding(
    source_path: str,
    reset: bool = False,
    batch_size: int = None,
    auto_config: bool = True,
) -> dict:
    """
    판례 임베딩 실행

    Args:
        source_path: JSON 파일 경로
        reset: 기존 데이터 삭제 후 시작
        batch_size: 배치 크기 (None=자동)
        auto_config: 디바이스에 따라 자동 설정

    Returns:
        처리 통계 딕셔너리
    """
    print("=" * 60)
    print("판례 임베딩 시작")
    print("=" * 60)

    device_info = get_device_info()
    optimal = get_optimal_config(device_info)

    if auto_config:
        batch_size = batch_size or optimal.batch_size
    else:
        batch_size = batch_size or CONFIG["BATCH_SIZE"]

    # 메모리 관리: 매 배치마다 gc 실행
    gc_interval = 1

    print(f"  Device: {device_info}")
    print(f"  Batch size: {batch_size}")
    print(f"  GC interval: every batch")
    print(f"  Source: {source_path}")
    print(f"  Reset: {reset}")

    chunk_config = PrecedentChunkConfig()
    store = LanceDBStore()

    stats = {
        "total_docs": 0,
        "processed_docs": 0,
        "total_chunks": 0,
        "skipped": 0,
        "errors": 0,
        "device": str(device_info),
    }

    print(f"\n[INFO] Loading precedent data from: {source_path}")

    if reset:
        deleted = store.delete_by_type("판례")
        print(f"[INFO] Deleted {deleted} existing precedent chunks")

    existing_source_ids = store.get_existing_source_ids("판례") if not reset else set()
    if existing_source_ids:
        print(f"[INFO] Found {len(existing_source_ids)} precedents already embedded")
        stats["skipped"] = len(existing_source_ids)

    # 스트리밍 또는 전체 로드
    use_streaming = IJSON_AVAILABLE
    file_handle = None

    if use_streaming:
        json_format = detect_json_format(source_path)
        print(f"[INFO] JSON format: {json_format}, using streaming (low memory)")

        # 전체 개수 파악 (스트리밍으로)
        print("[INFO] Counting total items...")
        total_count = 0
        result = load_json_streaming(source_path)
        if result:
            f_count, items_count = result
            for _ in items_count:
                total_count += 1
            f_count.close()
        stats["total_docs"] = total_count
        print(f"[INFO] Total precedents: {total_count:,}")

        # 실제 처리용 스트리밍
        result = load_json_streaming(source_path)
        if result:
            file_handle, items = result
        else:
            items = load_json_full(source_path)
    else:
        print("[INFO] Using full JSON load")
        items = load_json_full(source_path)
        stats["total_docs"] = len(items)
        print(f"[INFO] Total precedents: {len(items):,}")

    if stats["total_docs"] == 0:
        return stats

    # 모델 로드
    get_embedding_model(device_info.device)

    # 배치 처리
    batch_data = {
        "source_ids": [], "chunk_indices": [], "embeddings": [],
        "titles": [], "contents": [], "decision_dates": [],
        "court_names": [], "total_chunks_list": [], "case_numbers": [],
        "case_types": [], "reference_provisions_list": [], "reference_cases_list": [],
    }

    start_time = datetime.now()
    batch_count = 0
    processed_count = 0

    for idx, item in enumerate(tqdm(items, desc="판례 처리", total=stats["total_docs"])):
        source_id = str(item.get("판례정보일련번호", item.get("id", idx)))

        if source_id in existing_source_ids:
            continue

        # 임베딩 텍스트: 판시사항 + 판결요지
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

        text = "\n".join(parts)
        if not text or len(text) < chunk_config.min_chunk_size:
            stats["skipped"] += 1
            continue

        chunks = chunk_precedent_text(text, chunk_config)
        if not chunks:
            stats["skipped"] += 1
            continue

        total_chunks = len(chunks)
        stats["processed_docs"] += 1

        for chunk_idx, chunk_content in chunks:
            batch_data["source_ids"].append(source_id)
            batch_data["chunk_indices"].append(chunk_idx)
            batch_data["titles"].append(case_name)
            batch_data["contents"].append(f"[판례] {chunk_content}")
            batch_data["decision_dates"].append(
                item.get("선고일자", item.get("decision_date", ""))
            )
            batch_data["court_names"].append(
                item.get("법원명", item.get("court_name", ""))
            )
            batch_data["total_chunks_list"].append(total_chunks)
            batch_data["case_numbers"].append(
                item.get("사건번호", item.get("case_number", ""))
            )
            batch_data["case_types"].append(
                item.get("사건종류명", item.get("case_type", ""))
            )
            batch_data["reference_provisions_list"].append(
                item.get("참조조문", item.get("reference_provisions", ""))
            )
            batch_data["reference_cases_list"].append(
                item.get("참조판례", item.get("reference_cases", ""))
            )

        # 배치 저장
        if len(batch_data["source_ids"]) >= batch_size:
            batch_count += 1
            try:
                embeddings = create_embeddings(
                    batch_data["contents"], device_info.device
                )
                batch_data["embeddings"] = embeddings
                store.add_precedent_documents(**batch_data)
                stats["total_chunks"] += len(batch_data["source_ids"])

                # 임베딩 메모리 즉시 해제
                del embeddings
            except Exception as e:
                stats["errors"] += len(batch_data["source_ids"])
                print(f"  [ERROR] Batch error: {e}")

            # 배치 데이터 명시적 해제
            for key in batch_data:
                batch_data[key].clear()
                batch_data[key] = []

            # 매 배치마다 메모리 정리
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 50 배치마다 LanceDB 압축 (메모리 최적화)
            if batch_count % 50 == 0:
                store.compact()
                print_memory_status()

    # 남은 배치 처리
    if batch_data["source_ids"]:
        try:
            embeddings = create_embeddings(
                batch_data["contents"], device_info.device
            )
            batch_data["embeddings"] = embeddings
            store.add_precedent_documents(**batch_data)
            stats["total_chunks"] += len(batch_data["source_ids"])
            del embeddings
        except Exception as e:
            stats["errors"] += len(batch_data["source_ids"])
            print(f"  [ERROR] Final batch error: {e}")

    # 파일 핸들 닫기
    if file_handle is not None:
        file_handle.close()

    # 최종 메모리 정리
    del batch_data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # LanceDB 테이블 압축
    store.compact()

    elapsed = datetime.now() - start_time
    print(f"\n완료! 소요시간: {elapsed}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    show_stats(store)

    # store 객체도 해제
    del store
    gc.collect()

    return stats


# ============================================================================
# 분할 파일 처리
# ============================================================================

def run_precedent_embedding_part(
    source_path: str,
    reset: bool = False,
    batch_size: int = 64,
) -> dict:
    """
    분할된 판례 파일 하나를 처리

    Args:
        source_path: 분할된 JSON 파일 경로
        reset: 기존 데이터 삭제 (첫 파일에만 True)
        batch_size: 배치 크기 (기본: 64, 메모리 절약)

    Returns:
        처리 통계 딕셔너리
    """
    print("=" * 60)
    print(f"판례 임베딩 (분할 파일): {source_path}")
    print("=" * 60)

    device_info = get_device_info()
    print(f"  Device: {device_info}")
    print(f"  Batch size: {batch_size}")
    print(f"  Reset: {reset}")

    # 데이터 로드
    with open(source_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    print(f"[INFO] Items: {len(items):,}")
    print_memory_status()

    chunk_config = PrecedentChunkConfig()
    store = LanceDBStore()

    if reset:
        deleted = store.delete_by_type("판례")
        print(f"[INFO] Deleted {deleted} existing precedent chunks")

    stats = {
        "total_docs": len(items),
        "processed_docs": 0,
        "total_chunks": 0,
        "skipped": 0,
        "errors": 0,
    }

    # 모델 로드
    get_embedding_model(device_info.device)

    # 배치 데이터
    batch_data = {
        "source_ids": [], "chunk_indices": [], "embeddings": [],
        "titles": [], "contents": [], "decision_dates": [],
        "court_names": [], "total_chunks_list": [], "case_numbers": [],
        "case_types": [], "reference_provisions_list": [], "reference_cases_list": [],
    }

    start_time = datetime.now()

    for item in tqdm(items, desc="판례 처리"):
        source_id = str(item.get("판례정보일련번호", item.get("id", "")))

        # 텍스트 추출
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

        text = "\n".join(parts)
        if not text or len(text) < chunk_config.min_chunk_size:
            stats["skipped"] += 1
            continue

        chunks = chunk_precedent_text(text, chunk_config)
        if not chunks:
            stats["skipped"] += 1
            continue

        total_chunks = len(chunks)
        stats["processed_docs"] += 1

        for chunk_idx, chunk_content in chunks:
            batch_data["source_ids"].append(source_id)
            batch_data["chunk_indices"].append(chunk_idx)
            batch_data["titles"].append(case_name)
            batch_data["contents"].append(f"[판례] {chunk_content}")
            batch_data["decision_dates"].append(
                item.get("선고일자", item.get("decision_date", ""))
            )
            batch_data["court_names"].append(
                item.get("법원명", item.get("court_name", ""))
            )
            batch_data["total_chunks_list"].append(total_chunks)
            batch_data["case_numbers"].append(
                item.get("사건번호", item.get("case_number", ""))
            )
            batch_data["case_types"].append(
                item.get("사건종류명", item.get("case_type", ""))
            )
            batch_data["reference_provisions_list"].append(
                item.get("참조조문", item.get("reference_provisions", ""))
            )
            batch_data["reference_cases_list"].append(
                item.get("참조판례", item.get("reference_cases", ""))
            )

        # 배치 저장
        if len(batch_data["source_ids"]) >= batch_size:
            try:
                embeddings = create_embeddings(
                    batch_data["contents"], device_info.device
                )
                batch_data["embeddings"] = embeddings
                store.add_precedent_documents(**batch_data)
                stats["total_chunks"] += len(batch_data["source_ids"])
                del embeddings
            except Exception as e:
                stats["errors"] += len(batch_data["source_ids"])
                print(f"  [ERROR] Batch error: {e}")

            # 배치 데이터 클리어
            for key in batch_data:
                batch_data[key] = []

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 남은 배치 처리
    if batch_data["source_ids"]:
        try:
            embeddings = create_embeddings(
                batch_data["contents"], device_info.device
            )
            batch_data["embeddings"] = embeddings
            store.add_precedent_documents(**batch_data)
            stats["total_chunks"] += len(batch_data["source_ids"])
            del embeddings
        except Exception as e:
            stats["errors"] += len(batch_data["source_ids"])
            print(f"  [ERROR] Final batch error: {e}")

    # 정리
    del items
    del batch_data
    store.compact()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    elapsed = datetime.now() - start_time
    print(f"\n완료! 소요시간: {elapsed}")
    print(f"  Processed: {stats['processed_docs']:,}")
    print(f"  Chunks: {stats['total_chunks']:,}")
    print(f"  Skipped: {stats['skipped']:,}")
    print_memory_status()

    return stats


def run_all_precedent_parts(
    pattern: str = "precedents_part_*.json",
    batch_size: int = 64,
) -> dict:
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
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
) -> dict:
    """
    분할된 법령 파일 하나를 처리

    Args:
        source_path: 분할된 JSON 파일 경로
        reset: 기존 데이터 삭제 (첫 파일에만 True)
        batch_size: 배치 크기 (기본: 64)

    Returns:
        처리 통계 딕셔너리
    """
    print("=" * 60)
    print(f"법령 임베딩 (분할 파일): {source_path}")
    print("=" * 60)

    device_info = get_device_info()
    print(f"  Device: {device_info}")
    print(f"  Batch size: {batch_size}")
    print(f"  Reset: {reset}")

    # 데이터 로드
    with open(source_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    print(f"[INFO] Items: {len(items):,}")
    print_memory_status()

    chunk_config = LawChunkConfig()
    store = LanceDBStore()

    if reset:
        deleted = store.delete_by_type("법령")
        print(f"[INFO] Deleted {deleted} existing law chunks")

    stats = {
        "total_docs": len(items),
        "processed_docs": 0,
        "total_chunks": 0,
        "skipped": 0,
        "errors": 0,
    }

    # 모델 로드
    get_embedding_model(device_info.device)

    # 배치 데이터
    batch_data = {
        "source_ids": [], "chunk_indices": [], "embeddings": [],
        "titles": [], "contents": [], "enforcement_dates": [],
        "departments": [], "total_chunks_list": [], "promulgation_dates": [],
        "promulgation_nos": [], "law_types": [], "article_nos": [],
    }

    start_time = datetime.now()

    for item in tqdm(items, desc="법령 처리"):
        source_id = item.get("law_id", "")

        content = item.get("content", "")
        if not content:
            stats["skipped"] += 1
            continue

        chunks = chunk_law_content(content, chunk_config)
        if not chunks:
            stats["skipped"] += 1
            continue

        total_chunks = len(chunks)
        stats["processed_docs"] += 1

        for chunk_idx, chunk_content, article_no in chunks:
            if article_no:
                prefixed_content = f"[법령] {article_no} {chunk_content}"
            else:
                prefixed_content = f"[법령] {chunk_content}"

            batch_data["source_ids"].append(source_id)
            batch_data["chunk_indices"].append(chunk_idx)
            batch_data["titles"].append(item.get("law_name", ""))
            batch_data["contents"].append(prefixed_content)
            batch_data["enforcement_dates"].append(item.get("enforcement_date", ""))
            batch_data["departments"].append(item.get("ministry", ""))
            batch_data["total_chunks_list"].append(total_chunks)
            batch_data["promulgation_dates"].append(item.get("promulgation_date", ""))
            batch_data["promulgation_nos"].append(item.get("promulgation_no", ""))
            batch_data["law_types"].append(item.get("law_type", ""))
            batch_data["article_nos"].append(article_no or "")

        # 배치 저장
        if len(batch_data["source_ids"]) >= batch_size:
            try:
                embeddings = create_embeddings(
                    batch_data["contents"], device_info.device
                )
                batch_data["embeddings"] = embeddings
                store.add_law_documents(**batch_data)
                stats["total_chunks"] += len(batch_data["source_ids"])
                del embeddings
            except Exception as e:
                stats["errors"] += len(batch_data["source_ids"])
                print(f"  [ERROR] Batch error: {e}")

            # 배치 데이터 클리어
            for key in batch_data:
                batch_data[key] = []

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 남은 배치 처리
    if batch_data["source_ids"]:
        try:
            embeddings = create_embeddings(
                batch_data["contents"], device_info.device
            )
            batch_data["embeddings"] = embeddings
            store.add_law_documents(**batch_data)
            stats["total_chunks"] += len(batch_data["source_ids"])
            del embeddings
        except Exception as e:
            stats["errors"] += len(batch_data["source_ids"])
            print(f"  [ERROR] Final batch error: {e}")

    # 정리
    del items
    del batch_data
    store.compact()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    elapsed = datetime.now() - start_time
    print(f"\n완료! 소요시간: {elapsed}")
    print(f"  Processed: {stats['processed_docs']:,}")
    print(f"  Chunks: {stats['total_chunks']:,}")
    print(f"  Skipped: {stats['skipped']:,}")
    print_memory_status()

    return stats


def run_all_law_parts(
    pattern: str = "laws_part_*.json",
    batch_size: int = 64,
) -> dict:
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
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

def show_stats(store: LanceDBStore = None):
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
        print("예시 (분할 처리):")
        print("=" * 60)
        print("  # 판례")
        print("  split_precedents('precedents_cleaned.json', chunk_size=5000)")
        print("  run_all_precedent_parts('precedents_part_*.json', batch_size=64)")
        print("")
        print("  # 법령")
        print("  split_laws('law_cleaned.json', chunk_size=2000)")
        print("  run_all_law_parts('laws_part_*.json', batch_size=64)")
