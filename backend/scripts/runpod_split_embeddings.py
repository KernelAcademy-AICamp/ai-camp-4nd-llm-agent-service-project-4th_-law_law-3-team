#!/usr/bin/env python3
"""
LanceDB 임베딩 생성 스크립트 (RunPod - 파일 분할 버전)

대용량 판례 데이터를 작은 파일로 분할하여 메모리 문제 없이 처리합니다.

=============================================================================
사용법
=============================================================================

1. 패키지 설치
   !pip install sentence-transformers pyarrow tqdm psutil -q

2. 데이터 분할
   split_precedents('precedents_cleaned.json', chunk_size=5000)
   # -> precedents_part_001.json, precedents_part_002.json, ... 생성

3. 분할 파일 개별 처리
   process_all_parts()
   # 또는 개별 실행
   process_precedent_part('precedents_part_001.json')

4. 결과 확인
   show_stats()

=============================================================================
"""

import gc
import json
import os
import re
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

import pyarrow as pa
from tqdm import tqdm
import torch

try:
    import ijson
    IJSON_AVAILABLE = True
except ImportError:
    IJSON_AVAILABLE = False
    print("[WARN] ijson not installed. Using full JSON load (higher memory usage)")
    print("       Install with: pip install ijson")


# ============================================================================
# 설정 (환경변수에서 로드)
# ============================================================================

def _get_config() -> dict:
    """환경변수에서 설정 로드"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    return {
        "LANCEDB_URI": os.getenv("LANCEDB_URI", "./lancedb_data"),
        "LANCEDB_TABLE_NAME": os.getenv("LANCEDB_TABLE_NAME", "legal_chunks"),
        "EMBEDDING_MODEL": os.getenv("LOCAL_EMBEDDING_MODEL", "nlpai-lab/KURE-v1"),
        "VECTOR_DIM": 1024,
        "BATCH_SIZE": 64,  # 메모리 절약을 위해 줄임
        "PRECEDENT_CHUNK_SIZE": 1250,
        "PRECEDENT_CHUNK_OVERLAP": 125,
        "PRECEDENT_MIN_CHUNK_SIZE": 100,
    }


CONFIG = _get_config()


# ============================================================================
# 디바이스 감지
# ============================================================================

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def print_device_info():
    device = get_device()
    print(f"Device: {device.upper()}")
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU: {props.name}")
        print(f"  VRAM: {props.total_memory / 1024**3:.1f} GB")
    return device


def print_memory_status():
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"[Memory] RAM: {mem.used / 1024**3:.1f}GB / {mem.total / 1024**3:.1f}GB ({mem.percent}%)")
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"[Memory] GPU: {allocated:.2f}GB allocated")
    except ImportError:
        pass


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================================
# 데이터 분할
# ============================================================================

def detect_json_format(file_path: str) -> str:
    """JSON 파일 형식 감지 (array 또는 object)"""
    with open(file_path, "rb") as f:
        first_char = f.read(1).decode("utf-8").strip()
        while first_char in ("\ufeff", " ", "\n", "\r", "\t", ""):
            first_char = f.read(1).decode("utf-8")
    return "array" if first_char == "[" else "object"


def split_precedents(
    source_path: str,
    chunk_size: int = 5000,
    output_dir: str = ".",
) -> List[str]:
    """
    판례 JSON을 작은 파일들로 분할 (스트리밍 방식 - 메모리 효율적)

    Args:
        source_path: 원본 JSON 파일 경로
        chunk_size: 파일당 항목 수
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
        chunk_size: 파일당 항목 수
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
# LanceDB 스키마
# ============================================================================
#
# ⚠️ 중요: 이 스키마는 백엔드의 schema_v2.py와 반드시 동기화되어야 합니다!
# 위치: backend/app/common/vectorstore/schema_v2.py
#
# 스키마 변경 시 체크리스트:
# 1. 백엔드 schema_v2.py 수정
# 2. runpod_lancedb_embeddings.py의 스키마 수정
# 3. 이 파일의 LEGAL_CHUNKS_SCHEMA 수정
# ============================================================================

VECTOR_DIM = CONFIG["VECTOR_DIM"]

LEGAL_CHUNKS_SCHEMA = pa.schema([
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
    pa.field("promulgation_date", pa.utf8()),
    pa.field("promulgation_no", pa.utf8()),
    pa.field("law_type", pa.utf8()),
    pa.field("article_no", pa.utf8()),
    pa.field("case_number", pa.utf8()),
    pa.field("case_type", pa.utf8()),
    pa.field("judgment_type", pa.utf8()),
    pa.field("judgment_status", pa.utf8()),
    pa.field("reference_provisions", pa.utf8()),
    pa.field("reference_cases", pa.utf8()),
])


# ============================================================================
# LanceDB Store (간소화)
# ============================================================================

class SimpleLanceDBStore:
    """간소화된 LanceDB Store"""

    def __init__(self, db_path: str = None):
        import lancedb

        db_path = db_path or CONFIG["LANCEDB_URI"]
        Path(db_path).mkdir(parents=True, exist_ok=True)

        self.db = lancedb.connect(db_path)
        self.table_name = CONFIG["LANCEDB_TABLE_NAME"]
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

    def add_chunks(self, chunks: List[Dict]) -> int:
        """청크 배치 추가"""
        if not chunks:
            return 0
        table = self._ensure_table()
        table.add(chunks)
        return len(chunks)

    def count(self) -> int:
        if self._table is None:
            return 0
        return len(self._table)

    def count_by_type(self, data_type: str) -> int:
        if self._table is None:
            return 0
        try:
            result = self._table.search().where(f"data_type = '{data_type}'").limit(2000000).to_arrow()
            return result.num_rows
        except Exception:
            return 0

    def compact(self):
        if self._table is not None:
            try:
                self._table.compact_files()
            except Exception:
                pass

    def reset(self):
        if self.table_name in self.db.table_names():
            self.db.drop_table(self.table_name)
        self._table = None


# ============================================================================
# 임베딩 모델
# ============================================================================

_model = None
_device = None


def get_model(device: str = None):
    global _model, _device

    if device is None:
        device = get_device()

    if _model is None or _device != device:
        from sentence_transformers import SentenceTransformer

        print(f"[INFO] Loading {CONFIG['EMBEDDING_MODEL']}...")
        print("[INFO] This may take a while on first run (downloading model)...")
        _model = SentenceTransformer(
            CONFIG["EMBEDDING_MODEL"],
            trust_remote_code=True,
            device=device,
        )
        _device = device
        print(f"[INFO] Model loaded successfully on {device.upper()}!")

    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """텍스트 임베딩 생성"""
    model = get_model()
    processed = [t.strip()[:4000] if t else "(내용 없음)" for t in texts]
    embeddings = model.encode(processed, show_progress_bar=False)
    result = [emb.tolist() for emb in embeddings]
    del embeddings, processed
    return result


def clear_model():
    global _model, _device
    if _model is not None:
        del _model
        _model = None
        _device = None
        clear_memory()
        print("[INFO] Model cleared")


# ============================================================================
# 청킹
# ============================================================================

def chunk_precedent_text(text: str) -> List[tuple]:
    """판례 텍스트 청킹"""
    chunk_size = CONFIG["PRECEDENT_CHUNK_SIZE"]
    overlap = CONFIG["PRECEDENT_CHUNK_OVERLAP"]
    min_size = CONFIG["PRECEDENT_MIN_CHUNK_SIZE"]

    if not text or len(text) < min_size:
        return [(0, text)] if text else []

    chunks = []
    start = 0
    idx = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            for sep in ['. ', '.\n', '\n\n', '\n', ' ']:
                pos = text.rfind(sep, start + min_size, end)
                if pos > start:
                    end = pos + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk and len(chunk) >= min_size:
            chunks.append((idx, chunk))
            idx += 1

        start = end - overlap
        if start >= len(text) - min_size:
            break

    return chunks


# ============================================================================
# 판례 처리
# ============================================================================

def process_precedent_part(
    source_path: str,
    batch_size: int = None,
    reset: bool = False,
) -> dict:
    """
    단일 판례 파일 처리

    Args:
        source_path: JSON 파일 경로
        batch_size: 배치 크기
        reset: True면 기존 데이터 삭제 (첫 파일에만 사용)
    """
    print("=" * 50)
    print(f"Processing: {source_path}")
    print("=" * 50)

    batch_size = batch_size or CONFIG["BATCH_SIZE"]
    device = print_device_info()

    # 데이터 로드
    with open(source_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    print(f"Items: {len(items):,}")
    print_memory_status()

    # Store 초기화
    store = SimpleLanceDBStore()

    if reset:
        store.reset()
        print("[INFO] Table reset")

    stats = {"total": len(items), "processed": 0, "chunks": 0, "skipped": 0}

    # 모델 로드
    get_model(device)

    # 배치 처리
    batch = []
    start_time = datetime.now()
    first_item_logged = False

    for item in tqdm(items, desc="Embedding"):
        # 첫 번째 데이터 로드 성공 확인 로그
        if not first_item_logged:
            print("\n>>> 첫 번째 데이터 로드 성공! 임베딩 진행 중...")
            first_item_logged = True

        source_id = str(item.get("판례정보일련번호", item.get("id", "")))

        # 텍스트 추출
        parts = []
        case_name = item.get("사건명", item.get("case_name", ""))
        if case_name:
            parts.append(f"[{case_name}]")

        summary = item.get("판시사항", item.get("summary", ""))
        if summary:
            parts.append(summary)

        judgment = item.get("판결요지", item.get("judgment_summary", ""))
        if judgment:
            parts.append(judgment)

        text = "\n".join(parts)
        if not text or len(text) < CONFIG["PRECEDENT_MIN_CHUNK_SIZE"]:
            stats["skipped"] += 1
            continue

        # 청킹
        chunks = chunk_precedent_text(text)
        if not chunks:
            stats["skipped"] += 1
            continue

        stats["processed"] += 1
        total_chunks = len(chunks)

        for chunk_idx, chunk_content in chunks:
            batch.append({
                "id": f"prec_{source_id}_{chunk_idx}",  # 접두사로 ID 충돌 방지
                "source_id": source_id,
                "data_type": "판례",
                "title": case_name,
                "content": f"[판례] {chunk_content}",
                "vector": None,  # 나중에 채움
                "date": item.get("선고일자", item.get("decision_date", "")),
                "source_name": item.get("법원명", item.get("court_name", "")),
                "chunk_index": chunk_idx,
                "total_chunks": total_chunks,
                "promulgation_date": None,
                "promulgation_no": None,
                "law_type": None,
                "article_no": None,
                "case_number": item.get("사건번호", item.get("case_number", "")),
                "case_type": item.get("사건종류명", item.get("case_type", "")),
                "judgment_type": None,
                "judgment_status": None,
                "reference_provisions": item.get("참조조문", ""),
                "reference_cases": item.get("참조판례", ""),
            })

        # 배치 저장
        if len(batch) >= batch_size:
            texts = [c["content"] for c in batch]
            embeddings = embed_texts(texts)

            for i, emb in enumerate(embeddings):
                batch[i]["vector"] = emb

            store.add_chunks(batch)
            stats["chunks"] += len(batch)

            # 메모리 정리
            del embeddings, texts
            batch.clear()
            clear_memory()

    # 남은 배치 처리
    if batch:
        texts = [c["content"] for c in batch]
        embeddings = embed_texts(texts)

        for i, emb in enumerate(embeddings):
            batch[i]["vector"] = emb

        store.add_chunks(batch)
        stats["chunks"] += len(batch)

        del embeddings, texts
        batch.clear()

    # 정리
    del items
    store.compact()
    clear_memory()

    elapsed = datetime.now() - start_time
    print(f"\nDone! Time: {elapsed}")
    print(f"  Processed: {stats['processed']:,}")
    print(f"  Chunks: {stats['chunks']:,}")
    print(f"  Skipped: {stats['skipped']:,}")
    print_memory_status()

    return stats


def process_all_parts(pattern: str = "precedents_part_*.json"):
    """모든 분할 파일 처리"""
    import glob

    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[ERROR] No files matching: {pattern}")
        return

    print(f"[INFO] Found {len(files)} files to process")

    total_stats = {"processed": 0, "chunks": 0, "skipped": 0}

    for i, filepath in enumerate(files):
        reset = (i == 0)  # 첫 파일에서만 reset

        stats = process_precedent_part(filepath, reset=reset)

        total_stats["processed"] += stats["processed"]
        total_stats["chunks"] += stats["chunks"]
        total_stats["skipped"] += stats["skipped"]

        # 파일 간 메모리 정리
        clear_memory()
        print(f"\n[Progress] {i+1}/{len(files)} files done")
        print("-" * 50)

    print("\n" + "=" * 50)
    print("ALL DONE!")
    print("=" * 50)
    print(f"Total processed: {total_stats['processed']:,}")
    print(f"Total chunks: {total_stats['chunks']:,}")
    print(f"Total skipped: {total_stats['skipped']:,}")

    show_stats()


# ============================================================================
# 법령 처리
# ============================================================================

PARAGRAPH_PATTERN = re.compile(r"([①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳])")


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    korean = len(re.findall(r"[가-힣]", text))
    other = len(text) - korean
    return int(korean / 1.5 + other / 4)


def chunk_law_content(content: str) -> List[tuple]:
    """법령 청킹"""
    max_tokens = 800
    min_tokens = 100

    if not content:
        return []

    articles = content.split("\n\n")
    chunks = []
    idx = 0
    article_pattern = re.compile(r"^(제\d+조(?:의\d+)?)")

    for article in articles:
        article = article.strip()
        if not article:
            continue

        match = article_pattern.match(article)
        article_no = match.group(1) if match else None
        tokens = estimate_tokens(article)

        if tokens <= max_tokens:
            if tokens >= min_tokens:
                chunks.append((idx, article, article_no))
                idx += 1
            elif chunks:
                prev_idx, prev_text, prev_no = chunks[-1]
                chunks[-1] = (prev_idx, prev_text + "\n\n" + article, prev_no)
            else:
                chunks.append((idx, article, article_no))
                idx += 1
        else:
            # 긴 조문은 항 단위로 분리
            parts = PARAGRAPH_PATTERN.split(article)
            current = ""
            current_no = article_no

            for part in parts:
                if not current:
                    current = part
                elif estimate_tokens(current + "\n" + part) <= max_tokens:
                    current += "\n" + part
                else:
                    if estimate_tokens(current) >= min_tokens:
                        chunks.append((idx, current, current_no))
                        idx += 1
                    current = part

            if current and estimate_tokens(current) >= min_tokens:
                chunks.append((idx, current, current_no))
                idx += 1

    return chunks


def process_law_part(source_path: str, batch_size: int = None, reset: bool = False) -> dict:
    """법령 파일 처리"""
    print("=" * 50)
    print(f"Processing Laws: {source_path}")
    print("=" * 50)

    batch_size = batch_size or CONFIG["BATCH_SIZE"]
    device = print_device_info()

    with open(source_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data if isinstance(data, list) else data.get("items", [])
    print(f"Items: {len(items):,}")

    store = SimpleLanceDBStore()
    if reset:
        # 법령만 삭제 (판례는 유지)
        print("[INFO] Deleting existing law data...")

    stats = {"total": len(items), "processed": 0, "chunks": 0, "skipped": 0}
    get_model(device)

    batch = []
    start_time = datetime.now()
    first_item_logged = False

    for item in tqdm(items, desc="Law Embedding"):
        # 첫 번째 데이터 로드 성공 확인 로그
        if not first_item_logged:
            print("\n>>> 첫 번째 법령 데이터 로드 성공! 임베딩 진행 중...")
            first_item_logged = True

        source_id = item.get("law_id", "")
        content = item.get("content", "")

        if not content:
            stats["skipped"] += 1
            continue

        chunks = chunk_law_content(content)
        if not chunks:
            stats["skipped"] += 1
            continue

        stats["processed"] += 1
        total_chunks = len(chunks)

        for chunk_idx, chunk_content, article_no in chunks:
            prefix = f"[법령] {article_no} " if article_no else "[법령] "

            batch.append({
                "id": f"law_{source_id}_{chunk_idx}",  # 접두사로 ID 충돌 방지
                "source_id": source_id,
                "data_type": "법령",
                "title": item.get("law_name", ""),
                "content": prefix + chunk_content,
                "vector": None,
                "date": item.get("enforcement_date", ""),
                "source_name": item.get("ministry", ""),
                "chunk_index": chunk_idx,
                "total_chunks": total_chunks,
                "promulgation_date": item.get("promulgation_date", ""),
                "promulgation_no": item.get("promulgation_no", ""),
                "law_type": item.get("law_type", ""),
                "article_no": article_no or "",
                "case_number": None,
                "case_type": None,
                "judgment_type": None,
                "judgment_status": None,
                "reference_provisions": None,
                "reference_cases": None,
            })

        if len(batch) >= batch_size:
            texts = [c["content"] for c in batch]
            embeddings = embed_texts(texts)

            for i, emb in enumerate(embeddings):
                batch[i]["vector"] = emb

            store.add_chunks(batch)
            stats["chunks"] += len(batch)

            del embeddings, texts
            batch.clear()
            clear_memory()

    if batch:
        texts = [c["content"] for c in batch]
        embeddings = embed_texts(texts)

        for i, emb in enumerate(embeddings):
            batch[i]["vector"] = emb

        store.add_chunks(batch)
        stats["chunks"] += len(batch)

    del items
    store.compact()
    clear_memory()

    elapsed = datetime.now() - start_time
    print(f"\nDone! Time: {elapsed}")
    print(f"  Processed: {stats['processed']:,}")
    print(f"  Chunks: {stats['chunks']:,}")

    return stats


# ============================================================================
# 통계
# ============================================================================

def show_stats():
    """LanceDB 통계"""
    store = SimpleLanceDBStore()

    print("\n" + "=" * 50)
    print("LanceDB Statistics")
    print("=" * 50)

    total = store.count()
    print(f"Total chunks: {total:,}")

    if total > 0:
        law_count = store.count_by_type("법령")
        precedent_count = store.count_by_type("판례")
        print(f"  - 법령: {law_count:,}")
        print(f"  - 판례: {precedent_count:,}")


# ============================================================================
# 메인
# ============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("LanceDB Split Embedding Creator")
    print("=" * 50)
    print_device_info()
    print("\nFunctions:")
    print("  split_precedents(path, chunk_size=5000)  # 데이터 분할")
    print("  split_laws(path, chunk_size=2000)")
    print("  process_precedent_part(path)            # 단일 파일 처리")
    print("  process_law_part(path)")
    print("  process_all_parts()                     # 모든 분할 파일 처리")
    print("  show_stats()                            # 통계")
    print("  clear_model()                           # 모델 메모리 해제")
    print("  print_memory_status()                   # 메모리 확인")
