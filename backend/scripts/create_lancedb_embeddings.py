#!/usr/bin/env python3
"""
LanceDB 임베딩 생성 스크립트 (v2 스키마)

JSON 파일에서 법령/판례 데이터를 읽어 LanceDB에 저장합니다.
스키마 v2 (단일 테이블 + NULL) 방식을 사용합니다.

사용법:
    # 판례 임베딩 생성
    uv run python scripts/create_lancedb_embeddings.py --type precedent --source ../data/precedents.json

    # 법령 임베딩 생성
    uv run python scripts/create_lancedb_embeddings.py --type law --source ../data/law_cleaned.json

    # 전체 재생성
    uv run python scripts/create_lancedb_embeddings.py --type precedent --source ../data/precedents.json --reset

    # 통계 확인
    uv run python scripts/create_lancedb_embeddings.py --stats

지원 환경:
    - CUDA GPU (NVIDIA)
    - Apple Silicon (MPS)
    - CPU
"""

import argparse
import gc
import json
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any, Iterator
from dataclasses import dataclass

import torch
from torch.utils.data import IterableDataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.common.vectorstore.lancedb import LanceDBStore
from app.common.vectorstore.schema_v2 import VECTOR_DIM

try:
    import ijson
    IJSON_AVAILABLE = True
except ImportError:
    IJSON_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, **kwargs):
        return iterable


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

    def __str__(self) -> str:
        return f"{self.name} ({self.device}, {self.vram_gb:.1f}GB)"


@dataclass
class OptimalConfig:
    """환경별 최적 설정"""
    batch_size: int
    num_workers: int
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
        is_laptop = any(
            kw in name.lower()
            for kw in ["laptop", "mobile", "max-q", "notebook"]
        )
        return DeviceInfo(device=device, name=name, vram_gb=vram_gb, is_laptop=is_laptop)

    elif device == "mps":
        try:
            import psutil
            total_ram = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            total_ram = 16.0
        usable_memory = total_ram * 0.75
        return DeviceInfo(device=device, name="Apple Silicon (MPS)", vram_gb=usable_memory, is_laptop=True)

    else:
        try:
            import psutil
            total_ram = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            total_ram = 8.0
        return DeviceInfo(device=device, name="CPU", vram_gb=total_ram)


def get_optimal_config(device_info: DeviceInfo = None) -> OptimalConfig:
    """디바이스에 따른 최적 설정 반환"""
    if device_info is None:
        device_info = get_device_info()

    device = device_info.device
    vram = device_info.vram_gb

    if device == "cuda":
        if vram >= 14:
            return OptimalConfig(batch_size=100, num_workers=4, gc_interval=20)
        elif vram >= 8:
            return OptimalConfig(batch_size=70, num_workers=2, gc_interval=15)
        elif vram >= 6:
            return OptimalConfig(batch_size=50, num_workers=2, gc_interval=10)
        else:
            return OptimalConfig(batch_size=30, num_workers=1, gc_interval=5)

    elif device == "mps":
        if vram >= 12:
            return OptimalConfig(batch_size=50, num_workers=0, gc_interval=10)
        else:
            return OptimalConfig(batch_size=30, num_workers=0, gc_interval=5)

    else:
        return OptimalConfig(batch_size=20, num_workers=2, gc_interval=5)


def print_device_info() -> tuple[DeviceInfo, OptimalConfig]:
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
    print(f"\nRecommended Settings:")
    print(f"  batch_size: {config.batch_size}")
    print(f"  num_workers: {config.num_workers}")
    print("=" * 60)

    return device_info, config


# ============================================================================
# 임베딩 모델 (KURE-v1)
# ============================================================================

KURE_MODEL_NAME = "nlpai-lab/KURE-v1"
_local_model = None
_current_device = None


def get_embedding_model(device: str = None):
    """KURE-v1 모델 로드"""
    global _local_model, _current_device

    if device is None:
        device = get_device()

    if _local_model is None or _current_device != device:
        from sentence_transformers import SentenceTransformer

        cache_dir = Path(__file__).parent.parent / "data" / "models"
        cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Loading model: {KURE_MODEL_NAME}")
        print(f"[INFO] Device: {device.upper()}")

        _local_model = SentenceTransformer(
            KURE_MODEL_NAME,
            cache_folder=str(cache_dir),
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
    return [emb.tolist() for emb in embeddings]


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


# ============================================================================
# 청킹 설정
# ============================================================================

@dataclass
class PrecedentChunkConfig:
    """판례 청크 설정"""
    chunk_size: int = 1250
    chunk_overlap: int = 125
    min_chunk_size: int = 100


@dataclass
class LawChunkConfig:
    """법령 청크 설정 (토큰 기반)"""
    max_tokens: int = 800
    min_tokens: int = 100


# 항 번호 패턴
PARAGRAPH_PATTERN = re.compile(r"([①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳])")


def estimate_tokens(text: str) -> int:
    """토큰 수 추정"""
    if not text:
        return 0
    korean_chars = len(re.findall(r"[가-힣]", text))
    other_chars = len(text) - korean_chars
    return int(korean_chars / 1.5 + other_chars / 4)


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


def split_by_paragraphs(article_text: str) -> List[str]:
    """조문을 항(①②③) 단위로 분리"""
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
# DataLoader용 Dataset 클래스
# ============================================================================

class PrecedentIterableDataset(IterableDataset):
    """판례 데이터를 스트리밍으로 읽어 청크를 생성하는 Dataset"""

    def __init__(
        self,
        source_path: str,
        chunk_config: PrecedentChunkConfig,
        existing_source_ids: set = None,
    ):
        self.source_path = source_path
        self.chunk_config = chunk_config
        self.existing_source_ids = existing_source_ids or set()

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        use_streaming = IJSON_AVAILABLE

        if use_streaming:
            f = open(self.source_path, "rb")
            items = ijson.items(f, "items.item")
        else:
            with open(self.source_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            items = data.get("items", data.get("precedents", []))

        for idx, item in enumerate(items):
            source_id = str(item.get("판례정보일련번호", item.get("id", item.get("source_id", idx))))

            if source_id in self.existing_source_ids:
                continue

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
            if not text or len(text) < self.chunk_config.min_chunk_size:
                continue

            chunks = chunk_precedent_text(text, self.chunk_config)
            if not chunks:
                continue

            total_chunks = len(chunks)

            decision_date = item.get("선고일자", item.get("decision_date", ""))
            court_name = item.get("법원명", item.get("court_name", ""))
            case_number = item.get("사건번호", item.get("case_number", ""))
            case_type = item.get("사건종류명", item.get("case_type", ""))
            ref_provisions = item.get("참조조문", item.get("reference_provisions", ""))
            ref_cases = item.get("참조판례", item.get("reference_cases", ""))
            ruling = item.get("주문", item.get("ruling", ""))
            claim = item.get("청구취지", item.get("claim", ""))
            reasoning = item.get("이유", item.get("reasoning", ""))

            for chunk_idx, chunk_content in chunks:
                yield {
                    "source_id": source_id,
                    "chunk_index": chunk_idx,
                    "title": case_name,
                    "content": f"[판례] {chunk_content}",
                    "decision_date": decision_date,
                    "court_name": court_name,
                    "total_chunks": total_chunks,
                    "case_number": case_number,
                    "case_type": case_type,
                    "reference_provisions": ref_provisions,
                    "reference_cases": ref_cases,
                    "ruling": ruling,
                    "claim": claim,
                    "reasoning": reasoning,
                }

        if use_streaming:
            f.close()


class LawIterableDataset(IterableDataset):
    """법령 데이터를 읽어 청크를 생성하는 Dataset"""

    def __init__(
        self,
        source_path: str,
        chunk_config: LawChunkConfig,
        existing_source_ids: set = None,
    ):
        self.source_path = source_path
        self.chunk_config = chunk_config
        self.existing_source_ids = existing_source_ids or set()

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        with open(self.source_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        items = data.get("items", [])

        for item in items:
            source_id = item.get("law_id", "")

            if source_id in self.existing_source_ids:
                continue

            content = item.get("content", "")
            if not content:
                continue

            chunks = chunk_law_content(content, self.chunk_config)
            if not chunks:
                continue

            total_chunks = len(chunks)
            law_name = item.get("law_name", "")
            enforcement_date = item.get("enforcement_date", "")
            ministry = item.get("ministry", "")
            promulgation_date = item.get("promulgation_date", "")
            promulgation_no = item.get("promulgation_no", "")
            law_type = item.get("law_type", "")

            for chunk_idx, chunk_content, article_no in chunks:
                if article_no:
                    prefixed_content = f"[법령] {article_no} {chunk_content}"
                else:
                    prefixed_content = f"[법령] {chunk_content}"

                yield {
                    "source_id": source_id,
                    "chunk_index": chunk_idx,
                    "title": law_name,
                    "content": prefixed_content,
                    "enforcement_date": enforcement_date,
                    "department": ministry,
                    "total_chunks": total_chunks,
                    "promulgation_date": promulgation_date,
                    "promulgation_no": promulgation_no,
                    "law_type": law_type,
                    "article_no": article_no or "",
                }


def collate_chunks(batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """DataLoader용 collate 함수"""
    if not batch:
        return {}
    result = {key: [] for key in batch[0].keys()}
    for item in batch:
        for key, value in item.items():
            result[key].append(value)
    return result


# ============================================================================
# 임베딩 생성 함수
# ============================================================================

def create_precedent_embeddings(
    source_path: str,
    store: LanceDBStore = None,
    batch_size: int = None,
    reset: bool = False,
    num_workers: int = None,
    auto_config: bool = True,
) -> dict:
    """판례 임베딩 생성"""
    device_info = get_device_info()
    optimal = get_optimal_config(device_info)

    if auto_config:
        batch_size = batch_size or optimal.batch_size
        num_workers = num_workers if num_workers is not None else optimal.num_workers
        gc_interval = optimal.gc_interval
    else:
        batch_size = batch_size or 100
        num_workers = num_workers if num_workers is not None else 0
        gc_interval = 10

    chunk_config = PrecedentChunkConfig()

    if store is None:
        store = LanceDBStore()

    stats = {
        "total_docs": 0,
        "processed_docs": 0,
        "total_chunks": 0,
        "skipped": 0,
        "errors": 0,
        "device": str(device_info),
    }

    print(f"[INFO] Loading precedent data from: {source_path}")

    if reset:
        deleted = store.delete_by_type("판례")
        print(f"[INFO] Deleted {deleted} existing precedent chunks")

    existing_source_ids = store.get_existing_source_ids("판례") if not reset else set()
    if existing_source_ids:
        print(f"[INFO] Found {len(existing_source_ids)} precedents already embedded")
        stats["skipped"] = len(existing_source_ids)

    # 전체 개수 파악
    print("[INFO] Counting total items...")
    if IJSON_AVAILABLE:
        total_items = 0
        with open(source_path, "rb") as f:
            for _ in ijson.items(f, "items.item"):
                total_items += 1
        stats["total_docs"] = total_items
    else:
        with open(source_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = data.get("items", data.get("precedents", []))
        stats["total_docs"] = len(items)

    print(f"[INFO] Total precedents: {stats['total_docs']:,}")

    if stats["total_docs"] == 0:
        return stats

    get_embedding_model(device_info.device)

    dataset = PrecedentIterableDataset(
        source_path=source_path,
        chunk_config=chunk_config,
        existing_source_ids=existing_source_ids,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "collate_fn": collate_chunks,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    dataloader = DataLoader(dataset, **loader_kwargs)

    print(f"[INFO] Device: {device_info}")
    print(f"[INFO] Settings: batch_size={batch_size}, num_workers={num_workers}, gc_interval={gc_interval}")

    processed_source_ids = set()
    batch_count = 0
    pbar = tqdm(dataloader, desc="판례 임베딩", unit="batch")

    for batch in pbar:
        if not batch:
            continue

        batch_count += 1

        try:
            embeddings = create_embeddings(batch["content"], device_info.device)

            store.add_precedent_documents(
                source_ids=batch["source_id"],
                chunk_indices=batch["chunk_index"],
                embeddings=embeddings,
                titles=batch["title"],
                contents=batch["content"],
                decision_dates=batch["decision_date"],
                court_names=batch["court_name"],
                total_chunks_list=batch["total_chunks"],
                case_numbers=batch["case_number"],
                case_types=batch["case_type"],
                reference_provisions_list=batch["reference_provisions"],
                reference_cases_list=batch["reference_cases"],
                rulings=batch["ruling"],
                claims=batch["claim"],
                reasonings=batch["reasoning"],
            )

            stats["total_chunks"] += len(batch["source_id"])
            for sid in batch["source_id"]:
                if sid not in processed_source_ids:
                    processed_source_ids.add(sid)
                    stats["processed_docs"] += 1

        except Exception as e:
            stats["errors"] += len(batch.get("source_id", []))
            if TQDM_AVAILABLE:
                tqdm.write(f"  [ERROR] Batch error: {e}")
            else:
                print(f"  [ERROR] Batch error: {e}")

        if batch_count % gc_interval == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if TQDM_AVAILABLE:
            pbar.set_postfix(docs=stats["processed_docs"], chunks=stats["total_chunks"])

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return stats


def create_law_embeddings(
    source_path: str,
    store: LanceDBStore = None,
    batch_size: int = None,
    reset: bool = False,
    num_workers: int = None,
    auto_config: bool = True,
) -> dict:
    """법령 임베딩 생성"""
    device_info = get_device_info()
    optimal = get_optimal_config(device_info)

    if auto_config:
        batch_size = batch_size or optimal.batch_size
        num_workers = num_workers if num_workers is not None else optimal.num_workers
        gc_interval = optimal.gc_interval
    else:
        batch_size = batch_size or 100
        num_workers = num_workers if num_workers is not None else 0
        gc_interval = 10

    chunk_config = LawChunkConfig()

    if store is None:
        store = LanceDBStore()

    stats = {
        "total_docs": 0,
        "processed_docs": 0,
        "total_chunks": 0,
        "skipped": 0,
        "errors": 0,
        "device": str(device_info),
    }

    print(f"[INFO] Loading law data from: {source_path}")
    with open(source_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("items", [])
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

    get_embedding_model(device_info.device)

    dataset = LawIterableDataset(
        source_path=source_path,
        chunk_config=chunk_config,
        existing_source_ids=existing_source_ids,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "collate_fn": collate_chunks,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    dataloader = DataLoader(dataset, **loader_kwargs)

    print(f"[INFO] Device: {device_info}")
    print(f"[INFO] Settings: batch_size={batch_size}, num_workers={num_workers}, gc_interval={gc_interval}")

    processed_source_ids = set()
    batch_count = 0
    pbar = tqdm(dataloader, desc="법령 임베딩", unit="batch")

    for batch in pbar:
        if not batch:
            continue

        batch_count += 1

        try:
            embeddings = create_embeddings(batch["content"], device_info.device)

            store.add_law_documents(
                source_ids=batch["source_id"],
                chunk_indices=batch["chunk_index"],
                embeddings=embeddings,
                titles=batch["title"],
                contents=batch["content"],
                enforcement_dates=batch["enforcement_date"],
                departments=batch["department"],
                total_chunks_list=batch["total_chunks"],
                promulgation_dates=batch["promulgation_date"],
                promulgation_nos=batch["promulgation_no"],
                law_types=batch["law_type"],
                article_nos=batch["article_no"],
            )

            stats["total_chunks"] += len(batch["source_id"])
            for sid in batch["source_id"]:
                if sid not in processed_source_ids:
                    processed_source_ids.add(sid)
                    stats["processed_docs"] += 1

        except Exception as e:
            stats["errors"] += len(batch.get("source_id", []))
            if TQDM_AVAILABLE:
                tqdm.write(f"  [ERROR] Batch error: {e}")
            else:
                print(f"  [ERROR] Batch error: {e}")

        if batch_count % gc_interval == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if TQDM_AVAILABLE:
            pbar.set_postfix(docs=stats["processed_docs"], chunks=stats["total_chunks"])

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return stats


# ============================================================================
# 통계
# ============================================================================

def show_stats():
    """LanceDB 통계 출력"""
    store = LanceDBStore()

    print("\n" + "=" * 60)
    print("LanceDB Statistics (v2 Schema)")
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
# 메인
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="LanceDB 임베딩 생성 (v2 스키마)")
    parser.add_argument(
        "--type",
        choices=["precedent", "law", "all"],
        default="precedent",
        help="임베딩할 문서 유형 (기본: precedent)"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=False,
        help="JSON 파일 경로"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="배치 크기 (None=자동)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader 워커 수 (None=자동)"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="기존 데이터 삭제 후 재생성"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="통계만 출력"
    )
    parser.add_argument(
        "--device-info",
        action="store_true",
        help="디바이스 정보 출력"
    )

    args = parser.parse_args()

    if args.device_info:
        print_device_info()
        return

    if args.stats:
        show_stats()
        return

    if not args.source:
        print("[ERROR] --source 옵션이 필요합니다.")
        print("[INFO] 예: --type precedent --source ../data/precedents.json")
        sys.exit(1)

    print("=" * 60)
    print("LanceDB Embedding Creator (v2 Schema)")
    print("=" * 60)

    device_info, optimal = print_device_info()

    store = LanceDBStore()
    start_time = datetime.now()
    stats = {}

    if args.type == "precedent":
        print("\n[판례 임베딩]")
        stats = create_precedent_embeddings(
            source_path=args.source,
            store=store,
            batch_size=args.batch_size,
            reset=args.reset,
            num_workers=args.num_workers,
        )

    elif args.type == "law":
        print("\n[법령 임베딩]")
        stats = create_law_embeddings(
            source_path=args.source,
            store=store,
            batch_size=args.batch_size,
            reset=args.reset,
            num_workers=args.num_workers,
        )

    else:
        # all - 판례와 법령 모두 (두 파일 필요)
        print("\n[INFO] --type all 사용 시 판례와 법령을 따로 실행하세요:")
        print("  --type precedent --source ../data/precedents.json")
        print("  --type law --source ../data/law_cleaned.json")
        sys.exit(1)

    elapsed = datetime.now() - start_time

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\nElapsed time: {elapsed}")

    show_stats()


if __name__ == "__main__":
    main()
