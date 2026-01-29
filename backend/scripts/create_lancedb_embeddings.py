#!/usr/bin/env python3
"""
LanceDB 임베딩 생성 스크립트 (v2 스키마)

PostgreSQL 새 테이블(law_documents, precedent_documents)에서 데이터를 읽어
LanceDB에 저장합니다. 스키마 v2 (단일 테이블 + NULL) 방식을 사용합니다.

사전 요구사항:
    # PostgreSQL에 데이터 로드 (먼저 실행 필요)
    uv run python scripts/load_lancedb_data.py --type all

사용법:
    # 판례 임베딩 생성 (PostgreSQL precedent_documents 테이블에서)
    uv run python scripts/create_lancedb_embeddings.py --type precedent

    # 법령 임베딩 생성 (PostgreSQL law_documents 테이블에서)
    uv run python scripts/create_lancedb_embeddings.py --type law

    # 전체 재생성
    uv run python scripts/create_lancedb_embeddings.py --type all --reset

    # 통계 확인
    uv run python scripts/create_lancedb_embeddings.py --stats

    # 배치 크기 수동 지정 (자동 감지 대신)
    uv run python scripts/create_lancedb_embeddings.py --type all --batch-size 30

NOTE:
    - ruling, claim, reasoning은 LanceDB에 저장하지 않음 (메모리 효율화)
    - 검색 후 PostgreSQL에서 원본 조회하여 해당 필드 접근
    - MPS(Mac), CUDA, CPU 자동 감지 및 배치 크기 최적화
"""

import argparse
import asyncio
import gc
import json
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select, func

from app.core.config import settings
from app.common.database import async_session_factory
from app.common.vectorstore.lancedb import LanceDBStore
from app.common.vectorstore.schema_v2 import VECTOR_DIM
from app.models.law_document import LawDocument
from app.models.precedent_document import PrecedentDocument


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
    gc_interval: int = 10  # gc.collect() 호출 간격 (배치 수)


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
        if vram >= 20:  # RTX 3090/4090
            return OptimalConfig(batch_size=128, num_workers=4, gc_interval=25)
        elif vram >= 14:  # RTX 4080, 3080 Ti
            return OptimalConfig(batch_size=100, num_workers=4, gc_interval=20)
        elif vram >= 8:  # RTX 3070, 4060
            return OptimalConfig(batch_size=70, num_workers=2, gc_interval=15)
        else:
            return OptimalConfig(batch_size=50, num_workers=2, gc_interval=10)

    elif device == "mps":
        # Mac - MPS는 보수적으로 설정
        if vram >= 12:  # M3 16GB
            return OptimalConfig(batch_size=40, num_workers=0, gc_interval=8)
        else:  # M1/M2 8GB
            return OptimalConfig(batch_size=25, num_workers=0, gc_interval=5)

    else:  # CPU
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
    if device_info.compute_capability:
        cc = device_info.compute_capability
        print(f"  Compute Capability: {cc[0]}.{cc[1]}")

    print("\nOptimal Settings:")
    print(f"  batch_size: {config.batch_size}")
    print(f"  gc_interval: {config.gc_interval} batches")
    print("=" * 60)

    return device_info, config


def clear_gpu_memory():
    """GPU 메모리 정리"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS는 명시적 캐시 정리 없음, gc만 호출
        pass


# ============================================================================
# 청킹 설정
# ============================================================================

@dataclass
class ChunkConfig:
    """청크 설정"""
    chunk_size: int = 1250
    chunk_overlap: int = 125
    min_chunk_size: int = 100


def chunk_text(text: str, config: ChunkConfig) -> List[tuple[int, str]]:
    """
    텍스트를 청크로 분할 (판례용)

    Returns:
        [(chunk_index, chunk_text), ...]
    """
    if not text or len(text) < config.min_chunk_size:
        return [(0, text)] if text else []

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = min(start + config.chunk_size, len(text))

        # 문장 경계에서 자르기
        if end < len(text):
            for sep in ['. ', '.\n', '\n\n', '\n', ' ']:
                sep_pos = text.rfind(sep, start + config.min_chunk_size, end)
                if sep_pos > start:
                    end = sep_pos + len(sep)
                    break

        chunk_text_content = text[start:end].strip()
        if chunk_text_content and len(chunk_text_content) >= config.min_chunk_size:
            chunks.append((chunk_index, chunk_text_content))
            chunk_index += 1

        start = end - config.chunk_overlap
        if start >= len(text) - config.min_chunk_size:
            break

    return chunks


# ============================================================================
# 법령 청킹 (조문 단위 → 항 단위)
# ============================================================================

@dataclass
class LawChunkConfig:
    """법령 청크 설정 (토큰 기반)"""
    max_tokens: int = 800
    min_tokens: int = 100


# 항 번호 패턴 (①②③④⑤⑥⑦⑧⑨⑩ 등)
PARAGRAPH_PATTERN = re.compile(r"([①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳])")


def estimate_tokens(text: str) -> int:
    """토큰 수 추정 (한글 기준 약 0.5~0.7자/토큰)"""
    if not text:
        return 0
    korean_chars = len(re.findall(r"[가-힣]", text))
    other_chars = len(text) - korean_chars
    return int(korean_chars / 1.5 + other_chars / 4)


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


def chunk_law_content(
    content: str,
    config: LawChunkConfig,
) -> List[tuple[int, str, Optional[str]]]:
    """
    법령 내용을 청크로 분할

    Returns:
        [(chunk_index, chunk_text, article_no), ...]
    """
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
# 임베딩 모델 (환경변수에서 로드)
# ============================================================================

def _get_embedding_model_name() -> str:
    """환경변수에서 임베딩 모델명 로드"""
    return os.getenv("LOCAL_EMBEDDING_MODEL", "nlpai-lab/KURE-v1")


KURE_MODEL_NAME = _get_embedding_model_name()
_local_model = None
_current_device = None


def get_local_model(device: str = None):
    """임베딩 모델 로드 (환경변수 LOCAL_EMBEDDING_MODEL 사용)"""
    global _local_model, _current_device

    if device is None:
        device = get_device()

    if _local_model is None or _current_device != device:
        from sentence_transformers import SentenceTransformer

        cache_dir = Path(__file__).parent.parent.parent / "data" / "models"
        cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Loading KURE model: {KURE_MODEL_NAME}")
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
    """임베딩 생성 (KURE-v1)"""
    model = get_local_model(device)
    processed = [t.strip()[:4000] if t else "(내용 없음)" for t in texts]
    embeddings = model.encode(processed, show_progress_bar=False)
    return [emb.tolist() for emb in embeddings]


def clear_model_cache():
    """모델 캐시 정리 (메모리 해제)"""
    global _local_model, _current_device

    if _local_model is not None:
        del _local_model
        _local_model = None
        _current_device = None
        clear_gpu_memory()
        print("[INFO] Model cache cleared")


# ============================================================================
# 판례 임베딩 (PrecedentDocument 테이블 사용)
# ============================================================================

async def get_precedent_count() -> int:
    """판례 문서 수 (precedent_documents 테이블)"""
    async with async_session_factory() as session:
        query = select(func.count(PrecedentDocument.id))
        result = await session.execute(query)
        return result.scalar() or 0


async def get_precedents(offset: int, limit: int) -> List[PrecedentDocument]:
    """판례 문서 조회 (precedent_documents 테이블)"""
    async with async_session_factory() as session:
        query = (
            select(PrecedentDocument)
            .order_by(PrecedentDocument.id)
            .offset(offset)
            .limit(limit)
        )
        result = await session.execute(query)
        return list(result.scalars().all())


def build_precedent_embedding_text(doc: PrecedentDocument) -> str:
    """판례 임베딩용 텍스트 생성 (판시사항 + 판결요지)"""
    parts = []

    if doc.case_name:
        parts.append(f"[{doc.case_name}]")

    if doc.summary:  # 판시사항
        parts.append(doc.summary)

    if doc.reasoning:  # 판결요지
        parts.append(doc.reasoning)

    return "\n".join(parts)


async def create_precedent_embeddings(
    store: LanceDBStore,
    batch_size: int = None,
    chunk_config: ChunkConfig = None,
    reset: bool = False,
    device_info: DeviceInfo = None,
    gc_interval: int = None,
) -> dict:
    """
    판례 임베딩 생성 (precedent_documents 테이블에서 읽음)

    NOTE: ruling, claim, reasoning은 LanceDB에 저장하지 않음 (메모리 효율화)
          검색 후 PostgreSQL precedent_documents 테이블에서 조회하여 접근
    """
    if chunk_config is None:
        chunk_config = ChunkConfig()

    # 디바이스 및 최적 설정
    if device_info is None:
        device_info = get_device_info()
    optimal = get_optimal_config(device_info)

    if batch_size is None:
        batch_size = optimal.batch_size
    if gc_interval is None:
        gc_interval = optimal.gc_interval

    stats = {
        "total_docs": 0,
        "processed_docs": 0,
        "total_chunks": 0,
        "skipped": 0,
        "errors": 0,
        "device": str(device_info),
    }

    print(f"[INFO] Device: {device_info}")
    print(f"[INFO] Batch size: {batch_size}, GC interval: {gc_interval}")

    if reset:
        print("[INFO] Resetting precedent data...")
        try:
            existing = store.count_by_type("판례")
            if existing > 0:
                store._table.delete("data_type = '판례'")
                print(f"[INFO] Deleted {existing} existing precedent chunks")
        except Exception:
            pass

    # 이미 저장된 source_id 조회
    existing_source_ids = set()
    if not reset and store._table is not None:
        try:
            df = store._table.search().where("data_type = '판례'").limit(1000000).to_pandas()
            existing_source_ids = set(df["source_id"].unique())
            print(f"[INFO] Found {len(existing_source_ids)} documents already embedded")
        except Exception:
            pass

    total_count = await get_precedent_count()
    stats["total_docs"] = total_count
    print(f"[INFO] Total precedents in precedent_documents: {total_count:,}")

    if total_count == 0:
        print("[WARN] No data in precedent_documents table. Run load_lancedb_data.py first.")
        return stats

    # 모델 사전 로드
    get_local_model(device_info.device)

    # 배치 처리
    offset = 0
    db_batch_size = 500
    batch_count = 0

    # 배치 버퍼
    batch_data = {
        "source_ids": [],
        "chunk_indices": [],
        "embeddings": [],
        "titles": [],
        "contents": [],
        "decision_dates": [],
        "court_names": [],
        "total_chunks_list": [],
        "case_numbers": [],
        "case_types": [],
        "reference_provisions_list": [],
        "reference_cases_list": [],
    }

    while offset < total_count:
        docs = await get_precedents(offset, db_batch_size)
        if not docs:
            break

        for doc in docs:
            source_id = doc.serial_number

            if source_id in existing_source_ids:
                stats["skipped"] += 1
                continue

            text = build_precedent_embedding_text(doc)
            if not text or len(text) < chunk_config.min_chunk_size:
                stats["skipped"] += 1
                continue

            chunks = chunk_text(text, chunk_config)
            if not chunks:
                stats["skipped"] += 1
                continue

            total_chunks = len(chunks)
            stats["processed_docs"] += 1

            for chunk_idx, chunk_content in chunks:
                prefixed_content = f"[판례] {chunk_content}"

                batch_data["source_ids"].append(source_id)
                batch_data["chunk_indices"].append(chunk_idx)
                batch_data["titles"].append(doc.case_name or "")
                batch_data["contents"].append(prefixed_content)
                batch_data["decision_dates"].append(
                    doc.decision_date.isoformat() if doc.decision_date else ""
                )
                batch_data["court_names"].append(doc.court_name or "")
                batch_data["total_chunks_list"].append(total_chunks)
                batch_data["case_numbers"].append(doc.case_number or "")
                batch_data["case_types"].append(doc.case_type or "")
                batch_data["reference_provisions_list"].append(doc.reference_provisions or "")
                batch_data["reference_cases_list"].append(doc.reference_cases or "")

            # 배치 크기 도달 시 저장
            if len(batch_data["source_ids"]) >= batch_size:
                batch_count += 1
                try:
                    batch_data["embeddings"] = create_embeddings(
                        batch_data["contents"], device_info.device
                    )
                    store.add_precedent_documents(**batch_data)
                    stats["total_chunks"] += len(batch_data["source_ids"])
                except Exception as e:
                    stats["errors"] += len(batch_data["source_ids"])
                    print(f"  [ERROR] Batch error: {e}")

                # 버퍼 초기화
                for key in batch_data:
                    batch_data[key] = []

                # 메모리 정리
                if batch_count % gc_interval == 0:
                    clear_gpu_memory()

        # 진행률 출력
        progress = min(offset + db_batch_size, total_count)
        pct = progress / total_count * 100
        print(
            f"  [PROGRESS] {progress:,}/{total_count:,} ({pct:.1f}%) - "
            f"Docs: {stats['processed_docs']:,}, Chunks: {stats['total_chunks']:,}"
        )

        offset += db_batch_size

    # 남은 배치 처리
    if batch_data["source_ids"]:
        try:
            batch_data["embeddings"] = create_embeddings(
                batch_data["contents"], device_info.device
            )
            store.add_precedent_documents(**batch_data)
            stats["total_chunks"] += len(batch_data["source_ids"])
        except Exception as e:
            stats["errors"] += len(batch_data["source_ids"])
            print(f"  [ERROR] Final batch error: {e}")

    # 최종 메모리 정리
    clear_gpu_memory()

    return stats


# ============================================================================
# 법령 임베딩 (LawDocument 테이블 사용)
# ============================================================================

async def get_law_count() -> int:
    """법령 문서 수 (law_documents 테이블)"""
    async with async_session_factory() as session:
        query = select(func.count(LawDocument.id))
        result = await session.execute(query)
        return result.scalar() or 0


async def get_laws(offset: int, limit: int) -> List[LawDocument]:
    """법령 문서 조회 (law_documents 테이블)"""
    async with async_session_factory() as session:
        query = (
            select(LawDocument)
            .order_by(LawDocument.id)
            .offset(offset)
            .limit(limit)
        )
        result = await session.execute(query)
        return list(result.scalars().all())


async def create_law_embeddings(
    store: LanceDBStore,
    batch_size: int = None,
    chunk_config: LawChunkConfig = None,
    reset: bool = False,
    device_info: DeviceInfo = None,
    gc_interval: int = None,
) -> dict:
    """
    법령 임베딩 생성 (law_documents 테이블에서 읽음)

    사전 요구사항:
        uv run python scripts/load_lancedb_data.py --type law
    """
    if chunk_config is None:
        chunk_config = LawChunkConfig()

    # 디바이스 및 최적 설정
    if device_info is None:
        device_info = get_device_info()
    optimal = get_optimal_config(device_info)

    if batch_size is None:
        batch_size = optimal.batch_size
    if gc_interval is None:
        gc_interval = optimal.gc_interval

    stats = {
        "total_docs": 0,
        "processed_docs": 0,
        "total_chunks": 0,
        "skipped": 0,
        "errors": 0,
        "device": str(device_info),
    }

    print(f"[INFO] Device: {device_info}")
    print(f"[INFO] Batch size: {batch_size}, GC interval: {gc_interval}")

    if reset:
        print("[INFO] Resetting law data...")
        try:
            existing = store.count_by_type("법령")
            if existing > 0:
                store._table.delete("data_type = '법령'")
                print(f"[INFO] Deleted {existing} existing law chunks")
        except Exception:
            pass

    # 이미 저장된 source_id 조회
    existing_source_ids = set()
    if not reset and store._table is not None:
        try:
            df = store._table.search().where("data_type = '법령'").limit(1000000).to_pandas()
            existing_source_ids = set(df["source_id"].unique())
            print(f"[INFO] Found {len(existing_source_ids)} laws already embedded")
        except Exception:
            pass

    total_count = await get_law_count()
    stats["total_docs"] = total_count
    print(f"[INFO] Total laws in law_documents: {total_count:,}")

    if total_count == 0:
        print("[WARN] No data in law_documents table. Run load_lancedb_data.py first.")
        return stats

    # 모델 사전 로드
    get_local_model(device_info.device)

    # 배치 처리
    offset = 0
    db_batch_size = 500
    batch_count = 0

    # 배치 버퍼
    batch_data = {
        "source_ids": [],
        "chunk_indices": [],
        "embeddings": [],
        "titles": [],
        "contents": [],
        "enforcement_dates": [],
        "departments": [],
        "total_chunks_list": [],
        "promulgation_dates": [],
        "promulgation_nos": [],
        "law_types": [],
        "article_nos": [],
    }

    while offset < total_count:
        docs = await get_laws(offset, db_batch_size)
        if not docs:
            break

        for doc in docs:
            source_id = doc.law_id

            if source_id in existing_source_ids:
                stats["skipped"] += 1
                continue

            content = doc.content or ""
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
                batch_data["titles"].append(doc.law_name or "")
                batch_data["contents"].append(prefixed_content)
                batch_data["enforcement_dates"].append(
                    doc.enforcement_date.isoformat() if doc.enforcement_date else ""
                )
                batch_data["departments"].append(doc.ministry or "")
                batch_data["total_chunks_list"].append(total_chunks)
                batch_data["promulgation_dates"].append(doc.promulgation_date or "")
                batch_data["promulgation_nos"].append(doc.promulgation_no or "")
                batch_data["law_types"].append(doc.law_type or "")
                batch_data["article_nos"].append(article_no or "")

            # 배치 크기 도달 시 저장
            if len(batch_data["source_ids"]) >= batch_size:
                batch_count += 1
                try:
                    batch_data["embeddings"] = create_embeddings(
                        batch_data["contents"], device_info.device
                    )
                    store.add_law_documents(**batch_data)
                    stats["total_chunks"] += len(batch_data["source_ids"])
                except Exception as e:
                    stats["errors"] += len(batch_data["source_ids"])
                    print(f"  [ERROR] Batch error: {e}")

                # 버퍼 초기화
                for key in batch_data:
                    batch_data[key] = []

                # 메모리 정리
                if batch_count % gc_interval == 0:
                    clear_gpu_memory()

        # 진행률 출력
        progress = min(offset + db_batch_size, total_count)
        pct = progress / total_count * 100
        print(
            f"  [PROGRESS] {progress:,}/{total_count:,} ({pct:.1f}%) - "
            f"Docs: {stats['processed_docs']:,}, Chunks: {stats['total_chunks']:,}"
        )

        offset += db_batch_size

    # 남은 배치 처리
    if batch_data["source_ids"]:
        try:
            batch_data["embeddings"] = create_embeddings(
                batch_data["contents"], device_info.device
            )
            store.add_law_documents(**batch_data)
            stats["total_chunks"] += len(batch_data["source_ids"])
        except Exception as e:
            stats["errors"] += len(batch_data["source_ids"])
            print(f"  [ERROR] Final batch error: {e}")

    # 최종 메모리 정리
    clear_gpu_memory()

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
        default="all",
        help="임베딩할 문서 유형 (기본: all)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="배치 크기 (기본: 디바이스에 따라 자동 설정)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1250,
        help="판례 청크 크기 (기본: 1250자)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=125,
        help="판례 청크 오버랩 (기본: 125, 10%%)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=800,
        help="법령 청크 최대 토큰 (기본: 800)"
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=100,
        help="법령 청크 최소 토큰 (기본: 100)"
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

    args = parser.parse_args()

    print("=" * 60)
    print("LanceDB Embedding Creator (v2 Schema)")
    print("=" * 60)
    print(f"Model: {KURE_MODEL_NAME}")
    print(f"Vector dimension: {VECTOR_DIM}")
    print(f"Data source: PostgreSQL (law_documents, precedent_documents)")

    if args.stats:
        show_stats()
        return

    # 디바이스 정보 출력
    device_info, optimal_config = print_device_info()

    batch_size = args.batch_size or optimal_config.batch_size
    gc_interval = optimal_config.gc_interval

    print(f"\nUsing batch_size: {batch_size}")

    store = LanceDBStore()
    start_time = datetime.now()
    stats = {}

    if args.type == "precedent":
        print(f"Precedent chunk size: {args.chunk_size}")
        print(f"Precedent chunk overlap: {args.chunk_overlap}")
        print("\n" + "=" * 60)
        print("Creating embeddings for 판례 (from precedent_documents)...")
        print("=" * 60)

        chunk_config = ChunkConfig(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )

        stats = asyncio.run(create_precedent_embeddings(
            store=store,
            batch_size=batch_size,
            chunk_config=chunk_config,
            reset=args.reset,
            device_info=device_info,
            gc_interval=gc_interval,
        ))

    elif args.type == "law":
        print(f"Law max tokens: {args.max_tokens}")
        print(f"Law min tokens: {args.min_tokens}")
        print("\n" + "=" * 60)
        print("Creating embeddings for 법령 (from law_documents)...")
        print("=" * 60)

        law_chunk_config = LawChunkConfig(
            max_tokens=args.max_tokens,
            min_tokens=args.min_tokens,
        )

        stats = asyncio.run(create_law_embeddings(
            store=store,
            batch_size=batch_size,
            chunk_config=law_chunk_config,
            reset=args.reset,
            device_info=device_info,
            gc_interval=gc_interval,
        ))

    else:
        # all - 판례와 법령 모두 처리
        print("\n" + "=" * 60)
        print("Creating embeddings for 판례 (from precedent_documents)...")
        print("=" * 60)

        chunk_config = ChunkConfig(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )

        precedent_stats = asyncio.run(create_precedent_embeddings(
            store=store,
            batch_size=batch_size,
            chunk_config=chunk_config,
            reset=args.reset,
            device_info=device_info,
            gc_interval=gc_interval,
        ))

        print("\n" + "=" * 60)
        print("Creating embeddings for 법령 (from law_documents)...")
        print("=" * 60)

        law_chunk_config = LawChunkConfig(
            max_tokens=args.max_tokens,
            min_tokens=args.min_tokens,
        )

        law_stats = asyncio.run(create_law_embeddings(
            store=store,
            batch_size=batch_size,
            chunk_config=law_chunk_config,
            reset=args.reset,
            device_info=device_info,
            gc_interval=gc_interval,
        ))

        stats = {
            "precedent": precedent_stats,
            "law": law_stats,
        }

    elapsed = datetime.now() - start_time

    # 결과 출력
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\nElapsed time: {elapsed}")

    # 모델 캐시 정리
    clear_model_cache()

    # 최종 통계
    show_stats()


if __name__ == "__main__":
    main()
