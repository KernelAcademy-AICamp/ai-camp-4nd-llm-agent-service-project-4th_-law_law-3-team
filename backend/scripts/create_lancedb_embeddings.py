#!/usr/bin/env python3
"""
LanceDB 임베딩 생성 스크립트 (Local & RunPod 호환)

PostgreSQL의 law_documents, precedent_documents 테이블에서 데이터를 읽어
KURE-v1 임베딩을 생성하고 LanceDB에 저장합니다.

특징:
- 스트리밍 처리 (메모리 효율적)
- 하드웨어 자동 감지 (CUDA, MPS, CPU) 및 최적 배치 설정
- 스키마 v2 (단일 테이블 + NULL) 완벽 지원
- 중단된 작업 이어하기 지원 (이미 저장된 source_id 건너뜀)

사용법:
    # 판례 임베딩 생성
    uv run python scripts/create_lancedb_embeddings.py --type precedent

    # 법령 임베딩 생성
    uv run python scripts/create_lancedb_embeddings.py --type law

    # 전체 재생성
    uv run python scripts/create_lancedb_embeddings.py --type all --reset

    # 통계 확인
    uv run python scripts/create_lancedb_embeddings.py --stats
"""

import argparse
import asyncio
import gc
import os
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from tqdm import tqdm
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select, func

from app.core.config import settings
from app.core.database import async_session_factory
from app.tools.vectorstore.lancedb import LanceDBStore
from app.tools.vectorstore.schema_v2 import VECTOR_DIM
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

    def __str__(self) -> str:
        return f"{self.name} ({self.device}, {self.vram_gb:.1f}GB)"


@dataclass
class OptimalConfig:
    """환경별 최적 설정"""
    batch_size: int
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
        vram_gb = props.total_memory / (1024**3)
        name = props.name
        is_laptop = any(kw in name.lower() for kw in ["laptop", "mobile", "max-q"])
        return DeviceInfo(device, name, vram_gb, is_laptop)

    elif device == "mps":
        # Mac - MPS
        return DeviceInfo(device, "Apple Silicon (MPS)", 16.0, True)  # 가정값

    else:
        # CPU
        return DeviceInfo(device, "CPU", 8.0)


def get_optimal_config(device_info: DeviceInfo) -> OptimalConfig:
    """디바이스에 따른 최적 설정 반환"""
    if device_info.device == "cuda":
        if device_info.vram_gb >= 20: return OptimalConfig(128, 25)
        elif device_info.vram_gb >= 12: return OptimalConfig(100, 20)
        elif device_info.vram_gb >= 8: return OptimalConfig(64, 15)
        else: return OptimalConfig(32, 10)
    elif device_info.device == "mps":
        return OptimalConfig(40, 10)
    else:
        return OptimalConfig(16, 5)


def clear_memory():
    """메모리 정리"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================================
# 청킹 로직
# ============================================================================

@dataclass
class ChunkConfig:
    """판례 청크 설정"""
    chunk_size: int = 1250
    chunk_overlap: int = 125
    min_chunk_size: int = 100

@dataclass
class LawChunkConfig:
    """법령 청크 설정"""
    max_tokens: int = 800
    min_tokens: int = 100


def chunk_precedent_text(text: str, config: ChunkConfig) -> List[tuple]:
    """판례 텍스트 청킹"""
    if not text or len(text) < config.min_chunk_size:
        return [(0, text)] if text else []

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = min(start + config.chunk_size, len(text))
        
        # 문장 경계 자르기
        if end < len(text):
            for sep in ['. ', '.\n', '\n\n', '\n', ' ']:
                sep_pos = text.rfind(sep, start + config.min_chunk_size, end)
                if sep_pos > start:
                    end = sep_pos + len(sep)
                    break
        
        chunk = text[start:end].strip()
        if chunk and len(chunk) >= config.min_chunk_size:
            chunks.append((chunk_index, chunk))
            chunk_index += 1
        
        start = end - config.chunk_overlap
        if start >= len(text) - config.min_chunk_size:
            break
            
    return chunks


def chunk_law_content(content: str, config: LawChunkConfig) -> List[tuple]:
    """법령 텍스트 청킹 (조문 단위)"""
    if not content: return []
    
    # 간단한 구현: \n\n 단위 분리 후 병합
    articles = content.split("\n\n")
    chunks = []
    idx = 0
    
    article_no_pattern = re.compile(r"^(제\d+조(?:의\d+)?)")

    for art in articles:
        art = art.strip()
        if not art: continue
        
        match = article_no_pattern.match(art)
        article_no = match.group(1) if match else None
        
        # 길이 체크 (단순화: 1토큰 ≈ 2자)
        if len(art) > config.max_tokens * 2:
            # 너무 길면 강제 분할
            parts = [art[i:i+config.max_tokens*2] for i in range(0, len(art), int(config.max_tokens*1.8))]
            for p in parts:
                chunks.append((idx, p, article_no))
                idx += 1
        else:
            chunks.append((idx, art, article_no))
            idx += 1
            
    return chunks


# ============================================================================
# 임베딩 모델
# ============================================================================

_local_model = None
KURE_MODEL_NAME = os.getenv("LOCAL_EMBEDDING_MODEL", "nlpai-lab/KURE-v1")

def get_model(device: str):
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        print(f"[INFO] Loading model: {KURE_MODEL_NAME} on {device}")
        _local_model = SentenceTransformer(KURE_MODEL_NAME, device=device, trust_remote_code=True)
    return _local_model

def create_embeddings(texts: List[str], device: str) -> List[List[float]]:
    model = get_model(device)
    # 4000자 제한
    processed = [t[:4000] for t in texts]
    embeddings = model.encode(processed, show_progress_bar=False, device=device)
    return [e.tolist() for e in embeddings]


# ============================================================================
# 메인 프로세서
# ============================================================================

async def process_precedents(reset: bool, batch_size: int = None):
    """판례 데이터 처리"""
    store = LanceDBStore()
    device_info = get_device_info()
    config = get_optimal_config(device_info)
    bs = batch_size or config.batch_size
    
    print(f"\n=== Processing Precedents ===")
    print(f"Device: {device_info}")
    print(f"Batch Size: {bs}")

    if reset:
        store.delete_by_type("판례")
        print("[INFO] Reset complete.")

    # 기존 ID 로드
    existing_ids = store.get_existing_source_ids("판례")
    print(f"[INFO] Found {len(existing_ids)} existing precedents.")

    chunk_config = ChunkConfig()
    
    # DB 조회
    async with async_session_factory() as session:
        # 전체 개수
        total = (await session.execute(select(func.count(PrecedentDocument.id)))).scalar()
        print(f"[INFO] Total documents in DB: {total:,}")

        # 배치 처리
        offset = 0
        db_batch = 1000
        
        buffer = {
            "source_ids": [], "chunk_indices": [], "contents": [], "titles": [],
            "decision_dates": [], "court_names": [], "total_chunks_list": [],
            "case_numbers": [], "case_types": [], 
            "judgment_types": [], "judgment_statuses": [],  # 추가된 필드
            "reference_provisions_list": [], "reference_cases_list": []
        }
        
        with tqdm(total=total, desc="Processing") as pbar:
            while offset < total:
                result = await session.execute(
                    select(PrecedentDocument)
                    .order_by(PrecedentDocument.id)
                    .offset(offset).limit(db_batch)
                )
                docs = result.scalars().all()
                if not docs: break
                
                for doc in docs:
                    if doc.serial_number in existing_ids:
                        pbar.update(1)
                        continue
                        
                    # 텍스트 구성
                    text = f"[{doc.case_name or ''}]\n{doc.summary or ''}\n{doc.reasoning or ''}"
                    chunks = chunk_precedent_text(text, chunk_config)
                    
                    total_chunks = len(chunks)
                    
                    for idx, content in chunks:
                        buffer["source_ids"].append(doc.serial_number)
                        buffer["chunk_indices"].append(idx)
                        buffer["contents"].append(f"[판례] {content}")
                        buffer["titles"].append(doc.case_name or "")
                        buffer["decision_dates"].append(doc.decision_date.isoformat() if doc.decision_date else "")
                        buffer["court_names"].append(doc.court_name or "")
                        buffer["total_chunks_list"].append(total_chunks)
                        buffer["case_numbers"].append(doc.case_number or "")
                        buffer["case_types"].append(doc.case_type or "")
                        buffer["judgment_types"].append(doc.judgment_type or "")     # 추가
                        buffer["judgment_statuses"].append(doc.judgment_status or "") # 추가
                        buffer["reference_provisions_list"].append(doc.reference_provisions or "")
                        buffer["reference_cases_list"].append(doc.reference_cases or "")

                        # 버퍼 꽉 차면 저장
                        if len(buffer["source_ids"]) >= bs:
                            embeddings = create_embeddings(buffer["contents"], device_info.device)
                            store.add_precedent_documents(
                                source_ids=buffer["source_ids"],
                                chunk_indices=buffer["chunk_indices"],
                                embeddings=embeddings,
                                titles=buffer["titles"],
                                contents=buffer["contents"],
                                decision_dates=buffer["decision_dates"],
                                court_names=buffer["court_names"],
                                total_chunks_list=buffer["total_chunks_list"],
                                case_numbers=buffer["case_numbers"],
                                case_types=buffer["case_types"],
                                judgment_types=buffer["judgment_types"],        # 전달
                                judgment_statuses=buffer["judgment_statuses"],  # 전달
                                reference_provisions_list=buffer["reference_provisions_list"],
                                reference_cases_list=buffer["reference_cases_list"]
                            )
                            # 초기화
                            for k in buffer: buffer[k] = []
                            clear_memory()
                    
                    pbar.update(1)
                
                offset += db_batch
                
        # 남은 버퍼 처리
        if buffer["source_ids"]:
            embeddings = create_embeddings(buffer["contents"], device_info.device)
            store.add_precedent_documents(
                source_ids=buffer["source_ids"],
                chunk_indices=buffer["chunk_indices"],
                embeddings=embeddings,
                titles=buffer["titles"],
                contents=buffer["contents"],
                decision_dates=buffer["decision_dates"],
                court_names=buffer["court_names"],
                total_chunks_list=buffer["total_chunks_list"],
                case_numbers=buffer["case_numbers"],
                case_types=buffer["case_types"],
                judgment_types=buffer["judgment_types"],
                judgment_statuses=buffer["judgment_statuses"],
                reference_provisions_list=buffer["reference_provisions_list"],
                reference_cases_list=buffer["reference_cases_list"]
            )
            
    print("[INFO] Processing complete.")


async def process_laws(reset: bool, batch_size: int = None):
    """법령 데이터 처리"""
    store = LanceDBStore()
    device_info = get_device_info()
    config = get_optimal_config(device_info)
    bs = batch_size or config.batch_size
    
    print(f"\n=== Processing Laws ===")
    print(f"Device: {device_info}")

    if reset:
        store.delete_by_type("법령")

    existing_ids = store.get_existing_source_ids("법령")
    chunk_config = LawChunkConfig()
    
    async with async_session_factory() as session:
        total = (await session.execute(select(func.count(LawDocument.id)))).scalar()
        
        offset = 0
        db_batch = 1000
        buffer = {
            "source_ids": [], "chunk_indices": [], "contents": [], "titles": [],
            "enforcement_dates": [], "departments": [], "total_chunks_list": [],
            "promulgation_dates": [], "promulgation_nos": [], "law_types": [], "article_nos": []
        }
        
        with tqdm(total=total, desc="Processing") as pbar:
            while offset < total:
                result = await session.execute(
                    select(LawDocument).order_by(LawDocument.id).offset(offset).limit(db_batch)
                )
                docs = result.scalars().all()
                if not docs: break
                
                for doc in docs:
                    if doc.law_id in existing_ids:
                        pbar.update(1)
                        continue
                        
                    chunks = chunk_law_content(doc.content or "", chunk_config)
                    total_chunks = len(chunks)
                    
                    for idx, content, art_no in chunks:
                        buffer["source_ids"].append(doc.law_id)
                        buffer["chunk_indices"].append(idx)
                        prefix = f"[법령] {art_no} " if art_no else "[법령] "
                        buffer["contents"].append(prefix + content)
                        buffer["titles"].append(doc.law_name or "")
                        buffer["enforcement_dates"].append(doc.enforcement_date.isoformat() if doc.enforcement_date else "")
                        buffer["departments"].append(doc.ministry or "")
                        buffer["total_chunks_list"].append(total_chunks)
                        buffer["promulgation_dates"].append(doc.promulgation_date or "")
                        buffer["promulgation_nos"].append(doc.promulgation_no or "")
                        buffer["law_types"].append(doc.law_type or "")
                        buffer["article_nos"].append(art_no or "")

                        if len(buffer["source_ids"]) >= bs:
                            embeddings = create_embeddings(buffer["contents"], device_info.device)
                            store.add_law_documents(
                                source_ids=buffer["source_ids"],
                                chunk_indices=buffer["chunk_indices"],
                                embeddings=embeddings,
                                titles=buffer["titles"],
                                contents=buffer["contents"],
                                enforcement_dates=buffer["enforcement_dates"],
                                departments=buffer["departments"],
                                total_chunks_list=buffer["total_chunks_list"],
                                promulgation_dates=buffer["promulgation_dates"],
                                promulgation_nos=buffer["promulgation_nos"],
                                law_types=buffer["law_types"],
                                article_nos=buffer["article_nos"]
                            )
                            for k in buffer: buffer[k] = []
                            clear_memory()
                            
                    pbar.update(1)
                offset += db_batch

        if buffer["source_ids"]:
            embeddings = create_embeddings(buffer["contents"], device_info.device)
            store.add_law_documents(
                source_ids=buffer["source_ids"],
                chunk_indices=buffer["chunk_indices"],
                embeddings=embeddings,
                titles=buffer["titles"],
                contents=buffer["contents"],
                enforcement_dates=buffer["enforcement_dates"],
                departments=buffer["departments"],
                total_chunks_list=buffer["total_chunks_list"],
                promulgation_dates=buffer["promulgation_dates"],
                promulgation_nos=buffer["promulgation_nos"],
                law_types=buffer["law_types"],
                article_nos=buffer["article_nos"]
            )


def show_stats():
    store = LanceDBStore()
    print("\n=== LanceDB Statistics ===")
    print(f"Total Chunks: {store.count():,}")
    print(f"Law Chunks: {store.count_by_type('법령'):,}")
    print(f"Precedent Chunks: {store.count_by_type('판례'):,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["precedent", "law", "all"], default="all")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()

    if args.stats:
        show_stats()
        sys.exit(0)

    if args.type in ["precedent", "all"]:
        asyncio.run(process_precedents(args.reset, args.batch_size))
    
    if args.type in ["law", "all"]:
        asyncio.run(process_laws(args.reset, args.batch_size))
        
    show_stats()