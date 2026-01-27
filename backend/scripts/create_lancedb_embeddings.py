#!/usr/bin/env python3
"""
LanceDB 임베딩 생성 스크립트 (v2 스키마)

PostgreSQL에서 판례/법령 데이터를 읽어 LanceDB에 저장합니다.
스키마 v2 (단일 테이블 + NULL) 방식을 사용합니다.

사용법:
    # 판례 임베딩 생성
    uv run python scripts/create_lancedb_embeddings.py --type precedent

    # 법령 임베딩 생성 (JSON 파일에서)
    uv run python scripts/create_lancedb_embeddings.py --type law --source ./data/laws.json

    # 전체 재생성
    uv run python scripts/create_lancedb_embeddings.py --type precedent --reset

    # 통계 확인
    uv run python scripts/create_lancedb_embeddings.py --stats
"""

import argparse
import asyncio
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select, func

from app.core.config import settings
from app.common.database import async_session_factory
from app.common.vectorstore.lancedb import LanceDBStore
from app.common.vectorstore.schema_v2 import VECTOR_DIM
from app.models.legal_document import LegalDocument, DocType


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
    텍스트를 청크로 분할

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

        chunk_text = text[start:end].strip()
        if chunk_text and len(chunk_text) >= config.min_chunk_size:
            chunks.append((chunk_index, chunk_text))
            chunk_index += 1

        start = end - config.chunk_overlap
        if start >= len(text) - config.min_chunk_size:
            break

    return chunks


# ============================================================================
# 임베딩 모델 (KURE-v1)
# ============================================================================

KURE_MODEL_NAME = "nlpai-lab/KURE-v1"
_local_model = None


def get_local_model():
    """KURE-v1 모델 로드 (1024 차원)"""
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer

        cache_dir = Path(__file__).parent.parent.parent / "data" / "models"
        cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Loading KURE model: {KURE_MODEL_NAME}")
        _local_model = SentenceTransformer(
            KURE_MODEL_NAME,
            cache_folder=str(cache_dir),
            trust_remote_code=True,
        )
        dim = _local_model.get_sentence_embedding_dimension()
        print(f"[INFO] Model loaded. Dimension: {dim}")

        if dim != VECTOR_DIM:
            raise ValueError(f"Model dimension ({dim}) != Schema dimension ({VECTOR_DIM})")

    return _local_model


def create_embeddings(texts: List[str]) -> List[List[float]]:
    """임베딩 생성 (KURE-v1)"""
    model = get_local_model()
    # KURE는 한국어에 최적화되어 있어 긴 텍스트도 잘 처리
    processed = [t.strip()[:4000] if t else "(내용 없음)" for t in texts]
    embeddings = model.encode(processed, show_progress_bar=False)
    return [emb.tolist() for emb in embeddings]


# ============================================================================
# 판례 임베딩
# ============================================================================

async def get_precedent_count() -> int:
    """판례 문서 수"""
    async with async_session_factory() as session:
        query = select(func.count(LegalDocument.id)).where(
            LegalDocument.doc_type == DocType.PRECEDENT.value
        )
        result = await session.execute(query)
        return result.scalar() or 0


async def get_precedents(offset: int, limit: int) -> List[LegalDocument]:
    """판례 문서 조회"""
    async with async_session_factory() as session:
        query = (
            select(LegalDocument)
            .where(LegalDocument.doc_type == DocType.PRECEDENT.value)
            .order_by(LegalDocument.id)
            .offset(offset)
            .limit(limit)
        )
        result = await session.execute(query)
        return list(result.scalars().all())


def build_precedent_embedding_text(doc: LegalDocument) -> str:
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
    batch_size: int = 100,
    chunk_config: ChunkConfig = None,
    reset: bool = False,
) -> dict:
    """판례 임베딩 생성"""
    if chunk_config is None:
        chunk_config = ChunkConfig()

    stats = {
        "total_docs": 0,
        "processed_docs": 0,
        "total_chunks": 0,
        "skipped": 0,
        "errors": 0,
    }

    if reset:
        print("[INFO] Resetting precedent data...")
        # data_type='판례'인 것만 삭제 (전체 테이블 삭제 아님)
        try:
            existing = store.count_by_type("판례")
            if existing > 0:
                # LanceDB는 조건부 삭제 지원
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
    print(f"[INFO] Total precedents: {total_count:,}")

    if total_count == 0:
        return stats

    # 모델 사전 로드
    get_local_model()

    # 배치 처리
    offset = 0
    db_batch_size = 500

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
            source_id = str(doc.id)

            # 이미 처리된 문서 스킵
            if source_id in existing_source_ids:
                stats["skipped"] += 1
                continue

            # 임베딩 텍스트 생성
            text = build_precedent_embedding_text(doc)
            if not text or len(text) < chunk_config.min_chunk_size:
                stats["skipped"] += 1
                continue

            # 청킹
            chunks = chunk_text(text, chunk_config)
            if not chunks:
                stats["skipped"] += 1
                continue

            total_chunks = len(chunks)
            stats["processed_docs"] += 1

            for chunk_idx, chunk_content in chunks:
                # prefix 추가
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
                batch_data["reference_provisions_list"].append(doc.reference_articles or "")
                batch_data["reference_cases_list"].append(doc.reference_cases or "")

            # 배치 크기 도달 시 저장
            if len(batch_data["source_ids"]) >= batch_size:
                try:
                    # 임베딩 생성
                    batch_data["embeddings"] = create_embeddings(batch_data["contents"])

                    # LanceDB에 저장
                    store.add_precedent_documents(**batch_data)
                    stats["total_chunks"] += len(batch_data["source_ids"])

                except Exception as e:
                    stats["errors"] += len(batch_data["source_ids"])
                    print(f"  [ERROR] Batch error: {e}")

                # 버퍼 초기화
                for key in batch_data:
                    batch_data[key] = []

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
            batch_data["embeddings"] = create_embeddings(batch_data["contents"])
            store.add_precedent_documents(**batch_data)
            stats["total_chunks"] += len(batch_data["source_ids"])
        except Exception as e:
            stats["errors"] += len(batch_data["source_ids"])
            print(f"  [ERROR] Final batch error: {e}")

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
        "--batch-size",
        type=int,
        default=100,
        help="배치 크기 (기본: 100)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1250,
        help="청크 크기 (기본: 1250)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=125,
        help="청크 오버랩 (기본: 125, 10%%)"
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
    print(f"Batch size: {args.batch_size}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Chunk overlap: {args.chunk_overlap}")

    if args.stats:
        show_stats()
        return

    chunk_config = ChunkConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    store = LanceDBStore()
    start_time = datetime.now()

    if args.type == "precedent":
        print("\n" + "=" * 60)
        print("Creating embeddings for 판례...")
        print("=" * 60)

        stats = asyncio.run(create_precedent_embeddings(
            store=store,
            batch_size=args.batch_size,
            chunk_config=chunk_config,
            reset=args.reset,
        ))

    elif args.type == "law":
        print("\n[INFO] 법령 임베딩은 별도 데이터 소스가 필요합니다.")
        print("[INFO] --source 옵션으로 법령 JSON 파일 경로를 지정하세요.")
        stats = {}

    else:
        # all
        print("\n" + "=" * 60)
        print("Creating embeddings for 판례...")
        print("=" * 60)

        stats = asyncio.run(create_precedent_embeddings(
            store=store,
            batch_size=args.batch_size,
            chunk_config=chunk_config,
            reset=args.reset,
        ))

    elapsed = datetime.now() - start_time

    # 결과 출력
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    import json
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\nElapsed time: {elapsed}")

    # 최종 통계
    show_stats()


if __name__ == "__main__":
    main()
