#!/usr/bin/env python3
"""
데이터 검증 스크립트

PostgreSQL과 ChromaDB의 데이터 무결성을 검증합니다.

사용법:
    # 전체 검증
    uv run python scripts/validate_data.py

    # PostgreSQL만 검증
    uv run python scripts/validate_data.py --pg-only

    # ChromaDB만 검증
    uv run python scripts/validate_data.py --chroma-only

    # 불일치 데이터 수정
    uv run python scripts/validate_data.py --fix
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Set

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import func, select

from app.core.database import async_session_factory
from app.tools.vectorstore import VectorStore
from app.models.law import Law
from app.models.legal_document import DocType, LegalDocument
from app.models.legal_reference import LegalReference


async def validate_postgresql() -> dict:
    """
    PostgreSQL 데이터 검증

    Returns:
        검증 결과 딕셔너리
    """
    print("\n" + "="*60)
    print("PostgreSQL Validation")
    print("="*60)

    stats = {
        "legal_documents": {},
        "laws": {},
        "legal_references": {},
        "issues": [],
    }

    async with async_session_factory() as session:
        # =========================================================
        # 1. LegalDocument 검증
        # =========================================================
        print("\n[LegalDocument Table]")

        total_docs = await session.execute(
            select(func.count(LegalDocument.id))
        )
        doc_count = total_docs.scalar() or 0
        stats["legal_documents"]["total"] = doc_count
        print(f"  Total records: {doc_count:,}")

        # doc_type + source 별 카운트
        type_source_result = await session.execute(
            select(
                LegalDocument.doc_type,
                LegalDocument.source,
                func.count(LegalDocument.id)
            ).group_by(LegalDocument.doc_type, LegalDocument.source)
        )

        print("  By type and source:")
        by_type_source = {}
        for doc_type, source, count in type_source_result.fetchall():
            key = f"{doc_type}/{source}"
            by_type_source[key] = count
            print(f"    - {key}: {count:,}")
        stats["legal_documents"]["by_type_source"] = by_type_source

        # 빈 콘텐츠 체크
        empty_content = await session.execute(
            select(func.count(LegalDocument.id)).where(
                (LegalDocument.summary.is_(None)) &
                (LegalDocument.reasoning.is_(None))
            )
        )
        empty_count = empty_content.scalar() or 0
        stats["legal_documents"]["empty_content"] = empty_count
        if empty_count > 0:
            stats["issues"].append(f"LegalDocument: {empty_count} records with empty content")
            print(f"  [WARN] Empty content: {empty_count}")

        # =========================================================
        # 2. Law 검증
        # =========================================================
        print("\n[Law Table]")

        total_laws = await session.execute(
            select(func.count(Law.id))
        )
        law_count = total_laws.scalar() or 0
        stats["laws"]["total"] = law_count
        print(f"  Total records: {law_count:,}")

        # law_type 별 카운트
        law_type_result = await session.execute(
            select(
                Law.law_type,
                func.count(Law.id)
            ).group_by(Law.law_type)
        )

        print("  By law_type:")
        by_law_type = {}
        for law_type, count in law_type_result.fetchall():
            by_law_type[law_type or "unknown"] = count
            print(f"    - {law_type or 'unknown'}: {count:,}")
        stats["laws"]["by_type"] = by_law_type

        # =========================================================
        # 3. LegalReference 검증
        # =========================================================
        print("\n[LegalReference Table]")

        total_refs = await session.execute(
            select(func.count(LegalReference.id))
        )
        ref_count = total_refs.scalar() or 0
        stats["legal_references"]["total"] = ref_count
        print(f"  Total records: {ref_count:,}")

        # ref_type 별 카운트
        ref_type_result = await session.execute(
            select(
                LegalReference.ref_type,
                func.count(LegalReference.id)
            ).group_by(LegalReference.ref_type)
        )

        print("  By ref_type:")
        by_ref_type = {}
        for ref_type, count in ref_type_result.fetchall():
            by_ref_type[ref_type] = count
            print(f"    - {ref_type}: {count:,}")
        stats["legal_references"]["by_type"] = by_ref_type

        # =========================================================
        # 총계
        # =========================================================
        grand_total = doc_count + law_count + ref_count
        stats["grand_total"] = grand_total
        print(f"\n  GRAND TOTAL: {grand_total:,}")

    return stats


def validate_chromadb() -> dict:
    """
    ChromaDB 데이터 검증

    Returns:
        검증 결과 딕셔너리
    """
    print("\n" + "="*60)
    print("ChromaDB Validation")
    print("="*60)

    stats = {
        "total_chunks": 0,
        "by_doc_type": {},
        "by_source": {},
        "unique_documents": 0,
        "issues": [],
    }

    try:
        store = VectorStore()
        stats["total_chunks"] = store.count()
        print(f"Total chunks: {stats['total_chunks']:,}")

        # doc_type 별 카운트
        print("\nBy doc_type:")
        unique_doc_ids = set()

        for doc_type in DocType:
            try:
                results = store.collection.get(
                    where={"doc_type": doc_type.value},
                    include=["metadatas"]
                )
                chunk_count = len(results["ids"]) if results["ids"] else 0
                stats["by_doc_type"][doc_type.value] = chunk_count

                # 고유 문서 ID 수집
                if results["metadatas"]:
                    doc_ids = set(m.get("doc_id") for m in results["metadatas"] if m.get("doc_id"))
                    unique_doc_ids.update(doc_ids)
                    print(f"  - {doc_type.value}: {chunk_count:,} chunks from {len(doc_ids):,} documents")
                else:
                    print(f"  - {doc_type.value}: {chunk_count:,} chunks")
            except Exception as e:
                stats["issues"].append(f"Error counting {doc_type.value}: {e}")
                print(f"  - {doc_type.value}: [ERROR] {e}")

        stats["unique_documents"] = len(unique_doc_ids)
        print(f"\nUnique documents with embeddings: {stats['unique_documents']:,}")

        # 위원회별 source 카운트
        print("\nBy source (for committee):")
        try:
            results = store.collection.get(
                where={"doc_type": "committee"},
                include=["metadatas"]
            )
            if results["metadatas"]:
                source_counts = {}
                for m in results["metadatas"]:
                    source = m.get("source", "unknown")
                    source_counts[source] = source_counts.get(source, 0) + 1
                stats["by_source"] = source_counts
                for source, count in sorted(source_counts.items()):
                    print(f"  - {source}: {count:,} chunks")
        except Exception:
            pass

        # 임베딩 차원 검증
        if stats["total_chunks"] > 0:
            sample = store.collection.get(limit=1, include=["embeddings"])
            if sample["embeddings"]:
                embedding_dim = len(sample["embeddings"][0])
                stats["embedding_dimension"] = embedding_dim
                print(f"\nEmbedding dimension: {embedding_dim}")

    except Exception as e:
        stats["issues"].append(f"ChromaDB connection error: {e}")
        print(f"\n[ERROR] ChromaDB error: {e}")

    return stats


async def validate_consistency() -> dict:
    """
    PostgreSQL과 ChromaDB 간 일관성 검증

    청크 기반 임베딩이므로 doc_id를 기준으로 비교

    Returns:
        검증 결과 딕셔너리
    """
    print("\n" + "="*60)
    print("Consistency Validation (Document Level)")
    print("="*60)

    stats = {
        "pg_only": 0,      # PostgreSQL에만 있는 문서
        "chroma_only": 0,  # ChromaDB에만 있는 문서
        "matched": 0,      # 양쪽에 있는 문서
        "issues": [],
    }

    # PostgreSQL doc IDs 수집
    pg_doc_ids: Set[int] = set()
    async with async_session_factory() as session:
        result = await session.execute(
            select(LegalDocument.id)
        )
        for (doc_id,) in result.fetchall():
            pg_doc_ids.add(doc_id)

    print(f"PostgreSQL documents: {len(pg_doc_ids):,}")

    # ChromaDB doc IDs 수집 (청크의 doc_id 메타데이터에서)
    try:
        store = VectorStore()
        chroma_result = store.collection.get(include=["metadatas"])
        chroma_doc_ids: Set[int] = set()
        if chroma_result["metadatas"]:
            for m in chroma_result["metadatas"]:
                doc_id = m.get("doc_id")
                if doc_id is not None:
                    chroma_doc_ids.add(doc_id)
        print(f"ChromaDB unique documents: {len(chroma_doc_ids):,}")
    except Exception as e:
        print(f"[ERROR] ChromaDB error: {e}")
        stats["issues"].append(f"ChromaDB error: {e}")
        return stats

    # 비교
    stats["pg_only"] = len(pg_doc_ids - chroma_doc_ids)
    stats["chroma_only"] = len(chroma_doc_ids - pg_doc_ids)
    stats["matched"] = len(pg_doc_ids & chroma_doc_ids)

    print(f"\nMatched: {stats['matched']:,}")
    print(f"PostgreSQL only (need embedding): {stats['pg_only']:,}")
    print(f"ChromaDB only (orphan): {stats['chroma_only']:,}")

    if stats["pg_only"] > 0:
        stats["issues"].append(
            f"{stats['pg_only']} documents in PostgreSQL need embeddings"
        )

    if stats["chroma_only"] > 0:
        stats["issues"].append(
            f"{stats['chroma_only']} orphan embeddings in ChromaDB"
        )

    return stats


async def fix_consistency():
    """
    PostgreSQL과 ChromaDB 간 불일치 수정

    - ChromaDB에만 있는 고아 청크 삭제
    """
    print("\n" + "="*60)
    print("Fixing Consistency Issues")
    print("="*60)

    # PostgreSQL doc IDs 수집
    pg_doc_ids: Set[int] = set()
    async with async_session_factory() as session:
        result = await session.execute(
            select(LegalDocument.id)
        )
        for (doc_id,) in result.fetchall():
            pg_doc_ids.add(doc_id)

    # ChromaDB에서 고아 청크 찾기
    store = VectorStore()
    chroma_result = store.collection.get(include=["metadatas"])

    orphan_chunk_ids = []
    if chroma_result["ids"] and chroma_result["metadatas"]:
        for chunk_id, metadata in zip(chroma_result["ids"], chroma_result["metadatas"]):
            doc_id = metadata.get("doc_id")
            if doc_id is not None and doc_id not in pg_doc_ids:
                orphan_chunk_ids.append(chunk_id)

    if orphan_chunk_ids:
        print(f"[INFO] Deleting {len(orphan_chunk_ids)} orphan chunks...")
        # 배치로 삭제
        batch_size = 1000
        for i in range(0, len(orphan_chunk_ids), batch_size):
            batch = orphan_chunk_ids[i:i + batch_size]
            store.delete_by_ids(batch)
            print(f"  Deleted {min(i + batch_size, len(orphan_chunk_ids))}/{len(orphan_chunk_ids)}")
        print("[DONE] Orphan chunks deleted")
    else:
        print("[INFO] No orphan chunks found")

    # PostgreSQL에만 있는 문서는 create_embeddings.py로 임베딩 생성 필요
    store = VectorStore()
    chroma_result = store.collection.get(include=["metadatas"])
    chroma_doc_ids: Set[int] = set()
    if chroma_result["metadatas"]:
        for m in chroma_result["metadatas"]:
            doc_id = m.get("doc_id")
            if doc_id is not None:
                chroma_doc_ids.add(doc_id)

    missing_count = len(pg_doc_ids - chroma_doc_ids)
    if missing_count > 0:
        print(f"\n[INFO] {missing_count} documents need embeddings")
        print("Run: uv run python scripts/create_embeddings.py")


async def main_async(args):
    """비동기 메인 함수"""
    all_stats = {}
    issues = []

    if not args.chroma_only:
        pg_stats = await validate_postgresql()
        all_stats["postgresql"] = pg_stats
        issues.extend(pg_stats.get("issues", []))

    if not args.pg_only:
        chroma_stats = validate_chromadb()
        all_stats["chromadb"] = chroma_stats
        issues.extend(chroma_stats.get("issues", []))

    if not args.pg_only and not args.chroma_only:
        consistency_stats = await validate_consistency()
        all_stats["consistency"] = consistency_stats
        issues.extend(consistency_stats.get("issues", []))

        if args.fix and (consistency_stats["chroma_only"] > 0):
            await fix_consistency()

    # 최종 결과
    print("\n" + "="*60)
    print("Validation Summary")
    print("="*60)

    if issues:
        print(f"\n[WARN] Found {len(issues)} issue(s):")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("\n[OK] All validations passed!")

    return all_stats


def main():
    parser = argparse.ArgumentParser(
        description="데이터 검증"
    )
    parser.add_argument(
        "--pg-only",
        action="store_true",
        help="PostgreSQL만 검증"
    )
    parser.add_argument(
        "--chroma-only",
        action="store_true",
        help="ChromaDB만 검증"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="불일치 데이터 수정 (고아 청크 삭제)"
    )

    args = parser.parse_args()

    print("="*60)
    print("Data Validation Script")
    print("="*60)

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
