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

from sqlalchemy import select, func

from app.common.database import async_session_factory
from app.common.vectorstore import VectorStore
from app.models.legal_document import LegalDocument, DocType


async def validate_postgresql() -> dict:
    """
    PostgreSQL 데이터 검증

    Returns:
        검증 결과 딕셔너리
    """
    print("\n" + "="*50)
    print("PostgreSQL Validation")
    print("="*50)

    stats = {
        "total_records": 0,
        "by_type": {},
        "null_serial_numbers": 0,
        "empty_content": 0,
        "missing_dates": 0,
        "issues": [],
    }

    async with async_session_factory() as session:
        # 전체 카운트
        total_result = await session.execute(
            select(func.count(LegalDocument.id))
        )
        stats["total_records"] = total_result.scalar() or 0
        print(f"Total records: {stats['total_records']:,}")

        # 타입별 카운트
        type_result = await session.execute(
            select(
                LegalDocument.doc_type,
                func.count(LegalDocument.id)
            ).group_by(LegalDocument.doc_type)
        )

        print("\nBy type:")
        for doc_type, count in type_result.fetchall():
            stats["by_type"][doc_type] = count
            print(f"  - {doc_type}: {count:,}")

        # NULL serial_number 체크
        null_serial = await session.execute(
            select(func.count(LegalDocument.id)).where(
                LegalDocument.serial_number == None
            )
        )
        stats["null_serial_numbers"] = null_serial.scalar() or 0

        if stats["null_serial_numbers"] > 0:
            stats["issues"].append(
                f"NULL serial_number: {stats['null_serial_numbers']} records"
            )
            print(f"\n[WARN] NULL serial_number: {stats['null_serial_numbers']}")

        # 빈 콘텐츠 체크 (summary와 reasoning 둘 다 없는 경우)
        empty_content = await session.execute(
            select(func.count(LegalDocument.id)).where(
                (LegalDocument.summary == None) &
                (LegalDocument.reasoning == None)
            )
        )
        stats["empty_content"] = empty_content.scalar() or 0

        if stats["empty_content"] > 0:
            stats["issues"].append(
                f"Empty content (no summary/reasoning): {stats['empty_content']} records"
            )
            print(f"[WARN] Empty content: {stats['empty_content']}")

        # 날짜 누락 체크
        missing_date = await session.execute(
            select(func.count(LegalDocument.id)).where(
                LegalDocument.decision_date == None
            )
        )
        stats["missing_dates"] = missing_date.scalar() or 0
        print(f"[INFO] Missing decision_date: {stats['missing_dates']}")

        # 중복 체크
        duplicate_result = await session.execute(
            select(
                LegalDocument.doc_type,
                LegalDocument.serial_number,
                func.count(LegalDocument.id).label("cnt")
            ).group_by(
                LegalDocument.doc_type,
                LegalDocument.serial_number
            ).having(func.count(LegalDocument.id) > 1)
        )

        duplicates = duplicate_result.fetchall()
        if duplicates:
            stats["duplicates"] = len(duplicates)
            stats["issues"].append(f"Duplicate records: {len(duplicates)}")
            print(f"\n[WARN] Duplicate records found: {len(duplicates)}")
            for dt, sn, cnt in duplicates[:5]:
                print(f"  - {dt}/{sn}: {cnt} copies")

    return stats


def validate_chromadb() -> dict:
    """
    ChromaDB 데이터 검증

    Returns:
        검증 결과 딕셔너리
    """
    print("\n" + "="*50)
    print("ChromaDB Validation")
    print("="*50)

    stats = {
        "total_embeddings": 0,
        "by_type": {},
        "issues": [],
    }

    try:
        store = VectorStore()
        stats["total_embeddings"] = store.count()
        print(f"Total embeddings: {stats['total_embeddings']:,}")

        # 타입별 카운트
        print("\nBy type:")
        for doc_type in DocType:
            try:
                results = store.collection.get(
                    where={"doc_type": doc_type.value},
                    include=[]
                )
                count = len(results["ids"]) if results["ids"] else 0
                stats["by_type"][doc_type.value] = count
                print(f"  - {doc_type.value}: {count:,}")
            except Exception as e:
                stats["issues"].append(f"Error counting {doc_type.value}: {e}")
                print(f"  - {doc_type.value}: [ERROR] {e}")

        # 샘플 데이터 검증
        if stats["total_embeddings"] > 0:
            sample = store.collection.get(limit=1, include=["embeddings"])
            if sample["embeddings"]:
                embedding_dim = len(sample["embeddings"][0])
                print(f"\nEmbedding dimension: {embedding_dim}")

                # text-embedding-3-small은 1536 차원
                if embedding_dim != 1536:
                    stats["issues"].append(
                        f"Unexpected embedding dimension: {embedding_dim} (expected 1536)"
                    )

    except Exception as e:
        stats["issues"].append(f"ChromaDB connection error: {e}")
        print(f"\n[ERROR] ChromaDB error: {e}")

    return stats


async def validate_consistency() -> dict:
    """
    PostgreSQL과 ChromaDB 간 일관성 검증

    Returns:
        검증 결과 딕셔너리
    """
    print("\n" + "="*50)
    print("Consistency Validation")
    print("="*50)

    stats = {
        "pg_only": 0,      # PostgreSQL에만 있는 문서
        "chroma_only": 0,  # ChromaDB에만 있는 문서
        "matched": 0,      # 양쪽에 있는 문서
        "issues": [],
    }

    # PostgreSQL IDs 수집
    pg_ids: Set[str] = set()
    async with async_session_factory() as session:
        result = await session.execute(
            select(LegalDocument.doc_type, LegalDocument.serial_number)
        )
        for doc_type, serial_number in result.fetchall():
            pg_ids.add(f"{doc_type}_{serial_number}")

    print(f"PostgreSQL documents: {len(pg_ids):,}")

    # ChromaDB IDs 수집
    try:
        store = VectorStore()
        chroma_result = store.collection.get(include=[])
        chroma_ids = set(chroma_result["ids"]) if chroma_result["ids"] else set()
        print(f"ChromaDB embeddings: {len(chroma_ids):,}")
    except Exception as e:
        print(f"[ERROR] ChromaDB error: {e}")
        stats["issues"].append(f"ChromaDB error: {e}")
        return stats

    # 비교
    stats["pg_only"] = len(pg_ids - chroma_ids)
    stats["chroma_only"] = len(chroma_ids - pg_ids)
    stats["matched"] = len(pg_ids & chroma_ids)

    print(f"\nMatched: {stats['matched']:,}")
    print(f"PostgreSQL only: {stats['pg_only']:,}")
    print(f"ChromaDB only: {stats['chroma_only']:,}")

    if stats["pg_only"] > 0:
        stats["issues"].append(
            f"{stats['pg_only']} documents in PostgreSQL but not in ChromaDB"
        )
        # 샘플 출력
        missing_in_chroma = list(pg_ids - chroma_ids)[:5]
        print("\nSample documents missing in ChromaDB:")
        for doc_id in missing_in_chroma:
            print(f"  - {doc_id}")

    if stats["chroma_only"] > 0:
        stats["issues"].append(
            f"{stats['chroma_only']} embeddings in ChromaDB but not in PostgreSQL"
        )
        # 샘플 출력
        orphan_embeddings = list(chroma_ids - pg_ids)[:5]
        print("\nSample orphan embeddings in ChromaDB:")
        for doc_id in orphan_embeddings:
            print(f"  - {doc_id}")

    return stats


async def fix_consistency():
    """
    PostgreSQL과 ChromaDB 간 불일치 수정

    - ChromaDB에만 있는 고아 임베딩 삭제
    """
    print("\n" + "="*50)
    print("Fixing Consistency Issues")
    print("="*50)

    # PostgreSQL IDs 수집
    pg_ids: Set[str] = set()
    async with async_session_factory() as session:
        result = await session.execute(
            select(LegalDocument.doc_type, LegalDocument.serial_number)
        )
        for doc_type, serial_number in result.fetchall():
            pg_ids.add(f"{doc_type}_{serial_number}")

    # ChromaDB IDs 수집
    store = VectorStore()
    chroma_result = store.collection.get(include=[])
    chroma_ids = set(chroma_result["ids"]) if chroma_result["ids"] else set()

    # 고아 임베딩 삭제
    orphan_ids = list(chroma_ids - pg_ids)

    if orphan_ids:
        print(f"[INFO] Deleting {len(orphan_ids)} orphan embeddings...")
        # 배치로 삭제
        batch_size = 1000
        for i in range(0, len(orphan_ids), batch_size):
            batch = orphan_ids[i:i + batch_size]
            store.delete_by_ids(batch)
            print(f"  Deleted {min(i + batch_size, len(orphan_ids))}/{len(orphan_ids)}")
        print("[DONE] Orphan embeddings deleted")
    else:
        print("[INFO] No orphan embeddings found")

    # PostgreSQL에만 있는 문서는 create_embeddings.py로 임베딩 생성 필요
    missing_count = len(pg_ids - chroma_ids)
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

        if args.fix and consistency_stats["chroma_only"] > 0:
            await fix_consistency()

    # 최종 결과
    print("\n" + "="*50)
    print("Validation Summary")
    print("="*50)

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
        help="불일치 데이터 수정 (고아 임베딩 삭제)"
    )

    args = parser.parse_args()

    print("="*60)
    print("Data Validation Script")
    print("="*60)

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
