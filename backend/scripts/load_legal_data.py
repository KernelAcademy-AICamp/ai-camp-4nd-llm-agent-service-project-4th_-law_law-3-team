#!/usr/bin/env python3
"""
법률 데이터 PostgreSQL 로드 스크립트

JSON 파일에서 법률 데이터를 읽어 PostgreSQL에 저장합니다.

사용법:
    # 모든 데이터 로드
    uv run python scripts/load_legal_data.py

    # 특정 유형만 로드
    uv run python scripts/load_legal_data.py --type precedent
    uv run python scripts/load_legal_data.py --type constitutional
    uv run python scripts/load_legal_data.py --type administration
    uv run python scripts/load_legal_data.py --type legislation

    # 배치 크기 조정
    uv run python scripts/load_legal_data.py --batch-size 500

    # 기존 데이터 삭제 후 재로드
    uv run python scripts/load_legal_data.py --reset
"""

import argparse
import asyncio
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Generator, Optional

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select, delete, func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.database import async_session_factory, engine
from app.models.legal_document import LegalDocument, DocType


# 데이터 파일 경로
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "law_data"

# 파일 매핑
DATA_FILES = {
    DocType.PRECEDENT: [
        DATA_DIR / "precedents_full.json",
        DATA_DIR / "precedents_full-1.json",
        DATA_DIR / "precedents_full-2.json",
        DATA_DIR / "precedents_full-3.json",
        DATA_DIR / "precedents_full-4.json",
        DATA_DIR / "precedents_full-5.json",
    ],
    DocType.CONSTITUTIONAL: [DATA_DIR / "constitutional_full.json"],
    DocType.ADMINISTRATION: [DATA_DIR / "administation_full.json"],
    DocType.LEGISLATION: [DATA_DIR / "legislation_full.json"],
}

# 팩토리 메서드 매핑
FACTORY_METHODS = {
    DocType.PRECEDENT: LegalDocument.from_precedent,
    DocType.CONSTITUTIONAL: LegalDocument.from_constitutional,
    DocType.ADMINISTRATION: LegalDocument.from_administration,
    DocType.LEGISLATION: LegalDocument.from_legislation,
}


def load_json_streaming(file_path: Path) -> Generator[dict, None, None]:
    """
    JSON 파일을 스트리밍 방식으로 로드

    메모리 효율을 위해 한 번에 전체를 로드하지 않고
    레코드 단위로 yield
    """
    with open(file_path, "r", encoding="utf-8") as f:
        # JSON 배열 시작
        data = json.load(f)

        if isinstance(data, list):
            for item in data:
                yield item
        elif isinstance(data, dict) and "lawyers" in data:
            # lawyers 데이터 형식
            for item in data.get("lawyers", []):
                yield item
        else:
            yield data


def count_records(file_path: Path) -> int:
    """파일의 레코드 수 카운트"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, list):
            return len(data)
        return 1


async def get_existing_serial_numbers(
    session: AsyncSession,
    doc_type: str
) -> set:
    """이미 DB에 있는 serial_number 조회"""
    result = await session.execute(
        select(LegalDocument.serial_number).where(
            LegalDocument.doc_type == doc_type
        )
    )
    return set(row[0] for row in result.fetchall())


async def load_data_for_type(
    doc_type: DocType,
    batch_size: int = 1000,
    reset: bool = False,
) -> dict:
    """
    특정 유형의 데이터 로드

    Args:
        doc_type: 문서 유형
        batch_size: 배치 크기
        reset: True면 기존 데이터 삭제 후 로드

    Returns:
        로드 결과 통계
    """
    files = DATA_FILES.get(doc_type, [])
    factory = FACTORY_METHODS.get(doc_type)

    if not factory:
        print(f"[ERROR] Unknown doc_type: {doc_type}")
        return {"error": f"Unknown doc_type: {doc_type}"}

    stats = {
        "doc_type": doc_type.value,
        "files_processed": 0,
        "total_records": 0,
        "inserted": 0,
        "skipped": 0,
        "errors": 0,
    }

    async with async_session_factory() as session:
        # 기존 데이터 삭제 (reset 옵션)
        if reset:
            print(f"[INFO] Deleting existing {doc_type.value} data...")
            await session.execute(
                delete(LegalDocument).where(
                    LegalDocument.doc_type == doc_type.value
                )
            )
            await session.commit()
            existing_serials = set()
        else:
            # 기존 serial_number 조회 (중복 방지)
            existing_serials = await get_existing_serial_numbers(
                session, doc_type.value
            )
            print(f"[INFO] Found {len(existing_serials)} existing records for {doc_type.value}")

        for file_path in files:
            if not file_path.exists():
                print(f"[WARN] File not found: {file_path}")
                continue

            print(f"\n[INFO] Processing: {file_path.name}")
            total_in_file = count_records(file_path)
            print(f"[INFO] Total records in file: {total_in_file:,}")

            batch = []
            processed = 0

            for record in load_json_streaming(file_path):
                processed += 1
                stats["total_records"] += 1

                try:
                    doc = factory(record)

                    # 중복 체크
                    if doc.serial_number in existing_serials:
                        stats["skipped"] += 1
                        continue

                    batch.append(doc)
                    existing_serials.add(doc.serial_number)

                    # 배치 처리
                    if len(batch) >= batch_size:
                        session.add_all(batch)
                        await session.commit()
                        stats["inserted"] += len(batch)
                        batch = []

                        # 진행률 출력
                        pct = processed / total_in_file * 100
                        print(f"  [PROGRESS] {processed:,}/{total_in_file:,} ({pct:.1f}%) - Inserted: {stats['inserted']:,}")

                except Exception as e:
                    stats["errors"] += 1
                    if stats["errors"] <= 5:  # 처음 5개 에러만 출력
                        print(f"  [ERROR] Record error: {e}")

            # 남은 배치 처리
            if batch:
                session.add_all(batch)
                await session.commit()
                stats["inserted"] += len(batch)

            stats["files_processed"] += 1
            print(f"  [DONE] {file_path.name} - Inserted: {stats['inserted']:,}")

    return stats


async def load_all_data(batch_size: int = 1000, reset: bool = False) -> dict:
    """모든 유형의 데이터 로드"""
    all_stats = {}
    start_time = datetime.now()

    for doc_type in DocType:
        print(f"\n{'='*60}")
        print(f"Loading {doc_type.value}...")
        print('='*60)

        stats = await load_data_for_type(doc_type, batch_size, reset)
        all_stats[doc_type.value] = stats

    elapsed = datetime.now() - start_time
    all_stats["elapsed_time"] = str(elapsed)

    return all_stats


async def show_stats():
    """현재 DB 통계 출력"""
    async with async_session_factory() as session:
        # 전체 카운트
        total = await session.execute(
            select(func.count(LegalDocument.id))
        )
        total_count = total.scalar()

        # 유형별 카운트
        type_counts = await session.execute(
            select(
                LegalDocument.doc_type,
                func.count(LegalDocument.id)
            ).group_by(LegalDocument.doc_type)
        )

        print("\n" + "="*50)
        print("📊 Database Statistics")
        print("="*50)
        print(f"Total records: {total_count:,}")
        print("\nBy type:")
        for doc_type, count in type_counts.fetchall():
            print(f"  - {doc_type}: {count:,}")


def main():
    parser = argparse.ArgumentParser(
        description="법률 데이터 PostgreSQL 로드"
    )
    parser.add_argument(
        "--type",
        choices=["precedent", "constitutional", "administration", "legislation", "all"],
        default="all",
        help="로드할 데이터 유형 (기본: all)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="배치 크기 (기본: 1000)"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="기존 데이터 삭제 후 재로드"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="현재 DB 통계만 출력"
    )

    args = parser.parse_args()

    print("="*60)
    print("🏛️  법률 데이터 로드 스크립트")
    print("="*60)
    print(f"Data directory: {DATA_DIR}")
    print(f"Batch size: {args.batch_size}")
    print(f"Reset mode: {args.reset}")

    if args.stats:
        asyncio.run(show_stats())
        return

    if args.type == "all":
        stats = asyncio.run(load_all_data(args.batch_size, args.reset))
    else:
        doc_type = DocType(args.type)
        stats = asyncio.run(load_data_for_type(doc_type, args.batch_size, args.reset))

    # 결과 출력
    print("\n" + "="*60)
    print("📊 Load Results")
    print("="*60)
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    # 최종 통계
    asyncio.run(show_stats())


if __name__ == "__main__":
    main()
