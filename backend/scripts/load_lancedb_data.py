#!/usr/bin/env python3
"""
LanceDB용 데이터 로드 스크립트

JSON 파일에서 PostgreSQL로 법령/판례 데이터를 로드합니다.

사용법:
    # 법령 데이터 로드
    uv run python scripts/load_lancedb_data.py --type law

    # 판례 데이터 로드
    uv run python scripts/load_lancedb_data.py --type precedent

    # 전체 로드 (법령 + 판례)
    uv run python scripts/load_lancedb_data.py --type all

    # 기존 데이터 삭제 후 재로드
    uv run python scripts/load_lancedb_data.py --type all --reset

    # 커스텀 파일 경로
    uv run python scripts/load_lancedb_data.py --type law --source ../data/custom_law.json

데이터 소스:
    - data/law_cleaned.json (법령 5,841건)
    - data/precedents_cleaned.json (판례 65,107건)
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select, func, delete
from sqlalchemy.dialects.postgresql import insert

from app.core.database import async_session_factory
from app.models.law_document import LawDocument
from app.models.precedent_document import PrecedentDocument


# ============================================================================
# 기본 경로
# ============================================================================

DEFAULT_LAW_PATH = Path(__file__).parent.parent.parent / "data" / "law_cleaned.json"
DEFAULT_PRECEDENT_PATH = Path(__file__).parent.parent.parent / "data" / "precedents_cleaned.json"


# ============================================================================
# 법령 데이터 로드
# ============================================================================

def load_law_json(source_path: Path) -> List[Dict[str, Any]]:
    """법령 JSON 파일 로드"""
    if not source_path.exists():
        raise FileNotFoundError(f"법령 파일을 찾을 수 없습니다: {source_path}")

    print(f"[INFO] Loading law data from: {source_path}")
    with open(source_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("items", [])
    print(f"[INFO] Loaded {len(items):,} law items")
    return items


async def load_laws_to_db(
    source_path: Path,
    batch_size: int = 500,
    reset: bool = False,
) -> Dict[str, int]:
    """법령 데이터를 PostgreSQL에 로드"""
    stats = {
        "total": 0,
        "inserted": 0,
        "skipped": 0,
        "errors": 0,
    }

    items = load_law_json(source_path)
    stats["total"] = len(items)

    if not items:
        return stats

    async with async_session_factory() as session:
        # 기존 데이터 삭제
        if reset:
            print("[INFO] Resetting law_documents table...")
            await session.execute(delete(LawDocument))
            await session.commit()
            print("[INFO] Table cleared")

        # 이미 존재하는 law_id 조회
        result = await session.execute(select(LawDocument.law_id))
        existing_ids = {row[0] for row in result.fetchall()}
        print(f"[INFO] Found {len(existing_ids):,} existing records")

        # 중복 ID 추적 (DB + 현재 배치)
        seen_ids = set(existing_ids)

        # 배치 처리
        batch = []
        for idx, item in enumerate(items):
            law_id = item.get("law_id", "")

            if not law_id:
                stats["errors"] += 1
                continue

            if law_id in seen_ids:
                stats["skipped"] += 1
                continue

            # 중복 방지: 현재 처리 중인 ID 추가
            seen_ids.add(law_id)

            try:
                doc = LawDocument.from_json(item)
                batch.append(doc)
            except Exception as e:
                stats["errors"] += 1
                print(f"  [ERROR] Failed to parse item {idx}: {e}")
                continue

            # 배치 저장
            if len(batch) >= batch_size:
                session.add_all(batch)
                await session.commit()
                stats["inserted"] += len(batch)
                batch = []

                # 진행률 출력
                progress = idx + 1
                pct = progress / len(items) * 100
                print(f"  [PROGRESS] {progress:,}/{len(items):,} ({pct:.1f}%)")

        # 남은 배치 저장
        if batch:
            session.add_all(batch)
            await session.commit()
            stats["inserted"] += len(batch)

    return stats


# ============================================================================
# 판례 데이터 로드
# ============================================================================

def load_precedent_json(source_path: Path) -> List[Dict[str, Any]]:
    """판례 JSON 파일 로드"""
    if not source_path.exists():
        raise FileNotFoundError(f"판례 파일을 찾을 수 없습니다: {source_path}")

    print(f"[INFO] Loading precedent data from: {source_path}")
    with open(source_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # precedents_cleaned.json은 리스트 형태
    if isinstance(data, list):
        items = data
    else:
        items = data.get("items", [])

    print(f"[INFO] Loaded {len(items):,} precedent items")
    return items


async def load_precedents_to_db(
    source_path: Path,
    batch_size: int = 1000,
    reset: bool = False,
) -> Dict[str, int]:
    """판례 데이터를 PostgreSQL에 로드"""
    stats = {
        "total": 0,
        "inserted": 0,
        "skipped": 0,
        "errors": 0,
    }

    items = load_precedent_json(source_path)
    stats["total"] = len(items)

    if not items:
        return stats

    async with async_session_factory() as session:
        # 기존 데이터 삭제
        if reset:
            print("[INFO] Resetting precedent_documents table...")
            await session.execute(delete(PrecedentDocument))
            await session.commit()
            print("[INFO] Table cleared")

        # 이미 존재하는 serial_number 조회
        result = await session.execute(select(PrecedentDocument.serial_number))
        existing_ids = {row[0] for row in result.fetchall()}
        print(f"[INFO] Found {len(existing_ids):,} existing records")

        # 중복 ID 추적 (DB + 현재 배치)
        seen_ids = set(existing_ids)

        # 배치 처리
        batch = []
        for idx, item in enumerate(items):
            serial_number = item.get("판례정보일련번호", "")

            if not serial_number:
                stats["errors"] += 1
                continue

            if serial_number in seen_ids:
                stats["skipped"] += 1
                continue

            # 중복 방지: 현재 처리 중인 ID 추가
            seen_ids.add(serial_number)

            try:
                doc = PrecedentDocument.from_json(item)
                batch.append(doc)
            except Exception as e:
                stats["errors"] += 1
                print(f"  [ERROR] Failed to parse item {idx}: {e}")
                continue

            # 배치 저장
            if len(batch) >= batch_size:
                session.add_all(batch)
                await session.commit()
                stats["inserted"] += len(batch)
                batch = []

                # 진행률 출력
                progress = idx + 1
                pct = progress / len(items) * 100
                print(f"  [PROGRESS] {progress:,}/{len(items):,} ({pct:.1f}%)")

        # 남은 배치 저장
        if batch:
            session.add_all(batch)
            await session.commit()
            stats["inserted"] += len(batch)

    return stats


# ============================================================================
# 통계 조회
# ============================================================================

async def show_stats() -> None:
    """테이블 통계 출력"""
    async with async_session_factory() as session:
        # law_documents 카운트
        law_count = await session.execute(select(func.count(LawDocument.id)))
        law_total = law_count.scalar() or 0

        # precedent_documents 카운트
        precedent_count = await session.execute(select(func.count(PrecedentDocument.id)))
        precedent_total = precedent_count.scalar() or 0

    print("\n" + "=" * 60)
    print("PostgreSQL Table Statistics (LanceDB용)")
    print("=" * 60)
    print(f"law_documents: {law_total:,} records")
    print(f"precedent_documents: {precedent_total:,} records")
    print(f"Total: {law_total + precedent_total:,} records")


# ============================================================================
# 메인
# ============================================================================

async def async_main(args):
    """비동기 메인 함수"""
    start_time = datetime.now()
    results = {}

    # 법령 로드
    if args.type in ("law", "all"):
        law_path = Path(args.source) if args.source and args.type == "law" else DEFAULT_LAW_PATH
        print("\n" + "=" * 60)
        print("Loading 법령 data...")
        print("=" * 60)
        results["law"] = await load_laws_to_db(
            source_path=law_path,
            batch_size=args.batch_size,
            reset=args.reset,
        )

    # 판례 로드
    if args.type in ("precedent", "all"):
        precedent_path = Path(args.source) if args.source and args.type == "precedent" else DEFAULT_PRECEDENT_PATH
        print("\n" + "=" * 60)
        print("Loading 판례 data...")
        print("=" * 60)
        results["precedent"] = await load_precedents_to_db(
            source_path=precedent_path,
            batch_size=args.batch_size,
            reset=args.reset,
        )

    elapsed = datetime.now() - start_time

    # 결과 출력
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    for data_type, stats in results.items():
        print(f"\n[{data_type}]")
        print(f"  Total: {stats['total']:,}")
        print(f"  Inserted: {stats['inserted']:,}")
        print(f"  Skipped: {stats['skipped']:,}")
        print(f"  Errors: {stats['errors']:,}")

    print(f"\nElapsed time: {elapsed}")

    # 최종 통계
    await show_stats()


def main():
    parser = argparse.ArgumentParser(description="LanceDB용 PostgreSQL 데이터 로드")
    parser.add_argument(
        "--type",
        choices=["law", "precedent", "all"],
        default="all",
        help="로드할 데이터 유형 (기본: all)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="JSON 파일 경로 (미지정 시 기본 경로 사용)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="배치 크기 (기본: 500)"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="기존 데이터 삭제 후 재로드"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="통계만 출력"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("LanceDB PostgreSQL Data Loader")
    print("=" * 60)
    print(f"Data type: {args.type}")
    print(f"Batch size: {args.batch_size}")
    print(f"Reset: {args.reset}")

    if args.stats:
        asyncio.run(show_stats())
        return

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
