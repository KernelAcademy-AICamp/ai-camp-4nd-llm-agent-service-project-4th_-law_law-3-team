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
    uv run python scripts/load_legal_data.py --type committee
    uv run python scripts/load_legal_data.py --type law
    uv run python scripts/load_legal_data.py --type treaty
    uv run python scripts/load_legal_data.py --type admin_rule
    uv run python scripts/load_legal_data.py --type law_term

    # 배치 크기 조정
    uv run python scripts/load_legal_data.py --batch-size 500

    # 기존 데이터 삭제 후 재로드
    uv run python scripts/load_legal_data.py --reset

    # 현재 DB 통계만 출력
    uv run python scripts/load_legal_data.py --stats
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Generator

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.database import async_session_factory
from app.models.law import Law
from app.models.legal_document import COMMITTEE_SOURCES, DocType, LegalDocument
from app.models.legal_reference import LegalReference, RefType

# 데이터 파일 경로
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "law_data"

# 문서 유형별 파일 매핑
DOCUMENT_FILES = {
    DocType.PRECEDENT: {
        "source": "precedents",
        "files": [
            DATA_DIR / "precedents_full.json",
            DATA_DIR / "precedents_full-1.json",
            DATA_DIR / "precedents_full-2.json",
            DATA_DIR / "precedents_full-3.json",
            DATA_DIR / "precedents_full-4.json",
            DATA_DIR / "precedents_full-5.json",
        ],
    },
    DocType.CONSTITUTIONAL: {
        "source": "constitutional",
        "files": [DATA_DIR / "constitutional_full.json"],
    },
    DocType.ADMINISTRATION: {
        "source": "administration",
        "files": [DATA_DIR / "administration_full.json"],
    },
    DocType.LEGISLATION: {
        "source": "legislation",
        "files": [DATA_DIR / "legislation_full.json"],
    },
}

# 위원회별 파일 매핑
COMMITTEE_FILES = {
    "ftc": DATA_DIR / "ftc-full.json",        # 공정거래위원회
    "nhrck": DATA_DIR / "nhrck-full.json",    # 국가인권위원회
    "acrc": DATA_DIR / "acr-full.json",       # 국민권익위원회
    "ppc": DATA_DIR / "ppc-full.json",        # 개인정보보호위원회
    "kcc": DATA_DIR / "kcc-full.json",        # 방송통신위원회
    "fsc": DATA_DIR / "fsc-full.json",        # 금융위원회
    "ecc": DATA_DIR / "ecc-full.json",        # 중앙선거관리위원회
    "eiac": DATA_DIR / "eiac-full.json",      # 환경분쟁조정위원회
    "sfc": DATA_DIR / "sfc-full.json",        # 해양환경관리공단
    "iaciac": DATA_DIR / "iaciac-full.json",  # 산업재해보상보험심사위원회
    "oclt": DATA_DIR / "oclt-full.json",      # 원자력안전위원회
}

# 참조 데이터 파일 매핑
REFERENCE_FILES = {
    RefType.TREATY: DATA_DIR / "treaty-full.json",
    RefType.ADMIN_RULE: DATA_DIR / "administrative_rules_full.json",
    RefType.LAW_TERM: DATA_DIR / "lawterms_full.json",
}

# 법령 파일
LAW_FILE = DATA_DIR / "law.json"


def load_json_streaming(file_path: Path) -> Generator[dict, None, None]:
    """
    JSON 파일을 스트리밍 방식으로 로드

    메모리 효율을 위해 한 번에 전체를 로드하지 않고
    레코드 단위로 yield
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

        if isinstance(data, list):
            for item in data:
                yield item
        elif isinstance(data, dict):
            # law.json 형식: {"metadata": {...}, "items": [...]}
            if "items" in data:
                for item in data.get("items", []):
                    yield item
            # lawyers 데이터 형식
            elif "lawyers" in data:
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
        elif isinstance(data, dict):
            if "items" in data:
                return len(data.get("items", []))
            elif "lawyers" in data:
                return len(data.get("lawyers", []))
        return 1


async def get_existing_serial_numbers(
    session: AsyncSession,
    model: type,
    filter_field: str,
    filter_value: str,
    serial_field: str = "serial_number",
) -> set:
    """이미 DB에 있는 serial_number 조회"""
    filter_col = getattr(model, filter_field)
    serial_col = getattr(model, serial_field)
    result = await session.execute(
        select(serial_col).where(filter_col == filter_value)
    )
    return set(row[0] for row in result.fetchall())


async def get_existing_law_ids(session: AsyncSession) -> set:
    """이미 DB에 있는 law_id 조회"""
    result = await session.execute(select(Law.law_id))
    return set(row[0] for row in result.fetchall())


# ============================================================================
# 법률 문서 (LegalDocument) 로드
# ============================================================================

async def load_documents_for_type(
    doc_type: DocType,
    batch_size: int = 1000,
    reset: bool = False,
) -> dict:
    """특정 유형의 법률 문서 로드"""
    config = DOCUMENT_FILES.get(doc_type)
    if not config:
        print(f"[ERROR] Unknown doc_type: {doc_type}")
        return {"error": f"Unknown doc_type: {doc_type}"}

    source = config["source"]
    files = config["files"]

    # 팩토리 메서드 선택
    factory_methods = {
        DocType.PRECEDENT: lambda data: LegalDocument.from_precedent(data, source),
        DocType.CONSTITUTIONAL: lambda data: LegalDocument.from_constitutional(data, source),
        DocType.ADMINISTRATION: lambda data: LegalDocument.from_administration(data, source),
        DocType.LEGISLATION: lambda data: LegalDocument.from_legislation(data, source),
    }
    factory = factory_methods.get(doc_type)

    stats = {
        "doc_type": doc_type.value,
        "source": source,
        "files_processed": 0,
        "total_records": 0,
        "inserted": 0,
        "skipped": 0,
        "errors": 0,
    }

    async with async_session_factory() as session:
        if reset:
            print(f"[INFO] Deleting existing {doc_type.value} data (source={source})...")
            await session.execute(
                delete(LegalDocument).where(
                    LegalDocument.doc_type == doc_type.value,
                    LegalDocument.source == source,
                )
            )
            await session.commit()
            existing_serials = set()
        else:
            existing_serials = await get_existing_serial_numbers(
                session, LegalDocument, "source", source
            )
            print(f"[INFO] Found {len(existing_serials)} existing records for {source}")

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

                    if doc.serial_number in existing_serials:
                        stats["skipped"] += 1
                        continue

                    batch.append(doc)
                    existing_serials.add(doc.serial_number)

                    if len(batch) >= batch_size:
                        session.add_all(batch)
                        await session.commit()
                        stats["inserted"] += len(batch)
                        batch = []

                        pct = processed / total_in_file * 100
                        print(f"  [PROGRESS] {processed:,}/{total_in_file:,} ({pct:.1f}%) - Inserted: {stats['inserted']:,}")

                except Exception as e:
                    stats["errors"] += 1
                    if stats["errors"] <= 5:
                        print(f"  [ERROR] Record error: {e}")

            if batch:
                session.add_all(batch)
                await session.commit()
                stats["inserted"] += len(batch)

            stats["files_processed"] += 1
            print(f"  [DONE] {file_path.name} - Inserted: {stats['inserted']:,}")

    return stats


async def load_committee_documents(
    batch_size: int = 1000,
    reset: bool = False,
) -> dict:
    """모든 위원회 결정문 로드"""
    all_stats = {
        "doc_type": "committee",
        "sources": {},
        "total_inserted": 0,
        "total_errors": 0,
    }

    for source, file_path in COMMITTEE_FILES.items():
        if not file_path.exists():
            print(f"[WARN] File not found: {file_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Loading committee: {source} ({COMMITTEE_SOURCES.get(source, source)})")
        print('='*60)

        stats = {
            "source": source,
            "committee_name": COMMITTEE_SOURCES.get(source, source),
            "total_records": 0,
            "inserted": 0,
            "skipped": 0,
            "errors": 0,
        }

        async with async_session_factory() as session:
            if reset:
                print(f"[INFO] Deleting existing {source} data...")
                await session.execute(
                    delete(LegalDocument).where(
                        LegalDocument.doc_type == DocType.COMMITTEE.value,
                        LegalDocument.source == source,
                    )
                )
                await session.commit()
                existing_serials = set()
            else:
                existing_serials = await get_existing_serial_numbers(
                    session, LegalDocument, "source", source
                )
                print(f"[INFO] Found {len(existing_serials)} existing records")

            total_in_file = count_records(file_path)
            print(f"[INFO] Total records in file: {total_in_file:,}")

            batch = []
            processed = 0

            for record in load_json_streaming(file_path):
                processed += 1
                stats["total_records"] += 1

                try:
                    doc = LegalDocument.from_committee(record, source)

                    if doc.serial_number in existing_serials:
                        stats["skipped"] += 1
                        continue

                    batch.append(doc)
                    existing_serials.add(doc.serial_number)

                    if len(batch) >= batch_size:
                        session.add_all(batch)
                        await session.commit()
                        stats["inserted"] += len(batch)
                        batch = []

                        pct = processed / total_in_file * 100
                        print(f"  [PROGRESS] {processed:,}/{total_in_file:,} ({pct:.1f}%)")

                except Exception as e:
                    stats["errors"] += 1
                    if stats["errors"] <= 5:
                        print(f"  [ERROR] Record error: {e}")

            if batch:
                session.add_all(batch)
                await session.commit()
                stats["inserted"] += len(batch)

        all_stats["sources"][source] = stats
        all_stats["total_inserted"] += stats["inserted"]
        all_stats["total_errors"] += stats["errors"]
        print(f"[DONE] {source} - Inserted: {stats['inserted']:,}")

    return all_stats


# ============================================================================
# 법령 (Law) 로드
# ============================================================================

async def load_laws(
    batch_size: int = 1000,
    reset: bool = False,
) -> dict:
    """법령 데이터 로드"""
    stats = {
        "type": "law",
        "total_records": 0,
        "inserted": 0,
        "skipped": 0,
        "errors": 0,
    }

    if not LAW_FILE.exists():
        print(f"[WARN] File not found: {LAW_FILE}")
        return stats

    print(f"\n[INFO] Processing: {LAW_FILE.name}")
    total_in_file = count_records(LAW_FILE)
    print(f"[INFO] Total records in file: {total_in_file:,}")

    async with async_session_factory() as session:
        if reset:
            print("[INFO] Deleting existing law data...")
            await session.execute(delete(Law))
            await session.commit()
            existing_ids = set()
        else:
            existing_ids = await get_existing_law_ids(session)
            print(f"[INFO] Found {len(existing_ids)} existing records")

        batch = []
        processed = 0

        for record in load_json_streaming(LAW_FILE):
            processed += 1
            stats["total_records"] += 1

            try:
                law = Law.from_law_data(record)

                if law.law_id in existing_ids:
                    stats["skipped"] += 1
                    continue

                batch.append(law)
                existing_ids.add(law.law_id)

                if len(batch) >= batch_size:
                    session.add_all(batch)
                    await session.commit()
                    stats["inserted"] += len(batch)
                    batch = []

                    pct = processed / total_in_file * 100
                    print(f"  [PROGRESS] {processed:,}/{total_in_file:,} ({pct:.1f}%)")

            except Exception as e:
                stats["errors"] += 1
                if stats["errors"] <= 5:
                    print(f"  [ERROR] Record error: {e}")

        if batch:
            session.add_all(batch)
            await session.commit()
            stats["inserted"] += len(batch)

    print(f"[DONE] law - Inserted: {stats['inserted']:,}")
    return stats


# ============================================================================
# 참조 데이터 (LegalReference) 로드
# ============================================================================

async def load_references_for_type(
    ref_type: RefType,
    batch_size: int = 1000,
    reset: bool = False,
) -> dict:
    """특정 유형의 참조 데이터 로드"""
    file_path = REFERENCE_FILES.get(ref_type)
    if not file_path or not file_path.exists():
        print(f"[WARN] File not found for {ref_type.value}")
        return {"error": f"File not found for {ref_type.value}"}

    # 팩토리 메서드 선택
    factory_methods = {
        RefType.TREATY: LegalReference.from_treaty,
        RefType.ADMIN_RULE: LegalReference.from_admin_rule,
        RefType.LAW_TERM: LegalReference.from_law_term,
    }
    factory = factory_methods.get(ref_type)

    stats = {
        "ref_type": ref_type.value,
        "total_records": 0,
        "inserted": 0,
        "skipped": 0,
        "errors": 0,
    }

    print(f"\n[INFO] Processing: {file_path.name}")
    total_in_file = count_records(file_path)
    print(f"[INFO] Total records in file: {total_in_file:,}")

    async with async_session_factory() as session:
        if reset:
            print(f"[INFO] Deleting existing {ref_type.value} data...")
            await session.execute(
                delete(LegalReference).where(
                    LegalReference.ref_type == ref_type.value
                )
            )
            await session.commit()
            existing_serials = set()
        else:
            existing_serials = await get_existing_serial_numbers(
                session, LegalReference, "ref_type", ref_type.value
            )
            print(f"[INFO] Found {len(existing_serials)} existing records")

        batch = []
        processed = 0

        for record in load_json_streaming(file_path):
            processed += 1
            stats["total_records"] += 1

            try:
                ref = factory(record)

                if ref.serial_number in existing_serials:
                    stats["skipped"] += 1
                    continue

                batch.append(ref)
                existing_serials.add(ref.serial_number)

                if len(batch) >= batch_size:
                    session.add_all(batch)
                    await session.commit()
                    stats["inserted"] += len(batch)
                    batch = []

                    pct = processed / total_in_file * 100
                    print(f"  [PROGRESS] {processed:,}/{total_in_file:,} ({pct:.1f}%)")

            except Exception as e:
                stats["errors"] += 1
                if stats["errors"] <= 5:
                    print(f"  [ERROR] Record error: {e}")

        if batch:
            session.add_all(batch)
            await session.commit()
            stats["inserted"] += len(batch)

    print(f"[DONE] {ref_type.value} - Inserted: {stats['inserted']:,}")
    return stats


# ============================================================================
# 전체 로드 및 통계
# ============================================================================

async def load_all_data(batch_size: int = 1000, reset: bool = False) -> dict:
    """모든 유형의 데이터 로드"""
    all_stats = {}
    start_time = datetime.now()

    # 1. 기존 법률 문서 (판례, 헌재, 행정심판, 법령해석)
    for doc_type in [DocType.PRECEDENT, DocType.CONSTITUTIONAL,
                     DocType.ADMINISTRATION, DocType.LEGISLATION]:
        print(f"\n{'='*60}")
        print(f"Loading {doc_type.value}...")
        print('='*60)
        stats = await load_documents_for_type(doc_type, batch_size, reset)
        all_stats[doc_type.value] = stats

    # 2. 위원회 결정문
    print(f"\n{'='*60}")
    print("Loading committee decisions...")
    print('='*60)
    stats = await load_committee_documents(batch_size, reset)
    all_stats["committee"] = stats

    # 3. 법령
    print(f"\n{'='*60}")
    print("Loading laws...")
    print('='*60)
    stats = await load_laws(batch_size, reset)
    all_stats["law"] = stats

    # 4. 참조 데이터
    for ref_type in RefType:
        print(f"\n{'='*60}")
        print(f"Loading {ref_type.value}...")
        print('='*60)
        stats = await load_references_for_type(ref_type, batch_size, reset)
        all_stats[ref_type.value] = stats

    elapsed = datetime.now() - start_time
    all_stats["elapsed_time"] = str(elapsed)

    return all_stats


async def show_stats():
    """현재 DB 통계 출력"""
    async with async_session_factory() as session:
        print("\n" + "="*60)
        print("Database Statistics")
        print("="*60)

        # LegalDocument 통계
        total_docs = await session.execute(
            select(func.count(LegalDocument.id))
        )
        total_doc_count = total_docs.scalar()

        type_counts = await session.execute(
            select(
                LegalDocument.doc_type,
                LegalDocument.source,
                func.count(LegalDocument.id)
            ).group_by(LegalDocument.doc_type, LegalDocument.source)
        )

        print(f"\n[LegalDocument] Total: {total_doc_count:,}")
        print("By type and source:")
        for doc_type, source, count in type_counts.fetchall():
            print(f"  - {doc_type}/{source}: {count:,}")

        # Law 통계
        total_laws = await session.execute(
            select(func.count(Law.id))
        )
        law_count = total_laws.scalar()
        print(f"\n[Law] Total: {law_count:,}")

        # LegalReference 통계
        total_refs = await session.execute(
            select(func.count(LegalReference.id))
        )
        ref_count = total_refs.scalar()

        ref_type_counts = await session.execute(
            select(
                LegalReference.ref_type,
                func.count(LegalReference.id)
            ).group_by(LegalReference.ref_type)
        )

        print(f"\n[LegalReference] Total: {ref_count:,}")
        print("By type:")
        for ref_type, count in ref_type_counts.fetchall():
            print(f"  - {ref_type}: {count:,}")

        # 총계
        grand_total = total_doc_count + law_count + ref_count
        print(f"\n{'='*60}")
        print(f"GRAND TOTAL: {grand_total:,}")
        print("="*60)


async def main_async(args):
    """비동기 메인 함수"""
    if args.stats:
        await show_stats()
        return

    if args.type == "all":
        stats = await load_all_data(args.batch_size, args.reset)
    elif args.type == "committee":
        stats = await load_committee_documents(args.batch_size, args.reset)
    elif args.type == "law":
        stats = await load_laws(args.batch_size, args.reset)
    elif args.type in ["treaty", "admin_rule", "law_term"]:
        ref_type = RefType(args.type)
        stats = await load_references_for_type(ref_type, args.batch_size, args.reset)
    else:
        doc_type = DocType(args.type)
        stats = await load_documents_for_type(doc_type, args.batch_size, args.reset)

    # 결과 출력
    print("\n" + "="*60)
    print("Load Results")
    print("="*60)
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    # 최종 통계
    await show_stats()


def main():
    parser = argparse.ArgumentParser(
        description="법률 데이터 PostgreSQL 로드"
    )
    parser.add_argument(
        "--type",
        choices=[
            "precedent", "constitutional", "administration", "legislation",
            "committee", "law", "treaty", "admin_rule", "law_term", "all"
        ],
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
    print("Legal Data Loader")
    print("="*60)
    print(f"Data directory: {DATA_DIR}")
    print(f"Batch size: {args.batch_size}")
    print(f"Reset mode: {args.reset}")

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
