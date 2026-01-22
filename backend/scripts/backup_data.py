#!/usr/bin/env python3
"""
데이터 백업 스크립트

PostgreSQL과 ChromaDB 데이터를 JSON 형식으로 백업합니다.
"""

import asyncio
import json
import shutil
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select
from app.common.database import async_session_factory
from app.models.legal_document import LegalDocument


BACKUP_DIR = Path(__file__).parent.parent.parent / "data" / "backup_20260120"


async def backup_postgresql():
    """PostgreSQL 데이터를 JSON으로 백업"""
    print("\n[1/2] Backing up PostgreSQL data...")

    backup_file = BACKUP_DIR / "legal_documents_backup.json"

    async with async_session_factory() as session:
        result = await session.execute(
            select(
                LegalDocument.id,
                LegalDocument.doc_type,
                LegalDocument.serial_number,
                LegalDocument.source,
                LegalDocument.case_name,
                LegalDocument.case_number,
            )
        )

        records = []
        for row in result.fetchall():
            records.append({
                "id": row.id,
                "doc_type": row.doc_type,
                "serial_number": row.serial_number,
                "source": row.source,
                "case_name": row.case_name,
                "case_number": row.case_number,
            })

        print(f"  Found {len(records):,} records")

        with open(backup_file, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        print(f"  Saved to: {backup_file}")
        print(f"  File size: {backup_file.stat().st_size / 1024 / 1024:.1f} MB")


def backup_chromadb():
    """ChromaDB 데이터를 백업 (디렉토리 복사)"""
    print("\n[2/2] Backing up ChromaDB data...")

    # ChromaDB 디렉토리 찾기
    chroma_dir = Path(__file__).parent.parent.parent / "data" / "chroma"

    if not chroma_dir.exists():
        print(f"  ChromaDB directory not found: {chroma_dir}")
        return

    backup_chroma_dir = BACKUP_DIR / "chroma_backup"

    print(f"  Source: {chroma_dir}")
    print(f"  Destination: {backup_chroma_dir}")

    # 기존 백업 삭제
    if backup_chroma_dir.exists():
        shutil.rmtree(backup_chroma_dir)

    # 복사
    shutil.copytree(chroma_dir, backup_chroma_dir)

    # 크기 계산
    total_size = sum(f.stat().st_size for f in backup_chroma_dir.rglob("*") if f.is_file())
    print(f"  Backup size: {total_size / 1024 / 1024:.1f} MB")
    print("  Done!")


async def main():
    print("="*60)
    print("Data Backup Script")
    print("="*60)
    print(f"Backup directory: {BACKUP_DIR}")

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    await backup_postgresql()
    backup_chromadb()

    print("\n" + "="*60)
    print("Backup completed!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
