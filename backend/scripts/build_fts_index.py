"""
FTS 인덱싱 스크립트 (PostgreSQL 원본 → fts_index 테이블)

PostgreSQL의 law_documents, precedent_documents 원본 테이블에서
전체 텍스트를 읽어 MeCab 토크나이징 → tsvector 생성 → fts_index 테이블 저장.
문서 단위 1행. 원문 텍스트는 저장하지 않음 (검색 인덱스 전용).

Usage:
    cd backend
    uv run python scripts/build_fts_index.py              # 전체 빌드
    uv run python scripts/build_fts_index.py --type law    # 법령만
    uv run python scripts/build_fts_index.py --type prec   # 판례만
    uv run python scripts/build_fts_index.py --verify      # 건수 검증
    uv run python scripts/build_fts_index.py --reset       # 기존 삭제 후 재빌드
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert

from app.core.config import settings
from app.core.database import sync_session_factory
from app.models.fts_index import FtsIndex
from app.models.law_document import LawDocument
from app.models.precedent_document import PrecedentDocument
from app.services.rag.tsvector_builder import build_tsvector_string
from scripts.common.logging_config import setup_logging

logger = setup_logging(__name__)

BATCH_SIZE = 1000


def _get_tokenizer() -> Any:
    """MeCab 토크나이저 인스턴스 생성 (userdic + decomposition_map 필수)."""
    import json

    from app.tools.vectorstore.mecab_tokenizer import MeCabTokenizer

    userdic_path = str(Path(settings.MECAB_USERDIC_PATH))
    decomp_path = Path(settings.MECAB_USERDIC_PATH).parent / "decomposition_map.json"

    decomposition_map: dict[str, list[str]] = {}
    if decomp_path.exists():
        with open(decomp_path, encoding="utf-8") as f:
            decomposition_map = json.load(f)
        logger.info("분해맵 로드: %d개", len(decomposition_map))

    return MeCabTokenizer(
        userdic_path=userdic_path,
        decomposition_map=decomposition_map,
    )


def _build_precedent_fulltext(row: Any) -> str:
    """판례의 전체 텍스트를 concat (tsvector 생성용)."""
    parts: list[str] = []

    if row.case_name:
        parts.append(f"[{row.case_name}]")
    if row.case_number:
        parts.append(f"사건번호: {row.case_number}")
    if row.summary:
        parts.append(row.summary)
    if row.reasoning:
        parts.append(row.reasoning)
    if row.ruling:
        parts.append(row.ruling)
    if row.claim:
        parts.append(row.claim)
    if row.full_reason:
        parts.append(row.full_reason)
    elif row.full_text:
        parts.append(row.full_text)
    if row.reference_provisions:
        parts.append(f"참조조문: {row.reference_provisions}")
    if row.reference_cases:
        parts.append(f"참조판례: {row.reference_cases}")

    return "\n".join(parts)


def _build_law_fulltext(row: Any) -> str:
    """법령의 전체 텍스트를 concat (tsvector 생성용)."""
    parts: list[str] = []

    if row.law_name:
        parts.append(f"[{row.law_name}]")
    if row.content:
        parts.append(row.content)
    if row.supplementary:
        parts.append(row.supplementary)

    return "\n".join(parts)


def _upsert_batch(session: Any, batch: list[dict[str, Any]]) -> int:
    """ON CONFLICT DO UPDATE 배치 upsert."""
    if not batch:
        return 0

    stmt = insert(FtsIndex).values(batch)
    stmt = stmt.on_conflict_do_update(
        index_elements=["source_id", "data_type"],
        set_={
            "data_type": stmt.excluded.data_type,
            "title": stmt.excluded.title,
            "date": stmt.excluded.date,
            "source_name": stmt.excluded.source_name,
            "case_number": stmt.excluded.case_number,
            "content_tsvector": stmt.excluded.content_tsvector,
        },
    )
    session.execute(stmt)
    session.commit()
    return len(batch)


def build_precedent_fts(tokenizer: Any) -> int:
    """판례 원본 → fts_index 테이블."""
    logger.info("=== 판례 FTS 인덱싱 시작 ===")

    with sync_session_factory() as session:
        total = session.execute(
            select(func.count()).select_from(PrecedentDocument)
        ).scalar_one()
        logger.info("판례 원본 건수: %d", total)

        offset = 0
        doc_count = 0
        batch: list[dict[str, Any]] = []

        while offset < total:
            rows = session.execute(
                select(PrecedentDocument)
                .order_by(PrecedentDocument.id)
                .offset(offset)
                .limit(BATCH_SIZE)
            ).scalars().all()

            if not rows:
                break

            for row in rows:
                fulltext = _build_precedent_fulltext(row)
                if not fulltext.strip():
                    continue

                date_str = (
                    row.decision_date.strftime("%Y%m%d")
                    if row.decision_date
                    else None
                )

                tokens = tokenizer.morphs(fulltext)
                tsvector_str = build_tsvector_string(tokens)

                batch.append({
                    "source_id": row.serial_number,
                    "data_type": "판례",
                    "title": row.case_name or "",
                    "date": date_str,
                    "source_name": row.court_name,
                    "case_number": row.case_number,
                    "content_tsvector": tsvector_str or None,
                })

                if len(batch) >= BATCH_SIZE:
                    doc_count += _upsert_batch(session, batch)
                    logger.info(
                        "판례 진행: %d/%d 문서",
                        min(offset + BATCH_SIZE, total),
                        total,
                    )
                    batch = []

            offset += BATCH_SIZE

        if batch:
            doc_count += _upsert_batch(session, batch)

    logger.info("=== 판례 FTS 인덱싱 완료: %d 문서 ===", doc_count)
    return doc_count


def build_law_fts(tokenizer: Any) -> int:
    """법령 원본 → fts_index 테이블."""
    logger.info("=== 법령 FTS 인덱싱 시작 ===")

    with sync_session_factory() as session:
        total = session.execute(
            select(func.count()).select_from(LawDocument)
        ).scalar_one()
        logger.info("법령 원본 건수: %d", total)

        offset = 0
        doc_count = 0
        batch: list[dict[str, Any]] = []

        while offset < total:
            rows = session.execute(
                select(LawDocument)
                .order_by(LawDocument.id)
                .offset(offset)
                .limit(BATCH_SIZE)
            ).scalars().all()

            if not rows:
                break

            for row in rows:
                fulltext = _build_law_fulltext(row)
                if not fulltext.strip():
                    continue

                date_str = (
                    row.enforcement_date.strftime("%Y%m%d")
                    if row.enforcement_date
                    else row.promulgation_date
                )

                tokens = tokenizer.morphs(fulltext)
                tsvector_str = build_tsvector_string(tokens)

                batch.append({
                    "source_id": row.law_id,
                    "data_type": "법령",
                    "title": row.law_name or "",
                    "date": date_str,
                    "source_name": row.ministry,
                    "case_number": None,
                    "content_tsvector": tsvector_str or None,
                })

                if len(batch) >= BATCH_SIZE:
                    doc_count += _upsert_batch(session, batch)
                    logger.info(
                        "법령 진행: %d/%d 문서",
                        min(offset + BATCH_SIZE, total),
                        total,
                    )
                    batch = []

            offset += BATCH_SIZE

        if batch:
            doc_count += _upsert_batch(session, batch)

    logger.info("=== 법령 FTS 인덱싱 완료: %d 문서 ===", doc_count)
    return doc_count


def verify_fts() -> None:
    """fts_index 테이블 건수 및 상태 검증."""
    with sync_session_factory() as session:
        total = session.execute(
            select(func.count()).select_from(FtsIndex)
        ).scalar_one()

        by_type = session.execute(
            select(FtsIndex.data_type, func.count())
            .group_by(FtsIndex.data_type)
        ).all()

        has_tsvector = session.execute(
            select(func.count()).select_from(FtsIndex)
            .where(FtsIndex.content_tsvector.isnot(None))
        ).scalar_one()

    logger.info("=== fts_index 테이블 검증 ===")
    logger.info("총 문서 수: %d", total)
    for dtype, cnt in by_type:
        logger.info("  %s: %d 문서", dtype, cnt)
    logger.info(
        "tsvector 보유: %d/%d (%.1f%%)",
        has_tsvector,
        total,
        has_tsvector / total * 100 if total else 0,
    )


def reset_fts(data_type: str | None = None) -> None:
    """fts_index 테이블 데이터 삭제."""
    with sync_session_factory() as session:
        if data_type:
            session.execute(
                delete(FtsIndex).where(FtsIndex.data_type == data_type)
            )
            logger.info("%s FTS 인덱스 삭제 완료", data_type)
        else:
            session.execute(delete(FtsIndex))
            logger.info("전체 FTS 인덱스 삭제 완료")
        session.commit()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FTS 인덱싱 (PostgreSQL 원본 → fts_index 테이블)"
    )
    parser.add_argument(
        "--type",
        choices=["all", "law", "prec"],
        default="all",
        help="빌드 대상 (default: all)",
    )
    parser.add_argument("--verify", action="store_true", help="건수 검증만 수행")
    parser.add_argument("--reset", action="store_true", help="기존 삭제 후 재빌드")
    args = parser.parse_args()

    if args.verify:
        verify_fts()
        return

    tokenizer = _get_tokenizer()
    logger.info("MeCab 토크나이저 초기화 완료 (available=%s)", tokenizer.is_available)

    start = time.time()

    if args.reset:
        data_type_map = {"law": "법령", "prec": "판례"}
        reset_fts(data_type_map.get(args.type))

    total_docs = 0

    if args.type in ("all", "prec"):
        total_docs += build_precedent_fts(tokenizer)

    if args.type in ("all", "law"):
        total_docs += build_law_fts(tokenizer)

    elapsed = time.time() - start
    logger.info("=== 전체 완료: %d 문서, %.1f초 ===", total_docs, elapsed)

    verify_fts()


if __name__ == "__main__":
    main()
