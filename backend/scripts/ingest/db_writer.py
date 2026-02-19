"""
PostgreSQL 적재 + FTS 동시 생성

1패스로 JSON → PostgreSQL ORM 인스턴스 + fts_index tsvector를 동시 처리합니다.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

# 백엔드 app 모듈 import를 위한 경로 추가
_backend_root = Path(__file__).parent.parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert

from app.core.database import sync_session_factory
from app.models.fts_index import FtsIndex
from app.services.rag.tsvector_builder import build_tsvector_string
from scripts.common.json_loader import (  # noqa: E402
    load_json_directory,
    load_json_file,
)
from scripts.ingest.config import IngestConfig
from scripts.ingest.shared import get_tokenizer, upsert_fts_batch

logger = logging.getLogger(__name__)

BATCH_SIZE = 1000

# PostgreSQL tsvector 최대 1MB (1,048,575 bytes)
# 한글 1자 ≈ 3 bytes UTF-8, 안전 마진 고려하여 300,000자 제한
_MAX_FULLTEXT_CHARS = 300_000

# auto-increment PK와 타임스탬프는 upsert SET 대상에서 제외
_UPSERT_EXCLUDE_COLUMNS = {"id", "created_at"}


def _upsert_orm_batch(
    session: Any,
    config: IngestConfig,
    orm_batch: list[Any],
) -> None:
    """ORM 인스턴스 배치를 ON CONFLICT DO UPDATE로 upsert."""
    if not orm_batch:
        return

    orm_table = config.orm_class.__table__
    value_columns = [
        c.name for c in orm_table.columns
        if c.name not in _UPSERT_EXCLUDE_COLUMNS
    ]

    values = [
        {col: getattr(instance, col, None) for col in value_columns}
        for instance in orm_batch
    ]

    stmt = insert(orm_table).values(values)
    update_columns = {
        col: stmt.excluded[col]
        for col in value_columns
        if col != config.orm_id_attr
    }

    stmt = stmt.on_conflict_do_update(
        index_elements=[config.orm_id_attr],
        set_=update_columns,
    )
    session.execute(stmt)


def _load_json(source_path: Path) -> list[dict[str, Any]]:
    """JSON 파일 또는 디렉토리 로드 (common.json_loader 위임).

    디렉토리이면 각 item에 __source_group__ 키를 추가합니다.
    """
    if not source_path.exists():
        raise FileNotFoundError(f"소스를 찾을 수 없습니다: {source_path}")

    if source_path.is_dir():
        logger.info("디렉토리 로드: %s", source_path)
        items = load_json_directory(source_path, group_key="__source_group__")
        logger.info("총 %d건 로드 (디렉토리)", len(items))
        return items

    logger.info("JSON 로드: %s", source_path)
    items = load_json_file(source_path)
    logger.info("총 %d건 로드", len(items))
    return items


def run_db_ingest(
    config: IngestConfig,
    source_path: Path | None = None,
    reset: bool = False,
    batch_size: int = BATCH_SIZE,
) -> dict[str, int]:
    """
    JSON → PostgreSQL ORM + fts_index 동시 적재

    Args:
        config: 인제스트 설정
        source_path: JSON 소스 경로 (None이면 config.source_path)
        reset: 기존 데이터 삭제 후 재실행
        batch_size: 배치 크기

    Returns:
        통계 dict: total, db_inserted, db_skipped, fts_inserted, errors
    """
    source = source_path or config.source_path
    items = _load_json(source)

    stats: dict[str, int] = {
        "total": len(items),
        "db_inserted": 0,
        "db_skipped": 0,
        "fts_inserted": 0,
        "errors": 0,
    }

    if not items:
        return stats

    # MeCab 토크나이저 초기화 (필수 — 실패 시 __init__에서 예외 발생)
    tokenizer = get_tokenizer()

    start_time = time.time()

    with sync_session_factory() as session:
        # 리셋
        if reset:
            logger.info("기존 데이터 삭제 중...")
            # fts_index에서 해당 data_type 삭제
            session.execute(
                delete(FtsIndex).where(
                    FtsIndex.data_type == config.data_type_label
                )
            )
            # ORM 테이블 전체 삭제
            orm_table = config.orm_class.__table__
            session.execute(delete(orm_table))
            session.commit()
            logger.info("기존 데이터 삭제 완료")
            existing_ids: set[str] = set()
        else:
            # 기존 ID 조회 (config.orm_id_attr 사용)
            id_col = getattr(config.orm_class, config.orm_id_attr)
            result = session.execute(select(id_col))
            existing_ids = {str(row[0]) for row in result.fetchall()}
            logger.info("기존 문서: %d건", len(existing_ids))

        # 중복 추적
        seen_ids: set[str] = set(existing_ids)

        # 배치 축적
        orm_batch: list[Any] = []
        fts_batch: list[dict[str, Any]] = []

        for idx, item in enumerate(items):
            doc_id = str(item.get(config.id_field, ""))

            if not doc_id:
                stats["errors"] += 1
                continue

            if doc_id in seen_ids:
                stats["db_skipped"] += 1
                continue

            seen_ids.add(doc_id)

            # 1. ORM 인스턴스 생성
            try:
                orm_instance = config.orm_factory_fn(item)
                orm_batch.append(orm_instance)
            except Exception as e:
                logger.error("ORM 생성 실패 (id=%s): %s", doc_id, e)
                stats["errors"] += 1
                continue

            # 2. FTS 데이터 생성
            try:
                fulltext = config.fulltext_fn(item)
                if fulltext.strip():
                    # PostgreSQL tsvector 1MB 제한 방지
                    if len(fulltext) > _MAX_FULLTEXT_CHARS:
                        fulltext = fulltext[:_MAX_FULLTEXT_CHARS]
                    tokens = tokenizer.morphs(fulltext)
                    tsvector_str = build_tsvector_string(tokens)

                    fts_meta = config.fts_metadata_fn(item)
                    fts_meta["content_tsvector"] = tsvector_str or None
                    fts_batch.append(fts_meta)
            except Exception as e:
                logger.error("FTS 생성 실패 (id=%s): %s", doc_id, e)
                # FTS 실패해도 DB 적재는 계속 진행

            # 배치 커밋
            if len(orm_batch) >= batch_size:
                _upsert_orm_batch(session, config, orm_batch)
                upsert_fts_batch(session, fts_batch)
                session.commit()

                stats["db_inserted"] += len(orm_batch)
                stats["fts_inserted"] += len(fts_batch)

                orm_batch = []
                fts_batch = []

                # 진행률
                progress = idx + 1
                pct = progress / len(items) * 100
                elapsed = time.time() - start_time
                speed = stats["db_inserted"] / elapsed if elapsed > 0 else 0
                logger.info(
                    "진행: %d/%d (%.1f%%) | DB: %d | FTS: %d | %.1f docs/s",
                    progress,
                    len(items),
                    pct,
                    stats["db_inserted"],
                    stats["fts_inserted"],
                    speed,
                )

        # 잔여 배치 커밋
        if orm_batch:
            _upsert_orm_batch(session, config, orm_batch)
            upsert_fts_batch(session, fts_batch)
            session.commit()
            stats["db_inserted"] += len(orm_batch)
            stats["fts_inserted"] += len(fts_batch)

    elapsed = time.time() - start_time
    logger.info(
        "DB+FTS 적재 완료: DB %d건, FTS %d건, %.1f초",
        stats["db_inserted"],
        stats["fts_inserted"],
        elapsed,
    )

    return stats


def verify_db(config: IngestConfig) -> dict[str, Any]:
    """PostgreSQL + fts_index 데이터 검증"""
    result: dict[str, Any] = {}

    with sync_session_factory() as session:
        # ORM 테이블 건수
        orm_count = session.execute(
            select(func.count()).select_from(config.orm_class)
        ).scalar_one()
        result["orm_count"] = orm_count

        # fts_index 건수 (해당 data_type)
        fts_count = session.execute(
            select(func.count())
            .select_from(FtsIndex)
            .where(FtsIndex.data_type == config.data_type_label)
        ).scalar_one()
        result["fts_count"] = fts_count

        # tsvector 보유 비율
        fts_with_tsvector = session.execute(
            select(func.count())
            .select_from(FtsIndex)
            .where(FtsIndex.data_type == config.data_type_label)
            .where(FtsIndex.content_tsvector.isnot(None))
        ).scalar_one()
        result["fts_with_tsvector"] = fts_with_tsvector

    logger.info("=== %s DB 검증 ===", config.data_type_label)
    logger.info("  ORM 테이블: %d건", result["orm_count"])
    logger.info("  FTS 인덱스: %d건", result["fts_count"])
    logger.info(
        "  tsvector 보유: %d/%d (%.1f%%)",
        result["fts_with_tsvector"],
        result["fts_count"],
        (result["fts_with_tsvector"] / result["fts_count"] * 100)
        if result["fts_count"]
        else 0,
    )

    return result
