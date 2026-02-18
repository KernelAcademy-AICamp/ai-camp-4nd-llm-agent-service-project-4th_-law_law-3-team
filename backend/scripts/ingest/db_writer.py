"""
PostgreSQL 적재 + FTS 동시 생성

1패스로 JSON → PostgreSQL ORM 인스턴스 + fts_index tsvector를 동시 처리합니다.
"""

from __future__ import annotations

import json
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


def _extract_group_from_filename(filename: str) -> str:
    """파일명에서 그룹명(위원회/부처/기관명) 추출

    패턴: prefix_그룹명_v숫자.json
    예: dec_comm_공정거래위원회_v1.json → 공정거래위원회
        intp_min_고용노동부_v1.json → 고용노동부
        sadm_case_조세심판원_v1.json → 조세심판원
    """
    import re

    m = re.search(r"(?:dec_comm|intp_min|sadm_case)_(.+?)_v\d+\.json", filename)
    return m.group(1) if m else Path(filename).stem


def _load_json(source_path: Path) -> list[dict[str, Any]]:
    """JSON 파일 또는 디렉토리 로드

    단일 파일이면 그대로 로드.
    디렉토리이면 내부 .json 파일을 모두 합산하고
    각 item에 __source_group__ 키를 추가 (파일명에서 추출).
    """
    if not source_path.exists():
        raise FileNotFoundError(f"소스를 찾을 수 없습니다: {source_path}")

    if source_path.is_dir():
        return _load_json_directory(source_path)

    logger.info("JSON 로드: %s", source_path)
    with open(source_path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        items = data
    else:
        items = data.get("items", [])

    logger.info("총 %d건 로드", len(items))
    return items


def _load_json_directory(dir_path: Path) -> list[dict[str, Any]]:
    """디렉토리 내 모든 .json 파일을 합산 로드

    각 item에 __source_group__ 키를 추가하여
    어느 파일(위원회/부처)에서 왔는지 식별 가능하게 함.
    JSON 파싱 실패 파일은 경고 후 건너뜀.
    """
    json_files = sorted(dir_path.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"디렉토리에 .json 파일이 없습니다: {dir_path}")

    logger.info("디렉토리 로드: %s (%d개 파일)", dir_path, len(json_files))

    all_items: list[dict[str, Any]] = []
    for json_file in json_files:
        group_name = _extract_group_from_filename(json_file.name)

        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.warning("JSON 파싱 실패, 건너뜀: %s (%s)", json_file.name, e)
            continue

        if isinstance(data, list):
            items = data
        else:
            items = data.get("items", [])

        for item in items:
            item["__source_group__"] = group_name

        logger.info("  %s: %d건 (%s)", json_file.name, len(items), group_name)
        all_items.extend(items)

    logger.info("총 %d건 로드 (디렉토리)", len(all_items))
    return all_items


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
