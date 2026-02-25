"""
FTS 독립 재빌드

토크나이저 설정(userdic 등) 변경 후 tsvector만 재빌드합니다.
데이터 재적재 없이 PostgreSQL 원본에서 읽어 fts_index만 갱신합니다.
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


def run_fts_rebuild(
    config: IngestConfig,
    reset: bool = False,
) -> dict[str, int]:
    """
    PostgreSQL 원본 → MeCab → tsvector → fts_index 재빌드

    데이터 재적재 없이 인덱스만 갱신합니다.

    Args:
        config: 인제스트 설정
        reset: 기존 FTS 인덱스 삭제 후 재빌드

    Returns:
        통계 dict: total, indexed, errors
    """
    stats: dict[str, int] = {
        "total": 0,
        "indexed": 0,
        "errors": 0,
    }

    # MeCab 토크나이저 초기화 (실패 시 __init__에서 예외 발생)
    tokenizer = get_tokenizer()
    logger.info("MeCab 토크나이저 초기화 완료")

    start_time = time.time()

    with sync_session_factory() as session:
        # 리셋
        if reset:
            stmt = delete(FtsIndex).where(
                FtsIndex.data_type == config.data_type_label
            )
            # dec_* 위원회결정례는 data_type을 공유하므로
            # source_id 접두사로 해당 타입만 삭제 (다른 dec_* 보호)
            if config.name.startswith("dec_"):
                stmt = stmt.where(
                    FtsIndex.source_id.like(f"{config.name}:%")
                )
            session.execute(stmt)
            session.commit()
            logger.info(
                "%s(%s) FTS 인덱스 삭제 완료",
                config.data_type_label,
                config.name,
            )

        # 원본 건수 조회
        total = session.execute(
            select(func.count()).select_from(config.orm_class)
        ).scalar_one()
        stats["total"] = total
        logger.info("%s 원본 건수: %d", config.data_type_label, total)

        if total == 0:
            logger.info("원본 데이터가 없습니다.")
            return stats

        # ORM 클래스에서 정렬 기준 컬럼 추출
        orm_cls = config.orm_class
        orm_id_col = getattr(orm_cls, config.orm_id_attr)
        orm_pk = getattr(orm_cls, "id", orm_id_col)

        # 배치 순회
        offset = 0
        batch: list[dict[str, Any]] = []

        while offset < total:
            rows = (
                session.execute(
                    select(orm_cls)
                    .order_by(orm_pk)
                    .offset(offset)
                    .limit(BATCH_SIZE)
                )
                .scalars()
                .all()
            )

            if not rows:
                break

            for row in rows:
                row_id = getattr(row, config.orm_id_attr, "?")
                try:
                    fulltext = config.orm_fulltext_fn(row)
                    if not fulltext.strip():
                        continue

                    if len(fulltext) > _MAX_FULLTEXT_CHARS:
                        fulltext = fulltext[:_MAX_FULLTEXT_CHARS]

                    tokens = tokenizer.morphs(fulltext)
                    tsvector_str = build_tsvector_string(tokens)

                    fts_meta = config.orm_fts_metadata_fn(row)
                    fts_meta["data_type"] = config.data_type_label
                    fts_meta["content_tsvector"] = tsvector_str or None
                    batch.append(fts_meta)
                except Exception as e:
                    logger.error("FTS 생성 실패 (id=%s): %s", row_id, e)
                    stats["errors"] += 1
                    continue

                if len(batch) >= BATCH_SIZE:
                    stats["indexed"] += upsert_fts_batch(session, batch)
                    session.commit()
                    batch = []

                    logger.info(
                        "FTS 진행: %d/%d 문서",
                        min(offset + BATCH_SIZE, total),
                        total,
                    )

            offset += BATCH_SIZE

        # 잔여 배치
        if batch:
            stats["indexed"] += upsert_fts_batch(session, batch)
            session.commit()

    elapsed = time.time() - start_time
    logger.info(
        "FTS 재빌드 완료: %d건, %.1f초",
        stats["indexed"],
        elapsed,
    )

    return stats
