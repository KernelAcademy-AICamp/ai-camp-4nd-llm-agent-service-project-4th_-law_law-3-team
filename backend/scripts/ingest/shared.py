"""
인제스트 공통 유틸리티

db_writer / fts_builder 양쪽에서 사용하는 함수를 한 곳에서 관리합니다.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from sqlalchemy.dialects.postgresql import insert

from app.core.config import settings
from app.models.fts_index import FtsIndex

logger = logging.getLogger(__name__)


def get_tokenizer() -> Any:
    """MeCab 토크나이저 인스턴스 생성 (userdic + decomposition_map 필수)."""
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


def upsert_fts_batch(session: Any, batch: list[dict[str, Any]]) -> int:
    """fts_index ON CONFLICT DO UPDATE 배치 upsert."""
    if not batch:
        return 0

    stmt = insert(FtsIndex).values(batch)
    stmt = stmt.on_conflict_do_update(
        index_elements=["source_id"],
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
    return len(batch)
