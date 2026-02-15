"""
부처 유권해석 인제스트 설정

data/interpretation_ministry/ 디렉토리 (28파일)을 대상으로:
- 벡터 DB: 해석요약 1문서=1벡터
- PostgreSQL: 원문 전체 + FTS 인덱스

디렉토리형 데이터: 동일 스키마의 JSON 파일이 부처별로 분리되어 있음.
source_path를 디렉토리로 지정하면 _load_json()이 자동으로 모든 파일을 합산.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any, Optional

_backend_root = Path(__file__).parent.parent.parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))

from app.models.ingest.interpretation_ministry_document import (  # noqa: E402
    InterpretationMinistryDocument,
)
from scripts.embedding_common.schema import create_chunk  # noqa: E402
from scripts.ingest.config import (  # noqa: E402
    IngestConfig,
    get_source_path,
    register_config,
)

_DEFAULT_SOURCE = get_source_path("interpretation_ministry")


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------


def _parse_date(date_str: Optional[str]) -> Optional[date]:
    """날짜 문자열 파싱 (YYYYMMDD 또는 YYYY-MM-DD)"""
    if not date_str:
        return None

    date_str = str(date_str).strip()

    if date_str.isdigit() and len(date_str) == 8:
        try:
            return date(
                int(date_str[:4]),
                int(date_str[4:6]),
                int(date_str[6:8]),
            )
        except ValueError:
            return None

    try:
        return date.fromisoformat(date_str[:10])
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# ORM 팩토리
# ---------------------------------------------------------------------------


def _orm_factory(item: dict[str, Any]) -> InterpretationMinistryDocument:
    """JSON item → InterpretationMinistryDocument 인스턴스"""
    return InterpretationMinistryDocument(
        serial_number=item.get("법령해석일련번호", ""),
        case_name=item.get("안건명"),
        case_number=item.get("안건번호"),
        interpretation_date=_parse_date(item.get("해석일자")),
        inquiry=item.get("질의요지"),
        related_law=item.get("관련법령"),
        answer=item.get("회답"),
        reason=item.get("이유"),
        business_field=item.get("업무분야"),
        ministry_name=item.get("__source_group__"),
        ai_summary=item.get("해석요약"),
    )


# ---------------------------------------------------------------------------
# 벡터 DB 함수
# ---------------------------------------------------------------------------


def _vector_metadata_fn(
    item: dict[str, Any],
    vector: list[float],
) -> dict[str, Any]:
    """JSON item + embedding vector → LanceDB record dict"""
    interpretation_date = str(item.get("해석일자", "") or "")
    ministry = item.get("__source_group__", "")
    return create_chunk(
        data_type="부처유권해석",
        source_id=str(item.get("법령해석일련번호", "")),
        title=item.get("안건명", "") or "",
        content=item.get("해석요약", "") or "",
        vector=vector,
        source_name=ministry or "",
        date=interpretation_date if interpretation_date else None,
        chunk_index=0,
        total_chunks=1,
    )


# ---------------------------------------------------------------------------
# FTS 함수 (JSON 소스)
# ---------------------------------------------------------------------------


def _fulltext_fn(item: dict[str, Any]) -> str:
    """JSON item → FTS용 원문 텍스트 concat"""
    parts: list[str] = []

    case_name = item.get("안건명")
    if case_name:
        parts.append(f"[{case_name}]")

    ministry = item.get("__source_group__")
    if ministry:
        parts.append(f"부처: {ministry}")

    for field in ("질의요지", "회답", "이유"):
        value = item.get(field)
        if value:
            parts.append(value)

    related_law = item.get("관련법령")
    if related_law:
        parts.append(f"관련법령: {related_law}")

    return "\n".join(parts)


def _fts_metadata_fn(item: dict[str, Any]) -> dict[str, Any]:
    """JSON item → fts_index 메타데이터 dict"""
    interpretation_date = str(item.get("해석일자", "") or "")
    return {
        "source_id": str(item.get("법령해석일련번호", "")),
        "data_type": "부처유권해석",
        "title": item.get("안건명", "") or "",
        "date": interpretation_date if interpretation_date else None,
        "source_name": item.get("__source_group__"),
        "case_number": None,
    }


# ---------------------------------------------------------------------------
# FTS 함수 (ORM 소스 — DB 재빌드용)
# ---------------------------------------------------------------------------


def _orm_fulltext_fn(row: Any) -> str:
    """InterpretationMinistryDocument ORM 인스턴스 → FTS용 원문 텍스트 concat"""
    parts: list[str] = []

    if row.case_name:
        parts.append(f"[{row.case_name}]")
    if row.ministry_name:
        parts.append(f"부처: {row.ministry_name}")
    if row.inquiry:
        parts.append(row.inquiry)
    if row.answer:
        parts.append(row.answer)
    if row.reason:
        parts.append(row.reason)
    if row.related_law:
        parts.append(f"관련법령: {row.related_law}")

    return "\n".join(parts)


def _orm_fts_metadata_fn(row: Any) -> dict[str, Any]:
    """InterpretationMinistryDocument ORM 인스턴스 → fts_index 메타데이터 dict"""
    date_str = (
        row.interpretation_date.strftime("%Y%m%d")
        if row.interpretation_date
        else None
    )
    return {
        "source_id": row.serial_number,
        "data_type": "부처유권해석",
        "title": row.case_name or "",
        "date": date_str,
        "source_name": row.ministry_name,
        "case_number": None,
    }


# ---------------------------------------------------------------------------
# 설정 등록
# ---------------------------------------------------------------------------

INTERPRETATION_MINISTRY_CONFIG = IngestConfig(
    name="interpretation_ministry",
    data_type_label="부처유권해석",
    source_path=_DEFAULT_SOURCE,
    id_field="법령해석일련번호",
    summary_field="해석요약",
    title_field="안건명",
    orm_class=InterpretationMinistryDocument,
    orm_id_attr="serial_number",
    orm_factory_fn=_orm_factory,
    vector_metadata_fn=_vector_metadata_fn,
    fulltext_fn=_fulltext_fn,
    fts_metadata_fn=_fts_metadata_fn,
    orm_fulltext_fn=_orm_fulltext_fn,
    orm_fts_metadata_fn=_orm_fts_metadata_fn,
)

register_config(INTERPRETATION_MINISTRY_CONFIG)
