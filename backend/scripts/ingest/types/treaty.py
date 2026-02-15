"""
조약 인제스트 설정

data/treaty_v2.json (3,589건)을 대상으로:
- 벡터 DB: 조약요약 1문서=1벡터
- PostgreSQL: 원문 전체 + FTS 인덱스
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any, Optional

_backend_root = Path(__file__).parent.parent.parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))

from app.models.ingest.treaty_document import TreatyDocument  # noqa: E402
from scripts.embedding_common.schema import create_chunk  # noqa: E402
from scripts.ingest.config import (  # noqa: E402
    IngestConfig,
    get_source_path,
    register_config,
)

_DEFAULT_SOURCE = get_source_path("treaty")


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


def _orm_factory(item: dict[str, Any]) -> TreatyDocument:
    """JSON item → TreatyDocument 인스턴스"""
    return TreatyDocument(
        serial_number=item.get("조약일련번호", ""),
        treaty_number=item.get("조약번호"),
        treaty_name_kr=item.get("조약명_한글", ""),
        treaty_name_en=item.get("조약명_영문"),
        treaty_type_code=item.get("조약구분코드"),
        counterpart_country=item.get("체결대상국가"),
        counterpart_country_kr=item.get("체결대상국가한글"),
        country_code=item.get("국가코드"),
        bilateral_field_code=item.get("양자조약분야코드"),
        bilateral_field=item.get("양자조약분야명"),
        signing_date=_parse_date(item.get("서명일자")),
        signing_place=item.get("서명장소"),
        effective_date=_parse_date(item.get("발효일자")),
        parliament_approval=item.get("국회비준동의여부"),
        parliament_approval_date=_parse_date(item.get("국회비준동의일자")),
        cabinet_review_date=_parse_date(item.get("국무회의심의일자")),
        cabinet_review_session=item.get("국무회의심의회차"),
        presidential_approval_date=_parse_date(item.get("대통령재가일자")),
        gazette_date=_parse_date(item.get("관보게재일자")),
        content=item.get("조약내용"),
        note=item.get("비고"),
        ai_summary=item.get("조약요약"),
    )


# ---------------------------------------------------------------------------
# 벡터 DB 함수
# ---------------------------------------------------------------------------


def _vector_metadata_fn(
    item: dict[str, Any],
    vector: list[float],
) -> dict[str, Any]:
    """JSON item + embedding vector → LanceDB record dict"""
    signing_date = str(item.get("서명일자", "") or "")
    return create_chunk(
        data_type="조약",
        source_id=str(item.get("조약일련번호", "")),
        title=item.get("조약명_한글", "") or "",
        content=item.get("조약요약", "") or "",
        vector=vector,
        source_name=item.get("체결대상국가한글", "") or "",
        date=signing_date if signing_date else None,
        chunk_index=0,
        total_chunks=1,
    )


# ---------------------------------------------------------------------------
# FTS 함수 (JSON 소스)
# ---------------------------------------------------------------------------


def _fulltext_fn(item: dict[str, Any]) -> str:
    """JSON item → FTS용 원문 텍스트 concat"""
    parts: list[str] = []

    name_kr = item.get("조약명_한글")
    if name_kr:
        parts.append(f"[{name_kr}]")

    name_en = item.get("조약명_영문")
    if name_en:
        parts.append(name_en)

    country = item.get("체결대상국가한글")
    if country:
        parts.append(f"체결국: {country}")

    content = item.get("조약내용")
    if content:
        parts.append(content)

    return "\n".join(parts)


def _fts_metadata_fn(item: dict[str, Any]) -> dict[str, Any]:
    """JSON item → fts_index 메타데이터 dict"""
    signing_date = str(item.get("서명일자", "") or "")
    return {
        "source_id": str(item.get("조약일련번호", "")),
        "data_type": "조약",
        "title": item.get("조약명_한글", "") or "",
        "date": signing_date if signing_date else None,
        "source_name": item.get("체결대상국가한글"),
        "case_number": item.get("조약번호"),
    }


# ---------------------------------------------------------------------------
# FTS 함수 (ORM 소스 — DB 재빌드용)
# ---------------------------------------------------------------------------


def _orm_fulltext_fn(row: Any) -> str:
    """TreatyDocument ORM 인스턴스 → FTS용 원문 텍스트 concat"""
    parts: list[str] = []

    if row.treaty_name_kr:
        parts.append(f"[{row.treaty_name_kr}]")
    if row.treaty_name_en:
        parts.append(row.treaty_name_en)
    if row.counterpart_country_kr:
        parts.append(f"체결국: {row.counterpart_country_kr}")
    if row.content:
        parts.append(row.content)

    return "\n".join(parts)


def _orm_fts_metadata_fn(row: Any) -> dict[str, Any]:
    """TreatyDocument ORM 인스턴스 → fts_index 메타데이터 dict"""
    date_str = (
        row.signing_date.strftime("%Y%m%d")
        if row.signing_date
        else None
    )
    return {
        "source_id": row.serial_number,
        "data_type": "조약",
        "title": row.treaty_name_kr or "",
        "date": date_str,
        "source_name": row.counterpart_country_kr,
        "case_number": row.treaty_number,
    }


# ---------------------------------------------------------------------------
# 설정 등록
# ---------------------------------------------------------------------------

TREATY_CONFIG = IngestConfig(
    name="treaty",
    data_type_label="조약",
    source_path=_DEFAULT_SOURCE,
    id_field="조약일련번호",
    summary_field="조약요약",
    title_field="조약명_한글",
    orm_class=TreatyDocument,
    orm_id_attr="serial_number",
    orm_factory_fn=_orm_factory,
    vector_metadata_fn=_vector_metadata_fn,
    fulltext_fn=_fulltext_fn,
    fts_metadata_fn=_fts_metadata_fn,
    orm_fulltext_fn=_orm_fulltext_fn,
    orm_fts_metadata_fn=_orm_fts_metadata_fn,
)

register_config(TREATY_CONFIG)
