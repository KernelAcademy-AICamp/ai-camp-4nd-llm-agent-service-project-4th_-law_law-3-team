"""
법령해석례 인제스트 설정

data/ingest_source/legislation_v1.json (8,597건)을 대상으로:
- 벡터 DB: 해석례요약 1문서=1벡터
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

from app.models.ingest.legislation_document import LegislationDocument  # noqa: E402
from scripts.embedding_common.schema import create_chunk  # noqa: E402
from scripts.ingest.config import DATA_DIR, IngestConfig, register_config  # noqa: E402

_DEFAULT_SOURCE = DATA_DIR / "ingest_source" / "legislation_v1.json"


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


def _orm_factory(item: dict[str, Any]) -> LegislationDocument:
    """JSON item → LegislationDocument 인스턴스"""
    return LegislationDocument(
        serial_number=item.get("법령해석례일련번호", ""),
        case_name=item.get("안건명"),
        case_number=item.get("안건번호"),
        interpretation_date=_parse_date(item.get("해석일자")),
        registration_date=item.get("등록일시"),
        interpretation_agency_code=item.get("해석기관코드"),
        interpretation_agency=item.get("해석기관명"),
        inquiry_agency_code=item.get("질의기관코드"),
        inquiry_agency=item.get("질의기관명"),
        management_agency_code=item.get("관리기관코드"),
        inquiry=item.get("질의요지"),
        answer=item.get("회답"),
        reason=item.get("이유"),
        ai_summary=item.get("해석례요약"),
    )


# ---------------------------------------------------------------------------
# 벡터 DB 함수
# ---------------------------------------------------------------------------


def _vector_metadata_fn(
    item: dict[str, Any],
    vector: list[float],
) -> dict[str, Any]:
    """JSON item + embedding vector → LanceDB record dict"""
    return create_chunk(
        data_type="법령해석례",
        source_id=str(item.get("법령해석례일련번호", "")),
        title=item.get("안건명", "") or "",
        content=item.get("해석례요약", "") or "",
        vector=vector,
        source_name=item.get("해석기관명", "") or "",
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

    for field in ("질의요지", "회답", "이유"):
        value = item.get(field)
        if value:
            parts.append(value)

    return "\n".join(parts)


def _fts_metadata_fn(item: dict[str, Any]) -> dict[str, Any]:
    """JSON item → fts_index 메타데이터 dict"""
    reg_date = item.get("등록일시")
    return {
        "source_id": str(item.get("법령해석례일련번호", "")),
        "data_type": "법령해석례",
        "title": item.get("안건명", "") or "",
        "date": str(reg_date) if reg_date else None,
        "source_name": item.get("해석기관명"),
        "case_number": item.get("안건번호"),
    }


# ---------------------------------------------------------------------------
# FTS 함수 (ORM 소스 — DB 재빌드용)
# ---------------------------------------------------------------------------


def _orm_fulltext_fn(row: Any) -> str:
    """LegislationDocument ORM 인스턴스 → FTS용 원문 텍스트 concat"""
    parts: list[str] = []

    if row.case_name:
        parts.append(f"[{row.case_name}]")
    if row.inquiry:
        parts.append(row.inquiry)
    if row.answer:
        parts.append(row.answer)
    if row.reason:
        parts.append(row.reason)

    return "\n".join(parts)


def _orm_fts_metadata_fn(row: Any) -> dict[str, Any]:
    """LegislationDocument ORM 인스턴스 → fts_index 메타데이터 dict"""
    date_str = (
        row.interpretation_date.strftime("%Y%m%d")
        if row.interpretation_date
        else row.registration_date
    )
    return {
        "source_id": row.serial_number,
        "data_type": "법령해석례",
        "title": row.case_name or "",
        "date": date_str,
        "source_name": row.interpretation_agency,
        "case_number": row.case_number,
    }


# ---------------------------------------------------------------------------
# 설정 등록
# ---------------------------------------------------------------------------

LEGISLATION_CONFIG = IngestConfig(
    name="legislation",
    data_type_label="법령해석례",
    source_path=_DEFAULT_SOURCE,
    id_field="법령해석례일련번호",
    summary_field="해석례요약",
    title_field="안건명",
    orm_class=LegislationDocument,
    orm_id_attr="serial_number",
    orm_factory_fn=_orm_factory,
    vector_metadata_fn=_vector_metadata_fn,
    fulltext_fn=_fulltext_fn,
    fts_metadata_fn=_fts_metadata_fn,
    orm_fulltext_fn=_orm_fulltext_fn,
    orm_fts_metadata_fn=_orm_fts_metadata_fn,
)

register_config(LEGISLATION_CONFIG)
