"""
특별행정심판 재결례 인제스트 설정

data/ingest_source/special_admin_appeal/ 디렉토리 (2파일)을 대상으로:
- 벡터 DB: 심판례요약 1문서=1벡터
- PostgreSQL: 원문 전체 + FTS 인덱스

디렉토리형 데이터: 조세심판원, 해양안전심판원 등 기관별 JSON 파일.
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

from app.models.ingest.special_admin_appeal_document import (  # noqa: E402
    SpecialAdminAppealDocument,
)
from scripts.embedding_common.schema import create_chunk  # noqa: E402
from scripts.ingest.config import DATA_DIR, IngestConfig, register_config  # noqa: E402

_DEFAULT_SOURCE = DATA_DIR / "ingest_source" / "special_admin_appeal"


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


def _orm_factory(item: dict[str, Any]) -> SpecialAdminAppealDocument:
    """JSON item → SpecialAdminAppealDocument 인스턴스"""
    return SpecialAdminAppealDocument(
        serial_number=item.get("특별행정심판재결례일련번호", ""),
        case_name=item.get("사건명"),
        case_number=item.get("재결번호"),
        decision_date=_parse_date(item.get("의결일자")),
        adjudication_agency=item.get("재결청"),
        case_type=item.get("재결례유형명"),
        ruling=item.get("주문"),
        claim=item.get("청구취지"),
        reason=item.get("이유"),
        # 조세심판원 특화 필드
        adjudication_summary=item.get("재결요지"),
        related_rulings=item.get("참조결정"),
        following_rulings=item.get("따른결정"),
        tax_category=item.get("세목"),
        related_law=item.get("관련법령"),
        # 해양안전심판원 특화 필드
        vessel_type=item.get("선박유형"),
        accident_type=item.get("사고유형"),
        tribunal_location=item.get("해심위치"),
        related_persons=item.get("해양사고관련자"),
        appendix=item.get("별지"),
        retrial_notice=item.get("재심청구안내"),
        ai_summary=item.get("심판례요약"),
    )


# ---------------------------------------------------------------------------
# 벡터 DB 함수
# ---------------------------------------------------------------------------


def _vector_metadata_fn(
    item: dict[str, Any],
    vector: list[float],
) -> dict[str, Any]:
    """JSON item + embedding vector → LanceDB record dict"""
    decision_date = str(item.get("의결일자", "") or "")
    return create_chunk(
        data_type="특별행정심판",
        source_id=str(item.get("특별행정심판재결례일련번호", "")),
        title=item.get("사건명", "") or "",
        content=item.get("심판례요약", "") or "",
        vector=vector,
        source_name=item.get("재결청", "") or "",
        date=decision_date if decision_date else None,
        chunk_index=0,
        total_chunks=1,
    )


# ---------------------------------------------------------------------------
# FTS 함수 (JSON 소스)
# ---------------------------------------------------------------------------


def _fulltext_fn(item: dict[str, Any]) -> str:
    """JSON item → FTS용 원문 텍스트 concat"""
    parts: list[str] = []

    case_name = item.get("사건명")
    if case_name:
        parts.append(f"[{case_name}]")

    case_number = item.get("재결번호")
    if case_number:
        parts.append(f"재결번호: {case_number}")

    for field in ("주문", "청구취지", "이유", "재결요지"):
        value = item.get(field)
        if value:
            parts.append(value)

    return "\n".join(parts)


def _fts_metadata_fn(item: dict[str, Any]) -> dict[str, Any]:
    """JSON item → fts_index 메타데이터 dict"""
    decision_date = str(item.get("의결일자", "") or "")
    return {
        "source_id": str(item.get("특별행정심판재결례일련번호", "")),
        "data_type": "특별행정심판",
        "title": item.get("사건명", "") or "",
        "date": decision_date if decision_date else None,
        "source_name": item.get("재결청"),
        "case_number": item.get("재결번호"),
    }


# ---------------------------------------------------------------------------
# FTS 함수 (ORM 소스 — DB 재빌드용)
# ---------------------------------------------------------------------------


def _orm_fulltext_fn(row: Any) -> str:
    """SpecialAdminAppealDocument ORM 인스턴스 → FTS용 원문 텍스트 concat"""
    parts: list[str] = []

    if row.case_name:
        parts.append(f"[{row.case_name}]")
    if row.case_number:
        parts.append(f"재결번호: {row.case_number}")
    if row.ruling:
        parts.append(row.ruling)
    if row.claim:
        parts.append(row.claim)
    if row.reason:
        parts.append(row.reason)
    if row.adjudication_summary:
        parts.append(row.adjudication_summary)

    return "\n".join(parts)


def _orm_fts_metadata_fn(row: Any) -> dict[str, Any]:
    """SpecialAdminAppealDocument ORM 인스턴스 → fts_index 메타데이터 dict"""
    date_str = (
        row.decision_date.strftime("%Y%m%d")
        if row.decision_date
        else None
    )
    return {
        "source_id": row.serial_number,
        "data_type": "특별행정심판",
        "title": row.case_name or "",
        "date": date_str,
        "source_name": row.adjudication_agency,
        "case_number": row.case_number,
    }


# ---------------------------------------------------------------------------
# 설정 등록
# ---------------------------------------------------------------------------

SPECIAL_ADMIN_APPEAL_CONFIG = IngestConfig(
    name="special_admin_appeal",
    data_type_label="특별행정심판",
    source_path=_DEFAULT_SOURCE,
    id_field="특별행정심판재결례일련번호",
    summary_field="심판례요약",
    title_field="사건명",
    orm_class=SpecialAdminAppealDocument,
    orm_id_attr="serial_number",
    orm_factory_fn=_orm_factory,
    vector_metadata_fn=_vector_metadata_fn,
    fulltext_fn=_fulltext_fn,
    fts_metadata_fn=_fts_metadata_fn,
    orm_fulltext_fn=_orm_fulltext_fn,
    orm_fts_metadata_fn=_orm_fts_metadata_fn,
)

register_config(SPECIAL_ADMIN_APPEAL_CONFIG)
