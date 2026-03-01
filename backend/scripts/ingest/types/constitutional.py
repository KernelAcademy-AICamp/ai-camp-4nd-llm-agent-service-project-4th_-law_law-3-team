"""
헌법재판소 결정례 인제스트 설정

data/constitutional_v2.json (31,718건)을 대상으로:
- 벡터 DB: 심판례요약 1문서=1벡터
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

from app.models.ingest.constitutional_document import (  # noqa: E402
    ConstitutionalDocument,
)
from scripts.embedding_common.schema import create_chunk  # noqa: E402
from scripts.ingest.config import (  # noqa: E402
    IngestConfig,
    get_source_path,
    register_config,
)

_DEFAULT_SOURCE = get_source_path("constitutional")


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


def _stringify(value: Any) -> Optional[str]:
    """dict/list 값을 문자열로 변환 (str/None은 그대로 반환)"""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(f"{k}: {v}" for k, v in value.items())
    if isinstance(value, list):
        return "\n".join(str(v) for v in value)
    return str(value)


def _orm_factory(item: dict[str, Any]) -> ConstitutionalDocument:
    """JSON item → ConstitutionalDocument 인스턴스"""
    return ConstitutionalDocument(
        serial_number=item.get("헌재결정례일련번호", ""),
        case_number=item.get("사건번호"),
        case_name=item.get("사건명"),
        case_type=item.get("사건종류명"),
        case_type_code=item.get("사건종류코드"),
        decision_date=_parse_date(item.get("종국일자")),
        court_division_code=item.get("재판부구분코드"),
        summary=_stringify(item.get("판시사항")),
        reasoning=_stringify(item.get("결정요지")),
        ruling=_stringify(item.get("주문")),
        full_text=_stringify(item.get("전문")),
        reason=_stringify(item.get("이유")),
        reference_provisions=_stringify(item.get("심판대상조문")),
        reference_statutes=_stringify(item.get("참조조문")),
        reference_cases=_stringify(item.get("참조판례")),
        ai_summary=_stringify(item.get("심판례요약")),
    )


# ---------------------------------------------------------------------------
# 벡터 DB 함수
# ---------------------------------------------------------------------------


def _vector_metadata_fn(
    item: dict[str, Any],
    vector: list[float],
) -> dict[str, Any]:
    """JSON item + embedding vector → LanceDB record dict"""
    decision_date = str(item.get("종국일자", "") or "")
    return create_chunk(
        data_type="헌재결정례",
        source_id=str(item.get("헌재결정례일련번호", "")),
        title=item.get("사건명", "") or "",
        content=item.get("심판례요약", "") or "",
        vector=vector,
        source_name="헌법재판소",
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

    for field in ("판시사항", "결정요지"):
        value = _stringify(item.get(field))
        if value:
            parts.append(value)

    return "\n".join(parts)


def _fts_metadata_fn(item: dict[str, Any]) -> dict[str, Any]:
    """JSON item → fts_index 메타데이터 dict"""
    decision_date = str(item.get("종국일자", "") or "")
    return {
        "source_id": str(item.get("헌재결정례일련번호", "")),
        "data_type": "헌재결정례",
        "title": item.get("사건명", "") or "",
        "date": decision_date if decision_date else None,
        "source_name": "헌법재판소",
        "case_number": item.get("사건번호"),
    }


# ---------------------------------------------------------------------------
# FTS 함수 (ORM 소스 — DB 재빌드용)
# ---------------------------------------------------------------------------


def _orm_fulltext_fn(row: Any) -> str:
    """ConstitutionalDocument ORM 인스턴스 → FTS용 원문 텍스트 concat"""
    parts: list[str] = []

    if row.case_name:
        parts.append(f"[{row.case_name}]")
    if row.summary:
        parts.append(row.summary)
    if row.reasoning:
        parts.append(row.reasoning)

    return "\n".join(parts)


def _orm_fts_metadata_fn(row: Any) -> dict[str, Any]:
    """ConstitutionalDocument ORM 인스턴스 → fts_index 메타데이터 dict"""
    date_str = (
        row.decision_date.strftime("%Y%m%d")
        if row.decision_date
        else None
    )
    return {
        "source_id": row.serial_number,
        "data_type": "헌재결정례",
        "title": row.case_name or "",
        "date": date_str,
        "source_name": "헌법재판소",
        "case_number": row.case_number,
    }


# ---------------------------------------------------------------------------
# 설정 등록
# ---------------------------------------------------------------------------

CONSTITUTIONAL_CONFIG = IngestConfig(
    name="constitutional",
    data_type_label="헌재결정례",
    source_path=_DEFAULT_SOURCE,
    id_field="헌재결정례일련번호",
    summary_field="심판례요약",
    title_field="사건명",
    orm_class=ConstitutionalDocument,
    orm_id_attr="serial_number",
    orm_factory_fn=_orm_factory,
    vector_metadata_fn=_vector_metadata_fn,
    fulltext_fn=_fulltext_fn,
    fts_metadata_fn=_fts_metadata_fn,
    orm_fulltext_fn=_orm_fulltext_fn,
    orm_fts_metadata_fn=_orm_fts_metadata_fn,
)

register_config(CONSTITUTIONAL_CONFIG)
