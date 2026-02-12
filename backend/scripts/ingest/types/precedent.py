"""
판례 인제스트 설정

data/raw/precedents.json (92,055건, 19필드)을 대상으로:
- 벡터 DB: 판례요약 1문서=1벡터
- PostgreSQL: 원문 전체 + FTS 인덱스
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any, Optional

# 백엔드 app 모듈 import를 위한 경로 추가
_backend_root = Path(__file__).parent.parent.parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))

from app.models.precedent_document import PrecedentDocument  # noqa: E402
from scripts.embedding_common.schema import create_precedent_chunk  # noqa: E402
from scripts.ingest.config import DATA_DIR, IngestConfig, register_config  # noqa: E402

# 기본 데이터 소스 경로
_DEFAULT_SOURCE = DATA_DIR / "raw" / "precedents.json"


# ---------------------------------------------------------------------------
# ORM 팩토리: JSON item → PrecedentDocument 인스턴스 (적재용 SSOT)
# ---------------------------------------------------------------------------


def _parse_date(date_str: Optional[str]) -> Optional[date]:
    """날짜 문자열 파싱 (YYYYMMDD 또는 YYYY-MM-DD)"""
    if not date_str:
        return None

    date_str = str(date_str).strip()

    # 숫자만 있는 경우 (20170731)
    if date_str.isdigit() and len(date_str) == 8:
        try:
            return date(
                int(date_str[:4]),
                int(date_str[4:6]),
                int(date_str[6:8]),
            )
        except ValueError:
            return None

    # ISO 형식 (2017-07-31)
    try:
        return date.fromisoformat(date_str[:10])
    except (ValueError, IndexError):
        return None


def _orm_factory(item: dict[str, Any]) -> PrecedentDocument:
    """JSON item → PrecedentDocument 인스턴스"""
    return PrecedentDocument(
        serial_number=item.get("판례정보일련번호", ""),
        case_name=item.get("사건명"),
        case_number=item.get("사건번호"),
        decision_date=_parse_date(item.get("선고일자")),
        court_name=item.get("법원명"),
        case_type=item.get("사건종류명"),
        judgment_type=item.get("판결유형"),
        summary=item.get("판시사항"),
        reasoning=item.get("판결요지"),
        ruling=item.get("주문"),
        claim=item.get("청구취지"),
        full_reason=item.get("이유"),
        full_text=item.get("판례내용"),
        ai_summary=item.get("판례요약"),
        reference_provisions=item.get("참조조문"),
        reference_cases=item.get("참조판례"),
    )


def _vector_metadata_fn(
    item: dict[str, Any],
    vector: list[float],
) -> dict[str, Any]:
    """JSON item + embedding vector → LanceDB record dict"""
    source_id = str(item.get("판례정보일련번호", ""))
    title = item.get("사건명", "") or ""
    content = item.get("판례요약", "") or ""
    decision_date = str(item.get("선고일자", "") or "")
    court_name = item.get("법원명", "") or ""

    return create_precedent_chunk(
        source_id=source_id,
        chunk_index=0,
        title=title,
        content=content,
        vector=vector,
        decision_date=decision_date,
        court_name=court_name,
        total_chunks=1,
        case_number=item.get("사건번호"),
        case_type=item.get("사건종류명"),
        judgment_type=item.get("판결유형"),
        judgment_status=item.get("판결상태"),
        reference_provisions=item.get("참조조문"),
        reference_cases=item.get("참조판례"),
    )


def _fulltext_fn(item: dict[str, Any]) -> str:
    """JSON item → FTS용 원문 텍스트 concat"""
    parts: list[str] = []

    case_name = item.get("사건명")
    if case_name:
        parts.append(f"[{case_name}]")

    case_number = item.get("사건번호")
    if case_number:
        parts.append(f"사건번호: {case_number}")

    for field in ("판시사항", "판결요지", "주문", "청구취지", "이유"):
        value = item.get(field)
        if value:
            parts.append(value)

    ref_provisions = item.get("참조조문")
    if ref_provisions:
        parts.append(f"참조조문: {ref_provisions}")

    ref_cases = item.get("참조판례")
    if ref_cases:
        parts.append(f"참조판례: {ref_cases}")

    return "\n".join(parts)


def _fts_metadata_fn(item: dict[str, Any]) -> dict[str, Any]:
    """JSON item → fts_index 메타데이터 dict"""
    decision_date = str(item.get("선고일자", "") or "")

    return {
        "source_id": str(item.get("판례정보일련번호", "")),
        "data_type": "판례",
        "title": item.get("사건명", "") or "",
        "date": decision_date if decision_date else None,
        "source_name": item.get("법원명"),
        "case_number": item.get("사건번호"),
    }


def _orm_fulltext_fn(row: Any) -> str:
    """PrecedentDocument ORM 인스턴스 → FTS용 원문 텍스트 concat"""
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


def _orm_fts_metadata_fn(row: Any) -> dict[str, Any]:
    """PrecedentDocument ORM 인스턴스 → fts_index 메타데이터 dict"""
    date_str = (
        row.decision_date.strftime("%Y%m%d")
        if row.decision_date
        else None
    )

    return {
        "source_id": row.serial_number,
        "data_type": "판례",
        "title": row.case_name or "",
        "date": date_str,
        "source_name": row.court_name,
        "case_number": row.case_number,
    }


PRECEDENT_CONFIG = IngestConfig(
    name="precedent",
    data_type_label="판례",
    source_path=_DEFAULT_SOURCE,
    id_field="판례정보일련번호",
    summary_field="판례요약",
    title_field="사건명",
    orm_class=PrecedentDocument,
    orm_id_attr="serial_number",
    orm_factory_fn=_orm_factory,
    vector_metadata_fn=_vector_metadata_fn,
    fulltext_fn=_fulltext_fn,
    fts_metadata_fn=_fts_metadata_fn,
    orm_fulltext_fn=_orm_fulltext_fn,
    orm_fts_metadata_fn=_orm_fts_metadata_fn,
)

# 자동 등록
register_config(PRECEDENT_CONFIG)
