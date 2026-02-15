"""
노동위원회 결정례 인제스트 설정

data/decisions_committee/dec_comm_노동위원회_v2.json 대상:
- 벡터 DB: 결정문요약 1문서=1벡터
- PostgreSQL: 원문 전체 + FTS 인덱스
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_backend_root = Path(__file__).parent.parent.parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))

from app.models.ingest.dec_labor_document import DecLaborDocument  # noqa: E402
from scripts.ingest.types._dec_comm_common import register_dec_comm  # noqa: E402

_COMMITTEE = "노동위원회"


# ---------------------------------------------------------------------------
# ORM 팩토리
# ---------------------------------------------------------------------------


def _orm_factory(item: dict[str, Any]) -> DecLaborDocument:
    """JSON item -> DecLaborDocument 인스턴스"""
    return DecLaborDocument(
        serial_number=item.get("결정문일련번호", ""),
        case_name=item.get("제목"),
        case_number=item.get("사건번호"),
        decision_date=item.get("등록일"),
        judgment_matter=item.get("판정사항"),
        judgment_summary=item.get("판정요지"),
        judgment_result=item.get("판정결과"),
        full_text=item.get("내용"),
        data_category=item.get("자료구분"),
        department=item.get("담당부서"),
        organization_name=item.get("기관명"),
        ai_summary=item.get("결정문요약"),
    )


# ---------------------------------------------------------------------------
# FTS 함수 (JSON 소스)
# ---------------------------------------------------------------------------


def _fulltext_fn(item: dict[str, Any]) -> str:
    """JSON item -> FTS용 원문 텍스트 concat"""
    parts: list[str] = []

    case_name = item.get("제목")
    if case_name:
        parts.append(f"[{case_name}]")

    case_number = item.get("사건번호")
    if case_number:
        parts.append(case_number)

    for field in ("판정사항", "판정요지", "판정결과", "내용"):
        value = item.get(field)
        if value:
            parts.append(value)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# FTS 함수 (ORM 소스 -- DB 재빌드용)
# ---------------------------------------------------------------------------


def _orm_fulltext_fn(row: Any) -> str:
    """DecLaborDocument ORM 인스턴스 -> FTS용 원문 텍스트 concat"""
    parts: list[str] = []

    if row.case_name:
        parts.append(f"[{row.case_name}]")
    if row.case_number:
        parts.append(row.case_number)
    if row.judgment_matter:
        parts.append(row.judgment_matter)
    if row.judgment_summary:
        parts.append(row.judgment_summary)
    if row.judgment_result:
        parts.append(row.judgment_result)
    if row.full_text:
        parts.append(row.full_text)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 설정 등록
# ---------------------------------------------------------------------------

register_dec_comm(
    name="dec_labor",
    title_field="제목",
    date_field="등록일",
    committee_name=_COMMITTEE,
    orm_class=DecLaborDocument,
    orm_factory_fn=_orm_factory,
    fulltext_fn=_fulltext_fn,
    orm_fulltext_fn=_orm_fulltext_fn,
    case_number_field="사건번호",
)
