"""
고용보험심사위원회 결정례 인제스트 설정

data/decisions_committee/dec_comm_고용보험심사위원회_v2.json 대상:
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

from app.models.ingest.dec_employment_document import (  # noqa: E402
    DecEmploymentDocument,
)
from scripts.ingest.types._dec_comm_common import register_dec_comm  # noqa: E402

_COMMITTEE = "고용보험심사위원회"


# ---------------------------------------------------------------------------
# ORM 팩토리
# ---------------------------------------------------------------------------


def _orm_factory(item: dict[str, Any]) -> DecEmploymentDocument:
    """JSON item -> DecEmploymentDocument 인스턴스"""
    return DecEmploymentDocument(
        serial_number=item.get("결정문일련번호", ""),
        case_name=item.get("사건명"),
        case_number=item.get("사건번호"),
        case_classification=item.get("사건의분류"),
        decision_date=item.get("의결일자"),
        resolution_type=item.get("의결서종류"),
        ruling=item.get("주문"),
        reason=item.get("이유"),
        claim=item.get("청구취지"),
        petitioner=item.get("청구인"),
        respondent=item.get("피청구인"),
        overview=item.get("개요"),
        organization_name=item.get("기관명"),
        ai_summary=item.get("결정문요약"),
    )


# ---------------------------------------------------------------------------
# FTS 함수 (JSON 소스)
# ---------------------------------------------------------------------------


def _fulltext_fn(item: dict[str, Any]) -> str:
    """JSON item -> FTS용 원문 텍스트 concat"""
    parts: list[str] = []

    case_name = item.get("사건명")
    if case_name:
        parts.append(f"[{case_name}]")

    for field in ("청구취지", "개요"):
        value = item.get(field)
        if value:
            parts.append(value)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# FTS 함수 (ORM 소스 -- DB 재빌드용)
# ---------------------------------------------------------------------------


def _orm_fulltext_fn(row: Any) -> str:
    """DecEmploymentDocument ORM 인스턴스 -> FTS용 원문 텍스트 concat"""
    parts: list[str] = []

    if row.case_name:
        parts.append(f"[{row.case_name}]")
    if row.claim:
        parts.append(row.claim)
    if row.overview:
        parts.append(row.overview)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 설정 등록
# ---------------------------------------------------------------------------

register_dec_comm(
    name="dec_employment",
    title_field="사건명",
    date_field="의결일자",
    committee_name=_COMMITTEE,
    orm_class=DecEmploymentDocument,
    orm_factory_fn=_orm_factory,
    fulltext_fn=_fulltext_fn,
    orm_fulltext_fn=_orm_fulltext_fn,
    case_number_field="사건번호",
)
