"""
중앙환경분쟁조정위원회 결정례 인제스트 설정

data/decisions_committee/dec_comm_중앙환경분쟁조정위원회_v2.json 대상:
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

from app.models.ingest.dec_environment_document import (  # noqa: E402
    DecEnvironmentDocument,
)
from scripts.ingest.types._dec_comm_common import register_dec_comm  # noqa: E402

_COMMITTEE = "중앙환경분쟁조정위원회"


# ---------------------------------------------------------------------------
# ORM 팩토리
# ---------------------------------------------------------------------------


def _orm_factory(item: dict[str, Any]) -> DecEnvironmentDocument:
    """JSON item -> DecEnvironmentDocument 인스턴스"""
    return DecEnvironmentDocument(
        serial_number=item.get("결정문일련번호", ""),
        case_name=item.get("사건명"),
        decision_number=item.get("의결번호"),
        ruling=item.get("주문"),
        evaluation_opinion=item.get("평가의견"),
        party_claims=item.get("당사자주장"),
        fact_investigation=item.get("사실조사결과"),
        case_overview=item.get("사건의개요"),
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

    for field in ("주문", "평가의견", "당사자주장", "사실조사결과", "사건의개요"):
        value = item.get(field)
        if value:
            parts.append(value)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# FTS 함수 (ORM 소스 - DB 재빌드용)
# ---------------------------------------------------------------------------


def _orm_fulltext_fn(row: Any) -> str:
    """DecEnvironmentDocument ORM 인스턴스 -> FTS용 원문 텍스트 concat"""
    parts: list[str] = []

    if row.case_name:
        parts.append(f"[{row.case_name}]")
    if row.ruling:
        parts.append(row.ruling)
    if row.evaluation_opinion:
        parts.append(row.evaluation_opinion)
    if row.party_claims:
        parts.append(row.party_claims)
    if row.fact_investigation:
        parts.append(row.fact_investigation)
    if row.case_overview:
        parts.append(row.case_overview)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 설정 등록
# ---------------------------------------------------------------------------

register_dec_comm(
    name="dec_environment",
    title_field="사건명",
    date_field=None,
    committee_name=_COMMITTEE,
    orm_class=DecEnvironmentDocument,
    orm_factory_fn=_orm_factory,
    fulltext_fn=_fulltext_fn,
    orm_fulltext_fn=_orm_fulltext_fn,
    case_number_field=None,
)
