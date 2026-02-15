"""
증권선물위원회 결정례 인제스트 설정

data/decisions_committee/dec_comm_증권선물위원회_v2.json 대상:
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

from app.models.ingest.dec_securities_document import (
    DecSecuritiesDocument,  # noqa: E402
)
from scripts.ingest.types._dec_comm_common import register_dec_comm  # noqa: E402

_COMMITTEE = "증권선물위원회"


# ---------------------------------------------------------------------------
# ORM 팩토리
# ---------------------------------------------------------------------------


def _orm_factory(item: dict[str, Any]) -> DecSecuritiesDocument:
    """JSON item -> DecSecuritiesDocument 인스턴스"""
    return DecSecuritiesDocument(
        serial_number=item.get("결정문일련번호", ""),
        case_name=item.get("안건명"),
        decision_number=item.get("의결번호"),
        action_reason=item.get("조치이유"),
        action_content=item.get("조치내용"),
        ai_summary=item.get("결정문요약"),
    )


# ---------------------------------------------------------------------------
# FTS 함수 (JSON 소스)
# ---------------------------------------------------------------------------


def _fulltext_fn(item: dict[str, Any]) -> str:
    """JSON item -> FTS용 원문 텍스트 concat"""
    parts: list[str] = []

    case_name = item.get("안건명")
    if case_name:
        parts.append(f"[{case_name}]")

    action_reason = item.get("조치이유")
    if action_reason:
        parts.append(action_reason)

    action_content = item.get("조치내용")
    if action_content:
        parts.append(action_content)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# FTS 함수 (ORM 소스 - DB 재빌드용)
# ---------------------------------------------------------------------------


def _orm_fulltext_fn(row: Any) -> str:
    """DecSecuritiesDocument ORM 인스턴스 -> FTS용 원문 텍스트 concat"""
    parts: list[str] = []

    if row.case_name:
        parts.append(f"[{row.case_name}]")
    if row.action_reason:
        parts.append(row.action_reason)
    if row.action_content:
        parts.append(row.action_content)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 설정 등록
# ---------------------------------------------------------------------------

register_dec_comm(
    name="dec_securities",
    title_field="안건명",
    date_field=None,
    committee_name=_COMMITTEE,
    orm_class=DecSecuritiesDocument,
    orm_factory_fn=_orm_factory,
    fulltext_fn=_fulltext_fn,
    orm_fulltext_fn=_orm_fulltext_fn,
    case_number_field=None,
)
