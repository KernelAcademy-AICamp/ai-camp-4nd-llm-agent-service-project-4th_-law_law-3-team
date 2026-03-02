"""
산업재해보상보험재심사위원회 결정례 인제스트 설정

data/decisions_committee/dec_comm_산업재해보상위험재심사위원회_v2.json 대상:
- 벡터 DB: 결정문요약 1문서=1벡터
- PostgreSQL: 원문 전체 + FTS 인덱스

NOTE: 이 위원회는 사건명/안건명/제목 필드가 없음.
      title_field=None으로 설정하여 _dec_comm_common이
      case_number를 벡터 title로 사용하도록 함.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_backend_root = Path(__file__).parent.parent.parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))

from app.models.ingest.dec_industrial_document import (  # noqa: E402
    DecIndustrialDocument,
)
from scripts.ingest.types._dec_comm_common import register_dec_comm  # noqa: E402

_COMMITTEE = "산업재해보상보험재심사위원회"


# ---------------------------------------------------------------------------
# ORM 팩토리
# ---------------------------------------------------------------------------


def _orm_factory(item: dict[str, Any]) -> DecIndustrialDocument:
    """JSON item -> DecIndustrialDocument 인스턴스"""
    return DecIndustrialDocument(
        serial_number=item.get("결정문일련번호", ""),
        case_number=item.get("사건번호"),
        case_label=item.get("사건"),
        case_major_category=item.get("사건대분류"),
        case_mid_category=item.get("사건중분류"),
        case_sub_category=item.get("사건소분류"),
        decision_date=item.get("의결일자"),
        ruling=item.get("주문"),
        reason=item.get("이유"),
        issue=item.get("쟁점"),
        claim=item.get("청구취지"),
        petitioner=item.get("청구인"),
        original_authority=item.get("원처분기관"),
        document_provision_type=item.get("문서제공구분"),
        ai_summary=item.get("결정문요약"),
    )


# ---------------------------------------------------------------------------
# FTS 함수 (JSON 소스)
# ---------------------------------------------------------------------------


def _fulltext_fn(item: dict[str, Any]) -> str:
    """JSON item -> FTS용 원문 텍스트 concat"""
    parts: list[str] = []

    # 사건명칭 우선, 없으면 사건번호를 제목 대체로 사용
    case_label = item.get("사건명칭")
    case_number = item.get("사건번호")
    title = case_label or case_number
    if title:
        parts.append(f"[{title}]")

    for field in ("사건대분류명", "사건중분류명", "사건소분류명"):
        value = item.get(field)
        if value:
            parts.append(value)

    for field in ("쟁점", "청구취지"):
        value = item.get(field)
        if value:
            parts.append(value)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# FTS 함수 (ORM 소스 -- DB 재빌드용)
# ---------------------------------------------------------------------------


def _orm_fulltext_fn(row: Any) -> str:
    """DecIndustrialDocument ORM 인스턴스 -> FTS용 원문 텍스트 concat"""
    parts: list[str] = []

    title = row.case_label or row.case_number or ""
    if title:
        parts.append(f"[{title}]")
    if row.case_major_category:
        parts.append(row.case_major_category)
    if row.case_mid_category:
        parts.append(row.case_mid_category)
    if row.case_sub_category:
        parts.append(row.case_sub_category)
    if row.issue:
        parts.append(row.issue)
    if row.claim:
        parts.append(row.claim)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 설정 등록
# ---------------------------------------------------------------------------

register_dec_comm(
    name="dec_industrial",
    title_field=None,
    date_field="의결일자",
    committee_name=_COMMITTEE,
    orm_class=DecIndustrialDocument,
    orm_factory_fn=_orm_factory,
    fulltext_fn=_fulltext_fn,
    orm_fulltext_fn=_orm_fulltext_fn,
    case_number_field="사건번호",
)
